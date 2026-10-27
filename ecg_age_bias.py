#!/usr/bin/env python3
"""
Age-regression bias in the AI ECG age gap: a replication on CODE-15%.

This is a REPLICATION. A linear age-bias correction of the AI-ECG predicted age
was applied by Hirota et al., Int J Cardiol Heart Vasc 44:101172, 2023, doi
10.1016/j.ijcha.2023.101172 (Japanese cohort). The correction implemented here
is the residual form used by Barthels et al., Eur Heart J Digit Health 7(2),
2026, doi 10.1093/ehjdh/ztaf137 (first published 28 November 2025), on a Belgian
cohort. This script repeats Barthels' method on the Brazilian CODE-15% cohort.
See README.md.

Standard library only. No numpy, no pandas, no GPU. Runs in under 20 seconds.

Usage
-----
    python ecg_age_bias.py              # downloads the data if absent, then runs
    python ecg_age_bias.py --download   # force a fresh download first
    python ecg_age_bias.py --json out.json
    python ecg_age_bias.py --threshold 8.0

Data
----
CODE-15%, Ribeiro et al., Zenodo record 4916206, CC BY 4.0.
Only exams.csv is used (35.5 MB). The 46 GB of raw waveforms is not needed:
the neural network's own age predictions are already a column in the CSV.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Data source. Both URLs serve the same object; the API form is the canonical
# one and the /records/ form is kept as a fallback because Zenodo has served
# each of them at different times.
# ---------------------------------------------------------------------------
CSV_URLS = (
    "https://zenodo.org/api/records/4916206/files/exams.csv/content",
    "https://zenodo.org/records/4916206/files/exams.csv?download=1",
)
CSV_MD5 = "0107516d3f63864498fb77d15799cc95"
CSV_BYTES = 35500900

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "exams.csv")

# The 2021 paper split on +/- 8 years because the model's mean absolute error in
# CODE is about 8 years. Kept as the default so the comparison is like for like.
DEFAULT_THRESHOLD = 8.0

# Kaplan-Meier horizons, in years. Chosen from the follow-up distribution, not
# from the results. At 2 years about three quarters of the cohort is still under
# observation. 4 years is just past the median follow-up and about two fifths are
# still observed there. Beyond 5 years fewer than three in ten are, so a later
# horizon would rest on a shrinking minority. The exact shares are written to
# results.json as cohort.pct_followed_beyond_years.
KM_HORIZONS = (2.0, 4.0)

# The CODE-15% cohort as the 2021 paper reports it (its Table 1 and Table 2), so
# that the difference from the public file is computed rather than retyped.
PUBLISHED_COHORT = {"n": 218169, "deaths": 8118}

Z95 = 1.959963984540054

OLDER, WITHIN, YOUNGER = "older", "within", "younger"


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def shown(path):
    """
    A path to print. Relative to the repository when it is inside it.

    Printing an absolute path would put the user's home directory, and so their
    username, into any output they paste into an issue or an email. It also
    makes two runs on two machines produce different text for the same result.
    """
    try:
        rel = os.path.relpath(os.path.abspath(path), HERE)
    except ValueError:  # different drive on Windows
        return os.path.abspath(path)
    if rel.startswith(".."):
        return os.path.abspath(path)
    return rel.replace(os.sep, "/")


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(path):
    """Fetch exams.csv and refuse to continue unless the bytes are the published ones."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    last_error = None
    ok = False
    for url in CSV_URLS:
        try:
            sys.stderr.write("downloading 35.5 MB from %s\n" % url)
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                seen = 0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    seen += len(block)
                    sys.stderr.write("\r  %6.1f MB" % (seen / 1e6))
                sys.stderr.write("\n")
            ok = True
            break
        except Exception as exc:  # network, TLS, 404, redirect loop
            last_error = exc
            sys.stderr.write("  failed: %s\n" % exc)
            if os.path.exists(tmp):
                os.remove(tmp)
    if not ok:
        raise SystemExit(
            "could not download exams.csv from any known URL. Last error: %s\n"
            "Download it by hand from https://zenodo.org/records/4916206 and put "
            "it at %s" % (last_error, shown(path))
        )

    size = os.path.getsize(tmp)
    got = md5_of(tmp)
    if size != CSV_BYTES or got != CSV_MD5:
        os.remove(tmp)
        raise SystemExit(
            "checksum or size mismatch. Expected %d bytes / md5 %s, got %d / %s. "
            "Refusing to analyse a file that is not the published one."
            % (CSV_BYTES, CSV_MD5, size, got)
        )
    os.replace(tmp, path)
    sys.stderr.write("checksum OK: md5 %s\n" % got)


def ensure_data(path, force=False):
    if force or not os.path.exists(path):
        download(path)
        return "downloaded"
    got = md5_of(path)
    if got != CSV_MD5:
        raise SystemExit(
            "%s exists but its md5 is %s, not the published %s. Delete it and "
            "re-run to fetch a clean copy." % (shown(path), got, CSV_MD5)
        )
    return "cached"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def as_bool(s):
    return s.strip().lower() == "true"


def load(path):
    """
    One row per patient.

    In exams.csv the mortality outcome (death) and the follow-up time (timey)
    are populated only on each patient's FIRST exam and are blank on the rest, so
    filtering on a non-empty death field selects exactly the analysable
    first-exam rows and does not double count anyone.

    Patients are counted as well as rows, because the two drops are different
    things: most blank rows are a patient's later exams, but a few patients have
    no row with an outcome at all and so leave the cohort entirely.
    """
    rows = []
    n_total = 0
    n_blank_outcome = 0
    n_unparsable = 0
    patients_in_file = set()
    patients_with_outcome = set()
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            n_total += 1
            patients_in_file.add(rec["patient_id"])
            if rec["death"].strip() == "":
                n_blank_outcome += 1
                continue
            patients_with_outcome.add(rec["patient_id"])
            try:
                age = float(rec["age"])
                pred = float(rec["nn_predicted_age"])
                t = float(rec["timey"])
            except (ValueError, KeyError):
                n_unparsable += 1
                continue
            rows.append(
                {
                    "age": age,
                    "pred": pred,
                    "pad": pred - age,
                    "timey": t,
                    "died": as_bool(rec["death"]),
                    "male": as_bool(rec["is_male"]),
                    "normal": as_bool(rec["normal_ecg"]),
                }
            )
    return rows, {
        "rows_in_file": n_total,
        "rows_without_outcome": n_blank_outcome,
        "rows_unparsable": n_unparsable,
        "rows_analysed": len(rows),
        "patients_in_file": len(patients_in_file),
        "patients_without_outcome_row": len(patients_in_file - patients_with_outcome),
    }


# ---------------------------------------------------------------------------
# Statistics, written out rather than imported so the script has no dependencies
# ---------------------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)


def ols(x, y):
    """Least squares fit of y on x. Returns (slope, intercept)."""
    mx, my = mean(x), mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    slope = sxy / sxx
    return slope, my - slope * mx


def wilson(k, n, z=Z95):
    """
    Wilson score interval for a proportion. Used instead of the normal
    approximation because several strata have death rates near 2 per cent.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


NAN_PAIR = [float("nan"), float("nan")]


def log_interval(est, se_log, z=Z95):
    """95% interval for a positive quantity whose log has standard error se_log."""
    return [est * math.exp(-z * se_log), est * math.exp(z * se_log)]


def risk_ratio(a, n1, c, n2):
    """
    Ratio of two crude risks, (a / n1) / (c / n2), with the standard error of
    its log in the usual large-sample form, sqrt(1/a - 1/n1 + 1/c - 1/n2).
    Returns (ratio, se_log, ci95).
    """
    if not (a and c and n1 and n2):
        return float("nan"), float("nan"), NAN_PAIR
    est = (a / n1) / (c / n2)
    se = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n2)
    return est, se, log_interval(est, se)


def rate_ratio(d1, py1, d2, py2):
    """
    Ratio of two death rates, deaths per person-year, with the standard error of
    its log for Poisson counts, sqrt(1/d1 + 1/d2). Returns (ratio, se_log, ci95).
    """
    if not (d1 and d2 and py1 and py2):
        return float("nan"), float("nan"), NAN_PAIR
    est = (d1 / py1) / (d2 / py2)
    se = math.sqrt(1.0 / d1 + 1.0 / d2)
    return est, se, log_interval(est, se)


def kaplan_meier(rows, horizons=KM_HORIZONS, z=Z95):
    """
    Kaplan-Meier cumulative mortality, 100 * (1 - S(h)), at each horizon h.

    The crude share who died ignores follow-up: a group watched for longer has
    longer in which to die. Kaplan-Meier counts each patient only while they are
    under observation, so groups with different follow-up compare fairly.

    Deaths at a tied time are counted before the censorings at that time, the
    usual convention. The 95% interval uses Greenwood's variance on the
    log(-log S) scale, which keeps both ends between 0 and 100 per cent and is
    the form lifelines reports by default.

    Keyed by the horizon in years. n_followed_beyond is how many patients in the
    group were still under observation after h, which is what the estimate at h
    rests on.
    """
    events = sorted((r["timey"], 1 if r["died"] else 0) for r in rows)
    n = len(events)
    at_risk = n
    s = 1.0
    greenwood = 0.0  # running sum of d / (n (n - d)) over death times
    pending = sorted(horizons)
    out = {}

    def record(h):
        if 0.0 < s < 1.0:
            log_s = math.log(s)
            se = math.sqrt(greenwood) / abs(log_s)
            centre = math.log(-log_s)
            ends = sorted(math.exp(-math.exp(centre + k * z * se)) for k in (-1, 1))
            ci = [100.0 * (1.0 - ends[1]), 100.0 * (1.0 - ends[0])]
        else:
            ci = NAN_PAIR
        out["%g" % h] = {"pct": 100.0 * (1.0 - s), "ci95_pct": ci,
                         "n_followed_beyond": at_risk}

    i = 0
    while i < n:
        t = events[i][0]
        # Every horizon before this time has seen all the deaths it is owed.
        while pending and pending[0] < t:
            record(pending.pop(0))
        j = i
        d = 0
        while j < n and events[j][0] == t:
            d += events[j][1]
            j += 1
        if d:
            s *= 1.0 - d / at_risk
            if at_risk > d:
                greenwood += d / (at_risk * (at_risk - d))
        at_risk -= j - i
        i = j
    for h in pending:
        record(h)
    return out


def share_matched_cuts(rows, ref_key, key, thr):
    """
    Cut points on rows[key] that select EXACTLY as many patients in each tail as
    the +/- thr cut selects on rows[ref_key].

    The raw and corrected gaps have different spreads, and the raw one is not
    centred on zero, so the same +/- 8 year cut takes a different share of the
    cohort from each. Comparing groups of different sizes mixes the effect of
    the correction with the effect of where the cut happened to fall. Matching
    the shares removes the second.

    Each cut sits halfway between the two sorted values either side of it. A tie
    across a cut would make an exact match impossible, and this stops rather
    than quietly selecting a different number of patients.
    """
    n_older = sum(1 for r in rows if r[ref_key] > thr)
    n_younger = sum(1 for r in rows if r[ref_key] < -thr)
    vals = sorted(r[key] for r in rows)
    n = len(vals)
    if not (0 < n_older < n and 0 < n_younger < n):
        raise SystemExit("cannot share-match an empty or full tail")
    a, b = vals[n - n_older - 1], vals[n - n_older]
    c, d = vals[n_younger - 1], vals[n_younger]
    if a == b or c == d:
        raise SystemExit("tied values across a share-matched cut on %s" % key)
    return {
        "n_older": n_older,
        "n_younger": n_younger,
        "older_share_pct": 100.0 * n_older / n,
        "younger_share_pct": 100.0 * n_younger / n,
        "upper_cut_years": 0.5 * (a + b),
        "lower_cut_years": 0.5 * (c + d),
    }


def bias_correct(rows, key_in="pad", key_out="pad_bc", subset=None):
    """
    The correction itself, and it is one line of statistics: regress the age gap
    on chronological age and keep the residual.

    Barthels et al. define PAD_bc as "the residual of PAD regressed on
    chronological age". Fitting on a subset and applying to rows is what makes
    the within-sex variant possible.
    """
    fit_on = rows if subset is None else subset
    slope, intercept = ols([r["age"] for r in fit_on], [r[key_in] for r in fit_on])
    for r in rows:
        r[key_out] = r[key_in] - (intercept + slope * r["age"])
    return slope, intercept


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def strata(rows, key, thr, lower=None):
    """
    Split rows into older (key > thr), within, and younger (key < lower) groups.
    lower defaults to -thr, the symmetric split; the share-matched analysis
    passes its own two cut points.

    A note on names, because two kinds of "rate" now sit side by side.
    death_rate_pct and older_vs_younger_rate_ratio are the CRUDE share of the
    group who died during follow-up, which is a risk and not a rate, and the
    ratio of two of them is a risk ratio. They keep their original names so that
    earlier results stay comparable. They ignore how long each group was
    followed. deaths_per_1000_py is a true rate, deaths over person-years of
    follow-up, and km_cum_mortality_pct handles censoring properly.
    """
    lo_cut = -thr if lower is None else lower
    buckets = {OLDER: [], WITHIN: [], YOUNGER: []}
    for r in rows:
        v = r[key]
        buckets[OLDER if v > thr else YOUNGER if v < lo_cut else WITHIN].append(r)
    summary = {}
    for name, bucket in buckets.items():
        n = len(bucket)
        d = sum(1 for r in bucket if r["died"])
        py = sum(r["timey"] for r in bucket)
        lo, hi = wilson(d, n) if n else (float("nan"), float("nan"))
        rate = 1000.0 * d / py if py else float("nan")
        summary[name] = {
            "n": n,
            "share_pct": 100.0 * n / len(rows),
            "deaths": d,
            "death_rate_pct": 100.0 * d / n if n else float("nan"),
            "ci95_pct": [100.0 * lo, 100.0 * hi],
            "mean_age": mean([r["age"] for r in bucket]) if n else float("nan"),
            "median_followup_years": median([r["timey"] for r in bucket]) if n else float("nan"),
            "person_years": py,
            "mean_followup_years": py / n if n else float("nan"),
            "deaths_per_1000_py": rate,
            # Log-normal approximation to the Poisson interval, fine at the
            # hundreds of deaths every group here has.
            "deaths_per_1000_py_ci95": log_interval(rate, 1.0 / math.sqrt(d)) if d else NAN_PAIR,
            "km_cum_mortality_pct": kaplan_meier(bucket),
        }
    o, y = summary[OLDER], summary[YOUNGER]
    rr, rr_se, rr_ci = risk_ratio(o["deaths"], o["n"], y["deaths"], y["n"])
    summary["older_vs_younger_rate_ratio"] = rr
    summary["older_vs_younger_rate_ratio_se_log"] = rr_se
    summary["older_vs_younger_rate_ratio_ci95"] = rr_ci
    pr, _, pr_ci = rate_ratio(o["deaths"], o["person_years"], y["deaths"], y["person_years"])
    summary["older_vs_younger_py_rate_ratio"] = pr
    summary["older_vs_younger_py_rate_ratio_ci95"] = pr_ci
    return summary


def print_strata(title, summary, indent="    "):
    print("%s:" % title)
    for name in (OLDER, WITHIN, YOUNGER):
        s = summary[name]
        print(
            "%s%-8s n=%7d (%4.1f%%)  deaths=%5d  died=%5.2f%% [%4.2f, %4.2f]  mean age=%4.1f"
            % (
                indent,
                name,
                s["n"],
                s["share_pct"],
                s["deaths"],
                s["death_rate_pct"],
                s["ci95_pct"][0],
                s["ci95_pct"][1],
                s["mean_age"],
            )
        )
        km = "  ".join(
            "KM %sy %4.2f%% [%4.2f, %4.2f]" % (h, v["pct"], v["ci95_pct"][0], v["ci95_pct"][1])
            for h, v in sorted(s["km_cum_mortality_pct"].items(), key=lambda kv: float(kv[0])))
        print(
            "%s%-8s %5.2f per 1000 PY [%5.2f, %5.2f]  mean follow-up %4.2f y  %s"
            % (indent, "", s["deaths_per_1000_py"], s["deaths_per_1000_py_ci95"][0],
               s["deaths_per_1000_py_ci95"][1], s["mean_followup_years"], km)
        )
    print(
        "%solder / younger, crude risk ratio %.2f [%.2f, %.2f]   person-year rate ratio %.2f [%.2f, %.2f]"
        % (indent, summary["older_vs_younger_rate_ratio"],
           summary["older_vs_younger_rate_ratio_ci95"][0],
           summary["older_vs_younger_rate_ratio_ci95"][1],
           summary["older_vs_younger_py_rate_ratio"],
           summary["older_vs_younger_py_rate_ratio_ci95"][0],
           summary["older_vs_younger_py_rate_ratio_ci95"][1])
    )


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------
def analyse(rows, thr=DEFAULT_THRESHOLD):
    res = {"threshold_years": thr}

    ages = [r["age"] for r in rows]
    preds = [r["pred"] for r in rows]
    pads = [r["pad"] for r in rows]
    deaths = sum(1 for r in rows if r["died"])

    # ---- cohort, and the numbers that show this is the file the paper used ----
    res["cohort"] = {
        "n_patients": len(rows),
        "deaths": deaths,
        "death_rate_pct": 100.0 * deaths / len(rows),
        "male_pct": 100.0 * sum(1 for r in rows if r["male"]) / len(rows),
        "mean_age": mean(ages),
        "sd_age": sd(ages),
        # The paper sampled ages 16 to 85, so these say whether an age filter
        # could explain the difference in cohort size. It cannot: almost no one
        # in the public file is outside that range.
        "min_age": min(ages),
        "n_older_than_85": sum(1 for a in ages if a > 85),
        "mean_predicted_age": mean(preds),
        "sd_predicted_age": sd(preds),
        "mae_years": mean([abs(p) for p in pads]),
        "median_followup_years": median([r["timey"] for r in rows]),
        "pct_followed_beyond_years": {
            "%d" % y: 100.0 * sum(1 for r in rows if r["timey"] > y) / len(rows)
            for y in range(1, 7)
        },
        "person_years": sum(r["timey"] for r in rows),
        "pearson_r_predicted_vs_chronological": pearson(preds, ages),
        "normal_ecg_pct": 100.0 * sum(1 for r in rows if r["normal"]) / len(rows),
    }

    # ---- the published cohort is smaller, and by how much ----
    res["cohort_vs_published"] = {
        "published_n": PUBLISHED_COHORT["n"],
        "published_deaths": PUBLISHED_COHORT["deaths"],
        "extra_patients_here": len(rows) - PUBLISHED_COHORT["n"],
        "extra_deaths_here": deaths - PUBLISHED_COHORT["deaths"],
    }

    # ---- the confound ----
    slope, intercept = bias_correct(rows)
    res["confound"] = {
        "pearson_r_pad_vs_age": pearson(pads, ages),
        "ols_slope": slope,
        "ols_intercept": intercept,
        "pearson_r_padbc_vs_age": pearson([r["pad_bc"] for r in rows], ages),
    }

    # ---- what it does to the crude survival split ----
    res["raw"] = strata(rows, "pad", thr)
    res["corrected"] = strata(rows, "pad_bc", thr)

    # ---- the same comparison with the corrected gap cut at the raw cut's shares ----
    cuts = share_matched_cuts(rows, "pad", "pad_bc", thr)
    matched = strata(rows, "pad_bc", cuts["upper_cut_years"], lower=cuts["lower_cut_years"])
    if (matched[OLDER]["n"], matched[YOUNGER]["n"]) != (cuts["n_older"], cuts["n_younger"]):
        raise SystemExit("share-matched groups do not have the raw groups' sizes")
    res["share_matched"] = dict(cuts, corrected=matched)

    # ---- is the POOLED correction sex fair? ----
    males = [r for r in rows if r["male"]]
    females = [r for r in rows if not r["male"]]
    m_mean = mean([r["pad_bc"] for r in males])
    f_mean = mean([r["pad_bc"] for r in females])
    res["pooled_correction_by_sex"] = {
        "male": {"n": len(males), "mean_padbc": m_mean},
        "female": {"n": len(females), "mean_padbc": f_mean},
        "gap_years": abs(m_mean - f_mean),
        "male_strata": strata(males, "pad_bc", thr),
        "female_strata": strata(females, "pad_bc", thr),
    }

    # ---- correct WITHIN each sex instead, and see whether anything changes ----
    m_slope, m_int = bias_correct(males, key_out="pad_bc_sex", subset=males)
    f_slope, f_int = bias_correct(females, key_out="pad_bc_sex", subset=females)
    m_strata = strata(males, "pad_bc_sex", thr)
    f_strata = strata(females, "pad_bc_sex", thr)
    # Do the sexes differ? The two risk ratios come from disjoint groups, so the
    # log of their ratio has variance equal to the sum of the two variances.
    f_over_m = f_strata["older_vs_younger_rate_ratio"] / m_strata["older_vs_younger_rate_ratio"]
    f_over_m_se = math.hypot(f_strata["older_vs_younger_rate_ratio_se_log"],
                             m_strata["older_vs_younger_rate_ratio_se_log"])
    res["within_sex_correction"] = {
        "male": {
            "slope": m_slope,
            "intercept": m_int,
            "mean_padbc_sex": mean([r["pad_bc_sex"] for r in males]),
            "strata": m_strata,
        },
        "female": {
            "slope": f_slope,
            "intercept": f_int,
            "mean_padbc_sex": mean([r["pad_bc_sex"] for r in females]),
            "strata": f_strata,
        },
        # Crude risk ratios, older-looking over younger-looking. See strata().
        "rate_ratio_male": m_strata["older_vs_younger_rate_ratio"],
        "rate_ratio_male_ci95": m_strata["older_vs_younger_rate_ratio_ci95"],
        "rate_ratio_female": f_strata["older_vs_younger_rate_ratio"],
        "rate_ratio_female_ci95": f_strata["older_vs_younger_rate_ratio_ci95"],
        "female_over_male_ratio_of_rate_ratios": f_over_m,
        "female_over_male_ratio_of_rate_ratios_ci95": log_interval(f_over_m, f_over_m_se),
    }

    # ---- the apparently normal ECGs ----
    normals = [r for r in rows if r["normal"]]
    bias_correct(normals, key_out="pad_bc_normal", subset=normals)
    res["normal_ecg_only"] = {
        "n": len(normals),
        "deaths": sum(1 for r in normals if r["died"]),
        "pearson_r_pad_vs_age": pearson(
            [r["pad"] for r in normals], [r["age"] for r in normals]
        ),
        "raw": strata(normals, "pad", thr),
        "corrected_within_normal": strata(normals, "pad_bc_normal", thr),
    }

    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
CAVEAT = """
It does NOT show that the 2021 CODE-15% paper is wrong. That paper fitted a Cox
model ADJUSTED FOR AGE AND SEX and reported HR 1.79 (95% CI 1.69 to 1.90) for the
older-looking group. Every rate above is CRUDE and unadjusted, so it is not
comparable to that headline. Run cox_model.py for the adjusted comparison.

What it does show is that the raw gap is heavily age-confounded, that the crude
direction reverses once the confound is removed, and therefore that the age
adjustment in the published model is carrying a great deal of weight. Part of
the reversal is built in: the corrected groups are age balanced by construction.
The direction of the corrected mortality difference is not built in, and it
survives person-years, Kaplan-Meier and cutting at matched shares.

It is a REPLICATION. Hirota et al. (2023, Japanese cohort) applied a linear bias
correction first, and Barthels et al. published the residual form on a Belgian
cohort in November 2025 and reported r = -0.54 and the same reversal. What is new
here is the cohort, the sex-fairness result in [4], and the null in [5].
"""


def report(res, load_stats, source):
    c = res["cohort"]
    thr = res["threshold_years"]
    print("=" * 74)
    print("AGE-REGRESSION BIAS IN THE AI ECG AGE GAP")
    print("A replication on CODE-15% of Barthels et al., Eur Heart J Digit Health")
    print("2026, doi 10.1093/ehjdh/ztaf137. A linear bias correction was applied")
    print("earlier by Hirota et al., doi 10.1016/j.ijcha.2023.101172.")
    print("=" * 74)

    print("\n[1] COHORT  (exams.csv %s, md5 verified)" % source)
    print("    rows in file            %d  (%d distinct patients)"
          % (load_stats["rows_in_file"], load_stats["patients_in_file"]))
    print("    rows without an outcome %d" % load_stats["rows_without_outcome"])
    print("      (follow-up is recorded only on a patient's first exam; %d patients"
          % load_stats["patients_without_outcome_row"])
    print("      have no row with an outcome at all and so leave the cohort)")
    print("    rows that failed to parse %d" % load_stats["rows_unparsable"])
    print("    patients analysed       %d  (every analysis below uses these, or a"
          % c["n_patients"])
    print("                                    named subgroup of them)")
    p = res["cohort_vs_published"]
    print("    2021 paper's CODE-15%%   %d patients, %d deaths: %d fewer patients"
          % (p["published_n"], p["published_deaths"], p["extra_patients_here"]))
    print("                            and %d fewer deaths than the public file"
          % p["extra_deaths_here"])
    print("    deaths                  %d  (%.2f%%)" % (c["deaths"], c["death_rate_pct"]))
    print("    male                    %.1f%%" % c["male_pct"])
    print("    chronological age       mean %.2f  sd %.2f" % (c["mean_age"], c["sd_age"]))
    print("    ECG-predicted age       mean %.2f  sd %.2f"
          % (c["mean_predicted_age"], c["sd_predicted_age"]))
    print("    mean absolute error     %.2f years" % c["mae_years"])
    print("    median follow-up        %.2f years" % c["median_followup_years"])
    print("    still followed after    "
          + "  ".join("%sy %.1f%%" % (y, v) for y, v in
                      sorted(c["pct_followed_beyond_years"].items(), key=lambda kv: int(kv[0]))))
    print("    r(predicted, chrono)    %+.4f" % c["pearson_r_predicted_vs_chronological"])

    f = res["confound"]
    print("\n[2] THE CONFOUND")
    print("    PAD = ECG-predicted age minus chronological age")
    print("    r(PAD, chronological age)     %+.4f   <- this is the problem"
          % f["pearson_r_pad_vs_age"])
    print("    fitted  PAD = %+.4f * age %+.4f" % (f["ols_slope"], f["ols_intercept"]))
    print("    r(PAD_bc, chronological age)  %+.4f   <- zero by construction"
          % f["pearson_r_padbc_vs_age"])

    print("\n[3] WHAT THAT DOES TO THE CRUDE SURVIVAL SPLIT  (threshold +/- %g years)" % thr)
    print_strata("    RAW PAD, as the 2021 paper thresholded it", res["raw"], indent="      ")
    print()
    print_strata("    BIAS-CORRECTED PAD", res["corrected"], indent="      ")
    print("\n    The sign flips. In the raw split the mean ages are %.1f and %.1f,"
          % (res["raw"][OLDER]["mean_age"], res["raw"][YOUNGER]["mean_age"]))
    print("    so it is sorting patients by age. After correction they are %.1f and %.1f."
          % (res["corrected"][OLDER]["mean_age"], res["corrected"][YOUNGER]["mean_age"]))
    print("    That age balance is BY CONSTRUCTION: the corrected gap is fitted to be")
    print("    uncorrelated with age, so its two tails come out close to age balanced")
    print("    almost automatically. The direction of the mortality difference is NOT")
    print("    forced by the correction. It is what the gap carries beyond age.")

    sm = res["share_matched"]
    print("\n[3b] SHARE-MATCHED: the corrected gap cut at the raw cut's shares")
    print("    The +/- %g cut takes %.2f%% / %.2f%% of the cohort from the raw gap"
          % (thr, res["raw"][OLDER]["share_pct"], res["raw"][YOUNGER]["share_pct"]))
    print("    but %.2f%% / %.2f%% from the corrected one, which has a narrower spread."
          % (res["corrected"][OLDER]["share_pct"], res["corrected"][YOUNGER]["share_pct"]))
    print("    Cutting the corrected gap at > %+.3f and < %+.3f years selects exactly"
          % (sm["upper_cut_years"], sm["lower_cut_years"]))
    print("    the raw groups' sizes, %d and %d." % (sm["n_older"], sm["n_younger"]))
    print_strata("    BIAS-CORRECTED PAD, share-matched", sm["corrected"], indent="      ")

    p = res["pooled_correction_by_sex"]
    print("\n[4] IS THE POOLED CORRECTION SEX FAIR?  (zero would mean yes)")
    print("    mean PAD_bc, male    %+.3f years   (n=%d)"
          % (p["male"]["mean_padbc"], p["male"]["n"]))
    print("    mean PAD_bc, female  %+.3f years   (n=%d)"
          % (p["female"]["mean_padbc"], p["female"]["n"]))
    print("    residual sex gap     %.3f years" % p["gap_years"])
    print("    Male ECGs read older. A correction fitted on everyone leaves that in place.")

    w = res["within_sex_correction"]
    print("\n[5] CORRECT WITHIN EACH SEX INSTEAD")
    print("    male   slope %+.4f  intercept %+.4f  mean PAD_bc %+.6f"
          % (w["male"]["slope"], w["male"]["intercept"], w["male"]["mean_padbc_sex"]))
    print("    female slope %+.4f  intercept %+.4f  mean PAD_bc %+.6f"
          % (w["female"]["slope"], w["female"]["intercept"], w["female"]["mean_padbc_sex"]))
    print()
    print_strata("    male, corrected within sex", w["male"]["strata"], indent="      ")
    print()
    print_strata("    female, corrected within sex", w["female"]["strata"], indent="      ")
    print("\n    older/younger crude risk ratio: male %.2f [%.2f, %.2f], female %.2f [%.2f, %.2f]."
          % (w["rate_ratio_male"], w["rate_ratio_male_ci95"][0], w["rate_ratio_male_ci95"][1],
             w["rate_ratio_female"], w["rate_ratio_female_ci95"][0],
             w["rate_ratio_female_ci95"][1]))
    print("    female / male ratio of the two: %.2f [%.2f, %.2f]; 1 means no difference."
          % (w["female_over_male_ratio_of_rate_ratios"],
             w["female_over_male_ratio_of_rate_ratios_ci95"][0],
             w["female_over_male_ratio_of_rate_ratios_ci95"][1]))
    print("    The slopes are nearly identical, so the pooled unfairness in [4] is an")
    print("    intercept difference and not a slope difference. Correcting within sex")
    print("    leaves the mortality gradient about the same in both. That is a NULL")
    print("    result and it is reported here as one.")

    ne = res["normal_ecg_only"]
    print("\n[6] APPARENTLY NORMAL ECGs ONLY  (n=%d, deaths=%d)" % (ne["n"], ne["deaths"]))
    print("    r(PAD, age) in this subgroup  %+.4f" % ne["pearson_r_pad_vs_age"])
    print_strata("    raw PAD", ne["raw"], indent="      ")
    print()
    print_strata("    corrected within this subgroup", ne["corrected_within_normal"],
                 indent="      ")

    print("\n" + "=" * 74)
    print("WHAT THIS DOES AND DOES NOT SHOW")
    print("=" * 74)
    print(CAVEAT)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Replication of the ECG age-gap bias correction on CODE-15%."
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path to exams.csv")
    ap.add_argument("--download", action="store_true", help="force a fresh download")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="years; the 2021 paper used 8.0")
    ap.add_argument("--json", default=os.path.join(HERE, "results", "results.json"),
                    help="where to write the machine-readable results")
    ap.add_argument("--quiet", action="store_true", help="write the JSON and print nothing")
    args = ap.parse_args(argv)

    source = ensure_data(args.csv, force=args.download)
    rows, load_stats = load(args.csv)
    if not rows:
        raise SystemExit("no analysable rows found in %s" % shown(args.csv))

    res = analyse(rows, thr=args.threshold)
    res["_provenance"] = {
        "dataset": "CODE-15%, Zenodo record 4916206, CC BY 4.0",
        "file": "exams.csv",
        "md5": CSV_MD5,
        "bytes": CSV_BYTES,
        "loaded_from": source,
        "load_stats": load_stats,
        "python": sys.version.split()[0],
    }

    if not args.quiet:
        report(res, load_stats, source)

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    if not args.quiet:
        print("machine-readable results written to %s" % shown(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
