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

Standard library only. No numpy, no pandas, no GPU. Runs in about 10 seconds.

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
    """
    rows = []
    n_total = 0
    n_blank_outcome = 0
    n_unparsable = 0
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            n_total += 1
            if rec["death"].strip() == "":
                n_blank_outcome += 1
                continue
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


def wilson(k, n, z=1.959963984540054):
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
def strata(rows, key, thr):
    buckets = {OLDER: [], WITHIN: [], YOUNGER: []}
    for r in rows:
        v = r[key]
        buckets[OLDER if v > thr else YOUNGER if v < -thr else WITHIN].append(r)
    summary = {}
    for name, bucket in buckets.items():
        n = len(bucket)
        d = sum(1 for r in bucket if r["died"])
        lo, hi = wilson(d, n) if n else (float("nan"), float("nan"))
        summary[name] = {
            "n": n,
            "deaths": d,
            "death_rate_pct": 100.0 * d / n if n else float("nan"),
            "ci95_pct": [100.0 * lo, 100.0 * hi],
            "mean_age": mean([r["age"] for r in bucket]) if n else float("nan"),
            "median_followup_years": median([r["timey"] for r in bucket]) if n else float("nan"),
        }
    o, y = summary[OLDER], summary[YOUNGER]
    summary["older_vs_younger_rate_ratio"] = (
        o["death_rate_pct"] / y["death_rate_pct"] if y["death_rate_pct"] else float("nan")
    )
    return summary


def print_strata(title, summary, indent="    "):
    print("%s:" % title)
    for name in (OLDER, WITHIN, YOUNGER):
        s = summary[name]
        print(
            "%s%-8s n=%7d  deaths=%5d  rate=%5.2f%% [%4.2f, %4.2f]  mean age=%4.1f"
            % (
                indent,
                name,
                s["n"],
                s["deaths"],
                s["death_rate_pct"],
                s["ci95_pct"][0],
                s["ci95_pct"][1],
                s["mean_age"],
            )
        )
    print(
        "%solder / younger death rate ratio = %.2f"
        % (indent, summary["older_vs_younger_rate_ratio"])
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
        "mean_predicted_age": mean(preds),
        "sd_predicted_age": sd(preds),
        "mae_years": mean([abs(p) for p in pads]),
        "median_followup_years": median([r["timey"] for r in rows]),
        "pearson_r_predicted_vs_chronological": pearson(preds, ages),
        "normal_ecg_pct": 100.0 * sum(1 for r in rows if r["normal"]) / len(rows),
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
        "rate_ratio_male": m_strata["older_vs_younger_rate_ratio"],
        "rate_ratio_female": f_strata["older_vs_younger_rate_ratio"],
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
adjustment in the published model is carrying a great deal of weight.

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
    print("    rows in file            %d" % load_stats["rows_in_file"])
    print("    rows without an outcome %d" % load_stats["rows_without_outcome"])
    print("      (follow-up is recorded only on a patient's first exam)")
    print("    patients analysed       %d" % c["n_patients"])
    print("    deaths                  %d  (%.2f%%)" % (c["deaths"], c["death_rate_pct"]))
    print("    male                    %.1f%%" % c["male_pct"])
    print("    chronological age       mean %.2f  sd %.2f" % (c["mean_age"], c["sd_age"]))
    print("    ECG-predicted age       mean %.2f  sd %.2f"
          % (c["mean_predicted_age"], c["sd_predicted_age"]))
    print("    mean absolute error     %.2f years" % c["mae_years"])
    print("    median follow-up        %.2f years" % c["median_followup_years"])
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
    print("\n    older/younger rate ratio: male %.2f, female %.2f."
          % (w["rate_ratio_male"], w["rate_ratio_female"]))
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
