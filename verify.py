#!/usr/bin/env python3
"""
Verify this repository.

Two jobs, and the first one is the point of the file.

NUMBERS. Every figure quoted in README.md is listed below with the code path
that produces it. If a number in the README and a number in the code ever
disagree, this script fails. A README is a claim; this makes it a testable one.

HYGIENE. Checks that no data file is about to be committed, that no absolute
path or credential-shaped string has crept in, and that no em dash has appeared
in prose.

    python verify.py              # everything, about a minute
    python verify.py --fast       # skip the Cox models
    python verify.py --hygiene    # hygiene only, no data needed

Exit code 0 means every check passed.
"""

import argparse
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The numbers README.md quotes. Left side is a path into the results structure,
# right side is (expected value, tolerance).
#
# Tolerances are set to the precision the README prints at: a rate quoted as
# "3.02%" is checked to 5e-3, a hazard ratio quoted as "1.774" to 5e-4. They are
# not there to absorb a real change, and a real change should fail this file and
# then be corrected in BOTH places.
# ---------------------------------------------------------------------------
EXPECTED = [
    # How 345,779 exam rows become 233,647 patients, and the published count.
    ("_provenance.load_stats.rows_in_file", 345779, 0),
    ("_provenance.load_stats.patients_in_file", 233770, 0),
    ("_provenance.load_stats.rows_without_outcome", 112132, 0),
    ("_provenance.load_stats.patients_without_outcome_row", 123, 0),
    ("_provenance.load_stats.rows_unparsable", 0, 0),
    ("cohort_vs_published.published_n", 218169, 0),
    ("cohort_vs_published.published_deaths", 8118, 0),
    ("cohort_vs_published.extra_patients_here", 15478, 0),
    ("cohort_vs_published.extra_deaths_here", 223, 0),
    ("cohort.min_age", 17, 0),
    ("cohort.n_older_than_85", 13, 0),

    ("cohort.n_patients", 233647, 0),
    ("cohort.deaths", 8341, 0),
    ("cohort.death_rate_pct", 3.57, 5e-3),
    ("cohort.male_pct", 40.5, 5e-2),
    ("cohort.mean_age", 50.74, 5e-3),
    ("cohort.sd_age", 19.82, 5e-3),
    ("cohort.mean_predicted_age", 51.90, 5e-3),
    ("cohort.sd_predicted_age", 18.61, 5e-3),
    ("cohort.mae_years", 8.45, 5e-3),
    ("cohort.median_followup_years", 3.46, 5e-3),
    ("cohort.pearson_r_predicted_vs_chronological", 0.8381, 5e-5),
    # Why the Kaplan-Meier horizons are 2 and 4 years.
    ("cohort.pct_followed_beyond_years.2", 77.2, 5e-2),
    ("cohort.pct_followed_beyond_years.4", 40.5, 5e-2),
    ("cohort.pct_followed_beyond_years.5", 27.2, 5e-2),

    ("confound.pearson_r_pad_vs_age", -0.3842, 5e-5),
    ("confound.ols_slope", -0.2131, 5e-5),
    ("confound.ols_intercept", 11.9755, 5e-5),

    # Section 2, the crude table, every cell.
    ("raw.older.n", 56400, 0),
    ("raw.older.deaths", 1702, 0),
    ("raw.older.death_rate_pct", 3.02, 5e-3),
    ("raw.older.ci95_pct.0", 2.88, 5e-3),
    ("raw.older.ci95_pct.1", 3.16, 5e-3),
    ("raw.older.mean_age", 42.2, 5e-2),
    ("raw.within.n", 134126, 0),
    ("raw.within.deaths", 4667, 0),
    ("raw.within.death_rate_pct", 3.48, 5e-3),
    ("raw.within.ci95_pct.0", 3.38, 5e-3),
    ("raw.within.ci95_pct.1", 3.58, 5e-3),
    ("raw.within.mean_age", 50.4, 5e-2),
    ("raw.younger.n", 43121, 0),
    ("raw.younger.deaths", 1972, 0),
    ("raw.younger.death_rate_pct", 4.57, 5e-3),
    ("raw.younger.ci95_pct.0", 4.38, 5e-3),
    ("raw.younger.ci95_pct.1", 4.77, 5e-3),
    ("raw.younger.mean_age", 62.9, 5e-2),

    ("corrected.older.n", 47263, 0),
    ("corrected.older.deaths", 2431, 0),
    ("corrected.older.death_rate_pct", 5.14, 5e-3),
    ("corrected.older.ci95_pct.0", 4.95, 5e-3),
    ("corrected.older.ci95_pct.1", 5.35, 5e-3),
    ("corrected.older.mean_age", 52.7, 5e-2),
    ("corrected.within.n", 139150, 0),
    ("corrected.within.deaths", 4698, 0),
    ("corrected.within.death_rate_pct", 3.38, 5e-3),
    ("corrected.within.ci95_pct.0", 3.28, 5e-3),
    ("corrected.within.ci95_pct.1", 3.47, 5e-3),
    ("corrected.within.mean_age", 49.6, 5e-2),
    ("corrected.younger.n", 47234, 0),
    ("corrected.younger.deaths", 1212, 0),
    ("corrected.younger.death_rate_pct", 2.57, 5e-3),
    ("corrected.younger.ci95_pct.0", 2.43, 5e-3),
    ("corrected.younger.ci95_pct.1", 2.71, 5e-3),
    ("corrected.younger.mean_age", 52.2, 5e-2),

    # Shares each cut selects, and the share-matched cut points.
    ("raw.older.share_pct", 24.14, 5e-3),
    ("raw.younger.share_pct", 18.46, 5e-3),
    ("corrected.older.share_pct", 20.23, 5e-3),
    ("corrected.younger.share_pct", 20.22, 5e-3),
    ("share_matched.upper_cut_years", 6.620, 5e-4),
    ("share_matched.lower_cut_years", -8.582, 5e-4),
    ("share_matched.corrected.older.n", 56400, 0),
    ("share_matched.corrected.younger.n", 43121, 0),
    ("share_matched.corrected.older.mean_age", 53.1, 5e-2),
    ("share_matched.corrected.younger.mean_age", 52.7, 5e-2),
]

# Section 2, the follow-up table: mean follow-up, deaths per 1,000 person-years,
# and Kaplan-Meier cumulative mortality with its interval at 2 and 4 years, for
# the older and younger groups under each of the three cuts. Generated rather
# than written out 48 times; the expected values are still typed by hand below.
_FOLLOWUP = {
    # group path: (mean follow-up, per 1000 PY, KM 2y, lo, hi, KM 4y, lo, hi)
    "raw.older": (3.62, 8.34, 1.86, 1.75, 1.98, 3.26, 3.10, 3.44),
    "raw.younger": (3.73, 12.28, 2.73, 2.58, 2.90, 4.79, 4.56, 5.02),
    "corrected.older": (3.65, 14.07, 3.13, 2.97, 3.29, 5.52, 5.29, 5.76),
    "corrected.younger": (3.76, 6.83, 1.56, 1.44, 1.68, 2.71, 2.54, 2.88),
    "share_matched.corrected.older": (3.65, 13.92, 3.09, 2.95, 3.24, 5.50, 5.28, 5.72),
    "share_matched.corrected.younger": (3.76, 6.91, 1.56, 1.44, 1.68, 2.73, 2.56, 2.92),
}
_FOLLOWUP_FIELDS = ("mean_followup_years", "deaths_per_1000_py",
                    "km_cum_mortality_pct.2.pct", "km_cum_mortality_pct.2.ci95_pct.0",
                    "km_cum_mortality_pct.2.ci95_pct.1", "km_cum_mortality_pct.4.pct",
                    "km_cum_mortality_pct.4.ci95_pct.0", "km_cum_mortality_pct.4.ci95_pct.1")
for _group, _values in _FOLLOWUP.items():
    for _field, _want in zip(_FOLLOWUP_FIELDS, _values):
        EXPECTED.append(("%s.%s" % (_group, _field), _want, 5e-3))

EXPECTED += [
    # Section 2, older against younger: crude risk ratio and person-year rate
    # ratio, each with its interval, under each cut.
    ("raw.older_vs_younger_rate_ratio", 0.66, 5e-3),
    ("raw.older_vs_younger_rate_ratio_ci95.0", 0.62, 5e-3),
    ("raw.older_vs_younger_rate_ratio_ci95.1", 0.70, 5e-3),
    ("raw.older_vs_younger_py_rate_ratio", 0.68, 5e-3),
    ("raw.older_vs_younger_py_rate_ratio_ci95.0", 0.64, 5e-3),
    ("raw.older_vs_younger_py_rate_ratio_ci95.1", 0.73, 5e-3),
    ("corrected.older_vs_younger_rate_ratio", 2.00, 5e-3),
    ("corrected.older_vs_younger_rate_ratio_ci95.0", 1.87, 5e-3),
    ("corrected.older_vs_younger_rate_ratio_ci95.1", 2.15, 5e-3),
    ("corrected.older_vs_younger_py_rate_ratio", 2.06, 5e-3),
    ("corrected.older_vs_younger_py_rate_ratio_ci95.0", 1.92, 5e-3),
    ("corrected.older_vs_younger_py_rate_ratio_ci95.1", 2.21, 5e-3),
    ("share_matched.corrected.older_vs_younger_rate_ratio", 1.96, 5e-3),
    ("share_matched.corrected.older_vs_younger_rate_ratio_ci95.0", 1.83, 5e-3),
    ("share_matched.corrected.older_vs_younger_rate_ratio_ci95.1", 2.10, 5e-3),
    ("share_matched.corrected.older_vs_younger_py_rate_ratio", 2.02, 5e-3),
    ("share_matched.corrected.older_vs_younger_py_rate_ratio_ci95.0", 1.88, 5e-3),
    ("share_matched.corrected.older_vs_younger_py_rate_ratio_ci95.1", 2.16, 5e-3),

    ("pooled_correction_by_sex.male.n", 94736, 0),
    ("pooled_correction_by_sex.female.n", 138911, 0),
    ("pooled_correction_by_sex.male.mean_padbc", 0.372, 5e-4),
    ("pooled_correction_by_sex.female.mean_padbc", -0.254, 5e-4),
    ("pooled_correction_by_sex.gap_years", 0.626, 5e-4),

    ("within_sex_correction.male.slope", -0.2100, 5e-5),
    ("within_sex_correction.female.slope", -0.2159, 5e-5),
    ("within_sex_correction.male.strata.older.death_rate_pct", 6.57, 5e-3),
    ("within_sex_correction.male.strata.younger.death_rate_pct", 3.48, 5e-3),
    ("within_sex_correction.female.strata.older.death_rate_pct", 4.03, 5e-3),
    ("within_sex_correction.female.strata.younger.death_rate_pct", 2.05, 5e-3),
    ("within_sex_correction.rate_ratio_male", 1.89, 5e-3),
    ("within_sex_correction.rate_ratio_male_ci95.0", 1.72, 5e-3),
    ("within_sex_correction.rate_ratio_male_ci95.1", 2.07, 5e-3),
    ("within_sex_correction.rate_ratio_female", 1.96, 5e-3),
    ("within_sex_correction.rate_ratio_female_ci95.0", 1.77, 5e-3),
    ("within_sex_correction.rate_ratio_female_ci95.1", 2.16, 5e-3),
    ("within_sex_correction.female_over_male_ratio_of_rate_ratios", 1.04, 5e-3),
    ("within_sex_correction.female_over_male_ratio_of_rate_ratios_ci95.0", 0.91, 5e-3),
    ("within_sex_correction.female_over_male_ratio_of_rate_ratios_ci95.1", 1.19, 5e-3),

    ("normal_ecg_only.n", 92138, 0),
    ("normal_ecg_only.deaths", 1358, 0),
    ("normal_ecg_only.pearson_r_pad_vs_age", -0.4160, 5e-5),
    ("normal_ecg_only.raw.older.death_rate_pct", 1.15, 5e-3),
    ("normal_ecg_only.raw.younger.death_rate_pct", 1.94, 5e-3),
    ("normal_ecg_only.raw.older_vs_younger_rate_ratio", 0.59, 5e-3),
    ("normal_ecg_only.raw.older_vs_younger_rate_ratio_ci95.0", 0.50, 5e-3),
    ("normal_ecg_only.raw.older_vs_younger_rate_ratio_ci95.1", 0.70, 5e-3),
    ("normal_ecg_only.raw.older_vs_younger_py_rate_ratio", 0.60, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older.death_rate_pct", 2.27, 5e-3),
    ("normal_ecg_only.corrected_within_normal.younger.death_rate_pct", 1.11, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older_vs_younger_rate_ratio", 2.04, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older_vs_younger_rate_ratio_ci95.0", 1.71, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older_vs_younger_rate_ratio_ci95.1", 2.43, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older_vs_younger_py_rate_ratio", 2.03, 5e-3),
]

EXPECTED_COX = [
    ("models.raw_categorical.terms.PAD > +8 (older).hr", 1.774, 5e-4),
    ("models.raw_categorical.terms.PAD > +8 (older).hr_lo95", 1.674, 5e-4),
    ("models.raw_categorical.terms.PAD > +8 (older).hr_hi95", 1.880, 5e-4),
    ("models.raw_categorical.terms.PAD < -8 (younger).hr", 0.791, 5e-4),
    ("models.raw_categorical.terms.PAD < -8 (younger).hr_lo95", 0.750, 5e-4),
    ("models.raw_categorical.terms.PAD < -8 (younger).hr_hi95", 0.834, 5e-4),
    ("models.raw_categorical.terms.age (per year).hr", 1.074, 5e-4),
    ("models.raw_categorical.terms.male.hr", 1.611, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc > +8 (older).hr", 1.586, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc > +8 (older).hr_lo95", 1.510, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc > +8 (older).hr_hi95", 1.666, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc < -8 (younger).hr", 0.771, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc < -8 (younger).hr_lo95", 0.724, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc < -8 (younger).hr_hi95", 0.822, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc older (matched).hr", 1.538, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc older (matched).hr_lo95",
     1.467, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc older (matched).hr_hi95",
     1.612, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc younger (matched).hr",
     0.784, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc younger (matched).hr_lo95",
     0.734, 5e-4),
    ("models.corrected_categorical_share_matched.terms.PAD_bc younger (matched).hr_hi95",
     0.837, 5e-4),
    # Older-looking against younger-looking directly, adjusted for age and sex.
    ("models.raw_categorical.older_vs_younger.hr", 2.244, 5e-4),
    ("models.raw_categorical.older_vs_younger.hr_lo95", 2.095, 5e-4),
    ("models.raw_categorical.older_vs_younger.hr_hi95", 2.404, 5e-4),
    ("models.corrected_categorical.older_vs_younger.hr", 2.056, 5e-4),
    ("models.corrected_categorical.older_vs_younger.hr_lo95", 1.919, 5e-4),
    ("models.corrected_categorical.older_vs_younger.hr_hi95", 2.203, 5e-4),
    ("models.corrected_categorical_share_matched.older_vs_younger.hr", 1.961, 5e-4),
    ("models.corrected_categorical_share_matched.older_vs_younger.hr_lo95", 1.830, 5e-4),
    ("models.corrected_categorical_share_matched.older_vs_younger.hr_hi95", 2.101, 5e-4),
    ("models.raw_continuous.terms.PAD (per year).hr", 1.0293, 5e-5),
    ("models.corrected_continuous.terms.PAD_bc (per year).hr", 1.0293, 5e-5),
]


def dig(obj, path):
    """
    Walk a dotted path, tolerating keys that themselves contain dots. A segment
    that meets a list is read as an index into it, so "ci95_pct.0" is the lower
    end of an interval.
    """
    cur = obj
    rest = path
    while rest:
        if isinstance(cur, list):
            head, _, rest = rest.partition(".")
            try:
                cur = cur[int(head)]
            except (ValueError, IndexError):
                raise KeyError("no such path: %s (bad index %r)" % (path, head))
            continue
        for key in sorted(cur.keys(), key=len, reverse=True):
            if rest == key:
                return cur[key]
            if rest.startswith(key + "."):
                cur = cur[key]
                rest = rest[len(key) + 1:]
                break
        else:
            raise KeyError("no such path: %s (stuck at %r)" % (path, rest))
    return cur


def km_selftest(base, rows, n=5000):
    """
    The Kaplan-Meier walk in ecg_age_bias.py against the definition written out
    the slow way: for every distinct death time t up to the horizon, recount who
    is at risk and who died, and multiply. Same idea as the Cox fast-vs-naive
    test. A deterministic stride, not random.sample, so every run checks the
    same rows; the stride keeps plenty of tied follow-up times, which is where
    a walk like this goes wrong.
    """
    sub = rows[::max(1, len(rows) // n)][:n]
    fast = base.kaplan_meier(sub)
    worst = 0.0
    for h in base.KM_HORIZONS:
        s, greenwood = 1.0, 0.0
        for t in sorted({r["timey"] for r in sub if r["died"] and r["timey"] <= h}):
            at_risk = sum(1 for r in sub if r["timey"] >= t)
            d = sum(1 for r in sub if r["died"] and r["timey"] == t)
            s *= 1.0 - d / at_risk
            greenwood += d / (at_risk * (at_risk - d))
        se = greenwood ** 0.5 / abs(math.log(s))
        ends = sorted(math.exp(-math.exp(math.log(-math.log(s)) + k * base.Z95 * se))
                      for k in (-1, 1))
        naive = (100.0 * (1.0 - s), 100.0 * (1.0 - ends[1]), 100.0 * (1.0 - ends[0]))
        f = fast["%g" % h]
        got = (f["pct"], f["ci95_pct"][0], f["ci95_pct"][1])
        worst = max([worst] + [abs(a - b) for a, b in zip(got, naive)])
        beyond = sum(1 for r in sub if r["timey"] > h)
        if f["n_followed_beyond"] != beyond:
            worst = float("inf")
    return {"n_subsample": len(sub), "worst_abs_difference_pct_points": worst,
            "passed": worst < 1e-9}


def readme_counts_match(n_numbers, n_selftests):
    """
    README.md says how many numbers and self-tests this file checks. That is a
    number in the README like any other, and it went stale once already, so it
    is checked too. Every place the README states either count must agree.
    """
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    stated_numbers = [int(v) for v in re.findall(r"\((\d+) of them\)", text)]
    pairs = re.findall(r"runs (\d+) numeric assertions and (\d+) self-tests", text)
    stated_numbers += [int(a) for a, _ in pairs]
    stated_selftests = [int(b) for _, b in pairs]
    good = (bool(stated_numbers) and bool(stated_selftests)
            and all(v == n_numbers for v in stated_numbers)
            and all(v == n_selftests for v in stated_selftests))
    print("\nREADME's COUNT OF CHECKS")
    print("  %-8s numbers: README says %s, ran %d; self-tests: README says %s, ran %d"
          % ("ok" if good else "MISMATCH", stated_numbers or "nothing", n_numbers,
             stated_selftests or "nothing", n_selftests))
    return good


def check_values(results, expected, label):
    ok = True
    print("\n%s" % label)
    for path, want, tol in expected:
        try:
            got = dig(results, path)
        except KeyError as exc:
            print("  MISSING  %-62s %s" % (path, exc))
            ok = False
            continue
        good = abs(got - want) <= tol
        print("  %-8s %-62s want %-12s got %s"
              % ("ok" if good else "MISMATCH", path, want,
                 got if isinstance(got, int) else "%.6f" % got))
        ok = ok and good
    return ok


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
TEXT_SUFFIXES = {".py", ".md", ".txt", ".cff", ".yml", ".yaml", ".toml", ".cfg",
                 ".sh", ".ps1", ".json", ".gitignore", ".in", ""}
SKIP_DIRS = {".git", "data", "results", "__pycache__", ".venv", "venv"}

# EVERY quantifier below is BOUNDED, and that is not a style preference.
# The email pattern was first written with unbounded "+" quantifiers either side
# of the @. On a single 200,000 character line that takes 170 seconds, because
# the engine tries every start position and backtracks across the whole line:
# the cost is quadratic in line length. Bounding the quantifiers brings the same
# line down to 0.09 seconds, a factor of about 1900. A scanner that hangs on a
# minified file or an embedded blob is a scanner nobody runs.
SECRET_PATTERNS = [
    ("generic api key assignment",
     re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|token)\b[ \t]{0,8}"
                r"[=:][ \t]{0,8}['\"][A-Za-z0-9_\-]{12,200}['\"]")),
    ("bearer token", re.compile(r"(?i)\bBearer[ \t]{1,4}[A-Za-z0-9._\-]{20,400}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,255}\b")),
    ("openai style key", re.compile(r"\bsk-[A-Za-z0-9]{20,255}\b")),
    ("groq style key", re.compile(r"\bgsk_[A-Za-z0-9]{20,255}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,4000}\.[A-Za-z0-9_\-]{10,4000}\.")),
    ("absolute windows path", re.compile(r"[A-Za-z]:\\{1,2}[Uu]sers\\")),
    ("unix home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._\-]{1,64}/")),
    ("email address",
     re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}")),
]

# Lines longer than this are not pattern scanned, because the cost of scanning
# them is not worth paying and a very long line in a source repository is itself
# worth looking at. It is REPORTED rather than skipped silently, so this is a
# visible limitation and not a hole.
MAX_SCAN_LINE = 4000

# Em dash, as a literal codepoint and as all three HTML entity spellings. On
# some systems grep will not match U+2014 at all, which is why this is done in
# Python rather than in a shell one-liner.
#
# The pattern is ASSEMBLED rather than written out, so that this file does not
# itself contain an em dash or an entity and therefore does not match itself.
# The alternative, excluding this file from the scan, would mean a real em dash
# typed into the prose here would never be caught. A scanner that cannot check
# itself is one file short of a scanner.
_EM_DASH_ALTERNATIVES = [chr(0x2014)] + ["&" + name for name in
                                         ("mdash;", "#8212;", "#x2014;")]
EM_DASH = re.compile("|".join(_EM_DASH_ALTERNATIVES), re.IGNORECASE)


def walk_every_file(root):
    """
    Every file at every depth. Written as an explicit walk rather than a glob
    because a shallow scan is exactly how a credential gets missed: a scan that
    stops at four directories deep reports CLEAN on a secret that is five deep.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


# ---------------------------------------------------------------------------
# Proving the scanner can fire
#
# A clean scan is worth nothing on its own. This builds a throwaway tree in a
# temporary directory, plants one violation of every class, at five directories
# deep, and asserts the scanner finds all of them. Nothing is planted in the
# repository itself and nothing is left behind.
#
# The planted strings are assembled from fragments so that THIS file does not
# contain any of them literally and therefore does not report itself. They are
# invalid by construction and are not credentials.
# ---------------------------------------------------------------------------
def scanner_selftest():
    import shutil
    import tempfile

    j = "".join
    planted = {
        "github token": j(["gh", "p_", "A" * 36]),
        "aws access key id": j(["AKIA", "ABCDEFGHIJKLMNOP"]),
        "openai style key": j(["sk", "-", "A" * 32]),
        "groq style key": j(["gsk", "_", "B" * 32]),
        "slack token": j(["xox", "b-", "1234567890-abcdef"]),
        "bearer token": j(["Bearer", " ", "C" * 30]),
        "private key block": j(["-----BEGIN", " RSA PRIVATE KEY", "-----"]),
        "jwt": j(["eyJ", "abcdefghijk", ".", "eyJabcdefghijk", ".", "sig"]),
        "generic api key assignment": j(['api_key', ' = ', '"', "D" * 24, '"']),
        "absolute windows path": j(["C", ":", chr(92), "Users", chr(92), "someone"]),
        "unix home path": j(["/home", "/", "someone", "/"]),
        "email address": j(["someone", "@", "example", ".", "org"]),
        "em dash": chr(0x2014),
    }

    root = tempfile.mkdtemp(prefix="ecgscan_")
    try:
        deep = os.path.join(root, "a", "b", "c", "d", "e")
        os.makedirs(deep)
        for i, (label, text) in enumerate(sorted(planted.items())):
            # .md so that the em dash rule applies too.
            with open(os.path.join(deep, "planted_%02d.md" % i), "w",
                      encoding="utf-8") as fh:
                fh.write("a line that should be caught: %s\n" % text)
        # And a data file, plus something over the size limit.
        with open(os.path.join(deep, "leaked.csv"), "w", encoding="utf-8") as fh:
            fh.write("a,b\n1,2\n")
        with open(os.path.join(deep, "big.txt"), "w", encoding="utf-8") as fh:
            fh.write("x" * 1_200_000)

        found = set()
        for path in walk_every_file(root):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if os.path.getsize(path) > 1_000_000:
                found.add("file over 1 MB")
            if rel.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
                found.add("data file inside the repository")
            try:
                text = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                if len(line) > MAX_SCAN_LINE:
                    found.add("long line")
                    continue
                for label, pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        found.add(label)
                if EM_DASH.search(line):
                    found.add("em dash")

        expected = set(planted) | {"file over 1 MB", "long line",
                                   "data file inside the repository"}
        missed = sorted(expected - found)
        return {"expected": len(expected), "found": len(found & expected),
                "missed": missed, "passed": not missed,
                "depth_tested": 5}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def hygiene(root):
    ok = True
    print("\nHYGIENE")

    st = scanner_selftest()
    print("  scanner self-test: planted %d violation classes at %d directories "
          "deep, caught %d" % (st["expected"], st["depth_tested"], st["found"]))
    if st["passed"]:
        print("    every class fires, so a clean result below means something")
    else:
        print("    MISSED: %s" % ", ".join(st["missed"]))
        ok = False

    n_scanned = 0
    n_binary = 0
    findings = []

    for path in walk_every_file(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        suffix = os.path.splitext(path)[1].lower()
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            findings.append((rel, 0, "unreadable", str(exc)))
            continue
        if b"\x00" in raw:
            n_binary += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            n_binary += 1
            continue
        n_scanned += 1

        for lineno, line in enumerate(text.splitlines(), 1):
            if len(line) > MAX_SCAN_LINE:
                findings.append((rel, lineno,
                                 "line over %d characters, NOT pattern scanned"
                                 % MAX_SCAN_LINE, "%d characters" % len(line)))
                continue
            for label, pattern in SECRET_PATTERNS:
                for m in pattern.finditer(line):
                    # Deliberately no exemption for this file. The patterns are
                    # written so that they do not match their own source, which
                    # was checked by running the scan over this file with no
                    # exemption in place. If a credential were ever pasted in
                    # here, it would be reported like any other.
                    findings.append((rel, lineno, label, m.group(0)[:70]))
            if suffix in (".md", ".txt", ".py", ".cff"):
                for m in EM_DASH.finditer(line):
                    findings.append((rel, lineno, "em dash", line.strip()[:70]))

    print("  files scanned as text: %d   binary or undecodable skipped: %d"
          % (n_scanned, n_binary))

    # No data file anywhere in the tree outside data/, and nothing large tracked.
    for path in walk_every_file(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        size = os.path.getsize(path)
        if size > 1_000_000:
            findings.append((rel, 0, "file over 1 MB", "%d bytes" % size))
        if rel.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
            findings.append((rel, 0, "data file inside the repository", ""))

    # And ask git directly, which is the only authority on what would be pushed.
    try:
        tracked = subprocess.run(
            ["git", "-C", root, "ls-files"],
            capture_output=True, text=True, timeout=30)
        if tracked.returncode == 0:
            for line in tracked.stdout.splitlines():
                if line.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
                    findings.append((line, 0, "data file TRACKED BY GIT", ""))
            print("  files tracked by git: %d"
                  % len([l for l in tracked.stdout.splitlines() if l]))
        else:
            print("  git reports: not a repository yet (nothing tracked)")
    except (OSError, subprocess.SubprocessError) as exc:
        print("  git unavailable, skipped the tracked-file check: %s" % exc)

    if findings:
        ok = False
        print("  %d FINDING(S):" % len(findings))
        for rel, lineno, label, detail in findings:
            print("    %s:%s  %s  %s" % (rel, lineno, label, detail))
    else:
        print("  no secrets, no absolute paths, no email addresses, no em dashes,")
        print("  no data file in the tree, nothing over 1 MB.")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify this repository.")
    ap.add_argument("--fast", action="store_true", help="skip the Cox models")
    ap.add_argument("--hygiene", action="store_true", help="hygiene checks only")
    args = ap.parse_args(argv)

    all_ok = True

    if not args.hygiene:
        import ecg_age_bias as base
        base.ensure_data(base.DEFAULT_CSV)
        rows, load_stats = base.load(base.DEFAULT_CSV)
        results = base.analyse(rows)
        # Where ecg_age_bias.py puts them in results.json, so the paths match.
        results["_provenance"] = {"load_stats": load_stats}
        all_ok &= check_values(results, EXPECTED,
                               "NUMBERS QUOTED IN README.md (core analysis)")
        n_numbers = len(EXPECTED)
        n_selftests = 0

        print("\nKAPLAN-MEIER SELF-TEST")
        km = km_selftest(base, rows)
        print("  %-8s fast walk vs definition, n=%d, worst difference %.3g points"
              % ("ok" if km["passed"] else "FAILED", km["n_subsample"],
                 km["worst_abs_difference_pct_points"]))
        all_ok &= km["passed"]
        n_selftests += 1

        if not args.fast:
            import cox_model
            sys.stderr.write("  fitting Cox models, about a minute ...\n")
            code = cox_model.main(["--json",
                                   os.path.join(HERE, "results", "cox.json")])
            if code != 0:
                print("\ncox_model.py exited %d" % code)
                all_ok = False
            else:
                import json
                with open(os.path.join(HERE, "results", "cox.json"),
                          encoding="utf-8") as fh:
                    cox = json.load(fh)
                all_ok &= check_values(cox, EXPECTED_COX,
                                       "NUMBERS QUOTED IN README.md (Cox models)")
                n_numbers += len(EXPECTED_COX)
                # The Cox implementation's own self-tests print PASS or FAIL
                # inside cox_model.py, which exits 0 either way outside
                # --selftest. Read their verdicts from the JSON so that a
                # failed self-test fails this script too.
                print("\nCOX IMPLEMENTATION SELF-TESTS")
                for name, chk in sorted(cox["checks"].items()):
                    if isinstance(chk, dict) and "passed" in chk:
                        good = chk["passed"] is True
                        print("  %-8s %s" % ("ok" if good else "FAILED", name))
                        all_ok &= good
                        n_selftests += 1
                # Also inside continuous_identity's verdict above. Kept as its
                # own line because it is the one a reader of the README is told
                # about by name.
                ident = cox["checks"]["continuous_identity"]
                good = ident["identical_to_6dp"]
                print("  %-8s continuous PAD and PAD_bc hazard ratios identical"
                      % ("ok" if good else "MISMATCH"))
                all_ok &= good
                n_selftests += 1

                # The three attenuation figures the README quotes are DERIVED
                # from the two hazard ratios above, so they cannot drift on
                # their own. They are still checked, because a number can be
                # mistyped in prose without any code changing, and that is the
                # failure mode this whole file exists to catch.
                raw = dig(cox, "models.raw_categorical.terms.PAD > +8 (older).hr")
                cor = dig(cox,
                          "models.corrected_categorical.terms."
                          "PAD_bc > +8 (older).hr")
                derived = [
                    ("attenuation, hazard ratio scale (%)",
                     100 * (1 - cor / raw), 10.6, 5e-2),
                    ("attenuation, log hazard ratio scale (%)",
                     100 * (1 - math.log(cor) / math.log(raw)), 19.6, 5e-2),
                    ("attenuation, excess hazard scale (%)",
                     100 * (1 - (cor - 1) / (raw - 1)), 24.3, 5e-2),
                    ("excess hazard, raw (%)", 100 * (raw - 1), 77.4, 5e-2),
                    ("excess hazard, corrected (%)", 100 * (cor - 1), 58.6, 5e-2),
                ]
                print("\nATTENUATION FIGURES QUOTED IN README.md")
                for label, got, want, tol in derived:
                    good = abs(got - want) <= tol
                    print("  %-8s %-42s want %-7s got %.4f"
                          % ("ok" if good else "MISMATCH", label, want, got))
                    all_ok &= good
                n_numbers += len(derived)

                all_ok &= readme_counts_match(n_numbers, n_selftests)
        else:
            print("\nREADME's count of checks: not compared under --fast, which "
                  "skips the Cox checks it counts")

    all_ok &= hygiene(HERE)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOMETHING FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
