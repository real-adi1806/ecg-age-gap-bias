"""Assert every Germany number the manuscript quotes, against the files the code wrote.
Same idea as ../verify.py: a manuscript that drifts away from the code fails here instead of quietly lying.
Standard library only, so it runs without the data or the network, against the committed results/.

    python verify_germany.py                     numbers against results/
    PAPER_A_MANUSCRIPT=path/to/manuscript.md python verify_germany.py
                                                 also check that each number is printed in the manuscript

Sources: results/germany.json (run_germany.py) and, for the clearly labelled 25 Oct plain-head note only,
results/results.json (analyze.py). Exit code 0 only if every assertion holds.
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = {"g": os.path.join(HERE, "results", "germany.json"), "heads": os.path.join(HERE, "results", "results.json")}
MANUSCRIPT = os.environ.get("PAPER_A_MANUSCRIPT")   # the manuscript is not part of this repository
TEXT_FILES = ["README.md", "expected-output/run-2026-10-30.txt"]
if MANUSCRIPT:
    TEXT_FILES += [MANUSCRIPT, os.path.join(os.path.dirname(MANUSCRIPT), "NUMBERS.md")]
BANNED = ["\u2014", "\u2013", " -- ", "&" + "mdash;", "&" + "ndash;", "&#" + "8212;", "&#" + "8211;"]   # em and en dash, and their HTML spellings

R3, M2, P1, N, S4, F3 = "{:+.3f}", "{:+.2f}", "{:.1f}%", "{:,}", "{:.4f}", "{:.3f}"

def stat_checks(prefix, key, fmt, *printed):
    """point, CI low, CI high of one statistic in an analyse() block; pass None to skip a value not quoted."""
    paths = [f"{prefix}.point.{key}", f"{prefix}.ci.{key}.0", f"{prefix}.ci.{key}.1"]
    return [(p, "g", path, fmt) for p, path in zip(printed, paths) if p is not None]

PRIM, TWO, SENS, SCALE = "primary_first_ecg", "two_step_brazil_first_ecg", "sensitivity_all_ecgs", "scale_matched_first_ecg.brazil_line"
NORM, NOTN, HS = "exploratory_norm_first_ecg", "exploratory_not_norm_first_ecg", "half_split"
CHECKS = [
    # data and reproduction (sections 2, 3, 5.1)
    ("21,799", "g", "ptbxl_records.in_database", N), ("18,869", "g", "ptbxl_records.patients_in_database", N),
    ("293", "g", "ptbxl_records.age_coded_300", "{}"), ("133", "g", "ptbxl_records.age_under_18", "{}"),
    ("21,373", "g", "gate_b.n_ecgs", N), ("18,495", "g", "gate_b.n_patients", N),
    ("42,747", "g", "inputs.ptbxl_files_sha256_checked", N),
    ("52.1%", "g", "sex_coding_check.pct_records_sex0", P1),
    ("173.5", "g", "sex_coding_check.mean_height_cm_sex0", "{:.1f}"), ("161.0", "g", "sex_coding_check.mean_height_cm_sex1", "{:.1f}"),
    ("11.9755", "g", "brazil_line.intercept", S4), ("-0.2131", "g", "brazil_line.slope", S4),
    ("20261028", "g", "provenance.seed", "{}"), ("2,000", "g", "provenance.n_bootstrap", N),
    ("3.12.15", "g", "provenance.python", "{}"), ("2.5.4", "g", "provenance.numpy", "{}"), ("3.1.0", "g", "provenance.pandas", "{}"),
    ("2.14.1", "g", "provenance.torch", "{}"), ("00aad2c0341942837019086cf5d94cd5", "g", "inputs.md5.model.zip", "{}"),
    ("0.0", "g", "ptb_rerun_vs_cache.max_abs_diff", "{:.1f}"),
    ("9.21", "g", "gate_b.mae", "{:.2f}"), ("0.767", "g", "gate_b.pearson_r_pred_age", F3), ("+3.01", "g", "gate_b.mean_gap", M2),
    ("233,770", "g", "gate_a_cached.n", N), ("0.00009", "g", "gate_a_cached.mean_abs_diff", "{:.5f}"),
    ("0.00063", "g", "gate_a_cached.max_abs_diff", "{:.5f}"),
    ("1,000", "g", "gate_a_rerun.n", N), ("0.00009", "g", "gate_a_rerun.mean_abs_diff", "{:.5f}"),
    ("0.00045", "g", "gate_a_rerun.max_abs_diff", "{:.5f}"),
    # primary cohort (5.2)
    ("59.4", "g", f"{PRIM}.cohort.mean_age", "{:.1f}"), ("16.6", "g", f"{PRIM}.cohort.sd_age", "{:.1f}"),
    ("51.5%", "g", f"{PRIM}.cohort.pct_male", P1),
    ("2,491", "g", f"{PRIM}.cohort.band_n.lt40", N), ("2,204", "g", f"{PRIM}.cohort.band_n.40-49", N),
    ("3,811", "g", f"{PRIM}.cohort.band_n.50-59", N), ("4,462", "g", f"{PRIM}.cohort.band_n.60-69", N),
    ("3,504", "g", f"{PRIM}.cohort.band_n.70-79", N), ("2,023", "g", f"{PRIM}.cohort.band_n.ge80", N),
    ("+8.03", "g", f"{PRIM}.point.band_lt40_raw", M2), ("+4.95", "g", f"{PRIM}.point.band_40-49_raw", M2),
    ("+4.46", "g", f"{PRIM}.point.band_50-59_raw", M2), ("+3.27", "g", f"{PRIM}.point.band_60-69_raw", M2),
    ("+0.01", "g", f"{PRIM}.point.band_70-79_raw", M2), ("-5.45", "g", f"{PRIM}.point.band_ge80_raw", M2),
    ("33.4%", "g", f"{PRIM}.point.pct_over_plus8", P1), ("13.7%", "g", f"{PRIM}.point.pct_under_minus8", P1),
    ("+4.69", "g", f"{PRIM}.point.male_corrected", M2), ("+4.48", "g", f"{PRIM}.ci.male_corrected.0", M2), ("+4.91", "g", f"{PRIM}.ci.male_corrected.1", M2),
    ("+2.15", "g", f"{PRIM}.point.female_corrected", M2), ("+1.92", "g", f"{PRIM}.ci.female_corrected.0", M2), ("+2.36", "g", f"{PRIM}.ci.female_corrected.1", M2),
    ("32.8%", "g", f"{TWO}.point.pct_over_plus8", P1), ("13.6%", "g", f"{TWO}.point.pct_under_minus8", P1),
    ("15.498", "g", "german_line_full.intercept", F3), ("14.996", "g", "german_line_full.ci.intercept.0", F3),
    ("16.005", "g", "german_line_full.ci.intercept.1", F3),
    ("-0.2142", "g", "german_line_full.slope", S4), ("-0.2221", "g", "german_line_full.ci.slope.0", S4),
    ("-0.2061", "g", "german_line_full.ci.slope.1", S4),
    ("-0.63", "g", "brazil_reference.40-49.mean", M2), ("-1.55", "g", "brazil_reference.ge80.mean", M2),
    ("+1.14", "g", "brazil_reference.60-69.mean", M2),
    ("-1.33", "g", "brazil_reference.normal_ecg.mean", M2), ("92,138", "g", "brazil_reference.normal_ecg.n", N),
    ("2,813", "g", "brazil_reference.age_level_n_min", N), ("3,564", "g", "brazil_reference.age_level_n_max", N),
    ("13", "g", "brazil_reference.n_older_than_pool", "{}"),
    ("+0.86", "g", "brazil_reference.not_normal_ecg.mean", M2),
    # half split (5.3)
    ("9,247", "g", f"{HS}.n_patients_fit", N), ("9,248", "g", f"{HS}.n_patients_test", N),
    ("15.789", "g", f"{HS}.german_line_on_fit_half.intercept", F3), ("-0.2194", "g", f"{HS}.german_line_on_fit_half.slope", S4),
    # amplitude and scale (5.5)
    ("998", "g", "amplitude_check.n_code15", "{}"), ("1.967", "g", "amplitude_check.matched_scale", F3),
    ("1.59", "g", "amplitude_check.ratio_min", "{:.2f}"), ("2.20", "g", "amplitude_check.ratio_max", "{:.2f}"),
    ("2.04", "g", "amplitude_check.per_lead.II.code15", "{:.2f}"), ("0.93", "g", "amplitude_check.per_lead.II.ptbxl", "{:.2f}"),
    ("-0.1737", "g", "scale_matched_first_ecg.german_line.slope", S4),
    ("-0.1821", "g", "scale_matched_first_ecg.german_line.ci.slope.0", S4),
    ("-0.1653", "g", "scale_matched_first_ecg.german_line.ci.slope.1", S4),
    ("8,581", "g", f"{NORM}.cohort.n_ecgs", N), ("9,914", "g", f"{NOTN}.cohort.n_ecgs", N),
    ("21,373", "g", f"{SENS}.cohort.n_ecgs", N),
    # the 25 Oct plain-head model (Supplement S4), a different model, quoted for the record only
    ("-0.042", "heads", "ptbxl_brazil_fit.methods.(c) plain + post-hoc.r_gap_age", R3),
    ("-0.057", "heads", "ptbxl_brazil_fit.methods.(c) plain + post-hoc.r_ci.0", R3),
    ("-0.027", "heads", "ptbxl_brazil_fit.methods.(c) plain + post-hoc.r_ci.1", R3),
    ("-0.001", "heads", "ptbxl_german_refit.methods.(c) plain + post-hoc.r_gap_age", R3),
    ("-0.022", "heads", "ptbxl_german_refit.methods.(c) plain + post-hoc.r_ci.0", R3),
    ("+0.020", "heads", "ptbxl_german_refit.methods.(c) plain + post-hoc.r_ci.1", R3),
]
CHECKS += (stat_checks(PRIM, "mean_gap_raw", M2, "+2.78", "+2.62", "+2.95")
           + stat_checks(PRIM, "mean_gap_corrected", M2, "+3.46", "+3.31", "+3.61")
           + stat_checks(PRIM, "r_raw", R3, "-0.317", "-0.328", "-0.306")
           + stat_checks(PRIM, "r_corrected", R3, "-0.002", "-0.015", "+0.011")
           + stat_checks(PRIM, "band_lt40_corrected", M2, "+2.29", "+1.94", "+2.66")
           + stat_checks(PRIM, "band_40-49_corrected", M2, "+2.52", "+2.04", "+2.97")
           + stat_checks(PRIM, "band_50-59_corrected", M2, "+4.18", "+3.83", "+4.55")
           + stat_checks(PRIM, "band_60-69_corrected", M2, "+5.05", "+4.73", "+5.36")
           + stat_checks(PRIM, "band_70-79_corrected", M2, "+3.84", "+3.53", "+4.18")
           + stat_checks(PRIM, "band_ge80_corrected", M2, "+0.40", "-0.01", "+0.85")
           + stat_checks(PRIM, "band_range_corrected", "{:.2f}", "4.65", "4.06", "5.18")
           + stat_checks(PRIM, "male_minus_female", M2, "+2.55", "+2.24", "+2.84")
           + stat_checks(PRIM, "male_minus_female_band_adjusted", M2, "+2.24", "+1.93", "+2.53")
           + stat_checks(TWO, "mean_gap_corrected", M2, "+3.30", "+3.15", "+3.45")
           + stat_checks(TWO, "r_corrected", R3, "+0.006", "-0.007", "+0.018")
           + stat_checks(TWO, "band_lt40_corrected", M2, "+2.69", "+2.32", "+3.03")
           + stat_checks(TWO, "band_40-49_corrected", M2, "+3.11", "+2.64", "+3.57")
           + stat_checks(TWO, "band_50-59_corrected", M2, "+3.57", "+3.21", "+3.93")
           + stat_checks(TWO, "band_60-69_corrected", M2, "+3.91", "+3.58", "+4.23")
           + stat_checks(TWO, "band_70-79_corrected", M2, "+3.42", "+3.10", "+3.74")
           + stat_checks(TWO, "band_ge80_corrected", M2, "+2.22", "+1.80", "+2.66")
           + stat_checks(TWO, "band_range_corrected", "{:.2f}", "1.70", "1.22", "2.25")
           + stat_checks(TWO, "male_minus_female", M2, "+2.35", "+2.04", "+2.65"))
for blk, mean, r, lt40, b6069, ge80, spread, mf, over, under in (
        ("brazil_line", ("+3.48", "+3.26", "+3.70"), ("+0.006", "-0.012", "+0.025"), ("+2.14", None, None), ("+5.08", None, None),
         ("+0.86", None, None), "4.21", "+2.81", "33.6%", "13.6%"),
        ("german_line", ("+0.04", "-0.28", "+0.34"), ("+0.016", "-0.010", "+0.042"), ("-1.49", "-2.12", "-0.82"),
         ("+1.67", "+1.12", "+2.15"), ("-2.42", "-3.17", "-1.70"), "4.09", "+2.80", "23.0%", "22.4%"),
        ("brazil_two_step", ("+3.32", "+3.10", "+3.55"), ("+0.013", "-0.005", "+0.032"), ("+2.57", None, None), ("+3.94", None, None),
         ("+2.67", None, None), "1.37", "+2.60", "33.0%", "13.5%"),
        ("german_two_step", ("+0.03", "-0.29", "+0.33"), ("+0.016", "-0.011", "+0.041"), ("-0.25", "-0.98", "+0.50"),
         ("+0.07", "-0.61", "+0.71"), ("+0.92", "+0.00", "+1.76"), "1.23", "+2.41", "23.0%", "22.0%")):
    pre = f"{HS}.{blk}"
    CHECKS += (stat_checks(pre, "mean_gap_corrected", M2, *mean) + stat_checks(pre, "r_corrected", R3, *r)
               + stat_checks(pre, "band_lt40_corrected", M2, *lt40) + stat_checks(pre, "band_60-69_corrected", M2, *b6069)
               + stat_checks(pre, "band_ge80_corrected", M2, *ge80)
               + [(spread, "g", f"{pre}.point.band_range_corrected", "{:.2f}"), (mf, "g", f"{pre}.point.male_minus_female", M2),
                  (over, "g", f"{pre}.point.pct_over_plus8", P1), (under, "g", f"{pre}.point.pct_under_minus8", P1)])
for pre, mean, r, spread, mf, over in (
        (SENS, ("+3.78", "+3.62", "+3.94"), ("-0.011", "-0.025", "+0.004"), "4.60", "+2.63", "34.6%"),
        (SCALE, ("+2.45", "+2.29", "+2.61"), ("+0.060", "+0.047", "+0.073"), "4.75", "+2.27", "30.7%"),
        (NORM, ("+0.15", "-0.05", "+0.36"), ("-0.070", "-0.090", "-0.051"), "5.17", "+2.06", "21.0%"),
        (NOTN, ("+6.32", "+6.12", "+6.53"), ("-0.198", "-0.218", "-0.179"), "7.01", "+1.75", "44.1%")):
    CHECKS += (stat_checks(pre, "mean_gap_corrected", M2, *mean) + stat_checks(pre, "r_corrected", R3, *r)
               + [(spread, "g", f"{pre}.point.band_range_corrected", "{:.2f}"), (mf, "g", f"{pre}.point.male_minus_female", M2),
                  (over, "g", f"{pre}.point.pct_over_plus8", P1)])
CHECKS += stat_checks(SCALE, "slope_raw", R3, "-0.174", "-0.182", "-0.165")

# Added 30 Oct after an outside audit: the raw-gap column of the primary table, filled so no cell is blank
for band, lo, hi in (("lt40", "+7.65", "+8.40"), ("40-49", "+4.47", "+5.40"), ("50-59", "+4.11", "+4.82"),
                     ("60-69", "+2.96", "+3.59"), ("70-79", "-0.31", "+0.35"), ("ge80", "-5.88", "-4.98")):
    CHECKS += stat_checks(PRIM, f"band_{band}_raw", M2, None, lo, hi)
CHECKS += (stat_checks(PRIM, "band_range_raw", "{:.2f}", "13.48", "12.90", "14.07")
           + stat_checks(PRIM, "male_minus_female_raw", M2, "+2.86", "+2.53", "+3.16")
           + [("31.7%", "g", f"{PRIM}.point.pct_over_plus8_raw", P1), ("16.0%", "g", f"{PRIM}.point.pct_under_minus8_raw", P1)])

# ... the Brazil line fitted on a random half of the Brazilian cohort (4.6, 5.5)
SH, AW, SX, R4, D2 = "brazil_split_half", "age_window", "brazil_sex_lines", "{:+.4f}", "{:.2f}"
CHECKS += [("116,823", "g", f"{SH}.n_fit", N), ("116,824", "g", f"{SH}.n_test", N),
           ("11.9459", "g", f"{SH}.intercept", S4), ("-0.2127", "g", f"{SH}.slope", S4), ("-0.3837", "g", f"{SH}.r_raw_test", R4),
           ("-0.0015", "g", f"{SH}.half_line.r_corrected", R4), ("-0.0007", "g", f"{SH}.full_line.r_corrected", R4)]
def split_checks(pre, older, younger, rr, rate):
    """n, died %, Wilson CI for each group, then the risk ratio and the person-year rate ratio, each with its CI."""
    out = []
    for grp, vals in (("older", older), ("younger", younger)):
        out += [(v, "g", f"{pre}.{grp}.{k}", fmt) for v, k, fmt in zip(vals, ("n", "died_pct", "died_ci.0", "died_ci.1"), (N, D2, D2, D2)) if v]
    out += [(v, "g", f"{pre}.{k}", D2) for v, k in zip(rr + rate, ("risk_ratio", "risk_ratio_ci.0", "risk_ratio_ci.1",
                                                                   "rate_ratio", "rate_ratio_ci.0", "rate_ratio_ci.1"))]
    return out
CHECKS += (split_checks(f"{SH}.raw", ("28,223", "3.04", "2.84", "3.24"), ("21,694", "4.57", "4.30", "4.85"),
                        ("0.66", "0.61", "0.73"), ("0.68", "0.62", "0.75"))
           + split_checks(f"{SH}.half_line.split", ("23,654", "5.15", "4.87", "5.44"), ("23,597", "2.59", "2.39", "2.80"),
                          ("1.99", "1.81", "2.19"), ("2.05", "1.86", "2.26"))
           + split_checks(f"{SH}.full_line.split", ("23,629", "5.15", "4.88", "5.44"), ("23,633", "2.59", "2.39", "2.80"),
                          ("1.99", "1.81", "2.19"), ("2.05", "1.86", "2.26")))

# ... both cohorts cut to ages 18 to 85 (4.6, 5.2, 5.5)
CHECKS += [("230,070", "g", f"{AW}.brazil.n", N), ("3,577", "g", f"{AW}.brazil.n_dropped", N),
           ("11.7879", "g", f"{AW}.brazil.intercept", S4), ("-0.2100", "g", f"{AW}.brazil.slope", S4),
           ("-0.3735", "g", f"{AW}.brazil.r_raw", R4), ("546", "g", f"{AW}.germany_n_dropped", "{}"),
           ("17,949", "g", f"{AW}.germany.cohort.n_patients", N)]
CHECKS += split_checks(f"{AW}.brazil.raw", (None,) * 4, (None,) * 4, ("0.68", "0.64", "0.72"), ("0.69", "0.65", "0.74"))
CHECKS += split_checks(f"{AW}.brazil.corrected", (None,) * 4, (None,) * 4, ("1.98", "1.85", "2.12"), ("2.03", "1.90", "2.18"))
CHECKS += (stat_checks(f"{AW}.germany", "slope_raw", R3, "-0.194", "-0.202", "-0.185")
           + stat_checks(f"{AW}.germany", "band_ge80_corrected", M2, "+1.44", "+0.95", "+1.89"))

# ... each German sex corrected by the line fitted in Brazil within that sex (4.4, 5.4, 5.5)
CHECKS += [("12.1915", "g", f"{SX}.lines.male.intercept", S4), ("11.8635", "g", f"{SX}.lines.female.intercept", S4)]
CHECKS += (stat_checks(f"{SX}.germany", "male_minus_female", M2, None, "+1.58", "+2.17")
           + stat_checks(f"{SX}.germany", "male_minus_female_band_adjusted", M2, "+1.57", "+1.27", "+1.87")
           + stat_checks(f"{SX}.germany", "male_corrected", M2, "+4.30", "+4.09", "+4.51")
           + stat_checks(f"{SX}.germany", "female_corrected", M2, "+2.43", "+2.21", "+2.65"))
for pre, mean, r, spread, mf, over in (
        (f"{SH}.germany", ("+3.47", "+3.32", "+3.62"), ("-0.002", "-0.016", "+0.010"), "4.65", "+2.55", "33.4%"),
        (f"{AW}.germany", ("+3.65", "+3.49", "+3.81"), ("+0.024", "+0.012", "+0.037"), "3.60", "+2.63", "34.0%"),
        (f"{SX}.germany", ("+3.39", "+3.24", "+3.54"), ("+0.000", "-0.013", "+0.013"), "4.48", "+1.87", "33.1%")):
    CHECKS += (stat_checks(pre, "mean_gap_corrected", M2, *mean) + stat_checks(pre, "r_corrected", R3, *r)
               + [(spread, "g", f"{pre}.point.band_range_corrected", "{:.2f}"), (mf, "g", f"{pre}.point.male_minus_female", M2),
                  (over, "g", f"{pre}.point.pct_over_plus8", P1)])

def lookup(obj, path):
    """Dotted path into nested JSON; a key that itself contains dots (model.zip, exams.csv) is matched whole."""
    parts, i = path.split("."), 0
    while i < len(parts):
        if isinstance(obj, list):
            obj, i = obj[int(parts[i])], i + 1
            continue
        j = next(j for j in range(len(parts), i, -1) if ".".join(parts[i:j]) in obj)
        obj, i = obj[".".join(parts[i:j])], j
    return obj

def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    data = {k: json.load(open(p)) for k, p in SOURCES.items()}
    failures = []
    for printed, src, path, fmt in CHECKS:
        got = fmt.format(lookup(data[src], path))
        if got != printed:
            failures.append(f"{src}:{path} formats to {got!r}, the manuscript quotes {printed!r}")
    print(f"{len(CHECKS) - len(failures)} of {len(CHECKS)} quoted Germany numbers match the code's output")

    if not MANUSCRIPT:
        print("manuscript check skipped (set PAPER_A_MANUSCRIPT to run it)")
    else:
        if not os.path.exists(MANUSCRIPT):
            failures.append(f"manuscript not found at {MANUSCRIPT}")
        else:
            text = re.sub(r"\s+", " ", open(MANUSCRIPT, encoding="utf-8").read())
            missing = [p for p, *_ in CHECKS if p not in text]
            failures += [f"quoted value {p!r} is not printed anywhere in the manuscript" for p in missing]
            print(f"{len(CHECKS) - len(missing)} of {len(CHECKS)} are printed in the manuscript")

    present = [f for f in TEXT_FILES if os.path.exists(os.path.join(HERE, f))]
    for rel in present:
        body = open(os.path.join(HERE, rel), encoding="utf-8").read()
        failures += [f"{os.path.basename(rel)}: contains {b!r}" for b in BANNED if b in body]
    print(f"dash hygiene checked on {len(present)} files")

    for f in failures:
        print("FAIL", f)
    print("VERIFY GERMANY: " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
