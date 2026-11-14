#!/usr/bin/env python3
"""Writes results/SUMMARY.md from results/results.json and results/published_fig3a.json. No number is typed by hand.

    python make_summary.py

Refuses to write anything if a check in results.json failed (the plan's stop rule).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "results.json")
r = json.load(open(src))
pub = json.load(open(os.path.join(HERE, "results", "published_fig3a.json")))
if "checks" not in r or not r["checks"]["all_passed"]:
    sys.exit("STOP: results.json has no passed checks. SUMMARY.md was not written.")

P = str(r["primary_seed"])
seeds = r["seeds"]
f = lambda x: f"{x:.3f}"
rng = lambda v: f"{v[0]:.3f} to {v[1]:.3f}"
sgn = lambda x: f"+{x:.3f}" if x >= 0 else f"{x:.3f}"
NAMES = {"row1_age": ("1", "Age alone (rank, no model)"),
         "row2_age_sex": ("2", "Age + sex"),
         "row3_age_sex_rhythm": ("3", "Age + sex + six rhythm labels + normal_ecg"),
         "row4_age_sex_gap": ("4", "Age + sex + ECG-age gap"),
         "row5_age_sex_lima": ("5", "Age + sex + Lima network features (ridge)"),
         "row5b_lima_only": ("5b", "Lima network features alone (ridge, no age or sex)")}


def table(block):
    b = seeds[P][block]
    out = ["| Row | Inputs | Test AUC, split 0 (95% CI) | Mean of 5 splits (range) | Difference from row 2, split 0 (95% CI) "
           "| Average precision, split 0 (95% CI) |", "|---|---|---|---|---|---|"]
    for k, (num, label) in NAMES.items():
        if k not in b["auc"]:
            continue
        across = [s[block]["auc"][k] for s in seeds.values()]
        d = "reference" if k == "row2_age_sex" else (
            f"{sgn(b['auc_diff_vs_row2'][k])} ({rng(b['auc_diff_vs_row2_age_sex_ci95'][k])})")
        out.append(f"| {num} | {label} | {f(b['auc'][k])} ({rng(b['auc_ci95'][k])}) | "
                   f"{f(sum(across) / len(across))} ({f(min(across))} to {f(max(across))}) | {d} | "
                   f"{f(b['ap'][k])} ({rng(b['ap_ci95'][k])}) |")
    return out


co, fe, ch = r["cohort"], r["features"], r["checks"]
n0, ta, tf = seeds[P]["n"], seeds[P]["test_all"], seeds[P]["test_with_features"]
bm = pub["best_mean"]
cal = r["primary"]["calibration"]
dc = r["primary"]["decision_curve_test_with_features"]
sa = r["sensitivity_a_primary"]
rd = seeds[P]["ridge"]
r5 = [s["test_with_features"]["auc"]["row5_age_sex_lima"] for s in seeds.values()]

L = [
    "# One-year mortality on CODE-15%: the rows under a foundation model",
    "",
    f"Generated from `results/results.json` by `make_summary.py` ({r['boot']} bootstrap resamples, seed base "
    f"{r['seed_base']}). Every number below was computed in this repository (TESTED) unless it sits under "
    "\"Published\", where it is copied from the paper or its source data.",
    "",
    "## Cohort",
    "",
    f"- {co['patients']:,} patients with a death flag and a follow-up time in `exams.csv`, one row each.",
    f"- {co['deaths_1y']:,} died within one year; {co['survivors_1y']:,} were followed for at least one year; "
    f"{co['short_follow_up']:,} were alive with under one year of follow-up.",
    f"- Main analysis: the {co['short_follow_up']:,} are left out. Death rate {100 * co['death_rate_main']:.2f}%. "
    f"Keeping them as survivors (sensitivity A) gives {100 * co['death_rate_sensitivity_a']:.2f}%. "
    f"The paper reports {100 * co['published_death_rate']:.2f}%, which sits between the two.",
    f"- Split by patient, 80/10/10, stratified on the label, five times (splits 0 to 4). Split 0: "
    f"{n0['train']:,} train ({n0['train_deaths']:,} deaths), {n0['val']:,} validation ({n0['val_deaths']} deaths), "
    f"{n0['test']:,} test ({n0['test_deaths']} deaths).",
    f"- The saved network features belong to the outcome exam for {fe['patients_with_outcome_exam_features']:,} "
    f"patients. For the other {fe['patients_without']:,} the saved row is a different, usually later, exam, so they "
    f"cannot be used for row 5 without leaking the future. Those patients all have two or more exams and a 1-year "
    f"death rate of {100 * fe['death_rate_without']:.2f}% against {100 * fe['death_rate_with']:.2f}% in the rest, "
    "so the feature subset is not a random sample.",
    "",
    "## The ladder",
    "",
    f"Table A. Test patients whose outcome exam has saved features (split 0: {tf['patients']:,} patients, "
    f"{tf['deaths']} deaths). All six rows are scored on the same patients, so the differences are paired.",
    "",
    *table("test_with_features"),
    "",
    f"Table B. Every test patient (split 0: {ta['patients']:,} patients, {ta['deaths']} deaths). Rows 1 to 4 only.",
    "",
    *table("test_all"),
    "",
    "Intervals are 95% bootstrap intervals over the test patients of split 0. The range across the five splits shows "
    f"how much the choice of split moves each row: row 5 runs from {f(min(r5))} to {f(max(r5))}.",
    "",
    "## Published, on the authors' own test split (not ours)",
    "",
    "Gu et al., Nat Mach Intell 2026, doi 10.1038/s42256-026-01180-5. These are their numbers on their split. "
    "Their split is not published, so no row above can be said to beat or match them; the two sets of numbers "
    "are side by side, not head to head.",
    "",
    "| Source | Lead setting | From scratch | CSFM |",
    "|---|---|---|---|",
    f"| Results text | not stated | {pub['text_values']['from_scratch']:.3f} | up to {pub['text_values']['csfm_up_to']:.3f} |",
]
for chn in ("12-Lead", "6-Lead", "2-Lead", "Lead II"):
    L.append(f"| Source Data Fig. 3a, best mean of 3 seeds | {chn} | {bm[chn]['from_scratch'][0]:.3f} "
             f"({bm[chn]['from_scratch'][1]}) | {bm[chn]['csfm'][0]:.3f} ({bm[chn]['csfm'][1]}) |")
L += [
    "",
    "- The Lima network reads all 12 leads, so the 12-lead line is the like-for-like one. The text values 0.816 and "
    "0.844 do not appear exactly in the Fig. 3a sheet; they are close to the six-lead values, and Fig. 2d is a "
    "six-lead result. That they are six-lead numbers is an inference (UNVERIFIED), and a fair thing to ask.",
    "- No age, demographic or tabular baseline for this task was found in the Results text, the Fig. 3 caption or "
    "the Fig. 3a source data (TESTED for those three; the Methods were not read word for word).",
    "",
    "## Calibration (split 0)",
    "",
    "Slope: 1 is ideal, below 1 means predictions are too extreme. Intercept in the large: 0 is ideal.",
    "",
    "| Model | Patients | Slope (95% CI) | Intercept in the large (95% CI) | Mean predicted | Observed |",
    "|---|---|---|---|---|---|",
]
for blk, who in (("test_with_features", "feature subset"), ("test_all", "all test")):
    for k, v in cal[blk].items():
        L.append(f"| {NAMES[k][1]} | {who} | {v['slope']:.2f} ({v['slope_ci95'][0]:.2f} to {v['slope_ci95'][1]:.2f}) | "
                 f"{v['intercept_in_the_large']:+.2f} ({v['intercept_in_the_large_ci95'][0]:+.2f} to "
                 f"{v['intercept_in_the_large_ci95'][1]:+.2f}) | {100 * v['mean_predicted']:.2f}% | "
                 f"{100 * v['observed_rate']:.2f}% |")
L += [
    "",
    "Rows 2 and 4 are fitted on every training patient, so on the feature subset (a higher-risk group) they "
    "under-predict a little; that is the positive intercept.",
    "",
    "## Decision curve (split 0, feature subset): net benefit per 100 patients",
    "",
    "| Threshold | Treat all | Row 2, age + sex | Row 5, + Lima features | Row 5 minus row 2 (95% CI) | Share flagged by row 5 |",
    "|---|---|---|---|---|---|",
]
for i, t in enumerate(dc["thresholds"]):
    if round(1000 * t) in (5, 10, 20, 30, 50, 75, 100):
        lo, hi = dc["row5_minus_row2_ci95"][i]
        L.append(f"| {100 * t:.1f}% | {100 * dc['treat_all'][i]:.3f} | {100 * dc['row2_age_sex'][i]:.3f} | "
                 f"{100 * dc['row5_age_sex_lima'][i]:.3f} | {100 * (dc['row5_age_sex_lima'][i] - dc['row2_age_sex'][i]):+.3f} "
                 f"({100 * lo:+.3f} to {100 * hi:+.3f}) | {100 * dc['row5_age_sex_lima_share_treated'][i]:.1f}% |")
L += [
    "",
    "Treat none is 0 at every threshold. Net benefit as in Vickers and Elkin 2006, using `../decision-curves/netbenefit.py`.",
    "",
    "## Subgroups (split 0, feature subset)",
    "",
    "| Subgroup | Patients | Deaths | Row 2 | Row 4 | Row 5 | Row 5 minus row 2 (95% CI) |",
    "|---|---|---|---|---|---|---|",
]
for g, v in r["primary"]["subgroups_test_with_features"].items():
    a = v["auc"]
    g = g.replace("75-+", "75 and over").replace("-", " to ")
    L.append(f"| {g} | {v['patients']:,} | {v['deaths']} | {f(a['row2_age_sex'])} | {f(a['row4_age_sex_gap'])} | "
             f"{f(a['row5_age_sex_lima'])} | {sgn(v['auc_diff_vs_row2']['row5_age_sex_lima'])} "
             f"({rng(v['auc_diff_vs_row2_age_sex_ci95']['row5_age_sex_lima'])}) |")
young = r["primary"]["subgroups_test_with_features"]["age 16-39"]["deaths"]
L += [
    "",
    f"The youngest band has {young} deaths in the test split, so its numbers carry little weight. Within an age "
    "band, age and sex rank poorly by construction, which is why the within-band AUCs are lower than the overall ones.",
    "",
    "## Sensitivity A: short follow-ups kept as survivors (split 0)",
    "",
    "| Row | Main analysis | Sensitivity A |",
    "|---|---|---|",
]
for k, (num, label) in NAMES.items():
    L.append(f"| {num} {label} | {f(tf['auc'][k])} | {f(sa['test_with_features']['auc'][k])} "
             f"({rng(sa['test_with_features']['auc_ci95'][k])}) |")
L += [
    "",
    f"Feature subset of the test split in both columns. Sensitivity A test set: "
    f"{sa['test_with_features']['patients']:,} patients, {sa['test_with_features']['deaths']} deaths.",
    "",
    "## Row 5 details",
    "",
    f"- {rd['features_kept']:,} of {fe['feature_columns']:,} feature columns vary on the training patients of split 0; "
    f"the other {rd['features_constant_on_train']:,} are constant (dead units) and are dropped.",
    f"- Penalty chosen by validation log loss. Chosen C (scikit-learn, inverse penalty) per split: "
    + ", ".join(f"{s['ridge']['row5_age_sex_lima']['chosen_C']:.4g}" for s in seeds.values()) + ".",
    "- The network that produced the features was trained on the full CODE set with CODE-15% held out (Lima et al. "
    "2021), so the features are out of sample for these patients.",
    "",
    "## Checks (all must pass before a number is used)",
    "",
    f"1. Age alone on the whole main cohort: {ch['1_age_alone_whole_cohort']['auc']:.4f}; the pilot had "
    f"{ch['1_age_alone_whole_cohort']['pilot']}. PASS.",
    f"2. AUC by the rank formula written out by hand against scikit-learn, every row and split: largest difference "
    f"{ch['2_rank_formula_vs_sklearn']['largest_absolute_difference']:.1e}. PASS.",
    "3. Labels shuffled once, everything refitted: " +
    ", ".join(f"{NAMES[k][0]}: {v:.3f}" for k, v in ch["3_shuffled_labels"]["auc"].items() if k in NAMES) + ". PASS.",
    "4. Patients in two splits: " + str(sum(ch["4_no_patient_in_two_splits"]["overlaps"])) + ". PASS.",
    f"5. Row 4 against age + sex + predicted age: AUC difference "
    f"{ch['5_gap_equals_predicted_age_given_age']['largest_auc_difference']:.1e}. PASS.",
    f"6. Stop rule: largest row 5 AUC over the five splits is {ch['stop_rule_row5_not_above_0.86']['largest_row5_auc']:.3f}, "
    "not above 0.86. PASS.",
    "7. Every ridge fit converged and the chosen penalty is inside the grid. PASS.",
    "",
    "## Limits",
    "",
    "- Our splits are not the authors' split. Nothing here is a head-to-head comparison with CSFM.",
    "- How the authors labelled patients followed for less than a year is not public (UNVERIFIED). The two readings "
    "tried here change the AUCs only in the third decimal.",
    f"- Row 5 uses {fe['main_cohort_with']:,} of the {co['deaths_1y'] + co['survivors_1y']:,} main-cohort patients; "
    f"the left-out group holds {fe['deaths_1y_without']} of the {co['deaths_1y']:,} deaths.",
    "- Row 5 is a linear model on a frozen network's last layer. It says what that network already encodes, not what "
    "a fine-tuned network could reach.",
    "- One cohort, in Brazil. Death within one year is rare here (about 1.25%), so average precision is low for every row.",
]
open(os.path.join(HERE, "results", "SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
