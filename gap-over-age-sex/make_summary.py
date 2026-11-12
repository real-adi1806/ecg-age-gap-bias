#!/usr/bin/env python3
"""Writes results/SUMMARY.md from results/results.json, so no number in the summary is typed by hand."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results", "results.json")))
c, ci, d, dci = r["c_index_test"], r["c_index_test_ci95"], r["c_diff_test"], r["c_diff_test_ci95"]
s, co, cv, sg = r["split"], r["cohort"], r["cv5"], r["subgroups_test"]
hr = r["hazard_ratios_train"]
f = lambda x: f"{x:.3f}"
rng = lambda v: f"{v[0]:.3f} to {v[1]:.3f}"
sgn = lambda x: f"+{x:.3f}" if x >= 0 else f"{x:.3f}"

lines = [
    "# ECG-age gap over age and sex on CODE-15%: results",
    "",
    f"Generated from `results/results.json` by `make_summary.py` ({r['boot']} bootstrap samples, seed {r['seed']}).",
    "",
    f"Cohort: {co['patients']:,} patients (one row each, first exam), {co['deaths']:,} deaths, median follow-up "
    f"{co['median_follow_up_years']:.2f} years. Split 80/20 by patient: held-out {s['test']:,} patients, "
    f"{s['test_deaths']:,} deaths. Cox models fitted on the training patients; Harrell's C on the held-out patients.",
    "",
    "| Model | Held-out C (95% CI) | 5-fold CV mean (SD) |",
    "|---|---|---|",
    f"| Age + sex | {f(c['age_sex'])} ({rng(ci['age_sex'])}) | {f(cv['age_sex']['mean'])} ({cv['age_sex']['sd']:.3f}) |",
    f"| Age + sex + ECG-age gap | {f(c['age_sex_gap'])} ({rng(ci['age_sex_gap'])}) | {f(cv['age_sex_gap']['mean'])} ({cv['age_sex_gap']['sd']:.3f}) |",
    f"| Age + sex + corrected gap | {f(c['age_sex_gapbc'])} (identical by construction) | |",
    f"| Corrected gap alone | {f(c['gap_bc'])} ({rng(ci['gap_bc'])}) | {f(cv['gap_bc']['mean'])} ({cv['gap_bc']['sd']:.3f}) |",
    f"| Raw gap alone | {f(c['gap_raw'])} ({rng(ci['gap_raw'])}) | {f(cv['gap_raw']['mean'])} ({cv['gap_raw']['sd']:.3f}) |",
    "",
    f"- Adding the gap to age and sex: {sgn(d['age_sex_gap - age_sex'])} (95% CI {rng(dci['age_sex_gap - age_sex'])}).",
    f"- Hazard ratio per year of gap, adjusted for age and sex: {hr['age_sex_gap']['pad']['hr']:.4f} "
    f"(raw) and {hr['age_sex_gapbc']['pad_bc']['hr']:.4f} (corrected): the same, as expected.",
    f"- Unadjusted, the raw gap's hazard ratio is {hr['gap_raw']['pad']['hr']:.4f} per year (below 1, the wrong "
    f"direction) and the corrected gap's is {hr['gap_bc']['pad_bc']['hr']:.4f}. The correction lifts the gap-alone C "
    f"by {sgn(d['gap_bc - gap_raw'])} (95% CI {rng(dci['gap_bc - gap_raw'])}).",
    "",
    "Gain from adding the gap to age + sex, within subgroups of the held-out patients:",
    "",
    "| Subgroup | Patients | Deaths | Age + sex | + gap | Gain (95% CI) |",
    "|---|---|---|---|---|---|",
]
for k, v in sg.items():
    lines.append(f"| {k} | {v['patients']:,} | {v['deaths']:,} | {f(v['c_age_sex'])} | {f(v['c_age_sex_gap'])} | "
                 f"{sgn(v['diff'])} ({rng(v['diff_ci95'])}) |")
lines += ["",
          "Checks: lifelines' C against a written-out Harrell's C on 2,500 held-out patients: " +
          "; ".join(f"{k} {v['lifelines']:.4f} vs {v['naive']:.4f}" for k, v in r["check_concordance_impl"].items()) + ".",
          f"Raw and corrected models with age: risk-score correlation "
          f"{r['check_same_scores_raw_vs_corrected_with_age']['corr_of_risk_scores']:.6f}, C difference "
          f"{r['check_same_scores_raw_vs_corrected_with_age']['c_difference']:.6f}."]
open(os.path.join(HERE, "results", "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
