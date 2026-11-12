#!/usr/bin/env python3
"""Figure for results/results.json: held-out concordance by model, and the gap's added value by subgroup.

    python make_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results", "results.json")))
INK, GREY, RED, GREEN = "#24282B", "#8A9095", "#C0392B", "#1E8A50"

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})

# left: held-out C for each model, with 95% bootstrap intervals
rows = [("Age + sex\n(demographic-only)", "age_sex", INK),
        ("Age + sex + ECG-age gap\n(raw or corrected: identical)", "age_sex_gap", GREEN),
        ("Corrected gap alone", "gap_bc", GREY),
        ("Raw gap alone", "gap_raw", RED)]
for i, (label, k, col) in enumerate(rows):
    c, (lo, hi) = r["c_index_test"][k], r["c_index_test_ci95"][k]
    a.errorbar(c, i, xerr=[[c - lo], [hi - c]], fmt="o", color=col, capsize=4, ms=7)
    a.text(hi + 0.008, i, f"{c:.3f}", va="center", fontsize=10, color=col)
a.axvline(0.5, color=GREY, lw=0.8, ls=":")
a.text(0.503, -0.42, "coin flip", fontsize=8, color=GREY)
a.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=9)
a.invert_yaxis()
a.set_xlim(0.48, 0.86)
a.set_xlabel("Harrell's C on held-out patients (95% bootstrap CI)")
s = r["split"]
a.set_title(f"CODE-15%, held-out {s['test']:,} patients, {s['test_deaths']:,} deaths", fontsize=10, loc="left")

# right: what the gap adds over age + sex, overall and within subgroups
order = ["overall", "male", "female", "age 16-39", "age 40-59", "age 60-74", "age 75-+"]
sg = dict(r["subgroups_test"])
d = r["c_diff_test"]["age_sex_gap - age_sex"]
sg["overall"] = {"diff": d, "diff_ci95": r["c_diff_test_ci95"]["age_sex_gap - age_sex"],
                 "deaths": s["test_deaths"]}
for i, k in enumerate(order):
    v = sg[k]
    lo, hi = v["diff_ci95"]
    col = INK if k == "overall" else GREEN
    b.errorbar(v["diff"], i, xerr=[[v["diff"] - lo], [hi - v["diff"]]], fmt="s" if k == "overall" else "o",
               color=col, capsize=4, ms=6)
    b.text(hi + 0.004, i, f"+{v['diff']:.3f}  ({v['deaths']} deaths)", va="center", fontsize=8.5, color=GREY)
b.axvline(0, color=GREY, lw=0.8)
b.set_yticks(range(len(order)), ["All held-out", "Men", "Women", "Age 16 to 39", "Age 40 to 59", "Age 60 to 74",
                                 "Age 75 and over"], fontsize=9)
b.invert_yaxis()
b.set_xlim(-0.01, 0.2)
b.set_xlabel("Gain in C from adding the gap to age + sex (95% CI)")
b.set_title("The gap adds most where age alone ranks risk poorly", fontsize=10, loc="left")

for ax in (a, b):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "results", "cindex.png")
fig.savefig(out, dpi=160)
print("saved", out)
