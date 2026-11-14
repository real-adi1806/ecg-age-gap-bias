#!/usr/bin/env python3
"""Figure for results/results.json: the ladder with the published values beside it, calibration, decision curve.

    python make_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "results.json")
r = json.load(open(src))
pub = json.load(open(os.path.join(HERE, "results", "published_fig3a.json")))
if "checks" not in r or not r["checks"]["all_passed"]:
    sys.exit("STOP: results.json has no passed checks. No figure was drawn.")
INK, GREY, RED, GREEN, BLUE = "#24282B", "#8A9095", "#C0392B", "#1E8A50", "#2C6FB7"
P = str(r["primary_seed"])
blk = r["seeds"][P]["test_with_features"]

fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw={"width_ratios": [1.45, 1, 1]})

# left: the ladder. Dot and bar: split 0 with its 95% bootstrap interval. Small ticks: the other four splits.
rows = [("1  Age alone", "row1_age", INK),
        ("2  Age + sex", "row2_age_sex", INK),
        ("3  Age + sex + rhythm labels", "row3_age_sex_rhythm", INK),
        ("4  Age + sex + ECG-age gap", "row4_age_sex_gap", INK),
        ("5  Age + sex + Lima features", "row5_age_sex_lima", GREEN),
        ("5b  Lima features alone", "row5b_lima_only", GREY)]
for i, (label, k, col) in enumerate(rows):
    v, (lo, hi) = blk["auc"][k], blk["auc_ci95"][k]
    a.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=4, ms=7, zorder=3)
    others = [s["test_with_features"]["auc"][k] for q, s in r["seeds"].items() if q != P]
    a.plot(others, [i + 0.22] * len(others), "|", color=col, ms=9, alpha=0.7)
    a.text(0.741, i, f"{v:.3f}", va="center", fontsize=9.5, color=col)
tv, bm = pub["text_values"], pub["best_mean"]["12-Lead"]
refs = [(tv["from_scratch"], "--", GREY, f"from scratch, text: {tv['from_scratch']:.3f}"),
        (bm["from_scratch"][0], ":", GREY, f"from scratch, 12-lead: {bm['from_scratch'][0]:.3f}"),
        (tv["csfm_up_to"], "--", BLUE, f"CSFM, text: {tv['csfm_up_to']:.3f}"),
        (bm["csfm"][0], ":", BLUE, f"CSFM, 12-lead: {bm['csfm'][0]:.3f}")]
for j, (x, ls, col, label) in enumerate(refs):
    a.axvline(x, color=col, lw=1.1, ls=ls, zorder=1)
    a.text(x, 6.1 + 0.5 * j, label, fontsize=7.5, color=col, ha="center", va="center",
           bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
a.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=9)
a.set_ylim(8.0, -0.7)
a.set_xlim(0.74, 0.885)
a.set_xlabel("AUC for death within 1 year, held-out patients (95% bootstrap CI; ticks: four other splits)")
a.set_title(f"CODE-15%: {blk['patients']:,} test patients, {blk['deaths']} deaths (split 0).\n"
            "Vertical lines: published values on the authors' own split, not this one", fontsize=9.5, loc="left")

# middle: calibration on a log scale, because the risks run from 0.05% to 20%
cal = r["primary"]["calibration"]["test_with_features"]
for k, col, label in (("row2_age_sex", INK, "Age + sex"), ("row5_age_sex_lima", GREEN, "Age + sex + Lima features")):
    v = cal[k]
    b.plot(v["smooth"]["predicted"], v["smooth"]["observed"], color=col, lw=1.4, alpha=0.8)
    b.plot([g["mean_predicted"] for g in v["groups"]], [g["observed"] for g in v["groups"]], "o", color=col, ms=5,
           label=f"{label}: slope {v['slope']:.2f} ({v['slope_ci95'][0]:.2f} to {v['slope_ci95'][1]:.2f})")
b.plot([3e-4, 0.3], [3e-4, 0.3], color=GREY, lw=0.8, ls=":")
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(3e-4, 0.3)
b.set_ylim(3e-4, 0.3)
ticks = [0.001, 0.01, 0.1]
b.set_xticks(ticks, ["0.1%", "1%", "10%"])
b.set_yticks(ticks, ["0.1%", "1%", "10%"])
b.set_xlabel("Predicted 1-year risk")
b.set_ylabel("Observed 1-year risk")
b.legend(fontsize=8, frameon=False, loc="upper left")
b.set_title("Calibration: ten equal groups and a smoothed curve", fontsize=9.5, loc="left")

# right: decision curve
dc = r["primary"]["decision_curve_test_with_features"]
t = [100 * x for x in dc["thresholds"]]
c.plot(t, [100 * x for x in dc["row5_age_sex_lima"]], color=GREEN, lw=1.8, label="Age + sex + Lima features")
c.plot(t, [100 * x for x in dc["row2_age_sex"]], color=INK, lw=1.8, label="Age + sex")
c.plot(t, [100 * x for x in dc["treat_all"]], color=RED, lw=1, ls="--", label="Treat all")
c.axhline(0, color=GREY, lw=1, label="Treat none")
c.set_ylim(-0.3, 1.1)
c.set_xlim(0.5, 10)
c.set_xlabel("Risk threshold (%)")
c.set_ylabel("Net benefit per 100 patients")
c.legend(fontsize=8, frameon=False, loc="upper right")
c.set_title("Decision curve", fontsize=9.5, loc="left")

for ax in (a, b, c):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "results", "ladder.png") if len(sys.argv) < 3 else sys.argv[2]
fig.savefig(out, dpi=160)
print("saved", out)
