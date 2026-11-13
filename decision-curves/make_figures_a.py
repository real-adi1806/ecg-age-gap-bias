"""
Paper A figures, drawn only from dca.npz, results_dca.json and results_ph.json.
Run from the repository root:  python decision-curves/make_figures_a.py
Saved at printed size (7.0 in wide), SVG and 300 dpi PNG.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
D = np.load(os.path.join(HERE, "dca.npz"))
R = json.load(open(os.path.join(HERE, "results_dca.json")))
P = json.load(open(os.path.join(HERE, "results_ph.json")))
thr = D["thr"]
BAND = R["band"]
C1, C2 = "#2a78d6", "#eb6834"             # categorical slots 1 and 2, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: ("%g%%" % round(100 * v, 2)))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def decision_figure(tag, name):
    h = R["horizons"][tag]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"wspace": 0.32})
    for ax in (a, b):
        ax.axvspan(BAND[0], BAND[1], color="#f1f0ec", zorder=0, lw=0)
    a.plot(thr, D["%s__treat_all" % tag], color=MUTED, lw=1.0, label="Treat all")
    a.axhline(0, color=INK, lw=0.8, label="Treat none")
    a.plot(thr, D["%s__curve__age_sex" % tag], color=C1, lw=1.6, label="Age and sex")
    a.plot(thr, D["%s__curve__age_sex_gap" % tag], color=C2, lw=1.6, ls="--", label="Age, sex and ECG-age gap")
    risk = h["km_risk_at_horizon"]
    a.set_ylim(-0.004, risk * 1.08)
    a.set_xlim(0, thr[-1])
    pct_axis(a)
    a.set_xlabel("Threshold (%g-year death risk)" % h["horizon_years"])
    a.set_ylabel("Net benefit")
    a.set_title("a  Held-out patients (n = %s)" % format(R["test_patients"], ","))
    grid(a)

    d = (D["%s__curve__age_sex_gap" % tag] - D["%s__curve__age_sex" % tag]) * 1000
    b.fill_between(thr, D["%s__diff_lo" % tag] * 1000, D["%s__diff_hi" % tag] * 1000, color=C2, alpha=0.18, lw=0)
    b.plot(thr, d, color=C2, lw=1.6, ls="--")
    b.axhline(0, color=INK, lw=0.8)
    b.set_xlim(0, thr[-1])
    pct_axis(b)
    b.set_xlabel("Threshold (%g-year death risk)" % h["horizon_years"])
    b.set_ylabel("Gain from the gap\n(net deaths identified per 1,000)")
    b.set_title("b  Gap model minus age and sex, 95% band")
    grid(b)
    hd, lab = a.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.28)
    fig.legend(hd, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.4)
    save(fig, name)


def calibration_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.28})
    for ax, tag in zip(axes, ("2y", "4y")):
        h = R["horizons"][tag]
        lo_v, hi_v = 1e9, 0
        for k, col, mk, lab in (("age_sex", C1, "o", "Age and sex"), ("age_sex_gap", C2, "s", "Age, sex and ECG-age gap")):
            rows = h["calibration_deciles"][k]
            x = np.array([r["mean_predicted"] for r in rows])
            y = np.array([r["observed_km"] for r in rows])
            ylo = np.array([r["observed_lo"] for r in rows])
            yhi = np.array([r["observed_hi"] for r in rows])
            ax.errorbar(x, y, yerr=[y - ylo, yhi - y], fmt=mk, ms=4.5, color=col, mfc=col if mk == "o" else "white",
                        mew=1.0, elinewidth=0.8, capsize=0, label=lab)
            lo_v, hi_v = min(lo_v, x.min(), max(ylo.min(), 1e-4)), max(hi_v, x.max(), yhi.max())
        ax.plot([lo_v * 0.7, hi_v * 1.3], [lo_v * 0.7, hi_v * 1.3], color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo_v * 0.7, hi_v * 1.3)
        ax.set_ylim(lo_v * 0.7, hi_v * 1.3)
        pct_axis(ax, "x")
        pct_axis(ax, "y")
        ax.set_xlabel("Mean predicted risk in decile")
        ax.set_ylabel("Observed (Kaplan-Meier)")
        ax.set_title("%s  %g-year death risk, O/E %.2f and %.2f" % ("a" if tag == "2y" else "b", h["horizon_years"],
                     h["observed_over_expected"]["age_sex"], h["observed_over_expected"]["age_sex_gap"]))
        grid(ax)
    hd, lab = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.26)
    fig.legend(hd, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "figure_a3_calibration_by_decile")


def ph_figure():
    panels = [
        ("raw_categorical", "pad_older", "Raw gap > +8"),
        ("raw_categorical", "pad_younger", "Raw gap < -8"),
        ("raw_continuous", "pad", "Gap per year"),
        ("corrected_categorical", "pad_bc_older", "Corrected > +8"),
        ("corrected_categorical", "pad_bc_younger", "Corrected < -8"),
        ("raw_continuous", "age", "Age per year"),
        ("corrected_categorical_share_matched", "pad_bc_matched_older", "Matched older"),
        ("corrected_categorical_share_matched", "pad_bc_matched_younger", "Matched younger"),
        ("raw_continuous", "male", "Male"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.4), gridspec_kw={"wspace": 0.32, "hspace": 0.55})
    tmax = 7.5
    for ax, (m, c, title) in zip(axes.flat, panels):
        M = P["models"][m]
        tv = M["time_varying"][c]
        x = [b_["t_mid"] for b_ in tv["bins"]]
        y = np.array([b_["hr"] for b_ in tv["bins"]])
        lo = np.array([b_["hr_lo"] for b_ in tv["bins"]])
        hi = np.array([b_["hr_hi"] for b_ in tv["bins"]])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=3.5, color=C1, elinewidth=0.8, capsize=0,
                    label="Binned scaled Schoenfeld, 95% CI")
        for row, (a_, b_) in zip(M["by_period"], P["periods"]):
            b_ = tmax if b_ is None or not np.isfinite(b_) else b_
            ax.plot([a_, b_], [row["terms"][c]["hr"]] * 2, color=C2, lw=1.8, solid_capstyle="butt",
                    label="HR fitted within period")
        ax.axhline(M["terms"][c]["hr"], color=INK, lw=0.8, ls="--", label="Whole-follow-up HR")
        ax.set_yscale("log")
        ax.set_xlim(0, tmax)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % round(v, 3)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % round(v, 3)))
        ax.tick_params(axis="y", which="minor", labelsize=6.5)
        p = M["ph_test"]["rank"][c]["p"]
        ax.set_title("%s, p = %.2f" % (title, p), fontsize=7.8)
        ax.set_xlabel("Years of follow-up", fontsize=7.5)
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hazard ratio")
    hd, lab = axes[0, 0].get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h_, l_ in zip(hd, lab):
        if l_ not in seen:
            seen.add(l_)
            H.append(h_)
            L.append(l_)
    fig.subplots_adjust(bottom=0.11)
    fig.legend(H, L, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "figure_a4_ph_time_varying")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    decision_figure("2y", "figure_a1_decision_curve_2y")
    decision_figure("4y", "figure_a2_decision_curve_4y")
    calibration_figure()
    ph_figure()
    print("wrote", sorted(os.listdir(FIG)))
