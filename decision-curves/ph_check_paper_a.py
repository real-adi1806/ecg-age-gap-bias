"""
Paper A: proportional-hazards check of every Cox model the manuscript reports, plus the two
held-out models behind the decision curve.

Run from the repository root:   python decision-curves/ph_check_paper_a.py
Needs results/cox.json first (make cox). Writes decision-curves/results_ph.json (read by make_figures_a.py).

Models (Efron ties, fitted with lifelines; the manuscript fitted them with its own
standard-library code, and the script stops unless lifelines reproduces those coefficients):
    raw_categorical                      PAD > +8, PAD < -8 (within +/-8 = reference), age, male
    corrected_categorical                same on the corrected gap
    corrected_categorical_share_matched  corrected gap cut at the raw cut's shares
    raw_continuous                       PAD per year, age, male
    corrected_continuous                 corrected PAD per year, age, male
    heldout_age_sex, heldout_age_sex_gap the 80% training fits of gap-over-age-sex/analysis.py

For each model:
    1. Grambsch-Therneau test on scaled Schoenfeld residuals (rank and KM time transforms).
       With 8,341 deaths these reject trivially, so the size of any departure is what counts:
    2. beta(t) = beta + scaled Schoenfeld residual, averaged in ten equal-count bins of death
       time, and the OLS slope of the residual on time in years (change in log HR per year);
    3. the hazard ratio fitted separately within follow-up periods 0-1, 1-2, 2-4 and 4+ years
       (left-truncated fits: everyone alive and in follow-up at the period start);
    4. a sensitivity fit stratified on 10-year age band and sex, which frees the baseline
       hazard to differ in shape between those strata, so a non-proportional age or sex
       effect cannot leak into the gap terms.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # the repository root
sys.path.insert(0, REPO)
PERIODS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf))
N_BINS = 10
Z = 1.959963984540054


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cohort():
    cm = load_module("cox_model", os.path.join(REPO, "cox_model.py"))
    base = cm.base
    rows, _ = base.load(os.path.join(REPO, "data", "exams.csv"))
    base.bias_correct(rows)
    thr = base.DEFAULT_THRESHOLD
    cm.group_flags(rows, "pad", thr)
    cm.group_flags(rows, "pad_bc", thr)
    cuts = base.share_matched_cuts(rows, "pad", "pad_bc", thr)
    cm.group_flags(rows, "pad_bc", cuts["upper_cut_years"], lower=cuts["lower_cut_years"], name="pad_bc_matched")
    df = pd.DataFrame(rows)
    keep = ["timey", "died", "age", "male", "pad", "pad_bc", "pad_older", "pad_younger",
            "pad_bc_older", "pad_bc_younger", "pad_bc_matched_older", "pad_bc_matched_younger"]
    df = df[keep].copy()
    df["died"] = df["died"].astype(int)
    df["male"] = df["male"].astype(float)
    return df, cuts


MANUSCRIPT = {
    "raw_categorical": (["pad_older", "pad_younger", "age", "male"],
                        ["PAD > +8 (older)", "PAD < -8 (younger)", "age (per year)", "male"]),
    "corrected_categorical": (["pad_bc_older", "pad_bc_younger", "age", "male"],
                              ["PAD_bc > +8 (older)", "PAD_bc < -8 (younger)", "age (per year)", "male"]),
    "corrected_categorical_share_matched": (["pad_bc_matched_older", "pad_bc_matched_younger", "age", "male"],
                                            ["PAD_bc older (matched)", "PAD_bc younger (matched)", "age (per year)", "male"]),
    "raw_continuous": (["pad", "age", "male"], ["PAD (per year)", "age (per year)", "male"]),
    "corrected_continuous": (["pad_bc", "age", "male"], ["PAD_bc (per year)", "age (per year)", "male"]),
}
CONTRASTS = {"raw_categorical": ("pad_older", "pad_younger"),
             "corrected_categorical": ("pad_bc_older", "pad_bc_younger"),
             "corrected_categorical_share_matched": ("pad_bc_matched_older", "pad_bc_matched_younger")}


def fit(df, cols, **kw):
    cph = CoxPHFitter()
    cph.fit(df[cols + ["timey", "died"]], duration_col="timey", event_col="died", **kw)
    return cph


def hr_row(cph, col):
    b = float(cph.params_[col])
    se = float(cph.standard_errors_[col])
    return {"coef": b, "se": se, "hr": float(np.exp(b)), "lo": float(np.exp(b - Z * se)), "hi": float(np.exp(b + Z * se))}


def contrast(cph, a, b):
    v = cph.variance_matrix_
    d = float(cph.params_[a] - cph.params_[b])
    se = float(np.sqrt(v.loc[a, a] + v.loc[b, b] - 2 * v.loc[a, b]))
    return {"coef": d, "se": se, "hr": float(np.exp(d)), "lo": float(np.exp(d - Z * se)), "hi": float(np.exp(d + Z * se))}


def schoenfeld_summary(cph, df, cols):
    """Binned beta(t) and the slope of the scaled residual on time, per covariate."""
    r = cph.compute_residuals(df[cols + ["timey", "died"]], kind="scaled_schoenfeld")
    t = df.loc[r.index, "timey"].values
    order = np.argsort(t, kind="stable")
    edges = np.linspace(0, len(t), N_BINS + 1).astype(int)
    out = {}
    for c in cols:
        beta = float(cph.params_[c])
        bt = beta + r[c].values            # r already holds the scaled residual for this term
        bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            idx = order[a:b]
            m, s = float(bt[idx].mean()), float(bt[idx].std(ddof=1) / np.sqrt(len(idx)))
            bins.append({"t_mid": float(np.median(t[idx])), "t_lo": float(t[idx].min()), "t_hi": float(t[idx].max()),
                         "deaths": int(len(idx)), "beta": m, "beta_lo": m - Z * s, "beta_hi": m + Z * s,
                         "hr": float(np.exp(m)), "hr_lo": float(np.exp(m - Z * s)), "hr_hi": float(np.exp(m + Z * s))})
        # OLS of beta(t) on time: slope = change in log HR per year of follow-up
        X = np.column_stack([np.ones(len(t)), t])
        coef, res, *_ = np.linalg.lstsq(X, bt, rcond=None)
        resid = bt - X @ coef
        s2 = resid @ resid / (len(t) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        slope, se = float(coef[1]), float(np.sqrt(cov[1, 1]))
        t10, t90 = np.percentile(t, [10, 90])
        out[c] = {"bins": bins, "slope_log_hr_per_year": slope, "slope_se": se,
                  "slope_lo": slope - Z * se, "slope_hi": slope + Z * se,
                  "death_time_p10_p90": [float(t10), float(t90)],
                  "hr_at_p10_time_fitted_line": float(np.exp(coef[0] + coef[1] * t10)),
                  "hr_at_p90_time_fitted_line": float(np.exp(coef[0] + coef[1] * t90))}
    return out


def by_period(df, cols, contrast_pair=None):
    out = []
    for a, b in PERIODS:
        # The first period keeps everyone: 47 patients have a death recorded at time 0, and a
        # "time > 0" filter would silently drop them. Later periods are left-truncated at a.
        d = (df if a == 0 else df[df["timey"] > a]).copy()
        d["stop"] = np.minimum(d["timey"], b)
        d["ev"] = ((d["died"] == 1) & (d["timey"] <= b)).astype(int)
        cph = CoxPHFitter()
        if a == 0:
            cph.fit(d[cols + ["stop", "ev"]], duration_col="stop", event_col="ev")
        else:
            d["entry"] = a
            cph.fit(d[cols + ["entry", "stop", "ev"]], duration_col="stop", event_col="ev", entry_col="entry")
        row = {"period": "%g to %s years" % (a, "end" if np.isinf(b) else "%g" % b),
               "at_risk_at_start": int(len(d)), "deaths": int(d["ev"].sum()),
               "terms": {c: hr_row(cph, c) for c in cols}}
        if contrast_pair:
            row["older_vs_younger"] = contrast(cph, *contrast_pair)
        out.append(row)
    return out


def stratified(df, cols, contrast_pair=None):
    d = df.copy()
    d["ageband"] = np.minimum((d["age"] // 10).astype(int), 9)
    d.loc[d["ageband"] < 2, "ageband"] = 2            # pool under 30
    d["sexs"] = d["male"].astype(int)
    keep = [c for c in cols if c != "male"]
    cph = CoxPHFitter()
    cph.fit(d[keep + ["timey", "died", "ageband", "sexs"]], duration_col="timey", event_col="died",
            strata=["ageband", "sexs"])
    row = {"strata": "10-year age band (under 30 pooled, 90+ with 80s) x sex; linear age kept within strata",
           "terms": {c: hr_row(cph, c) for c in keep}}
    if contrast_pair:
        row["older_vs_younger"] = contrast(cph, *contrast_pair)
    return row


def ph_tests(cph, df, cols):
    out = {}
    for tt in ("rank", "km"):
        res = proportional_hazard_test(cph, df[cols + ["timey", "died"]], time_transform=tt)
        s = res.summary
        out[tt] = {c: {"chi2": float(s.loc[c, "test_statistic"]), "p": float(s.loc[c, "p"])} for c in cols}
    return out


def analyse(name, cph, df, cols, contrast_pair=None):
    t0 = time.time()
    r = {"n": int(len(df)), "deaths": int(df["died"].sum()),
         "terms": {c: hr_row(cph, c) for c in cols},
         "ph_test": ph_tests(cph, df, cols),
         "time_varying": schoenfeld_summary(cph, df, cols),
         "by_period": by_period(df, cols, contrast_pair),
         "stratified_age_band_sex": stratified(df, cols, contrast_pair)}
    if contrast_pair:
        r["older_vs_younger"] = contrast(cph, *contrast_pair)
    print("  %-38s done in %.1f s" % (name, time.time() - t0))
    return r


def main():
    t0 = time.time()
    df, cuts = build_cohort()
    cox_json = json.load(open(os.path.join(REPO, "results", "cox.json")))
    results = {"what": "Paper A proportional-hazards check. Every number from ph_check_paper_a.py.",
               "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "cohort": {"patients": int(len(df)), "deaths": int(df["died"].sum())},
               "share_matched_cuts": cuts, "periods": [list(p) for p in PERIODS], "models": {},
               "reproduction": {}}
    print("cohort %d patients, %d deaths" % (len(df), df["died"].sum()))
    for name, (cols, labels) in MANUSCRIPT.items():
        cph = fit(df, cols)
        ref = cox_json["models"][name]["terms"]
        worst = 0.0
        for c, lab in zip(cols, labels):
            d = abs(float(cph.params_[c]) - ref[lab]["coef"])
            worst = max(worst, d)
        results["reproduction"][name] = {"max_abs_coef_diff_vs_cox_json": worst}
        assert worst < 1e-5, "lifelines did not reproduce %s (max coef diff %.2e)" % (name, worst)
        results["models"][name] = analyse(name, cph, df, cols, CONTRASTS.get(name))
        results["models"][name]["labels"] = dict(zip(cols, labels))
    print("reproduced all five manuscript models: max |coef diff| %.2e"
          % max(v["max_abs_coef_diff_vs_cox_json"] for v in results["reproduction"].values()))

    # held-out models of gap-over-age-sex (fitted on the 80% training patients)
    an = load_module("gap_analysis", os.path.join(REPO, "gap-over-age-sex", "analysis.py"))
    an.import_base(REPO)
    full, _ = an.load(os.path.join(REPO, "data", "exams.csv"))
    rng = np.random.default_rng(an.SEED)
    order = rng.permutation(len(full))
    train = full.iloc[order[:int(0.8 * len(full))]].copy()
    an.correct(train, [train])
    train["died"] = train["died"].astype(int)
    for name, key in (("heldout_age_sex", "age_sex"), ("heldout_age_sex_gap", "age_sex_gap")):
        cols = an.MODELS[key]
        cph = fit(train, cols)
        results["models"][name] = analyse(name, cph, train.reset_index(drop=True), cols)
        results["models"][name]["labels"] = {c: c for c in cols}
    results["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "results_ph.json"), "w") as f:
        json.dump(results, f, indent=1)

    # console summary
    for name, m in results["models"].items():
        print("\n%s" % name)
        for c, v in m["terms"].items():
            tv = m["time_varying"][c]
            p_rank = m["ph_test"]["rank"][c]
            per = " ".join("%.3f" % row["terms"][c]["hr"] for row in m["by_period"])
            st = m["stratified_age_band_sex"]["terms"].get(c)
            print("  %-24s HR %.4f | GT rank chi2 %8.1f p %.1e | slope logHR/yr %+.4f [%+.4f, %+.4f] | "
                  "by period %s | stratified %s"
                  % (c, v["hr"], p_rank["chi2"], p_rank["p"], tv["slope_log_hr_per_year"], tv["slope_lo"],
                     tv["slope_hi"], per, "%.4f" % st["hr"] if st else "-"))
        if "older_vs_younger" in m:
            per = " ".join("%.3f (%.3f-%.3f)" % (r["older_vs_younger"]["hr"], r["older_vs_younger"]["lo"],
                                                r["older_vs_younger"]["hi"]) for r in m["by_period"])
            print("  older vs younger HR %.4f | by period %s | stratified %.4f"
                  % (m["older_vs_younger"]["hr"], per, m["stratified_age_band_sex"]["older_vs_younger"]["hr"]))
        print("  deaths by period: %s" % [r["deaths"] for r in m["by_period"]])
    print("runtime %.1f s" % results["runtime_s"])


if __name__ == "__main__":
    main()
