"""
Paper A: decision curve and horizon calibration for "does the ECG-age gap add anything
over age and sex?" on CODE-15% (Brazil; the German cohort has no outcomes).

Run from the repository root:   python decision-curves/dca_paper_a.py [--boot 1000]
Writes decision-curves/results_dca.json and decision-curves/dca.npz (read by make_figures_a.py).

The cohort, the 80/20 split by patient, the Barthels correction and the Cox fits are
taken from ecg-age-bias/gap-over-age-sex/analysis.py, imported read-only, and the
script stops unless its held-out concordance comes back exactly (0.789967, 0.798810).

HORIZON, fixed before any curve was computed (see METHODS.md):
    primary     2 years: 77.2% of patients are still under observation at 2 years,
                and it is the shorter of the two Kaplan-Meier horizons the manuscript
                already uses (2 and 4 years).
    sensitivity 4 years: 40.5% still under observation, so the Kaplan-Meier risk in
                each threshold group leans more on censored follow-up.
THRESHOLDS: 0.25% to 20% two-year death risk in steps of 0.25%; summary band 0.5% to
10%, also fixed in advance. Net benefit uses the Kaplan-Meier risk among patients above
each threshold (Vickers et al. 2008; McLernon et al. 2023).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
from lifelines import KaplanMeierFitter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netbenefit as nb  # noqa: E402

REPO = os.path.dirname(HERE)  # the repository root
GAP = os.path.join(REPO, "gap-over-age-sex")
HORIZONS = (2.0, 4.0)
THRESHOLDS = np.round(np.arange(0.0025, 0.2000001, 0.0025), 4)
BAND = (0.005, 0.10)
BOOT_SEED = 20261112


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("gap_analysis", os.path.join(GAP, "analysis.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.import_base(REPO)
    return mod


def rebuild_split(an):
    """Exactly the split in analysis.py main(): first draw of default_rng(SEED)."""
    df, stats = an.load(os.path.join(REPO, "data", "exams.csv"))
    rng = np.random.default_rng(an.SEED)
    order = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    train, test = df.iloc[order[:cut]].copy(), df.iloc[order[cut:]].copy()
    an.correct(train, [train, test])
    return df, train, test


def risk_at(cph, X, horizon):
    """1 - S(horizon | x) from the fitted Cox model (Breslow baseline)."""
    s = cph.predict_survival_function(X, times=[horizon]).values[0]
    return 1.0 - np.asarray(s, dtype=float)


def band_mask(thr):
    return (thr >= BAND[0] - 1e-12) & (thr <= BAND[1] + 1e-12)


def summarise(boot, point, thr):
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    bm = band_mask(thr)
    bb, fb = boot[:, bm].mean(axis=1), boot.mean(axis=1)
    return {
        "per_threshold": {"%.4f" % t: {"diff": float(d), "lo": float(a), "hi": float(b)}
                          for t, d, a, b in zip(thr, point, lo, hi)},
        "band_mean": {"diff": float(point[bm].mean()), "lo": float(np.percentile(bb, 2.5)),
                      "hi": float(np.percentile(bb, 97.5))},
        "full_range_mean": {"diff": float(point.mean()), "lo": float(np.percentile(fb, 2.5)),
                            "hi": float(np.percentile(fb, 97.5))},
        "thresholds_ci_above_0": [float(t) for t in thr[lo > 0]],
        "thresholds_ci_below_0": [float(t) for t in thr[hi < 0]],
    }, lo, hi


def calibration_by_decile(p, time_, event, horizon):
    """Observed (Kaplan-Meier, log(-log) Greenwood interval) against mean predicted, by decile of p."""
    q = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    out = []
    for b in range(10):
        m = bins == b
        km = KaplanMeierFitter().fit(time_[m], event[m])
        s = float(km.survival_function_at_times(horizon).iloc[0])
        ci = km.confidence_interval_survival_function_
        idx = ci.index.searchsorted(horizon, side="right") - 1
        s_lo, s_hi = float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
        out.append({"decile": b + 1, "n": int(m.sum()), "deaths_by_horizon": int(((time_ <= horizon) & (event == 1) & m).sum()),
                    "mean_predicted": float(p[m].mean()), "observed_km": 1 - s,
                    "observed_lo": 1 - s_hi, "observed_hi": 1 - s_lo})
    return out


def run_horizon(horizon, test, preds, n_boot, thr):
    t, e = test["timey"].values, test["died"].values.astype(int)
    frame = nb.SurvivalFrame(t, e, horizon)
    km_risk = nb.km_risk_at_horizon(frame)
    curves = {k: nb.net_benefit_survival(frame, p, thr) for k, p in preds.items()}
    treat_all = nb.net_benefit_treat_all_survival(frame, thr)
    assert abs(nb.net_benefit_treat_all_survival(frame, [km_risk])[0]) < 1e-12, \
        "treat all must cross zero at the horizon risk"

    # Real-data cross-check: in patients with no censoring before the horizon (died by it, or
    # followed past it), the survival formula must equal the binary formula exactly.
    known = (t > horizon) | (e == 1)
    yk = ((t <= horizon) & (e == 1)).astype(int)[known]
    fk = nb.SurvivalFrame(t[known], e[known], horizon)
    cross = {}
    for k, p in preds.items():
        a = nb.net_benefit_survival(fk, p[known], thr)
        b = nb.net_benefit_binary(yk, p[known], thr)
        cross[k] = float(np.max(np.abs(a - b)))
        assert cross[k] < 1e-12, "survival and binary net benefit disagree without censoring"

    # Paired bootstrap of the held-out patients (models fixed), as multinomial weights.
    rng = np.random.default_rng(BOOT_SEED + int(horizon * 10))
    n = len(t)
    keys = list(preds)
    boot = {k: np.empty((n_boot, len(thr))) for k in keys}
    boot_all = np.empty((n_boot, len(thr)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k in keys:
            boot[k][b] = nb.net_benefit_survival(frame, preds[k], thr, w=w)
        boot_all[b] = nb.net_benefit_treat_all_survival(frame, thr, w=w)
    diff, lo, hi = summarise(boot["age_sex_gap"] - boot["age_sex"],
                             curves["age_sex_gap"] - curves["age_sex"], thr)
    vs_default = {}
    for k in keys:
        s, _, _ = summarise(boot[k] - np.maximum(boot_all, 0), curves[k] - np.maximum(treat_all, 0), thr)
        vs_default[k] = {kk: vv for kk, vv in s.items() if kk != "per_threshold"}

    calib = {k: calibration_by_decile(p, t, e, horizon) for k, p in preds.items()}
    return {
        "horizon_years": horizon,
        "under_observation_past_horizon_pct": float(100 * (t > horizon).mean()),
        "censored_before_horizon_pct": float(100 * ((t <= horizon) & (e == 0)).mean()),
        "deaths_by_horizon": int(((t <= horizon) & (e == 1)).sum()),
        "km_risk_at_horizon": float(km_risk),
        "mean_predicted_risk": {k: float(p.mean()) for k, p in preds.items()},
        "observed_over_expected": {k: float(km_risk / p.mean()) for k, p in preds.items()},
        "predicted_risk_quantiles": {k: {q: float(np.quantile(p, float(q))) for q in ("0.01", "0.1", "0.5", "0.9", "0.99")}
                                     for k, p in preds.items()},
        "share_above_threshold_pct": {k: {"%.4f" % tt: float(100 * (p >= tt).mean()) for tt in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)}
                                      for k, p in preds.items()},
        "treat_all_zero_crossing": float(km_risk),
        "crosscheck_uncensored_max_abs_diff": cross,
        "curves": {k: v.tolist() for k, v in curves.items()},
        "treat_all": treat_all.tolist(),
        "gap_minus_age_sex": diff,
        "model_minus_best_default": vs_default,
        "calibration_deciles": calib,
        "_lo": lo, "_hi": hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    an = load_analysis_module()
    data_csv = os.path.join(REPO, "data", "exams.csv")
    df, train, test = rebuild_split(an)
    published = json.load(open(os.path.join(GAP, "results", "results.json")))

    fits = {k: an.fit(train, an.MODELS[k]) for k in ("age_sex", "age_sex_gap", "age_sex_gapbc")}
    lp = {k: f.predict_log_partial_hazard(test[an.MODELS[k]]).values for k, f in fits.items()}
    repro = {}
    for k in fits:
        got = float(an.cindex(test, lp[k]))
        want = float(published["c_index_test"][k])
        repro["c_index_" + k] = {"got": got, "published": want}
        assert abs(got - want) < 1e-9, "held-out concordance for %s did not reproduce" % k
    for k, v in published["split"].items():
        got = {"train": len(train), "test": len(test), "train_deaths": int(train["died"].sum()),
               "test_deaths": int(test["died"].sum())}[k]
        repro["split_" + k] = {"got": got, "published": v}
        assert got == v
    hr_gap = float(np.exp(fits["age_sex_gap"].params_["pad"]))
    repro["hr_gap_per_year"] = {"got": hr_gap, "published": published["hazard_ratios_train"]["age_sex_gap"]["pad"]["hr"]}
    assert abs(hr_gap - repro["hr_gap_per_year"]["published"]) < 1e-9
    print("REPRODUCED: split %s, held-out C age+sex %.6f, age+sex+gap %.6f, HR gap %.6f"
          % (published["split"], repro["c_index_age_sex"]["got"], repro["c_index_age_sex_gap"]["got"], hr_gap))

    out = {"what": "Paper A decision curve and horizon calibration. Every number from dca_paper_a.py.",
           "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exams_csv_md5": md5(data_csv), "reproduction": repro,
           "thresholds": THRESHOLDS.tolist(), "band": list(BAND), "n_boot": args.boot,
           "boot_seed": BOOT_SEED, "test_patients": int(len(test)), "test_deaths_all_follow_up": int(test["died"].sum()),
           "follow_up_all_patients": {"median_years": float(df["timey"].median()),
                                      **{"pct_under_observation_past_%dy" % y: float(100 * (df["timey"] > y).mean())
                                         for y in (1, 2, 3, 4, 5)}},
           "horizons": {}}
    arrays = {"thr": THRESHOLDS}
    for h in HORIZONS:
        preds = {k: risk_at(fits[k], test[an.MODELS[k]], h) for k in fits}
        same = float(np.max(np.abs(preds["age_sex_gap"] - preds["age_sex_gapbc"])))
        print("horizon %g: max |risk raw-gap model - risk corrected-gap model| = %.2e" % (h, same))
        assert same < 1e-9, "raw and corrected gap should give the same risks once age is in the model"
        preds.pop("age_sex_gapbc")
        r = run_horizon(h, test, preds, args.boot, THRESHOLDS)
        r["max_abs_risk_diff_raw_vs_corrected_gap"] = same
        tag = "%gy" % h
        arrays.update({"%s__curve__%s" % (tag, k): np.asarray(v) for k, v in r["curves"].items()})
        arrays["%s__treat_all" % tag] = np.asarray(r["treat_all"])
        arrays["%s__diff_lo" % tag], arrays["%s__diff_hi" % tag] = r.pop("_lo"), r.pop("_hi")
        arrays.update({"%s__pred__%s" % (tag, k): v for k, v in preds.items()})
        out["horizons"][tag] = r
        g = r["gap_minus_age_sex"]
        print("  KM risk %.4f, deaths by horizon %d, censored before %.1f%%"
              % (r["km_risk_at_horizon"], r["deaths_by_horizon"], r["censored_before_horizon_pct"]))
        print("  O/E age+sex %.3f, age+sex+gap %.3f" % tuple(r["observed_over_expected"].values()))
        print("  gap minus age+sex: band %.3f-%.3f mean %+.5f [%+.5f, %+.5f]; full %+.5f [%+.5f, %+.5f]"
              % (BAND[0], BAND[1], g["band_mean"]["diff"], g["band_mean"]["lo"], g["band_mean"]["hi"],
                 g["full_range_mean"]["diff"], g["full_range_mean"]["lo"], g["full_range_mean"]["hi"]))
        print("  CI above 0 at %s" % g["thresholds_ci_above_0"])
        print("  CI below 0 at %s" % g["thresholds_ci_below_0"])
        for k, v in r["model_minus_best_default"].items():
            print("  %s minus best default: band %+.5f [%+.5f, %+.5f]" % (k, v["band_mean"]["diff"], v["band_mean"]["lo"], v["band_mean"]["hi"]))
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "results_dca.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(HERE, "dca.npz"), y_time=test["timey"].values, y_event=test["died"].values.astype(int), **arrays)
    print("runtime %.1f s" % out["runtime_s"])


if __name__ == "__main__":
    main()
