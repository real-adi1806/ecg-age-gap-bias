#!/usr/bin/env python3
"""
Does the ECG-age gap add anything over age and sex on CODE-15%? (8 Nov 2026)

Uses the loader and the Barthels correction from the repository root (ecg_age_bias.py), so the
cohort and the corrected gap are defined exactly as in that replication. Every number in results/ comes from
this file.

    python analysis.py --boot 1000
    python analysis.py --boot 1000 --data /path/to/exams.csv

Lives inside the ecg-age-gap-bias repository and imports ecg_age_bias.py from the folder above. By default it
reads exams.csv from that repository's data/ folder. See README.md.

Models (Cox proportional hazards, fitted on the training patients, scored on held-out patients):
    age_sex        age + male                         same specification as the published demographic-only model
    age_sex_gap    age + male + raw gap (pred - age)
    age_sex_gapbc  age + male + corrected gap         identical risk scores to age_sex_gap, by construction
    gap_raw        raw gap alone
    gap_bc         corrected gap alone
    age            age alone                          for reference
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPER_A = os.path.join(HERE, "..")  # the repository root, one level up
base = None  # the ecg_age_bias module, imported in main() once its folder is known

SEED = 20261108
MODELS = {
    "age_sex": ["age", "male"],
    "age_sex_gap": ["age", "male", "pad"],
    "age_sex_gapbc": ["age", "male", "pad_bc"],
    "gap_raw": ["pad"],
    "gap_bc": ["pad_bc"],
    "age": ["age"],
}
AGE_BANDS = [(16, 40), (40, 60), (60, 75), (75, 200)]


def import_base(paper_a):
    """Import ecg_age_bias.py from the repository root (or another folder given by --paper-a)."""
    global base
    if not os.path.isfile(os.path.join(paper_a, "ecg_age_bias.py")):
        sys.exit(f"ecg_age_bias.py not found in {paper_a}. Run this from inside the repository, or pass --paper-a.")
    sys.path.insert(0, paper_a)
    import ecg_age_bias
    base = ecg_age_bias


def load(data_path):
    if not os.path.isfile(data_path):
        sys.exit(f"exams.csv not found at {data_path}. Download it from Zenodo record 4916206 or pass --data.")
    rows, stats = base.load(data_path)
    df = pd.DataFrame(rows)
    df["male"] = df["male"].astype(float)
    df["died"] = df["died"].astype(int)
    return df, stats


def correct(train, frames):
    """Barthels' residual correction, fitted on the training patients only (bias_correct in ecg_age_bias.py)."""
    tr = train.to_dict("records")
    slope, intercept = base.bias_correct(tr)            # fits pad ~ age on train
    for f in frames:
        f["pad_bc"] = f["pad"] - (intercept + slope * f["age"])
    return slope, intercept


def fit(train, cols):
    cph = CoxPHFitter()
    cph.fit(train[cols + ["timey", "died"]], duration_col="timey", event_col="died")
    return cph


def cindex(df, risk):
    # higher risk should mean earlier death, so pass the negative risk as the "predicted survival"
    return concordance_index(df["timey"].values, -np.asarray(risk), df["died"].values)


def harrell_naive(t, risk, e):
    """O(n^2) Harrell's C, written out, to check lifelines on a subsample. Ties in risk count one half."""
    conc = disc = tied = 0.0
    n = len(t)
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if t[j] > t[i]:                      # j outlived i, so i should have the higher risk
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] < risk[j]:
                    disc += 1
                else:
                    tied += 1
    return (conc + 0.5 * tied) / (conc + disc + tied)


def boot_ci(test, risks, pairs, B, rng):
    """Patient-level bootstrap of the held-out set: C for each model, and paired differences."""
    n = len(test)
    t, e = test["timey"].values, test["died"].values
    out = {k: [] for k in risks}
    diffs = {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        cs = {k: concordance_index(t[idx], -r[idx], e[idx]) for k, r in risks.items()}
        for k, c in cs.items():
            out[k].append(c)
        for a, b in pairs:
            diffs[f"{a} - {b}"].append(cs[a] - cs[b])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci(v) for k, v in out.items()}, {k: ci(v) for k, v in diffs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200, help="bootstrap samples (the reported run used 1000)")
    ap.add_argument("--paper-a", default=DEFAULT_PAPER_A, help="folder holding ecg_age_bias.py")
    ap.add_argument("--data", default=None, help="path to exams.csv (default: <paper-a>/data/exams.csv)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"), help="where to write the results")
    args = ap.parse_args()
    if args.boot < 1:
        sys.exit("--boot must be at least 1")
    import_base(args.paper_a)
    data_path = args.data or os.path.join(args.paper_a, "data", "exams.csv")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "boot": args.boot}

    df, stats = load(data_path)
    res["cohort"] = {**stats, "patients": int(len(df)), "deaths": int(df["died"].sum()),
                     "median_follow_up_years": float(df["timey"].median()),
                     "male_share": float(df["male"].mean()), "median_age": float(df["age"].median())}

    # ---- 80/20 split by patient (one row per patient already) ----
    order = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    train, test = df.iloc[order[:cut]].copy(), df.iloc[order[cut:]].copy()
    slope, intercept = correct(train, [train, test])
    res["correction_fitted_on_train"] = {"slope": slope, "intercept": intercept}
    res["split"] = {"train": int(len(train)), "test": int(len(test)),
                    "train_deaths": int(train["died"].sum()), "test_deaths": int(test["died"].sum())}

    # ---- fit on train, score on held-out patients ----
    fits, risks, held = {}, {}, {}
    for name, cols in MODELS.items():
        cph = fit(train, cols)
        fits[name] = {c: {"hr": float(np.exp(cph.params_[c])),
                          "hr_lo95": float(np.exp(cph.confidence_intervals_.loc[c].iloc[0])),
                          "hr_hi95": float(np.exp(cph.confidence_intervals_.loc[c].iloc[1]))} for c in cols}
        risks[name] = cph.predict_log_partial_hazard(test[cols]).values
        held[name] = cindex(test, risks[name])
    res["hazard_ratios_train"] = fits
    res["c_index_test"] = held

    # by construction: the raw-gap and corrected-gap models give the same ranking once age is in the model
    res["check_same_scores_raw_vs_corrected_with_age"] = {
        "corr_of_risk_scores": float(np.corrcoef(risks["age_sex_gap"], risks["age_sex_gapbc"])[0, 1]),
        "c_difference": float(held["age_sex_gap"] - held["age_sex_gapbc"])}

    # ---- check lifelines' concordance against the slow, written-out version on a subsample ----
    sub = rng.choice(len(test), 2500, replace=False)
    ts, es = test["timey"].values[sub], test["died"].values[sub]
    res["check_concordance_impl"] = {k: {"lifelines": float(concordance_index(ts, -risks[k][sub], es)),
                                         "naive": float(harrell_naive(ts, risks[k][sub], es))}
                                     for k in ("age_sex", "age_sex_gap", "gap_raw", "gap_bc")}

    # ---- bootstrap intervals on the held-out set ----
    pairs = [("age_sex_gap", "age_sex"), ("gap_bc", "gap_raw"), ("age_sex", "age")]
    res["c_index_test_ci95"], res["c_diff_test_ci95"] = boot_ci(test, risks, pairs, args.boot, rng)
    res["c_diff_test"] = {f"{a} - {b}": float(held[a] - held[b]) for a, b in pairs}

    # ---- subgroups of the held-out set: does the gap help within sex and within narrow age bands? ----
    sg = {}
    groups = {"male": test["male"] == 1, "female": test["male"] == 0}
    for lo, hi in AGE_BANDS:
        groups[f"age {lo}-{hi - 1 if hi < 200 else '+'}"] = (test["age"] >= lo) & (test["age"] < hi)
    for g, m in groups.items():
        m = m.values
        sub_t = test[m]
        r0, r1 = risks["age_sex"][m], risks["age_sex_gap"][m]
        c0, c1 = cindex(sub_t, r0), cindex(sub_t, r1)
        diffs = []
        tt, ee = sub_t["timey"].values, sub_t["died"].values
        for _ in range(args.boot):
            idx = rng.integers(0, len(sub_t), len(sub_t))
            diffs.append(concordance_index(tt[idx], -r1[idx], ee[idx]) - concordance_index(tt[idx], -r0[idx], ee[idx]))
        sg[g] = {"patients": int(m.sum()), "deaths": int(ee.sum()), "c_age_sex": float(c0), "c_age_sex_gap": float(c1),
                 "diff": float(c1 - c0), "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    res["subgroups_test"] = sg

    # ---- 5-fold cross-validation, as a check that the one split is not a fluke ----
    folds = np.array_split(rng.permutation(len(df)), 5)
    cv = {k: [] for k in ("age_sex", "age_sex_gap", "gap_raw", "gap_bc")}
    for k in range(5):
        te = df.iloc[folds[k]].copy()
        tr = df.iloc[np.concatenate([folds[j] for j in range(5) if j != k])].copy()
        correct(tr, [tr, te])
        for name in cv:
            cph = fit(tr, MODELS[name])
            cv[name].append(float(cindex(te, cph.predict_log_partial_hazard(te[MODELS[name]]).values)))
    res["cv5"] = {k: {"folds": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in cv.items()}

    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("cohort", "split", "c_index_test", "c_diff_test", "c_diff_test_ci95",
                                          "check_same_scores_raw_vs_corrected_with_age", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
