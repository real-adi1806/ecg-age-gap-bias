#!/usr/bin/env python3
"""
What sits under a foundation model's 1-year mortality AUC on CODE-15%? (14 Nov 2026)

A ladder of simple models for death within one year of the ECG, on the cohort used for that task in Gu et al.,
Nat Mach Intell 2026 (doi 10.1038/s42256-026-01180-5). Every number in results/ comes from this file.

    python analysis.py --boot 1000
    python analysis.py --boot 1000 --data /path/to/exams.csv --features /path/to/features

Rows (fitted on the training patients, any penalty chosen on the validation patients, scored once on test):
    row1_age              age used as a rank, no model
    row2_age_sex          logistic regression on age, age squared, male
    row3_age_sex_rhythm   row 2 plus 1dAVb, RBBB, LBBB, SB, ST, AF, normal_ecg
    row4_age_sex_gap      row 2 plus the ECG-age gap (network-predicted age minus age)
    row5_age_sex_lima     row 2 plus the 5,120 saved features of the Lima et al. age network; standardised on train,
                          ridge logistic regression, penalty chosen by validation log loss
    row5b_lima_only       the same ridge model without age and sex (the ECG alone, as a network sees it)
    check_age_sex_predicted_age   row 2 plus the predicted age itself: must equal row 4 (same information)

Row 5 needs the saved features of the exam the outcome is attached to. They exist for 217,932 of the 233,647
patients (see README.md), so rows 5 and 5b are scored on that subset of each test split, and rows 1 to 4 are scored
both on the whole test split and on the subset.

Results are written to disk after every block, so an interrupted run leaves usable output.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decision-curves"))   # reuse the repository's net-benefit code
import ladder as L  # noqa: E402
import netbenefit as nb  # noqa: E402

SEED_BASE = 20261114
SEEDS = [0, 1, 2, 3, 4]
PRIMARY = 0                                   # the split whose intervals, calibration and curves are reported
EXPECT = {"patients": 233647, "deaths_1y": 2812, "survivors_1y": 222057, "short_follow_up": 8778}
PILOT_AGE_AUC = 0.787                         # age alone on the whole cohort, from the pilot of 13 Nov
SUSPECT_AUC = 0.86                            # the plan's stop rule: above this, suspect leakage first
C_GRID = [float(10 ** e) for e in np.arange(-5, 1.01, 0.5)]   # sklearn's C (inverse penalty), weakest fit first
THRESHOLDS = [round(0.005 * k, 3) for k in range(1, 21)]      # 0.5% to 10%
BASE = ["age", "age2", "male"]
SMALL = {"row2_age_sex": BASE,
         "row3_age_sex_rhythm": BASE + L.RHYTHM_COLS,
         "row4_age_sex_gap": BASE + ["gap"],
         "check_age_sex_predicted_age": BASE + ["nn_predicted_age"]}
REF = "row2_age_sex"


def save(res, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)


def small_rows(c, y, tr, idx):
    """Rows 1 to 4: fit on the training rows `tr`, return predicted risk for the rows `idx`."""
    out = {"row1_age": c["age"].values[idx]}
    for name, cols in SMALL.items():
        beta = L.fit_logistic(c[cols].values[tr], y[tr])
        out[name] = L.predict_logistic(beta, c[cols].values[idx])
    return out


def standardised(X, row, c, tr, others):
    """
    Features for the training rows and for each index set in `others`, standardised with the TRAINING mean and SD.
    Columns that are constant on train (dead units of the network) are dropped. The last three columns are age,
    age squared and male, standardised the same way.
    """
    A = X[row[tr]]
    mu, sd = A.mean(axis=0, dtype=np.float64), A.std(axis=0, dtype=np.float64)
    keep = np.where(sd > 0)[0]
    mu, sd = mu[keep].astype(np.float32), sd[keep].astype(np.float32)
    D = c[BASE].values
    dmu, dsd = D[tr].mean(axis=0), D[tr].std(axis=0)

    def build(A, idx):
        Z = np.empty((len(idx), len(keep) + len(BASE)), dtype=np.float32)
        for s in range(0, len(idx), 20000):
            Z[s:s + 20000, :len(keep)] = (A[s:s + 20000][:, keep] - mu) / sd
        Z[:, len(keep):] = (D[idx] - dmu) / dsd
        return Z

    Ztr = build(A, tr)
    del A
    return Ztr, [build(X[row[i]], i) for i in others], int(len(keep))


def ridge_path(Ztr, ytr, Zva, yva, grid):
    """
    Ridge logistic regression along the penalty grid (warm started). The penalty is chosen by validation log loss.
    Stops once the validation log loss has been worse than its best for two grid steps in a row.
    """
    m = LogisticRegression(C=grid[0], warm_start=True, max_iter=5000, tol=1e-5)
    path, best, worse = [], None, 0
    for C in grid:
        m.C = C
        m.fit(Ztr, ytr)
        p = m.predict_proba(Zva)[:, 1]
        ll = float(log_loss(yva, p))
        path.append({"C": C, "val_log_loss": ll, "val_auc": L.auc_rank(yva, p), "iterations": int(m.n_iter_[0]),
                     "converged": bool(m.n_iter_[0] < m.max_iter)})
        if best is None or ll < best["ll"]:
            best, worse = {"ll": ll, "C": C, "coef": m.coef_.copy(), "intercept": m.intercept_.copy()}, 0
        else:
            worse += 1
            if worse >= 2:
                break
    at_edge = best["C"] == grid[-1]
    predict = lambda Z: 1 / (1 + np.exp(-(Z @ best["coef"].ravel() + best["intercept"][0]).astype(np.float64)))
    return predict, {"chosen_C": best["C"], "chosen_at_grid_edge": bool(at_edge), "path": path,
                     "all_converged": all(s["converged"] for s in path)}


def ridge_rows(c, X, row, y, tr, va, te):
    """Rows 5b (features only) and 5 (features plus age and sex) for the test rows `te`. All index sets have features."""
    Ztr, (Zva, Zte), n_kept = standardised(X, row, c, tr, [va, te])
    nb_ = len(BASE)
    demo = [Z[:, -nb_:].copy() for Z in (Ztr, Zva, Zte)]
    for Z in (Ztr, Zva, Zte):
        Z[:, -nb_:] = 0                        # age and sex switched off: their coefficients stay exactly zero
    pred_b, info_b = ridge_path(Ztr, y[tr], Zva, y[va], C_GRID)
    scores = {"row5b_lima_only": pred_b(Zte)}
    for Z, d in zip((Ztr, Zva, Zte), demo):
        Z[:, -nb_:] = d
    pred, info = ridge_path(Ztr, y[tr], Zva, y[va], C_GRID)
    scores["row5_age_sex_lima"] = pred(Zte)
    return scores, {"features_kept": n_kept, "features_constant_on_train": int(X.shape[1] - n_kept),
                    "train_patients": int(len(tr)), "train_deaths": int(y[tr].sum()),
                    "row5_age_sex_lima": info, "row5b_lima_only": info_b}


def evaluate(y, scores, B, rng):
    """AUC two ways, average precision, and bootstrap intervals, for a set of test patients."""
    out = {"patients": int(len(y)), "deaths": int(y.sum()), "death_rate": float(y.mean()),
           "auc": {k: L.auc_rank(y, s) for k, s in scores.items()},
           "auc_sklearn": {k: float(roc_auc_score(y, s)) for k, s in scores.items()},
           "ap": {k: float(average_precision_score(y, s)) for k, s in scores.items()}}
    out["auc_diff_vs_row2"] = {k: out["auc"][k] - out["auc"][REF] for k in scores if k != REF}
    if B:
        out.update(L.bootstrap(y, scores, REF, B, rng))
    return out


def run_split(c, X, row, y, in_cohort, split, B, rng, keep_scores=False):
    """The whole ladder for one split and one definition of the cohort (`in_cohort`) and of the label (`y`)."""
    has = row >= 0
    tr, va, te = (np.where(in_cohort & (split == s))[0] for s in ("train", "val", "test"))
    trf, vaf, tef = (np.where(in_cohort & has & (split == s))[0] for s in ("train", "val", "test"))
    pid = c["patient_id"].values
    overlap = len(set(pid[tr]) & set(pid[va])) + len(set(pid[tr]) & set(pid[te])) + len(set(pid[va]) & set(pid[te]))
    out = {"n": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                 "train_deaths": int(y[tr].sum()), "val_deaths": int(y[va].sum()), "test_deaths": int(y[te].sum())},
           "patients_in_two_splits": int(overlap)}
    s_all = small_rows(c, y, tr, te)                       # rows 1 to 4, fitted on every training patient
    out["test_all"] = evaluate(y[te], s_all, B, rng)
    s_sub = small_rows(c, y, tr, tef)                      # the same fitted models, scored on the feature subset
    s_ridge, out["ridge"] = ridge_rows(c, X, row, y, trf, vaf, tef)
    s_sub.update(s_ridge)
    out["test_with_features"] = evaluate(y[tef], s_sub, B, rng)
    return (out, {"te": te, "tef": tef, "all": s_all, "sub": s_sub}) if keep_scores else out


def primary_extras(c, y, sc, B, rng):
    """Calibration, decision curve and subgroups on the primary split's test patients."""
    yt, yf = y[sc["te"]], y[sc["tef"]]
    out = {"calibration": {"test_all": {}, "test_with_features": {}}}
    for k in ("row2_age_sex", "row4_age_sex_gap"):
        out["calibration"]["test_all"][k] = {**L.calibration(yt, sc["all"][k]),
                                             **L.calibration_slope_ci(yt, sc["all"][k], B, rng)}
    for k in ("row2_age_sex", "row4_age_sex_gap", "row5_age_sex_lima", "row5b_lima_only"):
        out["calibration"]["test_with_features"][k] = {**L.calibration(yf, sc["sub"][k]),
                                                       **L.calibration_slope_ci(yf, sc["sub"][k], B, rng)}
    # decision curve, rows 2 and 5 on the same patients (netbenefit.py from ../decision-curves, Vickers and Elkin 2006)
    dc = {"thresholds": THRESHOLDS, "treat_all": nb.net_benefit_treat_all_binary(yf, THRESHOLDS).tolist(),
          "treat_none": [0.0] * len(THRESHOLDS)}
    for k in ("row2_age_sex", "row5_age_sex_lima"):
        dc[k] = nb.net_benefit_binary(yf, sc["sub"][k], THRESHOLDS).tolist()
        dc[k + "_share_treated"] = [float((sc["sub"][k] >= t).mean()) for t in THRESHOLDS]
    diffs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(yf), len(yf)), minlength=len(yf)).astype(float)
        diffs.append(nb.net_benefit_binary(yf, sc["sub"]["row5_age_sex_lima"], THRESHOLDS, w)
                     - nb.net_benefit_binary(yf, sc["sub"]["row2_age_sex"], THRESHOLDS, w))
    if diffs:
        dc["row5_minus_row2_ci95"] = np.percentile(np.array(diffs), [2.5, 97.5], axis=0).T.tolist()
    out["decision_curve_test_with_features"] = dc
    # subgroups of the feature subset: sex and four age bands
    age, male = c["age"].values[sc["tef"]], c["male"].values[sc["tef"]]
    groups = {"male": male == 1, "female": male == 0}
    for lo, hi in L.AGE_BANDS:
        groups[f"age {lo}-{hi - 1 if hi < 200 else '+'}"] = (age >= lo) & (age < hi)
    sg = {}
    for g, m in groups.items():
        sub = {k: sc["sub"][k][m] for k in ("row2_age_sex", "row4_age_sex_gap", "row5_age_sex_lima")}
        e = evaluate(yf[m], sub, B, rng)
        sg[g] = {k: e[k] for k in ("patients", "deaths", "auc", "auc_ci95", "auc_diff_vs_row2",
                                   "auc_diff_vs_row2_age_sex_ci95") if k in e}
    out["subgroups_test_with_features"] = sg
    return out


def run_checks(res, c, y, main):
    """Every check must pass before any number is written into SUMMARY.md."""
    ch = {}
    whole = L.auc_rank(y[main], c["age"].values[main])
    ch["1_age_alone_whole_cohort"] = {"auc": whole, "pilot": PILOT_AGE_AUC, "pass": round(whole, 3) == PILOT_AGE_AUC}
    worst = max(abs(s[blk]["auc"][k] - s[blk]["auc_sklearn"][k])
                for s in res["seeds"].values() for blk in ("test_all", "test_with_features") for k in s[blk]["auc"])
    ch["2_rank_formula_vs_sklearn"] = {"largest_absolute_difference": worst, "pass": worst < 5e-5}
    sh = res["shuffled_labels_primary"]
    aucs = {**sh["test_all"]["auc"], **sh["test_with_features"]["auc"]}
    ch["3_shuffled_labels"] = {"auc": aucs, "pass": all(0.44 < v < 0.56 for v in aucs.values())}
    ch["4_no_patient_in_two_splits"] = {"overlaps": [s["patients_in_two_splits"] for s in res["seeds"].values()],
                                        "pass": all(s["patients_in_two_splits"] == 0 for s in res["seeds"].values())}
    d5 = max(abs(s["test_all"]["auc"]["row4_age_sex_gap"] - s["test_all"]["auc"]["check_age_sex_predicted_age"])
             for s in res["seeds"].values())
    ch["5_gap_equals_predicted_age_given_age"] = {"largest_auc_difference": d5, "pass": d5 < 1e-6}
    r5 = [s["test_with_features"]["auc"]["row5_age_sex_lima"] for s in res["seeds"].values()]
    ch["stop_rule_row5_not_above_0.86"] = {"largest_row5_auc": max(r5), "pass": max(r5) <= SUSPECT_AUC}
    ridge = [s["ridge"][k] for s in res["seeds"].values() for k in ("row5_age_sex_lima", "row5b_lima_only")]
    ch["ridge_converged_and_penalty_inside_grid"] = {
        "pass": all(r["all_converged"] and not r["chosen_at_grid_edge"] for r in ridge)}
    ch["all_passed"] = all(v["pass"] for v in ch.values())
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (the reported run used 1000)")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "exams.csv"), help="path to exams.csv")
    ap.add_argument("--features", default=os.path.join(HERE, "..", "..", "ecg-age-model", "features"),
                    help="folder holding code_part{k}_feat.npy and code_part{k}_ids.npy")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    args = ap.parse_args()
    if args.boot < 1:
        sys.exit("--boot must be at least 1")
    for p in (args.data, args.features):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    res = {"seed_base": SEED_BASE, "seeds_run": SEEDS, "primary_seed": PRIMARY, "boot": args.boot, "seeds": {}}

    # ---- step 1: the cohort. Stop if it does not match the pilot. ----
    c = L.build_cohort(args.data)
    y = c["died1"].values
    short = c["short"].values
    main_cohort = ~short
    got = {"patients": int(len(c)), "deaths_1y": int(y.sum()), "survivors_1y": int((main_cohort & (y == 0)).sum()),
           "short_follow_up": int(short.sum())}
    res["cohort"] = {**got, "expected": EXPECT,
                     "death_rate_main": float(y[main_cohort].mean()), "death_rate_sensitivity_a": float(y.mean()),
                     "published_death_rate": 0.0123, "median_age": float(c["age"].median()),
                     "male_share": float(c["male"].mean())}
    save(res, args.out)
    if got != EXPECT:
        sys.exit(f"STOP: cohort counts {got} do not match the pilot {EXPECT}. Nothing was fitted.")

    # ---- the saved features, matched on the outcome exam ----
    X, row = L.load_features(args.features, c["exam_id"].values)
    has = row >= 0
    res["features"] = {"patients_with_outcome_exam_features": int(has.sum()), "patients_without": int((~has).sum()),
                       "main_cohort_with": int((has & main_cohort).sum()),
                       "deaths_1y_with": int(y[has & main_cohort].sum()), "deaths_1y_without": int(y[~has].sum()),
                       "death_rate_with": float(y[has & main_cohort].mean()),
                       "death_rate_without": float(y[~has & main_cohort].mean()),
                       "feature_columns": int(X.shape[1])}
    save(res, args.out)
    print(json.dumps({"cohort": res["cohort"], "features": res["features"]}, indent=1), flush=True)

    # ---- steps 2 and 3: five patient-level splits, the ladder on each ----
    splits = {s: L.split_patients(y, SEED_BASE + s) for s in SEEDS}       # all 233,647 patients, stratified on y
    primary_scores = None
    for s in SEEDS:
        rng = np.random.default_rng(SEED_BASE + 1000 + s)
        out = run_split(c, X, row, y, main_cohort, splits[s], args.boot, rng, keep_scores=(s == PRIMARY))
        if s == PRIMARY:
            out, primary_scores = out
        res["seeds"][str(s)] = out
        res["seconds"] = round(time.time() - t0, 1)
        save(res, args.out)
        print(f"seed {s}: " + ", ".join(f"{k} {v:.4f}" for k, v in out["test_with_features"]["auc"].items()),
              f"[{res['seconds']}s]", flush=True)

    # ---- step 4: calibration, decision curve, subgroups on the primary split ----
    rng = np.random.default_rng(SEED_BASE + 2000)
    res["primary"] = primary_extras(c, y, primary_scores, args.boot, rng)
    save(res, args.out)
    print("primary extras done", flush=True)

    # ---- sensitivity A: the patients alive with under a year of follow-up kept as survivors ----
    res["sensitivity_a_primary"] = run_split(c, X, row, y, np.ones(len(c), dtype=bool), splits[PRIMARY], args.boot,
                                             np.random.default_rng(SEED_BASE + 3000))
    res["sensitivity_a_primary"]["age_alone_whole_cohort"] = L.auc_rank(y, c["age"].values)
    save(res, args.out)
    print("sensitivity A done", flush=True)

    # ---- step 5: checks ----
    y_shuffled = y.copy()
    idx = np.where(main_cohort)[0]
    y_shuffled[idx] = y[np.random.default_rng(SEED_BASE + 4000).permutation(idx)]
    res["shuffled_labels_primary"] = run_split(c, X, row, y_shuffled, main_cohort, splits[PRIMARY], 0, None)
    res["checks"] = run_checks(res, c, y, main_cohort)
    res["seconds"] = round(time.time() - t0, 1)
    save(res, args.out)
    print(json.dumps(res["checks"], indent=1))
    if not res["checks"]["all_passed"]:
        sys.exit("STOP: a check failed. Do not use these numbers. See results.json, key 'checks'.")


if __name__ == "__main__":
    main()
