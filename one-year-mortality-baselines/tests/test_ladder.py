"""
Tests for ladder.py. Run from this folder with either
    python -m pytest tests -q
    python tests/test_ladder.py          (no pytest needed)
Expected values are worked by hand in the comment beside them.
"""
import os
import sys
import tempfile

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "decision-curves"))
import ladder as L  # noqa: E402
import netbenefit as nb  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_auc_hand():
    # deaths score 3 and 1; survivors 2 and 0. Pairs: (3,2) win, (3,0) win, (1,2) loss, (1,0) win: 3 of 4.
    assert close(L.auc_rank([1, 0, 1, 0], [3, 2, 1, 0]), 0.75)
    assert close(L.auc_pairs([1, 0, 1, 0], [3, 2, 1, 0]), 0.75)


def test_auc_ties_count_half():
    # one death and one survivor share a score: that pair counts one half. Pairs: (1 vs 1) half, (1 vs 0) win: 0.75.
    assert close(L.auc_rank([1, 0, 0], [1, 1, 0]), 0.75)
    # every score equal: 0.5
    assert close(L.auc_rank([1, 0, 1, 0, 0], [7, 7, 7, 7, 7]), 0.5)


def test_auc_three_implementations_agree():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.05).astype(int)
    s = np.round(rng.normal(size=3000) + 0.8 * y, 1)          # rounded, so there are many ties
    a, b, c = L.auc_rank(y, s), L.auc_pairs(y, s), roc_auc_score(y, s)
    assert close(a, b, 1e-12) and close(a, c, 1e-12)


def test_auc_one_class_is_nan():
    assert np.isnan(L.auc_rank([0, 0, 0], [1, 2, 3]))


def test_split_is_disjoint_complete_and_stratified():
    rng = np.random.default_rng(2)
    y = (rng.random(10000) < 0.02).astype(int)
    s = L.split_patients(y, seed=0)
    assert set(np.unique(s)) == {"train", "val", "test"}
    assert (s == "train").sum() + (s == "val").sum() + (s == "test").sum() == len(y)
    d = int(y.sum())
    assert abs(y[s == "train"].sum() - 0.8 * d) <= 1 and abs(y[s == "val"].sum() - 0.1 * d) <= 1
    assert abs((s == "train").mean() - 0.8) < 0.001
    assert np.array_equal(s, L.split_patients(y, seed=0))      # same seed, same split
    assert not np.array_equal(s, L.split_patients(y, seed=1))


def test_fit_logistic_matches_sklearn_and_recovers_truth():
    rng = np.random.default_rng(3)
    n = 20000
    age = rng.uniform(17, 100, n)
    male = (rng.random(n) < 0.4).astype(float)
    X = np.column_stack([age, age ** 2, male])
    y = (rng.random(n) < expit(-6 + 0.05 * age + 0.4 * male)).astype(int)
    beta = L.fit_logistic(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=5000).fit(Z, y)
    assert np.max(np.abs(L.predict_logistic(beta, X) - sk.predict_proba(Z)[:, 1])) < 1e-5
    assert abs(beta[3] - 0.4) < 0.15
    # at the maximum-likelihood fit the mean prediction equals the event rate exactly
    assert close(L.predict_logistic(beta, X).mean(), y.mean(), 1e-9)


def test_fit_logistic_intercept_only_with_offset():
    # no offset, no columns: the intercept is the log odds of the event rate. 2 of 10: log(0.25).
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    b = L.fit_logistic(np.empty((10, 0)), y)
    assert close(b[0], np.log(0.25), 1e-8)


def test_calibration_of_perfect_and_distorted_predictions():
    rng = np.random.default_rng(4)
    lp = rng.normal(-4, 1.2, 200000)
    y = (rng.random(len(lp)) < expit(lp)).astype(int)
    good = L.calibration(y, expit(lp))
    assert abs(good["slope"] - 1) < 0.03 and abs(good["intercept_in_the_large"]) < 0.05
    # predictions twice as extreme on the logit scale: the slope must come out near one half
    assert abs(L.calibration(y, expit(2 * lp))["slope"] - 0.5) < 0.02
    # every prediction's odds multiplied by e: the intercept in the large must come out near -1
    assert abs(L.calibration(y, expit(lp + 1))["intercept_in_the_large"] + 1) < 0.05
    assert sum(g["n"] for g in good["groups"]) == len(y) and len(good["groups"]) == 10


def test_bootstrap_brackets_the_estimate_and_pairs_the_difference():
    rng = np.random.default_rng(5)
    y = (rng.random(4000) < 0.1).astype(int)
    a = rng.normal(size=4000) + y
    scores = {"a": a, "same": a.copy(), "noise": rng.normal(size=4000)}
    out = L.bootstrap(y, scores, "a", 200, np.random.default_rng(6))
    lo, hi = out["auc_ci95"]["a"]
    assert lo < L.auc_rank(y, a) < hi
    assert out["auc_diff_vs_a_ci95"]["same"] == [0.0, 0.0]     # identical scores: the paired difference is exactly 0
    assert out["auc_diff_vs_a_ci95"]["noise"][1] < 0


def test_build_cohort_labels():
    # five exams, four patients. Patient 1 has two exams, the outcome on one of them only.
    csv = ("exam_id,age,is_male,nn_predicted_age,1dAVb,RBBB,LBBB,SB,ST,AF,patient_id,death,timey,normal_ecg,trace_file\n"
           "10,50,True,55.0,False,False,False,False,False,False,1,True,0.5,False,a\n"      # died at 0.5 y: died1
           "11,53,True,58.0,False,False,False,False,False,False,1,,,False,a\n"             # later exam, no outcome
           "12,60,False,58.0,False,True,False,False,False,False,2,True,2.0,False,a\n"      # died at 2 y: survivor at 1 y
           "13,40,False,41.0,False,False,False,False,False,False,3,False,0.4,True,a\n"     # alive, 0.4 y: short
           "14,30,True,33.0,False,False,False,False,False,False,4,False,3.0,True,a\n")     # alive, 3 y: survivor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exams.csv")
        open(path, "w").write(csv)
        c = L.build_cohort(path)
    assert list(c["exam_id"]) == [10, 12, 13, 14]
    assert list(c["died1"]) == [1, 0, 0, 0]
    assert list(c["short"]) == [False, False, True, False]
    assert list(c["gap"]) == [5.0, -2.0, 1.0, 3.0]
    assert list(c["RBBB"]) == [0.0, 1.0, 0.0, 0.0]


def test_load_features_matches_on_exam_id():
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/code_part0_ids.npy", np.array([5, 7, 9]))
        np.save(f"{d}/code_part0_feat.npy", np.array([[5, 5], [7, 7], [9, 9]], dtype=np.float32))
        np.save(f"{d}/code_part1_ids.npy", np.array([2]))
        np.save(f"{d}/code_part1_feat.npy", np.array([[2, 2]], dtype=np.float32))
        X, row = L.load_features(d, [9, 2, 100, 5], parts=range(2), n_features=2)
    assert list(row >= 0) == [True, True, False, True]         # exam 100 has no saved features
    assert X.shape == (3, 2)
    assert [float(X[row[i], 0]) for i in (0, 1, 3)] == [9.0, 2.0, 5.0]


def test_netbenefit_import_and_hand_value():
    # the reused decision-curve code: p >= 0.5 treats rows 0,1,2; TP 2, FP 1, n 10, odds 1: 0.1
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.55, 0.3, 0.25, 0.2, 0.1, 0.1, 0.05, 0.02])
    assert close(nb.net_benefit_binary(y, p, [0.5])[0], 0.1, 1e-12)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
