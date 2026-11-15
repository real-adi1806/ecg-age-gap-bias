"""
Tests for exclusions.py, on tiny made-up frames. Run from this folder with either
    python -m pytest tests -q
    python tests/test_exclusions.py          (no pytest needed)
Expected values are worked by hand in the comment beside them.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decision-curves"))
import exclusions as X  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def tiny():
    """
    Patient 1: exams 10 (age 40), 20 (age 43). Outcome on exam 20. Lowest id is 10, a YOUNGER exam than the outcome.
    Patient 2: exams 11 (age 50), 5 (age 55). Outcome on exam 11. Lowest id is 5, an OLDER exam.
    Patient 3: exams 7 (age 60), 8 (age 60). Outcome on exam 8. Lowest id is 7, SAME age.
    Patient 4: one exam, 30 (age 70). Outcome on exam 30. Lowest id is the outcome exam.
    """
    return pd.DataFrame({
        "exam_id": [10, 20, 11, 5, 7, 8, 30],
        "patient_id": [1, 1, 2, 2, 3, 3, 4],
        "age": [40, 43, 50, 55, 60, 60, 70],
        "death": [None, "False", "True", None, None, "False", "False"],
        "timey": [None, 2.0, 0.5, None, None, 3.0, 1.0]})


def outcome_rows(df):
    return df[df["death"].notna()].rename(columns={})[["patient_id", "exam_id", "age"]]


def test_lowest_id_exam():
    first = X.lowest_id_exam(tiny())
    # lowest ids: patient 1 -> 10, patient 2 -> 5, patient 3 -> 7, patient 4 -> 30
    assert first["feat_exam_id"].to_dict() == {1: 10, 2: 5, 3: 7, 4: 30}
    assert first["feat_age"].to_dict() == {1: 40, 2: 55, 3: 60, 4: 70}


def test_age_order():
    # 41 vs 40 older, 40 vs 40 same, 39 vs 40 younger
    assert list(X.age_order([41, 40, 39], [40, 40, 40])) == ["older", "same", "younger"]


def test_classify_mismatch():
    df = tiny()
    cls = X.classify_feature_exams(df, outcome_rows(df)).set_index("patient_id")
    # patients 1, 2, 3 have a saved exam that is not the outcome exam; patient 4 does not
    assert cls["same_exam"].to_dict() == {1: False, 2: False, 3: False, 4: True}
    # saved age 40 vs outcome 43 younger; 55 vs 50 older; 60 vs 60 same; patient 4 same exam, same age
    assert cls["age_order"].to_dict() == {1: "younger", 2: "older", 3: "same", 4: "same"}
    # saved ids 10, 5, 7, 30 against outcome ids 20, 11, 8, 30: lower, lower, lower, same exam
    assert cls["id_order"].to_dict() == {1: "lower", 2: "lower", 3: "lower", 4: "same exam"}
    # outcome age equals the patient's lowest age? patient 1: 43 vs min 40 no; 2: 50 vs 50 yes; 3: 60 vs 60 yes; 4 yes
    assert cls["outcome_is_min_age"].to_dict() == {1: False, 2: True, 3: True, 4: True}


def test_smd_by_hand():
    # a = [0, 2], b = [2, 4]: means 1 and 3; sample variances 2 and 2; pooled SD sqrt(2); (1 - 3) / sqrt(2)
    assert close(X.smd([0, 2], [2, 4]), -2 / np.sqrt(2))
    # identical groups give 0
    assert close(X.smd([1, 2, 3], [1, 2, 3]), 0.0)
    # two constant groups have no spread: undefined, returned as nan rather than a division error
    assert np.isnan(X.smd([1, 1], [2, 2]))


def test_neither_share():
    # rows 1 and 3 have no label at all; row 2 has AF; row 4 has normal_ecg: 2 of 4
    f = pd.DataFrame({k: [0, 0, 0, 0] for k in X.LABELS})
    f.loc[1, "AF"] = 1
    f.loc[3, "normal_ecg"] = 1
    assert close(X.neither_share(f), 0.5)


def test_count_mismatches():
    assert X.count_mismatches({"a": 1, "b": 2}, {"a": 1, "b": 3}) == {"b": (2, 3)}
    assert X.count_mismatches({"a": 1}, {"a": 1}) == {}
    # a missing key counts as a mismatch
    assert X.count_mismatches({}, {"a": 1}) == {"a": (None, 1)}


def fake_results(sub, all14, diff5):
    """A results.json with two splits; `sub` and `all14` give AUCs by row, the same in both splits plus 0.01 in split 1."""
    seeds = {}
    for s, bump in (("0", 0.0), ("1", 0.01)):
        tw = {"auc": {k: v + bump for k, v in sub.items()}, "auc_ci95": {k: [v - 0.02, v + 0.02] for k, v in sub.items()},
              "auc_diff_vs_row2": {"row5_age_sex_lima": diff5[0]},
              "auc_diff_vs_row2_age_sex_ci95": {"row5_age_sex_lima": diff5[1:]}}
        seeds[s] = {"test_with_features": tw, "test_all": {"auc": {k: v + bump for k, v in all14.items()}}}
    return {"seeds": seeds}


def test_compare_versions():
    rows = dict(zip(X.ROWS, [0.80, 0.81, 0.82, 0.815, 0.84, 0.83]))
    v1 = fake_results(rows, {k: rows[k] - 0.003 for k in X.ROWS14}, [0.03, 0.01, 0.05])
    v2 = fake_results({k: rows[k] - 0.003 for k in X.ROWS}, {k: rows[k] - 0.003 for k in X.ROWS14}, [0.031, 0.012, 0.05])
    out = X.compare_versions(v1, v2)
    # row 5: 0.84 in the first version, 0.837 in the second; split 1 adds 0.01, so the means are 0.845 and 0.842
    assert close(out["by_split"]["0"]["v1"]["row5_age_sex_lima"], 0.84)
    assert close(out["by_split"]["0"]["v2"]["row5_age_sex_lima"], 0.837)
    assert close(out["mean"]["v1"]["row5_age_sex_lima"], 0.845) and close(out["mean"]["v2"]["row5_age_sex_lima"], 0.842)
    # rows 1 to 4 on the whole test split are the same in both versions
    assert close(out["rows14_all_test_v1_vs_v2_max_abs_diff"], 0.0)
    # in the first version, all minus kept is -0.003 for every row 1 to 4, in both splits
    assert all(close(d, 0.003) for d in out["v1_rows14_all_minus_kept_max_abs_diff"].values())
    assert out["split0_row5_minus_row2"]["v2"] == [0.031, 0.012, 0.05]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
