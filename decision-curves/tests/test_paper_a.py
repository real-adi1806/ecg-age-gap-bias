"""
Checks on the decision-curve and proportional-hazards outputs. Run after dca_paper_a.py and
ph_check_paper_a.py, from the repository root:   python -m pytest decision-curves/tests -q -p no:cacheprovider
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import netbenefit as nb  # noqa: E402

R = json.load(open(os.path.join(HERE, "results_dca.json")))
P = json.load(open(os.path.join(HERE, "results_ph.json")))
D = np.load(os.path.join(HERE, "dca.npz"))
HORIZONS = {"2y": 2.0, "4y": 4.0}


def test_published_held_out_analysis_reproduced():
    for k, v in R["reproduction"].items():
        assert abs(v["got"] - v["published"]) < 1e-9, k


@pytest.mark.parametrize("tag", list(HORIZONS))
def test_treat_all_crosses_zero_at_horizon_risk(tag):
    f = nb.SurvivalFrame(D["y_time"], D["y_event"], HORIZONS[tag])
    risk = nb.km_risk_at_horizon(f)
    assert risk == pytest.approx(R["horizons"][tag]["km_risk_at_horizon"], abs=1e-12)
    assert nb.net_benefit_treat_all_survival(f, [risk])[0] == pytest.approx(0, abs=1e-12)
    v = nb.net_benefit_treat_all_survival(f, [risk - 1e-3, risk + 1e-3])
    assert v[0] > 0 > v[1]


@pytest.mark.parametrize("tag", list(HORIZONS))
def test_curves_recompute_from_saved_predictions(tag):
    f = nb.SurvivalFrame(D["y_time"], D["y_event"], HORIZONS[tag])
    for k in ("age_sex", "age_sex_gap"):
        got = nb.net_benefit_survival(f, D["%s__pred__%s" % (tag, k)], D["thr"])
        assert np.allclose(got, D["%s__curve__%s" % (tag, k)], atol=1e-14)


@pytest.mark.parametrize("tag", list(HORIZONS))
def test_uncensored_crosscheck_and_identity_recorded(tag):
    h = R["horizons"][tag]
    assert max(h["crosscheck_uncensored_max_abs_diff"].values()) < 1e-12
    assert h["max_abs_risk_diff_raw_vs_corrected_gap"] < 1e-9


def test_survival_equals_binary_on_real_patients_without_censoring_before_horizon():
    t, e = D["y_time"], D["y_event"]
    known = (t > 2.0) | (e == 1)
    y = ((t <= 2.0) & (e == 1)).astype(int)[known]
    p = D["2y__pred__age_sex_gap"][known]
    f = nb.SurvivalFrame(t[known], e[known], 2.0)
    assert np.allclose(nb.net_benefit_survival(f, p, D["thr"]), nb.net_benefit_binary(y, p, D["thr"]), atol=1e-12)


def test_band_interval_contains_point():
    for tag in HORIZONS:
        b = R["horizons"][tag]["gap_minus_age_sex"]["band_mean"]
        assert b["lo"] <= b["diff"] <= b["hi"]


def test_ph_models_reproduce_manuscript_and_periods_add_up():
    for k, v in P["reproduction"].items():
        assert v["max_abs_coef_diff_vs_cox_json"] < 1e-5, k
    for name, m in P["models"].items():
        assert sum(r["deaths"] for r in m["by_period"]) == m["deaths"], name
