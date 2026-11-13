"""
Tests for decision-curves/netbenefit.py. Run from the repository root:
    python -m pytest decision-curves/tests -q
Every expected value below is worked by hand in the comment beside it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import netbenefit as nb  # noqa: E402

# ---------------------------------------------------------------------------
# binary, hand computed
# ---------------------------------------------------------------------------
Y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.6, 0.55, 0.3, 0.25, 0.2, 0.1, 0.1, 0.05, 0.02])


def test_binary_hand_t050():
    # p >= 0.5: rows 0,1,2. TP = 2 (rows 0,1), FP = 1 (row 2). n = 10, odds = 1.
    # NB = 2/10 - 1/10 * 1 = 0.1
    assert nb.net_benefit_binary(Y, P, [0.5])[0] == pytest.approx(0.1, abs=1e-12)


def test_binary_hand_t020():
    # p >= 0.2: rows 0..5. TP = 3 (rows 0,1,4), FP = 3 (rows 2,3,5). odds = 0.25.
    # NB = 0.3 - 0.3 * 0.25 = 0.225
    assert nb.net_benefit_binary(Y, P, [0.2])[0] == pytest.approx(0.225, abs=1e-12)


def test_binary_threshold_is_inclusive():
    # p >= 0.1: rows 0..7. TP = 3, FP = 5. odds = 1/9. NB = 0.3 - 0.5/9 = 0.244444...
    assert nb.net_benefit_binary(Y, P, [0.1])[0] == pytest.approx(0.3 - 0.5 / 9, abs=1e-12)


def test_binary_treat_all_hand():
    # prevalence 0.3. t = 0.2: 0.3 - 0.7 * 0.25 = 0.125
    assert nb.net_benefit_treat_all_binary(Y, [0.2])[0] == pytest.approx(0.125, abs=1e-12)


def test_binary_treat_all_crosses_zero_at_prevalence():
    # t = prevalence = 0.3: 0.3 - 0.7 * (0.3/0.7) = 0 exactly
    assert nb.net_benefit_treat_all_binary(Y, [0.3])[0] == pytest.approx(0.0, abs=1e-12)
    v = nb.net_benefit_treat_all_binary(Y, [0.29, 0.31])
    assert v[0] > 0 > v[1]


def test_binary_everyone_above_threshold_equals_treat_all():
    t = [0.01, 0.015]
    assert np.allclose(nb.net_benefit_binary(Y, P, t), nb.net_benefit_treat_all_binary(Y, t))


def test_binary_nobody_above_threshold_is_zero():
    assert nb.net_benefit_binary(Y, P, [0.95])[0] == 0.0


def test_binary_weights_equal_repeated_rows():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    p = rng.random(40)
    w = rng.integers(0, 4, 40)
    t = np.linspace(0.05, 0.9, 18)
    rep = np.repeat(np.arange(40), w)
    assert np.allclose(nb.net_benefit_binary(y, p, t, w=w), nb.net_benefit_binary(y[rep], p[rep], t))
    assert np.allclose(nb.net_benefit_treat_all_binary(y, t, w=w),
                       nb.net_benefit_treat_all_binary(y[rep], t))


def test_from_rates_matches_binary_when_prev_is_the_sample_prev():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    t = np.linspace(0.05, 0.95, 19)
    pos = p[None, :] >= t[:, None]
    tpr = pos[:, y == 1].mean(axis=1)
    fpr = pos[:, y == 0].mean(axis=1)
    assert np.allclose(nb.net_benefit_from_rates(y.mean(), tpr, fpr, t), nb.net_benefit_binary(y, p, t))


def test_threshold_out_of_range_rejected():
    with pytest.raises(ValueError):
        nb.net_benefit_binary(Y, P, [0.0])
    with pytest.raises(ValueError):
        nb.net_benefit_binary(Y, P, [1.0])


# ---------------------------------------------------------------------------
# survival, hand computed
# ---------------------------------------------------------------------------
#   row  time  event  p
#    0    2     1    0.8
#    1    3     0    0.7   censored
#    2    4     1    0.6
#    3    6     0    0.4   censored
#    4    7     1    0.3
#    5    8     0    0.2   censored
T = np.array([2, 3, 4, 6, 7, 8], dtype=float)
E = np.array([1, 0, 1, 0, 1, 0])
PS = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2])
HOR = 5.0


def test_km_all_hand():
    # time 2: 6 at risk, 1 death -> 5/6. time 3: censoring. time 4: 4 at risk, 1 death -> *3/4.
    # S(5) = 5/6 * 3/4 = 0.625
    f = nb.SurvivalFrame(T, E, HOR)
    assert f.km_at_horizon(np.ones(6))[0] == pytest.approx(0.625, abs=1e-12)


def test_survival_nb_hand_t035():
    # p >= 0.35: rows 0..3. KM in that group: time 2: 4 at risk, 1 death -> 3/4.
    # time 3 censored. time 4: 2 at risk, 1 death -> 3/4 * 1/2 = 3/8. S_pos = 0.375.
    # n_pos = 4, n = 6. TP = 0.625 * 4 = 2.5, FP = 0.375 * 4 = 1.5. odds = 0.35/0.65.
    # NB = 2.5/6 - 1.5/6 * 0.35/0.65 = 0.2820512820...
    f = nb.SurvivalFrame(T, E, HOR)
    expect = 2.5 / 6 - 1.5 / 6 * (0.35 / 0.65)
    assert nb.net_benefit_survival(f, PS, [0.35])[0] == pytest.approx(expect, abs=1e-12)


def test_survival_treat_all_hand_and_zero_crossing():
    # S = 0.625, horizon risk 0.375. t = 0.35: 0.375 - 0.625 * 0.35/0.65 = 0.0384615...
    f = nb.SurvivalFrame(T, E, HOR)
    assert nb.net_benefit_treat_all_survival(f, [0.35])[0] == pytest.approx(
        0.375 - 0.625 * 0.35 / 0.65, abs=1e-12)
    # zero exactly at t = horizon risk = 0.375
    assert nb.net_benefit_treat_all_survival(f, [0.375])[0] == pytest.approx(0.0, abs=1e-12)
    assert nb.km_risk_at_horizon(f) == pytest.approx(0.375, abs=1e-12)


def test_survival_nobody_treated_is_zero():
    f = nb.SurvivalFrame(T, E, HOR)
    assert nb.net_benefit_survival(f, PS, [0.9])[0] == 0.0


def test_km_matches_lifelines_with_ties_and_censoring():
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(3)
    n = 500
    time = rng.integers(1, 60, n).astype(float)          # integer times force ties
    event = rng.integers(0, 2, n)
    for hor in (10, 30.5, 59):
        f = nb.SurvivalFrame(time, event, hor)
        mine = f.km_at_horizon(np.ones(n))[0]
        km = KaplanMeierFitter().fit(time, event)
        theirs = float(km.survival_function_at_times(hor).iloc[0])
        assert mine == pytest.approx(theirs, abs=1e-12)
        # and in a subgroup
        g = rng.random(n) < 0.4
        mine_g = f.km_at_horizon(g.astype(float))[0]
        theirs_g = float(KaplanMeierFitter().fit(time[g], event[g]).survival_function_at_times(hor).iloc[0])
        assert mine_g == pytest.approx(theirs_g, abs=1e-12)


def test_survival_weights_equal_repeated_rows():
    rng = np.random.default_rng(4)
    n = 80
    time = rng.integers(1, 30, n).astype(float)
    event = rng.integers(0, 2, n)
    p = rng.random(n)
    w = rng.integers(0, 4, n)
    t = np.linspace(0.05, 0.9, 18)
    rep = np.repeat(np.arange(n), w)
    a = nb.net_benefit_survival(nb.SurvivalFrame(time, event, 12), p, t, w=w)
    b = nb.net_benefit_survival(nb.SurvivalFrame(time[rep], event[rep], 12), p[rep], t)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# the required cross-check: no censoring -> survival NB == binary NB
# ---------------------------------------------------------------------------

def test_no_censoring_survival_equals_binary():
    rng = np.random.default_rng(5)
    n = 2000
    time = rng.exponential(10, n)
    event = np.ones(n, dtype=int)                       # nobody censored
    p = 1 / (1 + np.exp(-(rng.normal(0, 1.2, n) + (10 - time) / 6)))
    hor = 7.0
    y = (time <= hor).astype(int)
    t = np.linspace(0.01, 0.99, 99)
    f = nb.SurvivalFrame(time, event, hor)
    assert np.allclose(nb.net_benefit_survival(f, p, t), nb.net_benefit_binary(y, p, t), atol=1e-12)
    assert np.allclose(nb.net_benefit_treat_all_survival(f, t), nb.net_benefit_treat_all_binary(y, t),
                       atol=1e-12)


def test_censoring_only_after_horizon_survival_equals_binary():
    # administrative censoring after the horizon does not change anyone's status at the horizon
    rng = np.random.default_rng(6)
    n = 1500
    time = rng.exponential(10, n)
    event = np.ones(n, dtype=int)
    late = time > 9
    event[late & (rng.random(n) < 0.5)] = 0
    hor = 8.0
    y = ((time <= hor) & (event == 1)).astype(int)
    p = rng.random(n)
    t = np.linspace(0.01, 0.99, 99)
    f = nb.SurvivalFrame(time, event, hor)
    assert np.allclose(nb.net_benefit_survival(f, p, t), nb.net_benefit_binary(y, p, t), atol=1e-12)
