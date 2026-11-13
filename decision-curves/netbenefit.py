"""
Net benefit for decision curve analysis, binary and time-to-event.

Formulas (written out in full in each paper's METHODS.md):

Binary outcome (Vickers and Elkin 2006, doi:10.1177/0272989X06295361):
    a patient is "treated" when predicted risk >= threshold t
    NB(t) = TP/n - FP/n * t/(1-t)
    treat all:  NB(t) = prev - (1-prev) * t/(1-t), which is zero at t = prev
    treat none: NB(t) = 0

Time-to-event outcome at horizon tau (Vickers, Cronin, Elkin, Gonen 2008,
doi:10.1186/1472-6947-8-53; the approach recommended by McLernon et al. 2023,
Ann Intern Med 176:105-114, PMID 36571841):
    among the n_pos patients with predicted risk >= t, S_pos = Kaplan-Meier
    survival at tau computed in that group only
    TP = (1 - S_pos) * n_pos      FP = S_pos * n_pos
    NB(t) = TP/n - FP/n * t/(1-t)
    treat all uses the Kaplan-Meier survival of everyone: (1-S) - S * t/(1-t),
    which is zero at t = 1 - S(tau), the horizon risk

Every function takes optional frequency weights. A bootstrap resample is the
same as the original rows with multinomial counts as weights, so the bootstrap
reuses one sort instead of re-sorting thousands of resamples. The tests check
that weighted results equal results on explicitly repeated rows.
"""
import numpy as np


def threshold_odds(t):
    """The exchange rate t/(1-t): the harm of one false positive in units of one true positive."""
    t = np.asarray(t, dtype=float)
    if np.any((t <= 0) | (t >= 1)):
        raise ValueError("thresholds must lie strictly between 0 and 1")
    return t / (1 - t)


def _weights(n, w):
    if w is None:
        return np.ones(n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,) or np.any(w < 0):
        raise ValueError("weights must be a non-negative vector, one per patient")
    return w


# ---------------------------------------------------------------------------
# binary outcome
# ---------------------------------------------------------------------------

def net_benefit_binary(y, p, thresholds, w=None):
    """Net benefit of 'treat if p >= t' for each t. y in {0,1}."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y must be 0 or 1")
    w = _weights(len(y), w)
    thr = np.atleast_1d(np.asarray(thresholds, dtype=float))
    odds = threshold_odds(thr)
    n = w.sum()
    pos = p[None, :] >= thr[:, None]                  # (n_thr, n_pat)
    tp = (pos * (w * y)[None, :]).sum(axis=1)
    fp = (pos * (w * (1 - y))[None, :]).sum(axis=1)
    return tp / n - fp / n * odds


def net_benefit_treat_all_binary(y, thresholds, w=None):
    y = np.asarray(y).astype(int)
    w = _weights(len(y), w)
    prev = (w * y).sum() / w.sum()
    return prev - (1 - prev) * threshold_odds(np.atleast_1d(thresholds))


def net_benefit_from_rates(prev, tpr, fpr, thresholds):
    """
    Net benefit in a population of prevalence `prev` given the true and false
    positive rates of a rule at each threshold: prev*TPR - (1-prev)*FPR*t/(1-t).
    Used to carry the held-out cases and non-cases to another prevalence; it is
    the expectation of the binary formula under resampling cases and non-cases
    to that prevalence.
    """
    return prev * np.asarray(tpr) - (1 - prev) * np.asarray(fpr) * threshold_odds(thresholds)


# ---------------------------------------------------------------------------
# time to event
# ---------------------------------------------------------------------------

class SurvivalFrame:
    """
    Pre-sorted follow-up data, so Kaplan-Meier at one horizon can be evaluated
    for many subgroups and many weight vectors without re-sorting.
    Ties: deaths and censorings at the same time are handled the standard way
    (everyone with time >= u is at risk at u; d deaths at u give factor 1 - d/n).
    """

    def __init__(self, time, event, horizon):
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(int)
        if time.shape != event.shape:
            raise ValueError("time and event must have the same length")
        if np.any(time < 0) or not np.all(np.isfinite(time)):
            raise ValueError("follow-up times must be finite and non-negative")
        if set(np.unique(event)) - {0, 1}:
            raise ValueError("event must be 0 or 1")
        self.order = np.argsort(time, kind="stable")
        ts = time[self.order]
        self.event_sorted = event[self.order].astype(float)
        self.uniq, self.start = np.unique(ts, return_index=True)
        self.horizon = float(horizon)
        self.use = self.uniq <= self.horizon       # only event times up to the horizon count
        self.n = len(time)

    def km_at_horizon(self, W):
        """
        Kaplan-Meier survival at the horizon for each row of W, a (k, n) array of
        non-negative weights in the ORIGINAL row order (0 = not in the group).
        Returns a length-k vector. A group with zero total weight returns nan.
        """
        W = np.atleast_2d(np.asarray(W, dtype=float))
        Ws = W[:, self.order]
        # at risk at each unique time: weight of rows with time >= u
        rev = np.cumsum(Ws[:, ::-1], axis=1)[:, ::-1]
        at_risk = rev[:, self.start]
        deaths = np.add.reduceat(Ws * self.event_sorted[None, :], self.start, axis=1)
        at_risk = at_risk[:, self.use]
        deaths = deaths[:, self.use]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(at_risk > 0, deaths / at_risk, 0.0)
        S = np.prod(1 - frac, axis=1)
        S[W.sum(axis=1) <= 0] = np.nan
        return S


def net_benefit_survival(frame, p, thresholds, w=None):
    """
    Net benefit at frame.horizon of 'treat if p >= t', with the event risk in the
    treated group taken from Kaplan-Meier in that group (handles censoring).
    """
    p = np.asarray(p, dtype=float)
    if len(p) != frame.n:
        raise ValueError("p must have one value per patient in the frame")
    w = _weights(frame.n, w)
    thr = np.atleast_1d(np.asarray(thresholds, dtype=float))
    odds = threshold_odds(thr)
    pos = (p[None, :] >= thr[:, None]).astype(float) * w[None, :]
    n_pos = pos.sum(axis=1)
    S_pos = frame.km_at_horizon(pos)
    n = w.sum()
    S_pos = np.where(n_pos > 0, S_pos, 1.0)           # nobody treated: NB is exactly 0
    tp = (1 - S_pos) * n_pos
    fp = S_pos * n_pos
    return tp / n - fp / n * odds


def net_benefit_treat_all_survival(frame, thresholds, w=None):
    w = _weights(frame.n, w)
    S = frame.km_at_horizon(w[None, :])[0]
    return (1 - S) - S * threshold_odds(np.atleast_1d(thresholds))


def km_risk_at_horizon(frame, w=None):
    """1 - KM survival at the horizon for everyone (the horizon risk)."""
    w = _weights(frame.n, w)
    return 1 - frame.km_at_horizon(w[None, :])[0]
