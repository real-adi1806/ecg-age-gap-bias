"""
Building blocks for analysis.py: the cohort, the splits, AUC written out by hand, an unpenalised logistic
regression (Newton's method), calibration, and the bootstrap. Kept apart so each piece can be tested on its own
(tests/test_ladder.py).
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

HORIZON_YEARS = 1.0
RHYTHM_COLS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg"]
AGE_BANDS = [(16, 40), (40, 60), (60, 75), (75, 200)]
EPS = 1e-12


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------
def build_cohort(csv_path):
    """
    One row per patient: the rows of exams.csv where death and timey are present (the outcome sits on one exam
    per patient). Adds:
        died1   death is True and timey <= 1 year
        short   alive at last contact and followed for less than 1 year (outcome at 1 year unknown)
    Sorted by exam_id so the row order, and so every split, is reproducible.
    """
    df = pd.read_csv(csv_path)
    c = df[df["death"].notna() & df["timey"].notna()].copy()
    if c["patient_id"].duplicated().any():
        raise ValueError("a patient has more than one outcome row")
    dead = c["death"].astype(str).str.strip().str.lower() == "true"
    c["died1"] = (dead & (c["timey"] <= HORIZON_YEARS)).astype(int)
    c["short"] = (~dead) & (c["timey"] < HORIZON_YEARS)
    c["male"] = c["is_male"].astype(float)
    c["age"] = c["age"].astype(float)
    c["age2"] = c["age"] ** 2
    c["gap"] = c["nn_predicted_age"] - c["age"]
    for k in RHYTHM_COLS:
        c[k] = c[k].astype(float)
    return c.sort_values("exam_id").reset_index(drop=True)


def split_patients(y, seed, fractions=(0.8, 0.1, 0.1)):
    """Patient-level split, stratified on y. Returns an array of 'train' / 'val' / 'test', one per row."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=object)
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        a = int(round(fractions[0] * len(idx)))
        b = int(round((fractions[0] + fractions[1]) * len(idx)))
        out[idx[:a]], out[idx[a:b]], out[idx[b:]] = "train", "val", "test"
    return out


# ---------------------------------------------------------------------------
# discrimination
# ---------------------------------------------------------------------------
def auc_rank(y, score):
    """
    AUC from the rank-sum formula: (sum of the ranks of the deaths - n1(n1+1)/2) / (n1 * n0), with tied scores
    given their average rank. This is the probability that a random death outranks a random survivor, ties one half.
    """
    y = np.asarray(y).astype(int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_pairs(y, score):
    """The same quantity by counting every death and survivor pair. O(n1 * n0); for tests and small checks only."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def bootstrap(y, scores, ref, B, rng, with_ap=True):
    """
    Resample patients with replacement B times. For every model in `scores`: 95% interval of the AUC and of the
    average precision, and of the paired AUC difference from the model named `ref`.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    auc = {k: [] for k in scores}
    ap = {k: [] for k in scores}
    diff = {k: [] for k in scores if k != ref}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        a = {k: auc_rank(yb, s[idx]) for k, s in scores.items()}
        for k in scores:
            auc[k].append(a[k])
            if with_ap:
                ap[k].append(average_precision_score(yb, scores[k][idx]))
        for k in diff:
            diff[k].append(a[k] - a[ref])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return {"auc_ci95": {k: ci(v) for k, v in auc.items()},
            "ap_ci95": {k: ci(v) for k, v in ap.items()} if with_ap else None,
            f"auc_diff_vs_{ref}_ci95": {k: ci(v) for k, v in diff.items()},
            "resamples_used": len(next(iter(auc.values())))}


# ---------------------------------------------------------------------------
# unpenalised logistic regression (rows 2 to 4, and calibration)
# ---------------------------------------------------------------------------
def fit_logistic(X, y, offset=None, max_iter=100, tol=1e-10):
    """
    Maximum-likelihood logistic regression by Newton's method, in float64. Returns coefficients on the ORIGINAL
    scale of X, intercept first. Columns are standardised inside for conditioning (age squared runs to 10,000) and
    transformed back. X may have zero columns (intercept only), which with an offset gives calibration in the large.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("a constant column cannot be fitted alongside the intercept")
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    b = np.zeros(Z.shape[1])
    b[0] = logit(np.clip(y.mean(), EPS, 1 - EPS)) - off.mean()
    for _ in range(max_iter):
        p = expit(Z @ b + off)
        grad = Z.T @ (y - p)
        H = (Z * (p * (1 - p))[:, None]).T @ Z
        step = np.linalg.solve(H, grad)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("Newton's method did not converge")
    coef = b[1:] / sd
    return np.concatenate([[b[0] - (coef * mu).sum()], coef])


def predict_logistic(beta, X):
    X = np.asarray(X, dtype=float).reshape(-1, len(beta) - 1)
    return expit(beta[0] + X @ beta[1:])


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibration(y, p, groups=10, grid=60):
    """
    slope       coefficient of logit(p) in a logistic regression of y on logit(p). 1 is ideal; below 1 means the
                predictions are too extreme.
    intercept   calibration in the large: the intercept with the slope fixed at 1 (logit(p) as an offset). 0 is ideal.
    groups      equal-sized groups by predicted risk: mean predicted against observed.
    smooth      observed risk against predicted, a Gaussian kernel smoother on logit(p) (bandwidth 0.35 logit units).
    """
    y = np.asarray(y).astype(int)
    lp = logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))
    slope = float(fit_logistic(lp[:, None], y)[1])
    citl = float(fit_logistic(np.empty((len(y), 0)), y, offset=lp)[0])
    order = np.argsort(lp, kind="stable")
    parts = np.array_split(order, groups)
    g = [{"n": int(len(i)), "deaths": int(y[i].sum()), "mean_predicted": float(expit(lp[i]).mean()),
          "observed": float(y[i].mean())} for i in parts]
    xs = np.linspace(np.percentile(lp, 1), np.percentile(lp, 99.9), grid)
    sm = []
    for x in xs:
        w = np.exp(-0.5 * ((lp - x) / 0.35) ** 2)
        sm.append(float((w * y).sum() / w.sum()))
    return {"slope": slope, "intercept_in_the_large": citl, "mean_predicted": float(expit(lp).mean()),
            "observed_rate": float(y.mean()), "groups": g,
            "smooth": {"predicted": [float(v) for v in expit(xs)], "observed": sm}}


def calibration_slope_ci(y, p, B, rng):
    """Bootstrap 95% intervals for the slope and the intercept in the large."""
    y = np.asarray(y).astype(int)
    lp = logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))
    n = len(y)
    s, c = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if y[idx].sum() < 2:
            continue
        s.append(fit_logistic(lp[idx][:, None], y[idx])[1])
        c.append(fit_logistic(np.empty((n, 0)), y[idx], offset=lp[idx])[0])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"slope_ci95": ci(s), "intercept_in_the_large_ci95": ci(c)}


# ---------------------------------------------------------------------------
# the saved network features
# ---------------------------------------------------------------------------
def load_features(feat_dir, exam_ids, parts=range(18), n_features=5120):
    """
    Rows of the saved Lima features (code_part{k}_feat.npy, with code_part{k}_ids.npy holding the exam ids) for the
    exams in `exam_ids`, matched on EXAM id, not patient id, so a feature row is only ever used for the exam the
    outcome is attached to. Parts are memory-mapped and copied one at a time.
    Returns (X float32 [n_found, n_features], row) where row[i] is the row of X for exam_ids[i], or -1 if absent.
    """
    exam_ids = np.asarray(exam_ids)
    where = pd.Series(np.arange(len(exam_ids)), index=exam_ids)
    if not where.index.is_unique:
        raise ValueError("exam ids must be unique")
    row = np.full(len(exam_ids), -1, dtype=np.int64)
    chunks, n = [], 0
    for k in parts:
        ids = np.load(f"{feat_dir}/code_part{k}_ids.npy")
        feat = np.load(f"{feat_dir}/code_part{k}_feat.npy", mmap_mode="r")
        if feat.shape != (len(ids), n_features):
            raise ValueError(f"part {k}: features {feat.shape} do not match {len(ids)} ids")
        keep = np.where(np.isin(ids, exam_ids))[0]
        target = where.loc[ids[keep]].values
        if np.any(row[target] != -1):
            raise ValueError(f"part {k}: an exam id appears twice in the feature files")
        row[target] = n + np.arange(len(keep))
        chunks.append(np.asarray(feat[keep], dtype=np.float32))
        n += len(keep)
    X = np.concatenate(chunks) if chunks else np.empty((0, n_features), dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in the features")
    return X, row
