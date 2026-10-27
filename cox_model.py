#!/usr/bin/env python3
"""
Cox proportional hazards on CODE-15%, in the standard library.

Why this file exists
--------------------
ecg_age_bias.py reports CRUDE death rates. The 2021 paper's headline (HR 1.79,
95% CI 1.69 to 1.90) came from a Cox model ADJUSTED FOR AGE AND SEX, so the crude
rates cannot be compared with it. This file fits that adjusted model, so the
replication can be stated on the same terms as the thing it replicates.

There is no third-party dependency here on purpose. lifelines would be the
obvious tool, but requiring an install would break the promise that this
repository runs on a clean Python with nothing added. Instead the partial
likelihood is written out, and correctness is established three ways:

  1. selftest_fast_vs_naive() checks the O(n) cumulative-sum implementation
     against a transparent O(n^2) risk-set loop on a random subsample. They must
     agree on the log-likelihood, the gradient and the information matrix.
  2. selftest_optimum() checks that the score is about zero at the solution and
     that the information matrix is positive definite.
  3. The fitted model is compared against a published number that this code did
     not produce: the 2021 paper's HR 1.79 on this same cohort.

Ties are handled by Efron's approximation, which is what lifelines and the R
survival package use by default.

Usage
-----
    python cox_model.py                  # full fit, about a minute
    python cox_model.py --selftest       # correctness checks only, a few seconds
    python cox_model.py --subsample 20000
"""

import argparse
import json
import math
import os
import random
import sys

import ecg_age_bias as base

HERE = os.path.dirname(os.path.abspath(__file__))

# How close to zero the score has to get before the fit is called converged.
#
# ONE constant, used both as the fitter's target and as the threshold the
# optimum check asserts against. They were briefly two different numbers, 1e-8
# in the fitter and 1e-9 in the check, and the result was a self-test that
# failed while the fit was working perfectly: the check was demanding something
# tighter than the fitter ever promised, and tighter than double precision
# reaches when the score is a sum over 200,000 terms. A guard that red-lines for
# the wrong reason is worse than no guard, because it teaches people to ignore
# it. Measured floor on this cohort: about 4e-9.
SCORE_TOL = 1e-8


# ---------------------------------------------------------------------------
# Small linear algebra, p is at most about 6 so none of this needs to be clever
# ---------------------------------------------------------------------------
def solve(a, b):
    """Solve a x = b by Gauss-Jordan with partial pivoting. a is destroyed."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ZeroDivisionError("singular information matrix")
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [v / d for v in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f:
                m[r] = [m[r][k] - f * m[col][k] for k in range(n + 1)]
    return [m[i][n] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve([row[:] for row in a], e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def is_positive_definite(a):
    """Cholesky without the square roots. Returns False the moment it is not."""
    n = len(a)
    m = [row[:] for row in a]
    for i in range(n):
        for j in range(i):
            s = m[i][j] - sum(m[i][k] * m[j][k] for k in range(j))
            m[i][j] = s / m[j][j]
        s = m[i][i] - sum(m[i][k] ** 2 for k in range(i))
        if s <= 0:
            return False
        m[i][i] = math.sqrt(s)
    return True


def norm_sf(z):
    """Two-sided normal tail probability."""
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# The partial likelihood
# ---------------------------------------------------------------------------
def prepare(rows, covariates):
    """
    Turn rows into the arrays the fitter wants, sorted by time DESCENDING so the
    risk set only ever grows as we walk forward.

    Covariates are centred. The Cox partial likelihood is invariant to adding a
    constant to every linear predictor, so centring changes no coefficient and it
    keeps exp() away from overflow during the first Newton steps.
    """
    p = len(covariates)
    xs = [[float(f(r)) for f in covariates] for r in rows]
    means = [sum(x[k] for x in xs) / len(xs) for k in range(p)]
    xs = [[x[k] - means[k] for k in range(p)] for x in xs]
    times = [r["timey"] for r in rows]
    events = [1 if r["died"] else 0 for r in rows]

    order = sorted(range(len(rows)), key=lambda i: -times[i])
    xs = [xs[i] for i in order]
    times = [times[i] for i in order]
    events = [events[i] for i in order]

    # The outer products are fixed across Newton iterations, so build them once.
    pairs = [(a, b) for a in range(p) for b in range(a, p)]
    xx = [[x[a] * x[b] for (a, b) in pairs] for x in xs]
    return xs, xx, pairs, times, events, means


def loglik_grad_info(beta, xs, xx, pairs, times, events):
    """
    Efron partial log-likelihood, its gradient, and the observed information.

    Walks time from the largest to the smallest, so the risk set is exactly the
    set of subjects already seen.
    """
    p = len(beta)
    npairs = len(pairs)
    n = len(xs)

    S0 = 0.0
    S1 = [0.0] * p
    S2 = [0.0] * npairs

    ll = 0.0
    grad = [0.0] * p
    info = [[0.0] * p for _ in range(p)]

    i = 0
    while i < n:
        t = times[i]
        j = i
        d0 = 0.0
        d1 = [0.0] * p
        d2 = [0.0] * npairs
        dsum = [0.0] * p
        d = 0
        # Every subject at this time enters the risk set. Those with an event
        # also enter the tied-death block.
        while j < n and times[j] == t:
            x = xs[j]
            xxj = xx[j]
            xb = 0.0
            for k in range(p):
                xb += beta[k] * x[k]
            w = math.exp(xb)
            S0 += w
            for k in range(p):
                S1[k] += w * x[k]
            for k in range(npairs):
                S2[k] += w * xxj[k]
            if events[j]:
                d += 1
                ll += xb
                d0 += w
                for k in range(p):
                    dsum[k] += x[k]
                    d1[k] += w * x[k]
                for k in range(npairs):
                    d2[k] += w * xxj[k]
            j += 1

        if d:
            for k in range(p):
                grad[k] += dsum[k]
            for l in range(d):
                f = l / d
                a0 = S0 - f * d0
                ll -= math.log(a0)
                a1 = [S1[k] - f * d1[k] for k in range(p)]
                for k in range(p):
                    grad[k] -= a1[k] / a0
                for idx, (a, b) in enumerate(pairs):
                    v = (S2[idx] - f * d2[idx]) / a0 - (a1[a] / a0) * (a1[b] / a0)
                    info[a][b] += v
                    if a != b:
                        info[b][a] += v
        i = j

    return ll, grad, info


def loglik_grad_info_naive(beta, xs, xx, pairs, times, events):
    """
    The same quantity computed the obvious, slow way: for every event time,
    loop over the whole risk set from scratch. O(n^2). Only ever used on a small
    subsample, to prove the fast version above is doing the right arithmetic.
    """
    p = len(beta)
    npairs = len(pairs)
    n = len(xs)
    ll = 0.0
    grad = [0.0] * p
    info = [[0.0] * p for _ in range(p)]

    distinct = sorted(set(times), reverse=True)
    for t in distinct:
        risk = [j for j in range(n) if times[j] >= t]
        died = [j for j in range(n) if times[j] == t and events[j]]
        d = len(died)
        if not d:
            continue
        S0 = 0.0
        S1 = [0.0] * p
        S2 = [0.0] * npairs
        for j in risk:
            w = math.exp(sum(beta[k] * xs[j][k] for k in range(p)))
            S0 += w
            for k in range(p):
                S1[k] += w * xs[j][k]
            for k in range(npairs):
                S2[k] += w * xx[j][k]
        d0 = 0.0
        d1 = [0.0] * p
        d2 = [0.0] * npairs
        for j in died:
            w = math.exp(sum(beta[k] * xs[j][k] for k in range(p)))
            ll += sum(beta[k] * xs[j][k] for k in range(p))
            for k in range(p):
                grad[k] += xs[j][k]
            d0 += w
            for k in range(p):
                d1[k] += w * xs[j][k]
            for k in range(npairs):
                d2[k] += w * xx[j][k]
        for l in range(d):
            f = l / d
            a0 = S0 - f * d0
            ll -= math.log(a0)
            a1 = [S1[k] - f * d1[k] for k in range(p)]
            for k in range(p):
                grad[k] -= a1[k] / a0
            for idx, (a, b) in enumerate(pairs):
                v = (S2[idx] - f * d2[idx]) / a0 - (a1[a] / a0) * (a1[b] / a0)
                info[a][b] += v
                if a != b:
                    info[b][a] += v
    return ll, grad, info


def fit(rows, covariates, names, max_iter=40, score_tol=SCORE_TOL, verbose=False):
    """
    Newton-Raphson with step halving.

    Convergence is judged on the SCORE (the gradient of the log-likelihood),
    not on the change in the log-likelihood. At a maximum the score is zero, so
    max|score| is a direct statement about how close to the solution we are,
    whereas a log-likelihood that has stopped moving only says the steps have
    become small.

    score_tol defaults to SCORE_TOL, which is the one place that number is
    written down. See the comment on that constant for why it is 1e-8 and not
    something tighter. The loop also stops when the score stops improving, so a
    tighter request cannot turn into a long silent stall.
    """
    xs, xx, pairs, times, events, means = prepare(rows, covariates)
    p = len(covariates)
    beta = [0.0] * p
    ll, grad, info = loglik_grad_info(beta, xs, xx, pairs, times, events)
    ll0 = ll
    iters = 0
    prev_score = float("inf")
    for it in range(max_iter):
        iters = it + 1
        try:
            step = solve([row[:] for row in info], grad)
        except ZeroDivisionError:
            raise SystemExit("information matrix is singular; check for a "
                             "collinear or constant covariate")
        scale = 1.0
        while True:
            cand = [beta[k] + scale * step[k] for k in range(p)]
            ll_new, grad_new, info_new = loglik_grad_info(
                cand, xs, xx, pairs, times, events)
            if ll_new >= ll - 1e-9 or scale < 1e-6:
                break
            scale /= 2.0
        beta, ll, grad, info = cand, ll_new, grad_new, info_new
        score = max(abs(g) for g in grad)
        if verbose:
            sys.stderr.write("  iter %2d  loglik %.9f  max|score| %.3g\n"
                             % (iters, ll, score))
        if score < score_tol:
            break
        if score > 0.5 * prev_score and it >= 3:
            # The score has stopped improving materially. Further iterations are
            # chasing floating-point noise, not the solution.
            break
        prev_score = score

    cov = inverse(info)
    out = {
        "n": len(rows),
        "events": sum(events),
        "loglik_null": ll0,
        "loglik": ll,
        "iterations": iters,
        "max_abs_score_at_optimum": max(abs(g) for g in grad),
        "information_positive_definite": is_positive_definite(info),
        # Kept so that a contrast between two terms can be given an interval.
        "term_order": list(names),
        "vcov": cov,
        "terms": {},
    }
    for k, nm in enumerate(names):
        se = math.sqrt(cov[k][k])
        z = beta[k] / se
        out["terms"][nm] = {
            "coef": beta[k],
            "se": se,
            "hr": math.exp(beta[k]),
            "hr_lo95": math.exp(beta[k] - 1.959963984540054 * se),
            "hr_hi95": math.exp(beta[k] + 1.959963984540054 * se),
            "z": z,
            "p": norm_sf(z),
        }
    return out


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def selftest_fast_vs_naive(rows, n=900):
    """
    Deterministic on purpose: an evenly spaced stride, not random.sample.

    random.sample with a fixed seed is NOT stable across Python versions, and
    measured here it is not: the same seed selected different rows on 3.11 and
    on 3.13. A correctness check that quietly tests different data on different
    machines is a check whose output cannot be compared. A stride gives every
    run the same subsample.

    EXPECT A SMALL RESIDUAL DIFFERENCE ON PYTHON 3.12 AND LATER, and do not
    treat it as a defect. Measured on identical data: 0 exactly on 3.11.15,
    2.4e-15 on 3.13.7. The cause is that from 3.12 the built-in sum() uses
    compensated (Neumaier) summation on floats while a manual += loop does not,
    verified directly:

        sum([0.1, 1e18, -1e18, 0.1, 0.3])   is 0.4 on 3.11 and 0.5 on 3.13

    The naive implementation below is written with sum() for readability and
    the fast one accumulates with +=, so on 3.12 and later the two take very
    slightly different rounding paths to the same answer. 2.4e-15 is five
    orders of magnitude inside the 1e-10 tolerance and, if anything, the newer
    result is the more accurate of the two.
    """
    step = max(1, len(rows) // n)
    sub = rows[::step][:n]
    covs = [lambda r: r["pad"], lambda r: r["age"], lambda r: 1.0 if r["male"] else 0.0]
    xs, xx, pairs, times, events, _ = prepare(sub, covs)
    worst = 0.0
    for beta in ([0.0, 0.0, 0.0], [0.02, 0.05, 0.3], [-0.03, 0.08, -0.2]):
        a = loglik_grad_info(beta, xs, xx, pairs, times, events)
        b = loglik_grad_info_naive(beta, xs, xx, pairs, times, events)
        worst = max(worst, abs(a[0] - b[0]) / (abs(b[0]) + 1))
        for k in range(len(beta)):
            worst = max(worst, abs(a[1][k] - b[1][k]) / (abs(b[1][k]) + 1))
            for m in range(len(beta)):
                worst = max(worst, abs(a[2][k][m] - b[2][k][m]) / (abs(b[2][k][m]) + 1))
    return {"n_subsample": len(sub), "worst_relative_difference": worst,
            "passed": worst < 1e-10, "deterministic": True}


def selftest_optimum(model):
    return {
        "tolerance": SCORE_TOL,
        "max_abs_score": model["max_abs_score_at_optimum"],
        "information_positive_definite": model["information_positive_definite"],
        "loglik_improved_over_null": model["loglik"] > model["loglik_null"],
        # Asserted against the fitter's OWN target, never a number invented
        # here, so the two can never disagree again.
        "passed": (model["max_abs_score_at_optimum"] < SCORE_TOL
                   and model["information_positive_definite"]
                   and model["loglik"] > model["loglik_null"]),
    }


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
def group_flags(rows, key, thr, lower=None, name=None):
    """
    Older and younger indicators, with the within group as the reference.
    older is key > thr and younger is key < lower, which defaults to -thr. The
    flags are called name + "_older" and name + "_younger", name defaulting to
    key, so a second cut on the same key can sit beside the first.
    """
    lo = -thr if lower is None else lower
    stem = key if name is None else name
    for r in rows:
        v = r[key]
        r[stem + "_older"] = 1.0 if v > thr else 0.0
        r[stem + "_younger"] = 1.0 if v < lo else 0.0


def contrast(model, a, b):
    """
    Hazard ratio of term a against term b in one fitted model, exp(beta_a -
    beta_b), with its interval from the covariance: var_a + var_b - 2 cov_ab.
    Used for older-looking against younger-looking directly, rather than each
    against the within group.
    """
    i, j = model["term_order"].index(a), model["term_order"].index(b)
    v = model["vcov"]
    diff = model["terms"][a]["coef"] - model["terms"][b]["coef"]
    se = math.sqrt(v[i][i] + v[j][j] - 2.0 * v[i][j])
    return {
        "of": a, "against": b, "coef": diff, "se": se,
        "hr": math.exp(diff),
        "hr_lo95": math.exp(diff - 1.959963984540054 * se),
        "hr_hi95": math.exp(diff + 1.959963984540054 * se),
        "z": diff / se, "p": norm_sf(diff / se),
    }


def print_model(title, model, note=""):
    print("\n%s" % title)
    if note:
        print("  %s" % note)
    print("  n=%d  events=%d  iterations=%d  log-likelihood %.3f"
          % (model["n"], model["events"], model["iterations"], model["loglik"]))
    print("  %-26s %9s %8s %-22s %10s" % ("term", "coef", "se", "HR (95% CI)", "p"))
    for nm, t in model["terms"].items():
        pv = "<1e-300" if t["p"] < 1e-300 else "%.3g" % t["p"]
        print("  %-26s %+9.5f %8.5f  %5.3f (%5.3f, %5.3f)   %10s"
              % (nm, t["coef"], t["se"], t["hr"], t["hr_lo95"], t["hr_hi95"], pv))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Age- and sex-adjusted Cox models.")
    ap.add_argument("--csv", default=base.DEFAULT_CSV)
    ap.add_argument("--threshold", type=float, default=base.DEFAULT_THRESHOLD)
    ap.add_argument("--subsample", type=int, default=0,
                    help="fit on a random subsample of this size, for a quick look")
    ap.add_argument("--selftest", action="store_true",
                    help="run the correctness checks and stop")
    ap.add_argument("--json", default=os.path.join(HERE, "results", "cox.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    base.ensure_data(args.csv)
    rows, _ = base.load(args.csv)

    print("=" * 74)
    print("COX PROPORTIONAL HAZARDS, ADJUSTED FOR AGE AND SEX")
    print("The specification the 2021 paper used, so that the replication can be")
    print("stated on the same terms as the result it replicates.")
    print("=" * 74)

    checks = {}
    print("\n[0] CORRECTNESS OF THIS IMPLEMENTATION")
    fvn = selftest_fast_vs_naive(rows)
    checks["fast_vs_naive"] = fvn
    print("    fast cumulative-sum vs naive O(n^2) risk-set loop, n=%d"
          % fvn["n_subsample"])
    print("    worst relative difference in loglik, gradient and information: %.3g"
          % fvn["worst_relative_difference"])
    print("    %s" % ("PASS" if fvn["passed"] else "FAIL"))
    if args.selftest and not fvn["passed"]:
        return 1

    if args.subsample:
        rng = random.Random(20260922)
        rows = rng.sample(rows, min(args.subsample, len(rows)))
        print("\n    NOTE: fitting on a random subsample of %d, not the full cohort."
              % len(rows))

    slope, _ = base.bias_correct(rows)
    group_flags(rows, "pad", args.threshold)
    group_flags(rows, "pad_bc", args.threshold)
    # The corrected gap cut at the shares the raw +/- threshold cut selects. The
    # same function as ecg_age_bias.py, so both files use the same cut points.
    cuts = base.share_matched_cuts(rows, "pad", "pad_bc", args.threshold)
    group_flags(rows, "pad_bc", cuts["upper_cut_years"], lower=cuts["lower_cut_years"],
                name="pad_bc_matched")

    sex = lambda r: 1.0 if r["male"] else 0.0
    age = lambda r: r["age"]

    models = {}

    models["raw_categorical"] = fit(
        rows,
        [lambda r: r["pad_older"], lambda r: r["pad_younger"], age, sex],
        ["PAD > +%g (older)" % args.threshold,
         "PAD < -%g (younger)" % args.threshold,
         "age (per year)", "male"],
        verbose=args.verbose,
    )
    checks["optimum_raw_categorical"] = selftest_optimum(models["raw_categorical"])

    if args.selftest:
        print("\n[0b] OPTIMUM CHECK on the raw categorical model")
        o = checks["optimum_raw_categorical"]
        print("    max |score| at the solution        %.3g  (target %g)"
              % (o["max_abs_score"], o["tolerance"]))
        print("    information matrix pos. definite   %s"
              % o["information_positive_definite"])
        print("    log-likelihood improved over null  %s" % o["loglik_improved_over_null"])
        print("    %s" % ("PASS" if o["passed"] else "FAIL"))
        return 0 if (fvn["passed"] and o["passed"]) else 1

    models["corrected_categorical"] = fit(
        rows,
        [lambda r: r["pad_bc_older"], lambda r: r["pad_bc_younger"], age, sex],
        ["PAD_bc > +%g (older)" % args.threshold,
         "PAD_bc < -%g (younger)" % args.threshold,
         "age (per year)", "male"],
        verbose=args.verbose,
    )
    models["corrected_categorical_share_matched"] = fit(
        rows,
        [lambda r: r["pad_bc_matched_older"], lambda r: r["pad_bc_matched_younger"], age, sex],
        ["PAD_bc older (matched)", "PAD_bc younger (matched)",
         "age (per year)", "male"],
        verbose=args.verbose,
    )
    models["raw_continuous"] = fit(
        rows, [lambda r: r["pad"], age, sex],
        ["PAD (per year)", "age (per year)", "male"], verbose=args.verbose)
    models["corrected_continuous"] = fit(
        rows, [lambda r: r["pad_bc"], age, sex],
        ["PAD_bc (per year)", "age (per year)", "male"], verbose=args.verbose)

    # Older-looking against younger-looking directly, in each categorical model.
    pairs_ovy = {
        "raw_categorical": ("PAD > +%g (older)" % args.threshold,
                            "PAD < -%g (younger)" % args.threshold),
        "corrected_categorical": ("PAD_bc > +%g (older)" % args.threshold,
                                  "PAD_bc < -%g (younger)" % args.threshold),
        "corrected_categorical_share_matched": ("PAD_bc older (matched)",
                                                "PAD_bc younger (matched)"),
    }
    for key, (a, b) in pairs_ovy.items():
        models[key]["older_vs_younger"] = contrast(models[key], a, b)

    # Check the contrast arithmetic by refitting with younger as the reference,
    # where older-versus-younger is an ordinary coefficient with its own se.
    refit = fit(
        rows,
        [lambda r: r["pad_bc_matched_older"],
         lambda r: 1.0 - r["pad_bc_matched_older"] - r["pad_bc_matched_younger"], age, sex],
        ["older vs younger", "within vs younger", "age (per year)", "male"],
        verbose=args.verbose,
    )
    via_cov = models["corrected_categorical_share_matched"]["older_vs_younger"]
    direct = refit["terms"]["older vs younger"]
    checks["contrast_vs_refit"] = {
        "coef_via_covariance": via_cov["coef"], "coef_refit": direct["coef"],
        "se_via_covariance": via_cov["se"], "se_refit": direct["se"],
        "passed": (abs(via_cov["coef"] - direct["coef"]) < 1e-6
                   and abs(via_cov["se"] - direct["se"]) < 1e-6),
    }

    checks["optimum_raw_categorical"] = selftest_optimum(models["raw_categorical"])
    o = checks["optimum_raw_categorical"]
    print("    max |score| at the solution %.3g, information positive definite: %s"
          % (o["max_abs_score"], o["information_positive_definite"]))

    print("\n[1] THE PUBLISHED SPECIFICATION, RAW PAD")
    print_model("    Cox: raw PAD group + age + sex",
                models["raw_categorical"],
                note="The 2021 paper reported HR 1.79 (1.69 to 1.90) for the older "
                     "group\n  and HR 0.78 (0.74 to 0.83) for the younger group.")

    pub_older, pub_younger = 1.79, 0.78
    got_older = models["raw_categorical"]["terms"][
        "PAD > +%g (older)" % args.threshold]["hr"]
    got_younger = models["raw_categorical"]["terms"][
        "PAD < -%g (younger)" % args.threshold]["hr"]
    checks["vs_published"] = {
        "published_hr_older": pub_older, "replicated_hr_older": got_older,
        "published_hr_younger": pub_younger, "replicated_hr_younger": got_younger,
        "abs_diff_older": abs(got_older - pub_older),
        "abs_diff_younger": abs(got_younger - pub_younger),
    }
    print("\n    Published 1.79 / 0.78 against this run %.3f / %.3f."
          % (got_older, got_younger))
    print("    These are close but they are not expected to be identical: the")
    print("    published analysis used %d participants and this one uses"
          % base.PUBLISHED_COHORT["n"])
    print("    %d, so the cohort definitions differ. See README.md."
          % models["raw_categorical"]["n"])

    print("\n[2] THE SAME MODEL ON THE BIAS-CORRECTED GAP")
    print_model("    Cox: corrected PAD group + age + sex",
                models["corrected_categorical"])

    print("\n[2b] THE SAME MODEL, CORRECTED GAP CUT AT THE RAW CUT'S SHARES")
    print("    older is PAD_bc > %+.3f (%.2f%% of patients), younger is PAD_bc < %+.3f"
          % (cuts["upper_cut_years"], cuts["older_share_pct"], cuts["lower_cut_years"]))
    print("    (%.2f%%), the same shares the raw +/- %g cut selects."
          % (cuts["younger_share_pct"], args.threshold))
    print_model("    Cox: corrected PAD group (share-matched) + age + sex",
                models["corrected_categorical_share_matched"])

    print("\n    Older-looking against younger-looking directly, adjusted for age and sex:")
    for key, label in (("raw_categorical", "raw gap, +/- %g" % args.threshold),
                       ("corrected_categorical", "corrected gap, +/- %g" % args.threshold),
                       ("corrected_categorical_share_matched",
                        "corrected gap, share-matched")):
        c = models[key]["older_vs_younger"]
        print("      %-30s HR %5.3f (%5.3f, %5.3f)" % (label, c["hr"], c["hr_lo95"], c["hr_hi95"]))
    cr = checks["contrast_vs_refit"]
    print("    contrast from the covariance against a refit with younger as reference:")
    print("      coef %.9f vs %.9f, se %.9f vs %.9f  %s"
          % (cr["coef_via_covariance"], cr["coef_refit"], cr["se_via_covariance"],
             cr["se_refit"], "PASS" if cr["passed"] else "FAIL"))

    print("\n[3] CONTINUOUS, PER YEAR OF GAP")
    print_model("    Cox: raw PAD continuous + age + sex", models["raw_continuous"])
    print_model("    Cox: corrected PAD continuous + age + sex",
                models["corrected_continuous"])

    raw_c = models["raw_continuous"]
    bc_c = models["corrected_continuous"]
    hr_raw = raw_c["terms"]["PAD (per year)"]["hr"]
    hr_bc = bc_c["terms"]["PAD_bc (per year)"]["hr"]
    b_gap = raw_c["terms"]["PAD (per year)"]["coef"]
    # PAD_bc = PAD - (intercept + slope * age), so
    #   b * PAD_bc + c_bc * age  =  b * PAD + (c_bc - b * slope) * age - b * intercept.
    # The constant drops out of a Cox model, so both fits share b and the
    # log-likelihood, and the age coefficients differ by exactly -b * slope.
    shift_observed = (raw_c["terms"]["age (per year)"]["coef"]
                      - bc_c["terms"]["age (per year)"]["coef"])
    shift_expected = -b_gap * slope
    identical = abs(hr_raw - hr_bc) < 1e-6
    checks["continuous_identity"] = {
        "hr_raw_per_year": hr_raw,
        "hr_corrected_per_year": hr_bc,
        "abs_difference": abs(hr_raw - hr_bc),
        "identical_to_6dp": identical,
        "loglik_difference": abs(raw_c["loglik"] - bc_c["loglik"]),
        "age_coef_shift_observed": shift_observed,
        "age_coef_shift_expected": shift_expected,
        "passed": (identical
                   and abs(raw_c["loglik"] - bc_c["loglik"]) < 1e-6
                   and abs(shift_observed - shift_expected) < 1e-6),
    }
    ci = checks["continuous_identity"]
    print("\n    NOTE: an algebraic identity, used here only as a check on the code.")
    print("    The two continuous hazard ratios are %.6f and %.6f." % (hr_raw, hr_bc))
    print("    They must be equal. PAD_bc is PAD minus a linear function of age, so")
    print("      b * PAD_bc + c * age  =  b * PAD + (c - b * slope) * age + constant,")
    print("    and a Cox model already adjusted for age fits the same model either way,")
    print("    with the age coefficient moved and the constant absorbed. That is not")
    print("    a finding about the data. What it does test is this implementation:")
    print("      log-likelihoods differ by          %.3g" % ci["loglik_difference"])
    print("      age coefficient shift, observed    %+.9f" % shift_observed)
    print("      age coefficient shift, -b * slope  %+.9f" % shift_expected)
    print("    %s" % ("PASS" if ci["passed"] else "FAIL"))
    print("    Barthels et al. report the same equality, HR 1.020 per year for both PAD")
    print("    and PAD_bc in their age-adjusted model, as the algebra requires.")

    payload = {"models": models, "checks": checks,
               "threshold_years": args.threshold,
               "share_matched_cuts": cuts,
               "subsample": args.subsample or None}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("\nmachine-readable results written to %s" % base.shown(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
