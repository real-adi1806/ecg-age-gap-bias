#!/usr/bin/env python3
"""
Optional: check cox_model.py against lifelines.

Nothing in this repository needs this file. cox_model.py is standard library
only and carries its own correctness checks. This script exists so that the Cox
implementation can also be compared against an established reference, which is
a stronger claim than an internal check. It is the only reason requirements.txt
has anything in it.

    python -m pip install -r requirements.txt
    python crosscheck_lifelines.py

WHAT IS ASSERTED, AND WHY IT IS NOT A COEFFICIENT TOLERANCE
-----------------------------------------------------------
The obvious test, "do the coefficients agree to 1e-6", was tried first and it
failed, for a reason worth writing down rather than tuning away.

Two independently converged optimisers cannot agree more closely than the
looser of them converges. Measured on this cohort, evaluating BOTH solutions in
the SAME likelihood function:

    cox_model.py    max|score| 4.1e-09    log-likelihood -94661.982283812
    lifelines       max|score| 2.1e-01    log-likelihood -94661.982283830

So the residual disagreement is lifelines' stopping rule, not this code's
arithmetic, and a coefficient tolerance measures the wrong thing. Both answers
are correct to every decimal place anyone would report.

This script therefore asserts two sharper things instead:

  1. STANDARD ERRORS agree to 1e-8. These come from the information matrix, a
     second-derivative quantity that is unforgiving of any error in the
     likelihood. A wrong Efron tie correction or a wrong outer-product term
     shows up here long before it shows up in a point estimate. This is the
     real evidence that the two implementations compute the same function.

  2. cox_model.py's solution sits AT LEAST AS HIGH on lifelines' own likelihood
     surface as lifelines' solution does. That is a tolerance-free statement:
     either this code found at least as good a fit, or it did not.

Coefficient differences are still printed, for information.
"""

import sys

import cox_model
import ecg_age_bias as base

SE_TOL = 1e-8


def main():
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        import lifelines
    except ImportError:
        print("lifelines is not installed. This check is optional:")
        print("    python -m pip install -r requirements.txt")
        return 0

    base.ensure_data(base.DEFAULT_CSV)
    rows, _ = base.load(base.DEFAULT_CSV)
    base.bias_correct(rows)
    cox_model.group_flags(rows, "pad", base.DEFAULT_THRESHOLD)
    cox_model.group_flags(rows, "pad_bc", base.DEFAULT_THRESHOLD)

    print("lifelines %s, pandas %s" % (lifelines.__version__, pd.__version__))
    print("n = %d rows, %d events\n"
          % (len(rows), sum(1 for r in rows if r["died"])))

    male = lambda r: 1.0 if r["male"] else 0.0
    age = lambda r: r["age"]
    specs = [
        ("raw categorical", ["pad_older", "pad_younger", "age", "male_num"],
         [lambda r: r["pad_older"], lambda r: r["pad_younger"], age, male]),
        ("corrected categorical", ["pad_bc_older", "pad_bc_younger", "age", "male_num"],
         [lambda r: r["pad_bc_older"], lambda r: r["pad_bc_younger"], age, male]),
        ("raw continuous", ["pad", "age", "male_num"],
         [lambda r: r["pad"], age, male]),
    ]

    columns = {
        "timey": [r["timey"] for r in rows],
        "died": [1 if r["died"] else 0 for r in rows],
        "pad": [r["pad"] for r in rows],
        "pad_bc": [r["pad_bc"] for r in rows],
        "age": [r["age"] for r in rows],
        "male_num": [male(r) for r in rows],
        "pad_older": [r["pad_older"] for r in rows],
        "pad_younger": [r["pad_younger"] for r in rows],
        "pad_bc_older": [r["pad_bc_older"] for r in rows],
        "pad_bc_younger": [r["pad_bc_younger"] for r in rows],
    }

    worst_se = 0.0
    failures = []
    for title, cols, covs in specs:
        frame = pd.DataFrame(columns)[["timey", "died"] + cols]
        cph = CoxPHFitter()
        cph.fit(frame, duration_col="timey", event_col="died")
        mine = cox_model.fit(rows, covs, cols)

        beta_me = [mine["terms"][c]["coef"] for c in cols]
        beta_lf = [float(cph.params_[c]) for c in cols]
        xs, xx, pairs, times, events, _ = cox_model.prepare(rows, covs)
        ll_me, g_me, _ = cox_model.loglik_grad_info(beta_me, xs, xx, pairs, times, events)
        ll_lf, g_lf, _ = cox_model.loglik_grad_info(beta_lf, xs, xx, pairs, times, events)

        print(title)
        print("  %-16s %15s %15s %11s" % ("term", "this repo", "lifelines", "abs diff"))
        for c in cols:
            a, b = mine["terms"][c]["coef"], float(cph.params_[c])
            print("  %-16s %+15.9f %+15.9f %11.2e" % (c, a, b, abs(a - b)))
            sa, sb = mine["terms"][c]["se"], float(cph.standard_errors_[c])
            d = abs(sa - sb)
            worst_se = max(worst_se, d)
            if d > SE_TOL:
                failures.append(("%s / %s standard error" % (title, c), d))
            print("  %-16s %+15.9f %+15.9f %11.2e" % ("  (se)", sa, sb, d))

        print("  max|score|      this repo %.3e   lifelines %.3e"
              % (max(abs(v) for v in g_me), max(abs(v) for v in g_lf)))
        print("  log-likelihood  this repo %.9f   lifelines %.9f"
              % (ll_me, ll_lf))
        if ll_me < ll_lf - 1e-9:
            failures.append(("%s: lifelines found a better fit" % title, ll_lf - ll_me))
            print("  -> lifelines sits HIGHER on the likelihood. That is a real problem.")
        else:
            print("  -> this repo sits at least as high on the likelihood. OK")
        print()

    print("worst standard-error disagreement: %.3e  (tolerance %g)" % (worst_se, SE_TOL))
    if failures:
        print("FAIL")
        for f in failures:
            print("   ", f)
        return 1
    print("PASS: standard errors match a reference implementation, and this "
          "repository's fit is never worse on the likelihood.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
