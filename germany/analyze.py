"""Step 1 analysis: gates, heads, evaluation. Reads cached features; writes results/*.json and results/results.md."""
import json, os, numpy as np, pandas as pd
from common import CODE_DIR, PTB_DIR, FEAT_DIR, HERE
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)
BANDS = [(18, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89)]
NBOOT = 1000

# ---------------- load CODE ----------------
df = pd.read_csv(f"{CODE_DIR}/exams.csv")
F, ids, P = [], [], []
for k in range(18):
    p = f"{FEAT_DIR}/code_part{k}"
    F.append(np.load(p + "_feat.npy")); ids.append(np.load(p + "_ids.npy")); P.append(np.load(p + "_pred.npy"))
F = np.concatenate(F); ids = np.concatenate(ids); P = np.concatenate(P)
first = df.sort_values("exam_id").drop_duplicates("patient_id").set_index("exam_id")
assert len(ids) == len(first) and set(ids) == set(first.index), "feature set does not match first-exam set"
m = first.loc[ids].reset_index()
age = m.age.values.astype(np.float64); nn_csv = m.nn_predicted_age.values.astype(np.float64)
gateA = dict(n=int(len(m)), pearson_r=float(np.corrcoef(P, nn_csv)[0, 1]), mean_abs_diff=float(np.abs(P - nn_csv).mean()),
             max_abs_diff=float(np.abs(P - nn_csv).max()))
gateA["pass"] = bool(gateA["pearson_r"] >= 0.99 and gateA["mean_abs_diff"] <= 1)
print("GATE A", gateA)
# also over every exam is not needed: first exams are 233,770 patients.

# ---------------- split by patient ----------------
pats = np.sort(m.patient_id.unique()); perm = np.random.default_rng(0).permutation(pats)
n = len(perm); tr_p, va_p, te_p = perm[:int(.7*n)], perm[int(.7*n):int(.8*n)], perm[int(.8*n):]
tr = m.patient_id.isin(tr_p).values; va = m.patient_id.isin(va_p).values; te = m.patient_id.isin(te_p).values
print("split", tr.sum(), va.sum(), te.sum())

# ---------------- heads ----------------
Xtr = F[tr].astype(np.float64); mu = Xtr.mean(0); Xc = Xtr - mu
ym = age[tr].mean(); yc = age[tr] - ym
G = Xc.T @ Xc; g = Xc.T @ yc; yy = float(yc @ yc)
ev, V = np.linalg.eigh(G); Vg = V.T @ g
scale = np.trace(G) / G.shape[0]
def ridge_w(lam): return V @ (Vg / (ev + lam))
def predict(w, X): return (X.astype(np.float64) - mu) @ w + ym
lams = scale * np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
def constrained_w(lam):
    w0 = ridge_w(lam)
    return w0 * (yy / float(g @ w0))     # exact Lagrange solution: w = (1+mu) A^-1 g with g'w = y'y
rows = []
best = {}
for name, fn in (("plain", ridge_w), ("constrained", constrained_w)):
    scores = [(np.abs(predict(fn(l), F[va]) - age[va]).mean(), l) for l in lams]
    best[name] = min(scores)[1]
    rows.append({name: [(float(s), float(l / scale)) for s, l in scores]})
print("ridge grid (val MAE, lambda/scale):", json.dumps(rows))
w_plain = ridge_w(best["plain"]); w_con = constrained_w(best["constrained"])
pred_tr_con = predict(w_con, F[tr])
cov_con = float(np.cov(pred_tr_con - age[tr], age[tr])[0, 1])
pred_tr_plain = predict(w_plain, F[tr]); cov_plain = float(np.cov(pred_tr_plain - age[tr], age[tr])[0, 1])
print("train cov(gap,age): plain", cov_plain, "constrained", cov_con)
# post-hoc line fitted in Brazil on train: gap ~ a + b*age, for plain head and for raw network
def fit_line(gap, a): b = np.polyfit(a, gap, 1); return float(b[1]), float(b[0])   # intercept, slope
line_plain = fit_line(pred_tr_plain - age[tr], age[tr]); line_net = fit_line(P[tr] - age[tr], age[tr])
heads_info = dict(lambda_over_scale={k: float(v / scale) for k, v in best.items()}, train_cov_gap_age=dict(plain=cov_plain, constrained=cov_con),
                  posthoc_line_plain=line_plain, posthoc_line_network=line_net)

def make_methods(Fx, P_net, a, lines):
    """returns dict method -> predicted age (c and d+posthoc use true age a)."""
    pp = predict(w_plain, Fx); pc = predict(w_con, Fx)
    out = {"(d) network raw": P_net, "(a) plain ridge": pp, "(b) constrained": pc}
    out["(c) plain + post-hoc"] = pp - (lines["plain"][0] + lines["plain"][1] * a)
    out["(d+) network + post-hoc"] = P_net - (lines["net"][0] + lines["net"][1] * a)
    return out

def metrics(pred, a, band_mask=None):
    gap = pred - a
    r = np.corrcoef(gap, a)[0, 1]; s = np.polyfit(a, gap, 1)[0]
    bands = {f"{lo}-{hi}": (float(gap[(a >= lo) & (a < hi + 1)].mean()) if ((a >= lo) & (a < hi + 1)).any() else None) for lo, hi in BANDS}
    return dict(MAE=float(np.abs(gap).mean()), r_gap_age=float(r), slope=float(s), gap_by_band=bands)

def boot(methods, a, cl):
    """cluster bootstrap by patient: CI for r, MAE of each method and r-differences vs (b)."""
    ucl, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(inv); starts = np.searchsorted(inv[order], np.arange(len(ucl))); ends = np.append(starts[1:], len(inv))
    r = {k: [] for k in methods}; mae = {k: [] for k in methods}
    rg = np.random.default_rng(1)
    for _ in range(NBOOT):
        pick = rg.integers(0, len(ucl), len(ucl))
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        aa = a[idx]
        for k, v in methods.items():
            gp = v[idx] - aa
            r[k].append(np.corrcoef(gp, aa)[0, 1]); mae[k].append(np.abs(gp).mean())
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {k: dict(r_ci=ci(r[k]), MAE_ci=ci(mae[k])) for k in methods}
    diffs = {}
    for k1, k2 in (("(b) constrained", "(c) plain + post-hoc"), ("(b) constrained", "(a) plain ridge"), ("(c) plain + post-hoc", "(a) plain ridge"),
                   ("(b) constrained", "(d) network raw"), ("(c) plain + post-hoc", "(d+) network + post-hoc")):
        d = np.array(r[k1]) - np.array(r[k2]); diffs[f"r[{k1}] - r[{k2}]"] = dict(mean=float(d.mean()), ci=ci(d))
        d = np.abs(np.array(r[k1])) - np.abs(np.array(r[k2])); diffs[f"|r|[{k1}] - |r|[{k2}]"] = dict(mean=float(d.mean()), ci=ci(d))
    return res, diffs

def evaluate(tag, Fx, Pn, a, cl, lines):
    meth = make_methods(Fx, Pn, a, lines)
    mt = {k: metrics(v, a) for k, v in meth.items()}
    ci, diffs = boot(meth, a, cl)
    for k in mt: mt[k].update(ci[k])
    return dict(n=int(len(a)), n_patients=int(len(np.unique(cl))), age_mean=float(a.mean()), methods=mt, r_differences=diffs)

lines_br = dict(plain=line_plain, net=line_net)
results = dict(gateA=gateA, heads=heads_info)
results["code_test"] = evaluate("code_test", F[te], P[te], age[te], m.patient_id.values[te], lines_br)

# ---------------- PTB-XL ----------------
pm = pd.read_csv(f"{FEAT_DIR}/ptb_meta.csv"); PF = np.load(f"{FEAT_DIR}/ptb_feat.npy"); PP = np.load(f"{FEAT_DIR}/ptb_pred.npy")
pa = pm.age.values.astype(np.float64)
gateB = dict(n=int(len(pm)), MAE=float(np.abs(PP - pa).mean()), pearson_r_pred_age=float(np.corrcoef(PP, pa)[0, 1]),
             mean_gap=float((PP - pa).mean()), CODE_test_net_MAE=float(np.abs(P[te] - age[te]).mean()),
             CODE_test_net_r=float(np.corrcoef(P[te], age[te])[0, 1]))
gateB["pass"] = bool(gateB["pearson_r_pred_age"] >= 0.5); results["gateB"] = gateB; print("GATE B", gateB)
results["ptbxl_brazil_fit"] = evaluate("ptb_br", PF, PP, pa, pm.patient_id.values, lines_br)
# German refit of the post-hoc line on half the patients (by patient), evaluate on the other half
pp_ids = np.sort(pm.patient_id.unique()); pp_perm = np.random.default_rng(0).permutation(pp_ids)
half = set(pp_perm[:len(pp_perm) // 2]); inA = pm.patient_id.isin(half).values
pp_plain = predict(w_plain, PF)
lines_de = dict(plain=fit_line(pp_plain[inA] - pa[inA], pa[inA]), net=fit_line(PP[inA] - pa[inA], pa[inA]))
results["ptbxl_german_refit"] = evaluate("ptb_de", PF[~inA], PP[~inA], pa[~inA], pm.patient_id.values[~inA], lines_de)
results["ptbxl_german_refit"]["lines_fitted_on_german_half"] = lines_de
results["ptbxl_german_refit"]["note"] = "only the post-hoc lines of (c) and (d+) are refitted; (a), (b), (d) are unchanged"

# ---------------- optional: mortality hazard ratio on CODE test ----------------
try:
    from lifelines import CoxPHFitter
    meth = make_methods(F[te], P[te], age[te], lines_br)
    sub = m[te].copy(); sub["sex"] = sub.is_male.astype(int)
    ok = sub.timey.notna().values & sub.death.notna().values
    surv = {}
    for k in ("(b) constrained", "(c) plain + post-hoc"):
        d = pd.DataFrame(dict(T=sub.timey.values[ok], E=sub.death.values[ok].astype(int), gap=(meth[k] - age[te])[ok],
                              age=age[te][ok], sex=sub.sex.values[ok]))
        cph = CoxPHFitter().fit(d, "T", "E"); s = cph.summary.loc["gap"]
        surv[k] = dict(HR_per_year=float(np.exp(s["coef"])), ci=[float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])],
                       n=int(ok.sum()), deaths=int(d.E.sum()))
    results["mortality_cox_code_test"] = surv; print("COX", surv)
except Exception as e:
    results["mortality_cox_code_test"] = f"failed: {e!r}"; print("cox failed", e)

json.dump(results, open(f"{RES}/results.json", "w"), indent=1)
def table(block):
    lines = ["| Method | MAE (years) | r (gap vs age) [95% CI] | slope |" + "".join(f" gap {lo}-{hi} |" for lo, hi in BANDS),
             "|---|---|---|---|" + "---|" * len(BANDS)]
    for k, v in block["methods"].items():
        lines.append(f"| {k} | {v['MAE']:.2f} | {v['r_gap_age']:+.3f} [{v['r_ci'][0]:+.3f}, {v['r_ci'][1]:+.3f}] | {v['slope']:+.3f} | " +
                     " | ".join("n/a" if x is None else f"{x:+.1f}" for x in v["gap_by_band"].values()) + " |")
    return "\n".join(lines)
with open(f"{RES}/results.md", "w") as f:
    for key, title in (("code_test", "CODE-15% test patients"), ("ptbxl_brazil_fit", "PTB-XL, corrections fitted in Brazil"),
                       ("ptbxl_german_refit", "PTB-XL second half, post-hoc line refitted on first half")):
        b = results[key]; f.write(f"### {title} (n={b['n']} ECGs, {b['n_patients']} patients, mean age {b['age_mean']:.1f})\n\n{table(b)}\n\n")
        for k, v in b["r_differences"].items(): f.write(f"- {k}: {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]\n")
        f.write("\n")
print(open(f"{RES}/results.md").read())
