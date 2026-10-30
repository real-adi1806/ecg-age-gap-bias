"""Germany: does the regression-bias correction of the AI-ECG age gap, fitted in Brazil, transfer to PTB-XL?

    python run_germany.py              the full run: re-runs the network on every analysed PTB-XL ECG
    python run_germany.py --use-cache  reads the 25 Oct cached predictions instead and skips the network
                                       (quick iteration only: no gate A re-run, no amplitude or scale check)

The correction is the one the Brazil replication in the parent folder uses (python ecg_age_bias.py): the
original network's own predicted age, gap = predicted - chronological, minus the line gap = a + b x age
fitted by OLS on 233,647 CODE-15% patients. That line is read from the replication's results file and is
NOT refitted on German data, except in the half-split analysis, which exists to show what a local refit buys.
Secondary: Barthels et al.'s second, age-level step (subtract the mean linearly corrected gap at each integer
age), and the network re-run at an input scale matched to CODE-15% amplitudes.

PTB-XL has no deaths. Everything here is about the bias of the age gap, never survival.
Writes results/germany.json. verify_germany.py asserts every number the manuscript quotes.
"""
import argparse, ast, hashlib, json, os, platform, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import CODE_DIR, PTB_DIR, FEAT_DIR, DATA_DIR, HERE, PTB_AGE_RANGE

BRAZIL_RESULTS = os.environ.get("BRAZIL_RESULTS", os.path.join(HERE, "..", "results", "results.json"))
BRAZIL_N = 233647                   # patients in the replication's cohort (exams.csv rows that carry an outcome)
MD5 = {"model.zip": "00aad2c0341942837019086cf5d94cd5",      # Zenodo 4892365, the published weights
       "exams.csv": "0107516d3f63864498fb77d15799cc95"}      # Zenodo 4916206, CODE-15%
BANDS = [("lt40", 0, 39), ("40-49", 40, 49), ("50-59", 50, 59), ("60-69", 60, 69), ("70-79", 70, 79), ("ge80", 80, 200)]
AGE_POOL = 85                       # age-level step: ages from 85 up share one mean (CODE-15% has 13 patients above 85)
NBOOT = 2000
SEED = 20261028
CODE_SPOT_N = 1000                  # CODE-15% exams re-run from raw waveforms as a spot check of gate A
MIN_SIGNAL = 1000                   # amplitude check skips tracings with fewer non-zero samples (empty records)
OUT = os.path.join(HERE, "results", "germany.json")
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

def file_digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------- inputs
def check_inputs(ptb_rows):
    """md5 of the weights and of exams.csv; sha256 of every PTB-XL file used, against PTB-XL's own SHA256SUMS.txt."""
    got = {"model.zip": file_digest(os.path.join(DATA_DIR, "model", "model.zip"), "md5"),
           "exams.csv": file_digest(os.path.join(CODE_DIR, "exams.csv"), "md5")}
    for k, v in got.items():
        assert v == MD5[k], f"{k}: md5 {v}, expected {MD5[k]}"
    sums = {}
    for line in open(os.path.join(PTB_DIR, "SHA256SUMS.txt")):
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    files = ["ptbxl_database.csv"] + [f"{fn}{ext}" for fn in ptb_rows.filename_hr for ext in (".hea", ".dat")]
    with ThreadPoolExecutor(8) as ex:
        digests = list(ex.map(lambda f: file_digest(os.path.join(PTB_DIR, f), "sha256"), files))
    bad = [f for f, d in zip(files, digests) if sums.get(f) != d]
    assert not bad, f"{len(bad)} PTB-XL files fail SHA256SUMS.txt, first: {bad[:3]}"
    return dict(md5=got, ptbxl_files_sha256_checked=len(files), ptbxl_sha256_failures=0)

# ---------------------------------------------------------------- Brazil
def brazil_line():
    """The line from the replication's results file, checked against an independent refit on exams.csv."""
    br = json.load(open(BRAZIL_RESULTS))
    assert br["_provenance"]["md5"] == MD5["exams.csv"] and br["cohort"]["n_patients"] == BRAZIL_N
    a, b = br["confound"]["ols_intercept"], br["confound"]["ols_slope"]
    ex = pd.read_csv(os.path.join(CODE_DIR, "exams.csv"))
    co = ex[ex.death.notna()]
    assert len(co) == BRAZIL_N, len(co)
    age = co.age.values.astype(float); gap = co.nn_predicted_age.values.astype(float) - age
    b2, a2 = np.polyfit(age, gap, 1)
    assert abs(a2 - a) < 1e-8 and abs(b2 - b) < 1e-10, (a, a2, b, b2)
    return (a, b), co, ex

def age_index(age):
    return np.minimum(age.astype(int), AGE_POOL)

def age_level_means(age, corr, w):
    """Barthels' second step: weighted mean of the linearly corrected gap at each integer age (ages >= AGE_POOL pooled).
    Returns a lookup array indexed by age_index; NaN where no patient was drawn."""
    i = age_index(age)
    s = np.bincount(i, w * corr, minlength=AGE_POOL + 1); n = np.bincount(i, w, minlength=AGE_POOL + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n

def brazil_reference(co, line):
    """What the line leaves behind in its own cohort: mean corrected gap by age band and by sex, normal 95% CIs."""
    age = co.age.values.astype(float); male = co.is_male.values.astype(bool)
    corr = co.nn_predicted_age.values - age - (line[0] + line[1] * age)
    def mean_ci(x):
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        return dict(n=int(len(x)), mean=float(m), ci=[float(m - 1.96 * se), float(m + 1.96 * se)])
    out = {k: mean_ci(corr[(age >= lo) & (age <= hi)]) for k, lo, hi in BANDS}
    out["male"] = mean_ci(corr[male]); out["female"] = mean_ci(corr[~male])
    out["male_minus_female"] = out["male"]["mean"] - out["female"]["mean"]
    normal = co.normal_ecg.values.astype(bool)                 # the dataset's own flag, not the same definition as PTB-XL NORM
    out["normal_ecg"] = mean_ci(corr[normal]); out["not_normal_ecg"] = mean_ci(corr[~normal])
    out["pct_over_plus8"] = float(100 * (corr > 8).mean()); out["pct_under_minus8"] = float(100 * (corr < -8).mean())
    mp = age_level_means(age, corr, np.ones_like(age))
    out["age_level_means"] = {str(k): float(mp[k]) for k in range(int(age.min()), AGE_POOL + 1)}
    counts = np.bincount(age.astype(int))[int(age.min()): AGE_POOL + 1]     # patients at each single age, before pooling
    out["age_level_n_min"] = int(counts.min()); out["age_level_n_max"] = int(counts.max())
    out["n_older_than_pool"] = int((age > AGE_POOL).sum())
    return out, mp

# ---------------------------------------------------------------- the network and the signal scale
def robust_amplitude(x):
    """Per lead: (99.5th - 0.5th percentile) after removing the median, over the non-zero (unpadded) samples."""
    x = x[np.any(x != 0, axis=1)]
    x = x - np.median(x, axis=0)
    return np.percentile(x, 99.5, axis=0) - np.percentile(x, 0.5, axis=0)

def gate_a_cached(ex):
    """The 25 Oct cached predictions for all 233,770 first exams against exams.csv nn_predicted_age."""
    ids = np.concatenate([np.load(f"{FEAT_DIR}/code_part{k}_ids.npy") for k in range(18)])
    P = np.concatenate([np.load(f"{FEAT_DIR}/code_part{k}_pred.npy") for k in range(18)]).astype(float)
    nn = ex.set_index("exam_id").loc[ids, "nn_predicted_age"].values.astype(float)
    return dict(n=int(len(ids)), pearson_r=float(np.corrcoef(P, nn)[0, 1]),
                mean_abs_diff=float(np.abs(P - nn).mean()), max_abs_diff=float(np.abs(P - nn).max()))

def gate_a_rerun(co, model, dev):
    """Re-run the network from the raw CODE-15% HDF5 on a random sample of the Brazil cohort."""
    import h5py
    from common import run_batches
    smp = co.sample(CODE_SPOT_N, random_state=SEED)
    X, nn = [], []
    for fname, grp in smp.groupby("trace_file"):
        with h5py.File(os.path.join(CODE_DIR, fname), "r") as f:
            pos = {e: i for i, e in enumerate(np.array(f["exam_id"]))}
            idx = sorted(pos[e] for e in grp.exam_id)
            want = dict(zip(grp.exam_id, grp.nn_predicted_age))
            X.append(np.array(f["tracings"][idx], np.float32)); nn += [want[e] for e in np.array(f["exam_id"])[idx]]
    X = np.concatenate(X); nn = np.array(nn, float)
    _, P = run_batches(model, dev, lambda a, b: X[a:b], len(X)); P = P.astype(float)
    amp = np.array([robust_amplitude(x) for x in X if np.any(x != 0, axis=1).sum() >= MIN_SIGNAL])
    return dict(n=int(len(P)), pearson_r=float(np.corrcoef(P, nn)[0, 1]), mean_abs_diff=float(np.abs(P - nn).mean()),
                max_abs_diff=float(np.abs(P - nn).max())), amp

def run_ptb(model, dev, rows, scale, amp=None):
    from common import load_ptb_record, run_batches
    def batch(a, b):
        x = np.stack([load_ptb_record(rows.filename_hr.iloc[i], scale) for i in range(a, b)])
        if amp is not None:
            amp.extend(robust_amplitude(e) for e in x)
        return x
    t = time.time(); _, P = run_batches(model, dev, batch, len(rows))
    log(f"network on {len(rows)} PTB-XL ECGs at scale x{scale:.3f}: {time.time() - t:.0f}s on {dev}")
    return P.astype(float)

def ptb_predictions(d, use_cache):
    """The original network on every analysed PTB-XL ECG, from the raw 500 Hz waveforms, at the README scale (mV x 1)."""
    cached = None
    if os.path.exists(f"{FEAT_DIR}/ptb_pred.npy"):
        cm = pd.read_csv(f"{FEAT_DIR}/ptb_meta.csv", usecols=["ecg_id"])
        assert (cm.ecg_id.values == d.ecg_id.values).all(), "cached PTB-XL predictions are in a different row order"
        cached = np.load(f"{FEAT_DIR}/ptb_pred.npy").astype(float)
    if use_cache:
        return cached, None, None, None, None
    from common import device, load_model
    dev = device(); model = load_model(dev); amp = []
    P = run_ptb(model, dev, d, 1.0, amp)
    vs_cache = None
    if cached is not None:
        vs_cache = dict(n=int(len(P)), pearson_r=float(np.corrcoef(P, cached)[0, 1]),
                        mean_abs_diff=float(np.abs(P - cached).mean()), max_abs_diff=float(np.abs(P - cached).max()))
    return P, vs_cache, np.array(amp), model, dev

def amplitude_check(amp_code, amp_ptb):
    leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    mc, mp = np.median(amp_code, axis=0), np.median(amp_ptb, axis=0)
    ratio = mc / mp
    return dict(stat="median over ECGs of (99.5th - 0.5th percentile) after removing the median, non-zero samples only, network input units",
                n_code15=int(len(amp_code)), n_ptbxl=int(len(amp_ptb)),
                per_lead={l: dict(code15=float(c), ptbxl=float(p), ratio=float(r)) for l, c, p, r in zip(leads, mc, mp, ratio)},
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()), matched_scale=float(np.median(ratio)))

# ---------------------------------------------------------------- statistics
def wmean(x, w):
    return float((w * x).sum() / w.sum())

def wcorr(x, y, w):
    mx, my = wmean(x, w), wmean(y, w)
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))

def wline(age, gap, w):
    """Weighted OLS of gap on age: (intercept, slope)."""
    ma, mg = wmean(age, w), wmean(gap, w)
    b = (w * (age - ma) * (gap - mg)).sum() / (w * (age - ma) ** 2).sum()
    return float(mg - b * ma), float(b)

def line_offset(line, age):
    return line[0] + line[1] * age

def summarise(pred, age, male, offset, w=None):
    """Every statistic quoted for one cohort and one correction (corrected gap = gap - offset), as a flat dict."""
    w = np.ones_like(age) if w is None else w
    gap = pred - age
    corr = gap - offset
    s = dict(mae_raw=wmean(np.abs(gap), w), mean_gap_raw=wmean(gap, w), mean_gap_corrected=wmean(corr, w),
             r_raw=wcorr(gap, age, w), r_corrected=wcorr(corr, age, w),
             slope_raw=wline(age, gap, w)[1], slope_corrected=wline(age, corr, w)[1],
             pct_over_plus8=100 * wmean((corr > 8).astype(float), w), pct_under_minus8=100 * wmean((corr < -8).astype(float), w))
    dummies = []
    for k, lo, hi in BANDS:
        m = ((age >= lo) & (age <= hi)).astype(float)
        s[f"band_{k}_raw"] = wmean(gap, w * m); s[f"band_{k}_corrected"] = wmean(corr, w * m)
        dummies.append(m)
    bands = [s[f"band_{k}_corrected"] for k, _, _ in BANDS]
    s["band_range_corrected"] = max(bands) - min(bands)        # spread of the band means: curvature plus any tilt
    s["male_corrected"] = wmean(corr, w * male); s["female_corrected"] = wmean(corr, w * ~male)
    s["male_minus_female"] = s["male_corrected"] - s["female_corrected"]
    raw_bands = [s[f"band_{k}_raw"] for k, _, _ in BANDS]      # the same rows for the uncorrected gap, so no table cell is blank
    s["band_range_raw"] = max(raw_bands) - min(raw_bands)
    s["male_minus_female_raw"] = wmean(gap, w * male) - wmean(gap, w * ~male)
    s["pct_over_plus8_raw"] = 100 * wmean((gap > 8).astype(float), w); s["pct_under_minus8_raw"] = 100 * wmean((gap < -8).astype(float), w)
    X = np.column_stack([male.astype(float)] + dummies)        # sex difference within age band
    XtW = X.T * w
    s["male_minus_female_band_adjusted"] = float(np.linalg.solve(XtW @ X, XtW @ corr)[0])
    return s

def boot_weights(cluster, rng):
    """Resample patients with replacement; every ECG of a drawn patient comes along."""
    n = cluster.max() + 1
    return np.bincount(rng.integers(0, n, n), minlength=n)[cluster].astype(float)

def ci(reps):
    return {k: [float(np.percentile([r[k] for r in reps], 2.5)), float(np.percentile([r[k] for r in reps], 97.5))] for k in reps[0]}

def describe(sub):
    age = sub.age.values
    return dict(n_ecgs=int(len(sub)), n_patients=int(sub.patient_id.nunique()), mean_age=float(age.mean()),
                sd_age=float(age.std(ddof=1)), min_age=float(age.min()), max_age=float(age.max()),
                pct_male=float(100 * sub.male.mean()),
                band_n={k: int(((age >= lo) & (age <= hi)).sum()) for k, lo, hi in BANDS})

def analyse(sub, offset_fn, seed):
    """Point estimates and patient-resampling bootstrap CIs for one cohort under one fixed correction."""
    return analyse_offset(sub, offset_fn(sub.age.values.astype(float)), seed)

def analyse_offset(sub, off, seed):
    """As analyse, with the correction given as one offset per row (for corrections that also depend on sex)."""
    age = sub.age.values.astype(float); args = (sub.pred.values, age, sub.male.values)
    cl = pd.factorize(sub.patient_id)[0]; rng = np.random.default_rng(seed)
    reps = [summarise(*args, off, boot_weights(cl, rng)) for _ in range(NBOOT)]
    return dict(cohort=describe(sub), point=summarise(*args, off), ci=ci(reps))

def german_line(sub, seed):
    age = sub.age.values.astype(float); gap = sub.pred.values - age
    a, b = wline(age, gap, np.ones_like(age))
    cl = pd.factorize(sub.patient_id)[0]; rng = np.random.default_rng(seed)
    reps = [dict(zip(("intercept", "slope"), wline(age, gap, boot_weights(cl, rng)))) for _ in range(NBOOT)]
    return dict(intercept=a, slope=b, ci=ci(reps))

def local_fit(ageA, gapA, w):
    """German line and German age-level means fitted on the fit half with weights w."""
    ln = wline(ageA, gapA, w)
    return ln, age_level_means(ageA, gapA - line_offset(ln, ageA), w)

def half_split(prim, brazil, brazil_mp, seed):
    """Refit on a random half of German patients, test on the other half: Brazil and German line, and both two-step versions.
    Each bootstrap replicate resamples both halves and refits on the resampled fit half, so the CIs carry the refit's own noise."""
    pats = np.sort(prim.patient_id.unique()); perm = np.random.default_rng(seed).permutation(pats)
    inA = prim.patient_id.isin(set(perm[: len(perm) // 2])).values
    A, B = prim[inA], prim[~inA]
    ageA = A.age.values.astype(float); gapA = A.pred.values - ageA
    ageB = B.age.values.astype(float); argsB = (B.pred.values, ageB, B.male.values); iB = age_index(ageB)
    de, de_mp = local_fit(ageA, gapA, np.ones_like(ageA))
    assert not np.isnan(de_mp[np.unique(iB)]).any(), "an integer age in the test half has no patient in the fit half"
    off = lambda ln, mp: line_offset(ln, ageB) + (0 if mp is None else mp[iB])
    clA, clB = pd.factorize(A.patient_id)[0], pd.factorize(B.patient_id)[0]
    rng = np.random.default_rng(seed + 1); reps = {k: [] for k in ("brazil_line", "german_line", "brazil_two_step", "german_two_step")}
    for _ in range(NBOOT):
        ln, mp = local_fit(ageA, gapA, boot_weights(clA, rng)); wB = boot_weights(clB, rng)
        mp = np.where(np.isnan(mp), de_mp, mp)                # an age absent from this resample keeps the point fit's mean
        reps["brazil_line"].append(summarise(*argsB, off(brazil, None), wB))
        reps["german_line"].append(summarise(*argsB, off(ln, None), wB))
        reps["brazil_two_step"].append(summarise(*argsB, off(brazil, brazil_mp), wB))
        reps["german_two_step"].append(summarise(*argsB, off(ln, mp), wB))
    pts = dict(brazil_line=off(brazil, None), german_line=off(de, None), brazil_two_step=off(brazil, brazil_mp), german_two_step=off(de, de_mp))
    out = dict(n_patients_fit=int(len(A)), n_patients_test=int(len(B)), german_line_on_fit_half=dict(intercept=de[0], slope=de[1]),
               test_cohort=describe(B))
    for k in reps:
        out[k] = dict(point=summarise(*argsB, pts[k]), ci=ci(reps[k]))
    return out

# ---------------------------------------------------------------- checks added after an outside audit (30 Oct 2026)
WINDOW = (18, 85)                   # ages both cohorts cover: PTB-XL is analysed from 18, CODE-15% has 13 patients above 85
Z95 = 1.959963984540054             # as in the replication

def wilson(k, n, z=Z95):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]

def log_ci(est, se):
    return [float(est * np.exp(-Z95 * se)), float(est * np.exp(Z95 * se))]

def split_at_8(gap, death, years):
    """The replication's crude survival split at plus or minus 8 years: deaths with Wilson CIs, risk ratio, person-year rate ratio."""
    out = {}
    for k, m in (("older", gap > 8), ("younger", gap < -8)):
        n, d = int(m.sum()), int(death[m].sum())
        out[k] = dict(n=n, deaths=d, died_pct=100 * d / n, died_ci=[100 * x for x in wilson(d, n)], person_years=float(years[m].sum()))
    o, y = out["older"], out["younger"]
    rr = (o["deaths"] / o["n"]) / (y["deaths"] / y["n"])
    pr = (o["deaths"] / o["person_years"]) / (y["deaths"] / y["person_years"])
    out.update(risk_ratio=rr, risk_ratio_ci=log_ci(rr, np.sqrt(1 / o["deaths"] - 1 / o["n"] + 1 / y["deaths"] - 1 / y["n"])),
               rate_ratio=pr, rate_ratio_ci=log_ci(pr, np.sqrt(1 / o["deaths"] + 1 / y["deaths"])))
    return out

def brazil_arrays(co):
    age = co.age.values.astype(float)
    return age, co.nn_predicted_age.values - age, co.death.values.astype(bool), co.timey.values.astype(float)

def brazil_split_half(co, line, seed):
    """The published Brazil line was fitted in-sample. Refit it on a random half of the cohort and test on the other half,
    next to the in-sample line on the same test patients."""
    age, gap, death, years = brazil_arrays(co)
    perm = np.random.default_rng(seed).permutation(len(age)); A, B = perm[: len(age) // 2], perm[len(age) // 2:]
    b, a = np.polyfit(age[A], gap[A], 1)
    out = dict(n_fit=int(len(A)), n_test=int(len(B)), intercept=float(a), slope=float(b),
               r_raw_test=float(np.corrcoef(gap[B], age[B])[0, 1]), raw=split_at_8(gap[B], death[B], years[B]))
    for k, ln in (("half_line", (a, b)), ("full_line", line)):
        corr = gap[B] - line_offset(ln, age[B])
        out[k] = dict(r_corrected=float(np.corrcoef(corr, age[B])[0, 1]), mean_corrected=float(corr.mean()),
                      split=split_at_8(corr, death[B], years[B]))
    return out

def age_window(co, prim, seed):
    """Both cohorts cut to the ages both cover, the line refitted in Brazil on that window and applied to Germany on it."""
    lo, hi = WINDOW
    c = co[(co.age >= lo) & (co.age <= hi)]
    age, gap, death, years = brazil_arrays(c)
    b, a = np.polyfit(age, gap, 1); corr = gap - (a + b * age)
    br = dict(n=int(len(c)), n_dropped=int(len(co) - len(c)), intercept=float(a), slope=float(b),
              r_raw=float(np.corrcoef(gap, age)[0, 1]), raw=split_at_8(gap, death, years), corrected=split_at_8(corr, death, years))
    p = prim[(prim.age >= lo) & (prim.age <= hi)].reset_index(drop=True)
    return dict(window=list(WINDOW), brazil=br, germany_n_dropped=int(len(prim) - len(p)),
                germany=analyse(p, lambda x: line_offset((a, b), x), seed))

def brazil_sex_lines(co):
    """The Brazil line fitted within each sex, checked against the replication's own within-sex fit."""
    rep = json.load(open(BRAZIL_RESULTS))["within_sex_correction"]
    age, gap, _, _ = brazil_arrays(co); male = co.is_male.values.astype(bool)
    out = {}
    for k, m in (("male", male), ("female", ~male)):
        b, a = np.polyfit(age[m], gap[m], 1)
        assert abs(a - rep[k]["intercept"]) < 1e-8 and abs(b - rep[k]["slope"]) < 1e-10, (k, a, b)
        out[k] = (float(a), float(b))
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--use-cache", action="store_true"); args = ap.parse_args()
    full = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    d = full[(full.age >= PTB_AGE_RANGE[0]) & (full.age <= PTB_AGE_RANGE[1])].reset_index(drop=True)
    from importlib.metadata import version
    res = dict(provenance=dict(run_date=time.strftime("%Y-%m-%d %H:%M"), python=platform.python_version(),
                               numpy=np.__version__, pandas=pd.__version__, torch=version("torch"), machine=platform.machine(),
                               seed=SEED, n_bootstrap=NBOOT, used_cached_predictions=args.use_cache,
                               brazil_results_file=os.path.relpath(BRAZIL_RESULTS, HERE)))
    res["inputs"] = check_inputs(d); log(f"inputs verified: {res['inputs']['ptbxl_files_sha256_checked']} PTB-XL files, md5 of weights and exams.csv")
    res["ptbxl_records"] = dict(in_database=int(len(full)), patients_in_database=int(full.patient_id.nunique()),
                                age_coded_300=int((full.age == 300).sum()), age_under_18=int((full.age < 18).sum()))

    # PTB-XL codes sex as 0/1. Its page says 52% of patients are male; check which code that is, by count and by height.
    s0 = float(100 * (full.sex == 0).mean()); h = full.groupby("sex").height.mean()
    assert 51 < s0 < 53 and h[0] > h[1] + 5, (s0, h.to_dict())
    res["sex_coding_check"] = dict(pct_records_sex0=s0, mean_height_cm_sex0=float(h[0]), mean_height_cm_sex1=float(h[1]), conclusion="0 = male")
    d["male"] = d.sex.values == 0

    line, co, ex = brazil_line()
    res["brazil_line"] = dict(intercept=line[0], slope=line[1], n_patients=BRAZIL_N, refit_check="exact to 1e-8")
    res["brazil_reference"], brazil_mp = brazil_reference(co, line)
    log(f"Brazil line gap = {line[0]:.4f} + ({line[1]:.4f}) x age, re-derived from exams.csv")

    res["gate_a_cached"] = gate_a_cached(ex); log(f"gate A, cached: {res['gate_a_cached']}")
    P, vs_cache, amp_ptb, model, dev = ptb_predictions(d, args.use_cache)
    res["ptb_rerun_vs_cache"] = vs_cache
    if vs_cache:
        log(f"PTB-XL re-run against 25 Oct cache: {vs_cache}")
    d["pred"] = P
    res["gate_b"] = dict(n_ecgs=int(len(d)), n_patients=int(d.patient_id.nunique()), mean_gap=float((P - d.age).mean()),
                         mae=float(np.abs(P - d.age).mean()), pearson_r_pred_age=float(np.corrcoef(P, d.age)[0, 1]))
    log(f"gate B: {res['gate_b']}")

    # Primary: first ECG per patient (earliest recording date; PTB-XL shifts dates by one offset per patient, so order holds)
    prim = d.sort_values(["patient_id", "recording_date", "ecg_id"]).drop_duplicates("patient_id").reset_index(drop=True)
    brazil = lambda age: line_offset(line, age)
    brazil2 = lambda age: line_offset(line, age) + brazil_mp[age_index(age)]
    res["primary_first_ecg"] = analyse(prim, brazil, SEED); log("primary analysis done")
    res["german_line_full"] = german_line(prim, SEED + 2)
    res["two_step_brazil_first_ecg"] = analyse(prim, brazil2, SEED + 7)
    res["half_split"] = half_split(prim, line, brazil_mp, SEED + 3); log("half split done")
    res["sensitivity_all_ecgs"] = analyse(d, brazil, SEED + 4); log("sensitivity (all ECGs) done")
    has_norm = prim.scp_codes.map(lambda s: "NORM" in ast.literal_eval(s)).values   # NORM statement at any likelihood
    res["exploratory_norm_first_ecg"] = analyse(prim[has_norm], brazil, SEED + 5)
    res["exploratory_not_norm_first_ecg"] = analyse(prim[~has_norm], brazil, SEED + 6); log("exploratory NORM subgroups done")

    # Added after an outside audit (30 Oct): the Brazil line fitted out of sample, a common age window, sex-specific lines.
    sh = res["brazil_split_half"] = brazil_split_half(co, line, SEED + 10)
    sh["germany"] = analyse(prim, lambda age: line_offset((sh["intercept"], sh["slope"]), age), SEED + 11)
    res["age_window"] = age_window(co, prim, SEED + 12)
    sl = brazil_sex_lines(co); age_p = prim.age.values.astype(float)
    off = np.where(prim.male.values, line_offset(sl["male"], age_p), line_offset(sl["female"], age_p))
    res["brazil_sex_lines"] = dict(lines={k: dict(intercept=a, slope=b) for k, (a, b) in sl.items()},
                                   germany=analyse_offset(prim, off, SEED + 13))
    log("audit checks done (split half, age window, sex-specific lines)")

    if model is not None:
        res["gate_a_rerun"], amp_code = gate_a_rerun(co, model, dev); log(f"gate A, re-run from raw HDF5: {res['gate_a_rerun']}")
        res["amplitude_check"] = amplitude_check(amp_code, amp_ptb)
        s = res["amplitude_check"]["matched_scale"]; log(f"amplitude ratio CODE-15% / PTB-XL: median over leads x{s:.3f}")
        prim_s = prim.copy(); prim_s["pred"] = run_ptb(model, dev, prim, s)
        res["scale_matched_first_ecg"] = dict(scale=s, brazil_line=analyse(prim_s, brazil, SEED + 8), german_line=german_line(prim_s, SEED + 9))
        log("scale-matched sensitivity done")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    report(res)
    log(f"wrote {os.path.relpath(OUT, HERE)}")

# ---------------------------------------------------------------- console report
def report(res):
    f = lambda v, c: f"{v:+.3f} ({c[0]:+.3f} to {c[1]:+.3f})"
    g = lambda v, c: f"{v:+.2f} ({c[0]:+.2f} to {c[1]:+.2f})"
    def block(title, blk):
        p, c, h = blk["point"], blk["ci"], blk.get("cohort")
        print(f"\n== {title}" + (f": {h['n_ecgs']} ECGs, {h['n_patients']} patients, mean age {h['mean_age']:.1f}, {h['pct_male']:.1f}% male" if h else ""))
        print(f"  MAE raw {p['mae_raw']:.2f}   mean gap raw {g(p['mean_gap_raw'], c['mean_gap_raw'])}   corrected {g(p['mean_gap_corrected'], c['mean_gap_corrected'])}")
        print(f"  r(gap, age) raw {f(p['r_raw'], c['r_raw'])}   corrected {f(p['r_corrected'], c['r_corrected'])}")
        print(f"  slope raw {f(p['slope_raw'], c['slope_raw'])}   corrected {f(p['slope_corrected'], c['slope_corrected'])}")
        for k, _, _ in BANDS:
            n = f"  n={h['band_n'][k]}" if h else ""
            print(f"  band {k:>5}: corrected {g(p[f'band_{k}_corrected'], c[f'band_{k}_corrected'])}   raw {p[f'band_{k}_raw']:+.2f}{n}")
        print(f"  band range (max - min) {g(p['band_range_corrected'], c['band_range_corrected'])}")
        print(f"  male {g(p['male_corrected'], c['male_corrected'])}   female {g(p['female_corrected'], c['female_corrected'])}"
              f"   male-female {g(p['male_minus_female'], c['male_minus_female'])}   band-adjusted {g(p['male_minus_female_band_adjusted'], c['male_minus_female_band_adjusted'])}")
        print(f"  corrected gap > +8: {p['pct_over_plus8']:.2f}%   < -8: {p['pct_under_minus8']:.2f}%")
        print(f"  raw gap: band range {g(p['band_range_raw'], c['band_range_raw'])}   male-female {g(p['male_minus_female_raw'], c['male_minus_female_raw'])}"
              f"   > +8 {p['pct_over_plus8_raw']:.2f}%   < -8 {p['pct_under_minus8_raw']:.2f}%")
    def gline(title, gl):
        print(f"\n== {title}: gap = {gl['intercept']:.3f} ({gl['ci']['intercept'][0]:.3f} to {gl['ci']['intercept'][1]:.3f})"
              f" + {gl['slope']:.4f} ({gl['ci']['slope'][0]:.4f} to {gl['ci']['slope'][1]:.4f}) x age")
    br = res["brazil_reference"]
    print(f"\n== Brazil line: gap = {res['brazil_line']['intercept']:.4f} + {res['brazil_line']['slope']:.4f} x age (n={res['brazil_line']['n_patients']})")
    print("  in its own cohort, mean corrected gap by band: " + "   ".join(f"{k} {br[k]['mean']:+.2f}" for k, _, _ in BANDS))
    print(f"  male {br['male']['mean']:+.3f}  female {br['female']['mean']:+.3f}   >+8 {br['pct_over_plus8']:.2f}%  <-8 {br['pct_under_minus8']:.2f}%"
          f"   normal_ecg flag {br['normal_ecg']['mean']:+.2f} (n={br['normal_ecg']['n']}), not flagged {br['not_normal_ecg']['mean']:+.2f}")
    block("PRIMARY, first ECG per patient, Brazil line", res["primary_first_ecg"])
    gline("German line, all primary patients", res["german_line_full"])
    block("SECONDARY, two-step (Brazil line + Brazil age-level means), first ECG per patient", res["two_step_brazil_first_ecg"])
    hs = res["half_split"]
    print(f"\n== HALF SPLIT: refitted on {hs['n_patients_fit']} patients: gap = {hs['german_line_on_fit_half']['intercept']:.3f}"
          f" + {hs['german_line_on_fit_half']['slope']:.4f} x age; tested on {hs['n_patients_test']}")
    for k in ("brazil_line", "german_line", "brazil_two_step", "german_two_step"):
        block(f"  test half, {k.replace('_', ' ')}", hs[k])
    block("SENSITIVITY, all ECGs (cluster bootstrap by patient), Brazil line", res["sensitivity_all_ecgs"])
    block("EXPLORATORY, NORM statement present, first ECG per patient, Brazil line", res["exploratory_norm_first_ecg"])
    block("EXPLORATORY, NORM statement absent, first ECG per patient, Brazil line", res["exploratory_not_norm_first_ecg"])
    def split(title, s):
        o, y = s["older"], s["younger"]
        print(f"  {title}: older-looking {o['deaths']}/{o['n']} died {o['died_pct']:.2f}%, younger-looking {y['deaths']}/{y['n']} {y['died_pct']:.2f}%,"
              f" risk ratio {s['risk_ratio']:.2f} ({s['risk_ratio_ci'][0]:.2f} to {s['risk_ratio_ci'][1]:.2f}),"
              f" rate ratio {s['rate_ratio']:.2f} ({s['rate_ratio_ci'][0]:.2f} to {s['rate_ratio_ci'][1]:.2f})")
    sh = res["brazil_split_half"]
    print(f"\n== BRAZIL SPLIT HALF: line fitted on {sh['n_fit']} patients: gap = {sh['intercept']:.4f} + {sh['slope']:.4f} x age;"
          f" tested on {sh['n_test']}, raw r {sh['r_raw_test']:+.4f}")
    split("raw gap", sh["raw"])
    for k in ("half_line", "full_line"):
        print(f"  {k.replace('_', ' ')}: r {sh[k]['r_corrected']:+.4f}, mean {sh[k]['mean_corrected']:+.4f}")
        split(k.replace("_", " "), sh[k]["split"])
    block("  Germany, first ECG per patient, Brazil half-sample line", sh["germany"])
    aw = res["age_window"]; bw = aw["brazil"]
    print(f"\n== AGE WINDOW {aw['window'][0]} to {aw['window'][1]}: Brazil n={bw['n']} ({bw['n_dropped']} dropped), gap = {bw['intercept']:.4f}"
          f" + {bw['slope']:.4f} x age, raw r {bw['r_raw']:+.4f}; Germany {aw['germany_n_dropped']} dropped")
    split("Brazil raw gap", bw["raw"]); split("Brazil corrected gap", bw["corrected"])
    block("  Germany, first ECG per patient, Brazil window line", aw["germany"])
    sx = res["brazil_sex_lines"]
    print("\n== SEX-SPECIFIC BRAZIL LINES: " + "   ".join(f"{k} gap = {v['intercept']:.4f} + {v['slope']:.4f} x age" for k, v in sx["lines"].items()))
    block("  Germany, first ECG per patient, each sex corrected by its own Brazil line", sx["germany"])
    if "amplitude_check" in res:
        a = res["amplitude_check"]
        print(f"\n== Amplitude, CODE-15% ({a['n_code15']} ECGs) against PTB-XL ({a['n_ptbxl']} ECGs), input units:")
        print("  " + "  ".join(f"{l} {v['code15']:.2f}/{v['ptbxl']:.2f}" for l, v in a["per_lead"].items()))
        print(f"  ratio {a['ratio_min']:.2f} to {a['ratio_max']:.2f}, median over leads {a['matched_scale']:.3f}")
        sm = res["scale_matched_first_ecg"]
        block(f"SCALE SENSITIVITY, PTB-XL fed at x{sm['scale']:.3f}, first ECG per patient, Brazil line", sm["brazil_line"])
        gline("  German line at the matched scale", sm["german_line"])

if __name__ == "__main__":
    main()
