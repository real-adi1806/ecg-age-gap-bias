"""
Writes results/exclusions.json and DATA_EXCLUSIONS.md: the cohort flow, the 15,715 patients the first version left
out because their saved network features were not from the outcome exam, why that was leak prevention, what changed
when their outcome exams were extracted and they came back, and a baseline table. Every number in DATA_EXCLUSIONS.md
comes from exclusions.json, which takes the AUCs from results/results.json (this version) and results/results-v1.json
(the first version); none is typed by hand. The cohort rules are ladder.build_cohort and the split is
ladder.split_patients with the seed analysis.py uses for split 0.

    python exclusions.py [--data ../data/exams.csv] [--features ../../ecg-age-model/features]

Stops before writing anything if a count differs from results.json, results-v1.json, outcome_extraction.json or the
figures below, or if rows 1 to 4 on the whole test split differ between the two versions (they use no network
features, so they must not).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decision-curves"))
import ladder as L  # noqa: E402
import analysis as A  # noqa: E402

EXPECTED = {"patients_with_outcome": 233647, "short_follow_up": 8778, "main": 224869, "deaths_main": 2812,
            "left_out_v1": 15715, "lowest_id_features": 217932, "kept_v1": 209154, "deaths_kept_v1": 2780,
            "test_patients_v1": 20892, "test_deaths_v1": 278, "test_patients": 22480, "test_deaths": 281,
            "extracted": 15715}
ROWS14 = ["row1_age", "row2_age_sex", "row3_age_sex_rhythm", "row4_age_sex_gap"]
ROWS = ROWS14 + ["row5_age_sex_lima", "row5b_lima_only"]
PARTS = range(18)
LABELS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg"]
SAME_TOL = 1e-9
ZENODO_QUOTE = ('"death": true if the patient dies in the follow-up time. This data is available only in the first '
                'exam of the patient. Other exams will have this as an empty field')
LIMA_QUOTE = ("Only the first patient exam is considered in all the analysis with this cohort and the remaining "
              "exams from the patients are removed from the remaining data and not used in the analysis.")


# ---------------------------------------------------------------------------
# pure helpers (tested in tests/test_exclusions.py)
# ---------------------------------------------------------------------------
def lowest_id_exam(df):
    """The exam extract_code.py saved features for: the lowest exam_id of each patient. Indexed by patient_id."""
    first = df.sort_values("exam_id").drop_duplicates("patient_id")
    return first.set_index("patient_id")[["exam_id", "age"]].rename(columns={"exam_id": "feat_exam_id",
                                                                             "age": "feat_age"})


def age_order(feat_age, outcome_age):
    """Age at the saved-feature exam against age at the outcome exam, whole years: older, same or younger."""
    d = np.asarray(feat_age, dtype=float) - np.asarray(outcome_age, dtype=float)
    return np.where(d > 0, "older", np.where(d < 0, "younger", "same"))


def classify_feature_exams(df, cohort):
    """
    One row per cohort patient: the saved-feature exam (lowest exam_id), whether it is the outcome exam, how its age
    and its id compare with the outcome exam, and whether the outcome exam has the lowest age among the patient's
    exams (ties count). `df` is every exam, `cohort` the outcome rows (ladder.build_cohort).
    """
    first = lowest_id_exam(df)
    min_age = df.groupby("patient_id")["age"].min().rename("min_age")
    out = cohort[["patient_id", "exam_id", "age"]].rename(columns={"exam_id": "outcome_exam_id",
                                                                    "age": "outcome_age"})
    out = out.join(first, on="patient_id").join(min_age, on="patient_id")
    out["same_exam"] = out["feat_exam_id"] == out["outcome_exam_id"]
    out["age_order"] = age_order(out["feat_age"], out["outcome_age"])
    out["id_order"] = np.where(out["feat_exam_id"] > out["outcome_exam_id"], "higher",
                               np.where(out["feat_exam_id"] < out["outcome_exam_id"], "lower", "same exam"))
    out["outcome_is_min_age"] = out["outcome_age"] <= out["min_age"]
    return out.reset_index(drop=True)


def neither_share(frame):
    """Share of rows with normal_ecg and all six rhythm labels equal to 0 (no label at all)."""
    return float((frame[LABELS].astype(float).sum(axis=1) == 0).mean())


def smd(a, b):
    """Standardised mean difference of a minus b: difference in means over sqrt of the mean of the two variances."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def count_mismatches(got, expected):
    """Keys where got differs from expected, as {key: (got, expected)}."""
    return {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}


def quartiles(v):
    return [float(np.percentile(v, p)) for p in (25, 50, 75)]


def column_stats(c, m):
    """Baseline characteristics of the rows of cohort frame `c` selected by boolean mask `m`."""
    s = c[m]
    out = {"n": int(m.sum()), "deaths": int(s["died1"].sum()), "deaths_pct": float(100 * s["died1"].mean()),
           "age_mean": float(s["age"].mean()), "age_sd": float(s["age"].std()), "age_q": quartiles(s["age"]),
           "male_pct": float(100 * s["male"].mean()), "timey_q": quartiles(s["timey"]),
           "timey_mean": float(s["timey"].mean()), "exams_q": quartiles(s["n_exams"]),
           "multi_pct": float(100 * (s["n_exams"] >= 2).mean()),
           "gap_mean": float(s["gap"].mean()), "gap_sd": float(s["gap"].std())}
    out.update({k: float(100 * s[k].mean()) for k in LABELS})
    return out


# ---------------------------------------------------------------------------
# the computation
# ---------------------------------------------------------------------------
def feature_ids(feat_dir, prefix="code_part"):
    return np.concatenate([np.load(f"{feat_dir}/{prefix}{k}_ids.npy") for k in PARTS])


def extract_line():
    path = os.path.join(HERE, "..", "germany", "extract_code.py")
    with open(path) as f:
        for line in f:
            if 'drop_duplicates("patient_id")' in line:
                return line.strip()
    sys.exit("germany/extract_code.py no longer contains the drop_duplicates line; update exclusions.py")


def compare_versions(r1, r2):
    """AUCs of every row in the first version (feature subset) and this one, by split, and the rows 1 to 4 check."""
    out = {"by_split": {}}
    for s in r2["seeds"]:
        a, b = r1["seeds"][s], r2["seeds"][s]
        out["by_split"][s] = {"v1": {k: a["test_with_features"]["auc"][k] for k in ROWS},
                              "v2": {k: b["test_with_features"]["auc"][k] for k in ROWS},
                              "v1_all_rows14": {k: a["test_all"]["auc"][k] for k in ROWS14}}
    p1, p2 = r1["seeds"]["0"]["test_with_features"], r2["seeds"]["0"]["test_with_features"]
    out["split0_ci"] = {"v1": {k: p1["auc_ci95"][k] for k in ROWS}, "v2": {k: p2["auc_ci95"][k] for k in ROWS}}
    out["split0_row5_minus_row2"] = {
        "v1": [p1["auc_diff_vs_row2"]["row5_age_sex_lima"], *p1["auc_diff_vs_row2_age_sex_ci95"]["row5_age_sex_lima"]],
        "v2": [p2["auc_diff_vs_row2"]["row5_age_sex_lima"], *p2["auc_diff_vs_row2_age_sex_ci95"]["row5_age_sex_lima"]]}
    out["mean"] = {v: {k: float(np.mean([d[v][k] for d in out["by_split"].values()])) for k in ROWS} for v in ("v1", "v2")}
    out["rows14_all_test_v1_vs_v2_max_abs_diff"] = max(
        abs(r1["seeds"][s]["test_all"]["auc"][k] - r2["seeds"][s]["test_all"]["auc"][k]) for s in r2["seeds"] for k in ROWS14)
    out["v1_rows14_all_minus_kept_max_abs_diff"] = {
        k: max(abs(d["v1_all_rows14"][k] - d["v1"][k]) for d in out["by_split"].values()) for k in ROWS14}
    return out


def compute(data_path, feat_dir, results_path, v1_path, extraction_path):
    df = pd.read_csv(data_path)
    c = L.build_cohort(data_path)
    y = c["died1"].values
    short = c["short"].values
    main = ~short
    cls = classify_feature_exams(df, c)
    low = np.isin(c["exam_id"].values, feature_ids(feat_dir))                      # saved: lowest exam id
    ext_ids = feature_ids(feat_dir, "code_outcome_part")                           # extracted: outcome exams
    if not (cls["same_exam"].values == low).all():
        sys.exit("STOP: 'saved features exist for the outcome exam' differs from 'outcome exam is the lowest exam_id'")
    if set(ext_ids.tolist()) != set(c.loc[~low, "exam_id"].tolist()):
        sys.exit("STOP: the extracted exams are not exactly the outcome exams without saved features")
    excl = main & ~low                     # left out of the first version
    kept = main & low                      # kept in the first version
    split = L.split_patients(y, A.SEED_BASE + 0)
    test = main & (split == "test")
    c = c.assign(n_exams=c["patient_id"].map(df.groupby("patient_id").size()).values)

    per_patient = df[df["death"].notna() & df["timey"].notna()].groupby("patient_id").size()
    flow = {"exams": int(len(df)), "patients": int(df["patient_id"].nunique()),
            "patients_with_outcome": int(len(c)), "patients_without_outcome": int(df["patient_id"].nunique() - len(c)),
            "exams_of_patients_without_outcome": int((~df["patient_id"].isin(c["patient_id"])).sum()),
            "outcome_rows_per_patient_max": int(per_patient.max()),
            "deaths_all": int(y.sum()), "deaths_all_pct": float(100 * y.mean()),
            "short_follow_up": int(short.sum()), "deaths_in_short": int(y[short].sum()),
            "main": int(main.sum()), "deaths_main": int(y[main].sum()), "deaths_main_pct": float(100 * y[main].mean()),
            "lowest_id_features": int(low.sum()), "extracted": int(len(ext_ids)),
            "left_out_v1": int(excl.sum()), "deaths_left_out_v1": int(y[excl].sum()),
            "deaths_left_out_v1_pct": float(100 * y[excl].mean()),
            "kept_v1": int(kept.sum()), "deaths_kept_v1": int(y[kept].sum()), "deaths_kept_v1_pct": float(100 * y[kept].mean()),
            "overlap_short_and_left_out": int((short & ~low).sum())}
    flow["splits"] = {s: {"n": int((main & (split == s)).sum()), "deaths": int(y[main & (split == s)].sum())}
                      for s in ("train", "val", "test")}
    flow["test_patients"], flow["test_deaths"] = flow["splits"]["test"]["n"], flow["splits"]["test"]["deaths"]
    flow["test_patients_v1"] = int((kept & (split == "test")).sum())
    flow["test_deaths_v1"] = int(y[kept & (split == "test")].sum())

    e = cls[excl]
    order = {"n": int(len(e)),
             "age": {k: int((e["age_order"] == k).sum()) for k in ("older", "same", "younger")},
             "id": {k: int((e["id_order"] == k).sum()) for k in ("higher", "lower", "same exam")},
             "outcome_is_min_age_excluded": int(e["outcome_is_min_age"].sum()),
             "kept_multi_n": int((kept & (c["n_exams"].values >= 2)).sum()),
             "outcome_is_min_age_kept_multi": int(cls[kept & (c["n_exams"].values >= 2)]["outcome_is_min_age"].sum()),
             "n_all": int(len(cls)),
             "age_gap_years_median_excluded": float(np.median(e["feat_age"] - e["outcome_age"])),
             "min_exams_excluded": int(c.loc[excl, "n_exams"].min()),
             "outcome_exam_is_lowest_id": int(cls["same_exam"].sum())}

    cols = {"all": np.ones(len(c), dtype=bool), "main": main, "excluded": excl, "kept": kept, "test": test}
    chars = {k: column_stats(c, m) for k, m in cols.items()}
    sm = {"deaths": smd(y[excl], y[kept]), "age": smd(c.loc[excl, "age"], c.loc[kept, "age"]),
          "male": smd(c.loc[excl, "male"], c.loc[kept, "male"]),
          "timey": smd(c.loc[excl, "timey"], c.loc[kept, "timey"]),
          "multi": smd(c.loc[excl, "n_exams"] >= 2, c.loc[kept, "n_exams"] >= 2),
          "gap": smd(c.loc[excl, "gap"], c.loc[kept, "gap"])}
    sm.update({k: smd(c.loc[excl, k], c.loc[kept, k]) for k in LABELS})
    used = ["age", "is_male", "nn_predicted_age", "timey", "death"] + LABELS
    missing = {k: int(df.loc[df["patient_id"].isin(c["patient_id"]) & df["exam_id"].isin(c["exam_id"]), k].isna().sum())
               for k in used}

    r2, r1, ex = (json.load(open(p)) for p in (results_path, v1_path, extraction_path))
    cmp_ = compare_versions(r1, r2)

    got = {k: flow[k] for k in EXPECTED}
    bad = count_mismatches(got, EXPECTED)
    rc, f1, f2 = r2["cohort"], r1["features"], r2["features"]
    t1, t2 = r1["seeds"]["0"]["test_with_features"], r2["seeds"]["0"]["test_with_features"]
    pairs = {"patients": (flow["patients_with_outcome"], rc["patients"]), "deaths_1y": (flow["deaths_all"], rc["deaths_1y"]),
             "short": (flow["short_follow_up"], rc["short_follow_up"]),
             "v2_with_features": (flow["patients_with_outcome"], f2["patients_with_outcome_exam_features"]),
             "v2_without": (0, f2["patients_without"]),
             "v1_with_features": (flow["lowest_id_features"], f1["patients_with_outcome_exam_features"]),
             "v1_without": (int((~low).sum()), f1["patients_without"]),
             "v1_main_with": (flow["kept_v1"], f1["main_cohort_with"]),
             "v1_deaths_without": (int(y[~low].sum()), f1["deaths_1y_without"]),
             "v1_split0_test": ((flow["test_patients_v1"], flow["test_deaths_v1"]), (t1["patients"], t1["deaths"])),
             "v2_split0_test": ((flow["test_patients"], flow["test_deaths"]), (t2["patients"], t2["deaths"])),
             "extracted": (flow["extracted"], ex["extracted"])}
    bad.update({k: v for k, v in pairs.items() if v[0] != v[1]})
    if cmp_["rows14_all_test_v1_vs_v2_max_abs_diff"] > SAME_TOL:
        bad["rows14_all_test_v1_vs_v2"] = (cmp_["rows14_all_test_v1_vs_v2_max_abs_diff"], 0)
    if not r2.get("checks", {}).get("all_passed"):
        bad["results_checks"] = ("not passed", "passed")
    if bad:
        sys.exit(f"STOP: counts differ, as (got, expected): {bad}. Nothing written.")

    multi = c["n_exams"].values >= 2
    nonout = df[df["death"].isna()]
    rate = lambda m: {"n": int(m.sum()), "normal_pct": float(100 * c.loc[m, "normal_ecg"].mean())}
    normal = {"multi_outcome_not_lowest": rate(multi & ~low), "multi_outcome_is_lowest": rate(multi & low),
              "non_outcome_exams": {"n": int(len(nonout)), "normal_pct": float(100 * nonout["normal_ecg"].astype(float).mean())},
              "neither_pct": {"dropped": 100 * neither_share(c[excl]), "kept_multi": 100 * neither_share(c[kept & multi]),
                              "kept_single": 100 * neither_share(c[kept & ~multi])}}

    filters = {"all exams": len(df), "exams of patients with an outcome": int(df["patient_id"].isin(c["patient_id"]).sum()),
               "exams with death and timey": int(df["death"].notna().sum()),
               "exams of patients with two or more exams": int(df["patient_id"].map(df.groupby("patient_id").size()).ge(2).sum()),
               "exams with age between 16 and 85": int(df["age"].between(16, 85).sum()),
               "exams with timey above 0 or missing": int((~(df["timey"] <= 0)).sum())}
    return {"flow": flow, "order": order, "chars": chars, "smd": sm, "missing": missing, "versions": cmp_,
            "extraction": ex, "filters_tried_for_344723": filters, "normal": normal,
            "seed": A.SEED_BASE, "extract_line": extract_line(), "zenodo_quote": ZENODO_QUOTE,
            "lima_quote": LIMA_QUOTE,
            "checks": "all counts match results.json, results-v1.json and outcome_extraction.json; rows 1 to 4 identical"}


# ---------------------------------------------------------------------------
# the document: a template, every number comes from R
# ---------------------------------------------------------------------------
NAMES = {"row1_age": "1. Age alone", "row2_age_sex": "2. Age + sex", "row3_age_sex_rhythm": "3. Age + sex + rhythm labels",
         "row4_age_sex_gap": "4. Age + sex + ECG-age gap", "row5_age_sex_lima": "5. Age + sex + Lima features",
         "row5b_lima_only": "5b. Lima features alone"}


def render(R):
    f, o, ch, sm, v, ex = R["flow"], R["order"], R["chars"], R["smd"], R["versions"], R["extraction"]
    n = lambda x: f"{int(x):,}"
    pc = lambda a, b, d=1: f"{100 * a / b:.{d}f}%"
    ci = lambda c: f"{c[0]:.3f} to {c[1]:.3f}"
    s0 = v["by_split"]["0"]
    tests = f["splits"]
    ag = ex["agreement_with_nn_predicted_age"]
    L_ = []
    a = L_.append
    a("# Data exclusions: who is left out of the CODE-15% ladder, and why")
    a("")
    a("Written by `exclusions.py` from `exams.csv`, the feature id files, `results/results.json` (this version), `results/results-v1.json` "
      "(the first published version, commit b952482) and `results/outcome_extraction.json`. Every number below is in `results/exclusions.json`; "
      "none is typed by hand. The script stops without writing if a count differs from those files.")
    a("")
    a(f"**In this version the only patients left out are the {n(f['short_follow_up'])} whose one-year outcome is unknown.** "
      f"The first version also left out {n(f['left_out_v1'])} patients whose saved network features came from a different exam than the one the outcome "
      f"is attached to. Their outcome exams have since been run through the network, and they are back in every row. Row 5 moved from "
      f"{s0['v1']['row5_age_sex_lima']:.3f} to {s0['v2']['row5_age_sex_lima']:.3f} on split 0 (section C).")
    a("")
    a("## A. The cohort, step by step")
    a("")
    a("```")
    a(f"{n(f['exams'])} exams in exams.csv")
    a(f"{n(f['patients'])} patients")
    a(f"  - {n(f['patients_without_outcome'])} patients with no exam that carries death and timey ({n(f['exams_of_patients_without_outcome'])} exams)")
    a(f"= {n(f['patients_with_outcome'])} patients with an outcome, {n(f['deaths_all'])} deaths within 1 year ({f['deaths_all_pct']:.2f}%)")
    a(f"  - {n(f['short_follow_up'])} alive with under 1 year of follow-up ({f['deaths_in_short']} deaths by definition)")
    a(f"= {n(f['main'])} main cohort, {n(f['deaths_main'])} deaths ({f['deaths_main_pct']:.2f}%), all with features from the outcome exam")
    a(f"  split 0 (seed {R['seed']}): train {n(tests['train']['n'])} ({n(tests['train']['deaths'])} deaths), "
      f"validation {n(tests['val']['n'])} ({tests['val']['deaths']}), test {n(tests['test']['n'])} ({tests['test']['deaths']})")
    a("```")
    a("")
    a(f"Each of the {n(f['patients_with_outcome'])} patients has exactly one exam with death and timey (the most on any patient is {f['outcome_rows_per_patient_max']}). "
      f"Network features for that exam: {n(f['lowest_id_features'])} patients from the saved features of each patient's lowest exam id, where that exam is the outcome exam, "
      f"and {n(f['extracted'])} extracted from the outcome exam's trace by `extract_outcome_exams.py`. "
      f"The extracted predicted ages match `nn_predicted_age` in `exams.csv`, the network's prediction that ships with the dataset, with a largest difference of "
      f"{ag['max_abs_diff_years']:.4f} years (Pearson r {ag['pearson_r']:.6f}), so the right traces were read. "
      "Missing values: " +
      ("none in any column used here, so no patient is left out for a missing value." if sum(R["missing"].values()) == 0
       else f"counts per column: {R['missing']}."))
    a("")
    a("The first version had one more step:")
    a("")
    a("```")
    a(f"  {n(f['main'])} main cohort")
    a(f"  - {n(f['left_out_v1'])} saved features not from the outcome exam ({f['deaths_left_out_v1']} deaths, {f['deaths_left_out_v1_pct']:.2f}%)")
    a(f"  = {n(f['kept_v1'])} kept, {n(f['deaths_kept_v1'])} deaths ({f['deaths_kept_v1_pct']:.2f}%); split 0 test {n(f['test_patients_v1'])} ({f['test_deaths_v1']} deaths)")
    a("```")
    a("")
    a(f"The short-follow-up patients and these {n(f['left_out_v1'])} do not overlap ({f['overlap_short_and_left_out']} in both). The split is drawn over all "
      f"{n(f['patients_with_outcome'])} patients with the same seed in both versions, so the first version's {n(f['test_patients_v1'])} test patients are a subset of this version's {n(f['test_patients'])}.")
    a("")
    a("## B. The patients the first version left out, and the exact logic")
    a("")
    a("CODE-15% has no exam dates or timestamps, so nothing below is date logic. The saved features come from `germany/extract_code.py`, "
      "which keeps one exam per patient, the lowest exam id:")
    a("")
    a("```")
    a(R["extract_line"])
    a("```")
    a("")
    a(f"The outcome (death, timey) sits on one exam per patient, and for {n(o['outcome_exam_is_lowest_id'])} of {n(o['n_all'])} patients that is the lowest exam id. "
      f"For the other {n(o['n'])}, `ladder.load_features`, which matches on exam id, found no saved features for the outcome exam, and the patient dropped out of every row. "
      f"All of them have two or more exams (fewest: {o['min_exams_excluded']}).")
    a("")
    a("What the sources say about which exam carries the outcome:")
    a("")
    a("Zenodo record 4916206, description of `exams.csv` (read through the Zenodo API):")
    a("")
    a(f"> {R['zenodo_quote']}")
    a("")
    a("Lima et al. 2021, Nature Communications 12:5117, article text on the journal page:")
    a("")
    a(f"> {R['lima_quote']}")
    a("")
    a("Neither defines \"first\" (by date, by id, or otherwise), and the sources read do not say whether exam id follows time. "
      "So the order has to be judged from the data. Age is in whole years, so it gives a direction only when the years differ:")
    a("")
    a(f"| Saved-feature exam compared with the outcome exam | Patients | Share of {n(o['n'])} |")
    a("|---|---|---|")
    for k, lab in (("older", "Older (higher age in years)"), ("same", "Same age in years"), ("younger", "Younger")):
        a(f"| {lab} | {n(o['age'][k])} | {pc(o['age'][k], o['n'])} |")
    for k, lab in (("lower", "Lower exam id"), ("higher", "Higher exam id")):
        a(f"| {lab} | {n(o['id'][k])} | {pc(o['id'][k], o['n'])} |")
    a("")
    a(f"The id rows are true by construction: the saved exam is the lowest id, so it can never be higher. The median age difference is {o['age_gap_years_median_excluded']:.0f} years. "
      f"In {n(o['outcome_is_min_age_excluded'])} of the {n(o['n'])} patients ({pc(o['outcome_is_min_age_excluded'], o['n'])}) the outcome exam has the lowest age among the patient's exams, "
      f"against {pc(o['outcome_is_min_age_kept_multi'], o['kept_multi_n'])} of the {n(o['kept_multi_n'])} kept patients with two or more exams. "
      "Exams.csv defines `age` as \"patient age in years at the moment of the exam\" (same Zenodo description). Read that way, the older group is a later exam, "
      f"the same-age group cannot be ordered ({n(o['age']['same'])} patients), and the younger group ({n(o['age']['younger'])} patients) is an earlier exam. "
      "Exam id order is not documented as chronological, and nothing here shows that it is.")
    a("")
    a("Why that mattered: an exam recorded after the outcome exam exists only if the patient was alive at that later date, so features taken from it carry information "
      "about survival after the outcome exam, which is what a one-year death label asks the model to predict. The first version removed these patients rather than estimate "
      "how large that effect would be. This version reads the outcome exam itself, so the question no longer arises.")
    a("")
    a("## C. Why leaving them out was leak prevention, and what changed when they came back")
    a("")
    a(f"The patients left out die less: {f['deaths_left_out_v1']} deaths in {n(f['left_out_v1'])} ({f['deaths_left_out_v1_pct']:.2f}%) against "
      f"{n(f['deaths_kept_v1'])} in {n(f['kept_v1'])} ({f['deaths_kept_v1_pct']:.2f}%). Removing a low-risk group could in principle move an AUC either way, "
      "so the test is to put them back and look. Split 0, AUC for death within one year with 95% bootstrap intervals:")
    a("")
    a(f"| Row | First version, {n(f['test_patients_v1'])} test patients ({f['test_deaths_v1']} deaths) | This version, {n(f['test_patients'])} ({f['test_deaths']} deaths) | Change | Mean of 5 splits, first version to this one |")
    a("|---|---|---|---|---|")
    for k in ROWS:
        a(f"| {NAMES[k]} | {s0['v1'][k]:.3f} ({ci(v['split0_ci']['v1'][k])}) | {s0['v2'][k]:.3f} ({ci(v['split0_ci']['v2'][k])}) | "
          f"{s0['v2'][k] - s0['v1'][k]:+.3f} | {v['mean']['v1'][k]:.3f} to {v['mean']['v2'][k]:.3f} |")
    a("")
    d1, d2 = v["split0_row5_minus_row2"]["v1"], v["split0_row5_minus_row2"]["v2"]
    a(f"Row 5 minus row 2 on split 0: {d1[0]:+.3f} ({d1[1]:.3f} to {d1[2]:.3f}) in the first version, {d2[0]:+.3f} ({d2[1]:.3f} to {d2[2]:.3f}) in this one. "
      "The two columns are different patient sets, so the change has no interval. "
      "Rows 1 to 4 use no network features: on the whole test split they are identical in the two versions (largest difference "
      + ("exactly 0" if v["rows14_all_test_v1_vs_v2_max_abs_diff"] == 0 else f"{v['rows14_all_test_v1_vs_v2_max_abs_diff']:.1e}")
      + ", checked by this script), "
      "so their change above comes only from scoring the added patients.")
    a("")
    a("Limits:")
    a("")
    a(f"- The added patients hold {f['deaths_left_out_v1']} of the {n(f['deaths_main'])} deaths and {f['test_deaths'] - f['test_deaths_v1']} of the {f['test_deaths']} in split 0's test set, "
      "so they move a pooled AUC only a little. The small change shows the exclusion did not flatter the results. It does not show how well any row ranks risk within that group.")
    a("- The extracted features come from the same network, read the same way, as the saved ones; the prediction check in section A is the evidence for that.")
    a("")
    a("## D. Baseline characteristics")
    a("")
    cols = [("all", "With an outcome"), ("main", "Main cohort"), ("excluded", "Left out of the first version"), ("kept", "Kept in the first version"),
            ("test", "Split 0 test, this version")]
    a("| | " + " | ".join(t for _, t in cols) + " | SMD, left out vs kept |")
    a("|---|" + "---|" * (len(cols) + 1))
    sd = lambda x: "" if x is None else f"{x:+.2f}"
    q = lambda x, d=1: f"{x[1]:.{d}f} ({x[0]:.{d}f} to {x[2]:.{d}f})"
    rows = [("Patients", lambda c: n(c["n"]), None),
            ("Deaths within 1 year", lambda c: f"{n(c['deaths'])} ({c['deaths_pct']:.2f}%)", sm["deaths"]),
            ("Age, mean (SD)", lambda c: f"{c['age_mean']:.1f} ({c['age_sd']:.1f})", sm["age"]),
            ("Age, median (Q1 to Q3)", lambda c: q(c["age_q"], 0), None),
            ("Male, %", lambda c: f"{c['male_pct']:.1f}", sm["male"]),
            ("Follow-up, years, median (Q1 to Q3)", lambda c: q(c["timey_q"], 2), sm["timey"]),
            ("Exams per patient, median (Q1 to Q3)", lambda c: q(c["exams_q"], 0), None),
            ("Two or more exams, %", lambda c: f"{c['multi_pct']:.1f}", sm["multi"])]
    rows += [(lab + ", %", (lambda c, k=k: f"{c[k]:.1f}"), sm[k]) for k, lab in
             (("1dAVb", "1st degree AV block"), ("RBBB", "Right bundle branch block"), ("LBBB", "Left bundle branch block"),
              ("SB", "Sinus bradycardia"), ("ST", "Sinus tachycardia"), ("AF", "Atrial fibrillation"), ("normal_ecg", "Normal ECG label"))]
    rows.append(("ECG-age gap, years, mean (SD)", lambda c: f"{c['gap_mean']:.1f} ({c['gap_sd']:.1f})", sm["gap"]))
    for lab, fn, sv in rows:
        a(f"| {lab} | " + " | ".join(fn(ch[k]) for k, _ in cols) + f" | {sd(sv)} |")
    a("")
    a("All characteristics are taken from the outcome exam. The gap is `nn_predicted_age` minus `age`. SMD is the left-out mean minus the kept mean over the pooled SD "
      "(for percentages, the same on the 0 or 1 values); for follow-up it uses means, not the medians shown. Follow-up is time to death for the dead and time to last "
      "contact for the rest. Exams per patient counts every exam of the patient in `exams.csv`, which is why the left-out group all have two or more.")
    a("")
    nm = R["normal"]
    extra = nm['multi_outcome_is_lowest']['n'] - o['kept_multi_n']
    multi = nm['multi_outcome_is_lowest']['n'] + nm['multi_outcome_not_lowest']['n']
    a(f"Normal ECG label. Take the {n(multi)} patients with an outcome and two or more exams (this counts the short follow-up patients too, "
      f"so the lowest-id group below has {n(extra)} more than the kept count in section B). The outcome exam carries `normal_ecg` in {nm['multi_outcome_not_lowest']['normal_pct']:.1f}% "
      f"({n(nm['multi_outcome_not_lowest']['n'])} patients) when it is not the patient's lowest exam id, and in {nm['multi_outcome_is_lowest']['normal_pct']:.1f}% "
      f"({n(nm['multi_outcome_is_lowest']['n'])} patients) when it is. The {n(nm['non_outcome_exams']['n'])} exams that do not carry the outcome have it in {nm['non_outcome_exams']['normal_pct']:.1f}%. "
      f"On the outcome exam, {nm['neither_pct']['dropped']:.1f}% of the left-out patients have neither `normal_ecg` nor any of the six rhythm labels, "
      f"against {nm['neither_pct']['kept_multi']:.1f}% of the kept patients with two or more exams and {nm['neither_pct']['kept_single']:.1f}% of the kept patients with one exam. "
      "The Zenodo description defines the label as: \"normal_ecg\": True if automatic annotation system say it is a normal ECG. "
      "We do not know why the outcome exam of these patients almost never carries it, and the dataset description does not say how exam ids were assigned. "
      f"Row 3 uses `normal_ecg`, so it is the row most exposed to this: on split 0 it moved from {s0['v1']['row3_age_sex_rhythm']:.3f} to {s0['v2']['row3_age_sex_rhythm']:.3f} "
      "when these patients came back (section C).")
    a("")
    a("## Reproduce")
    a("")
    a("```")
    a("cd one-year-mortality-baselines")
    a("python extract_outcome_exams.py    # once: writes the code_outcome_part files, needs the CODE-15% HDF5 traces")
    a("python analysis.py --boot 1000")
    a("python exclusions.py               # needs ../data/exams.csv and the _ids.npy files; reads no _feat.npy")
    a("```")
    a("")
    return "\n".join(L_)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "exams.csv"))
    ap.add_argument("--features", default=os.path.join(HERE, "..", "..", "ecg-age-model", "features"))
    ap.add_argument("--results", default=os.path.join(HERE, "results", "results.json"))
    ap.add_argument("--v1", default=os.path.join(HERE, "results", "results-v1.json"))
    ap.add_argument("--extraction", default=os.path.join(HERE, "results", "outcome_extraction.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "exclusions.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "DATA_EXCLUSIONS.md"))
    a = ap.parse_args()
    R = compute(a.data, a.features, a.results, a.v1, a.extraction)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    with open(a.md, "w") as f:
        f.write(render(R))
    print(f"wrote {a.out} and {a.md}")
