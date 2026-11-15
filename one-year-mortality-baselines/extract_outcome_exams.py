#!/usr/bin/env python3
"""
Extract the Lima network's features for the outcome exams that the saved feature files do not hold.

The saved files (code_part{k}_feat.npy, written by ../germany/extract_code.py) hold one exam per patient: the
lowest exam id. CODE-15% attaches the death flag and follow-up time to one exam per patient, and for 15,715 patients
that is a different exam (DATA_EXCLUSIONS.md). This script reads those exams' traces from the CODE-15% HDF5 files,
runs the same network, and writes code_outcome_part{k}_feat.npy, _ids.npy and _pred.npy beside the saved files.
After it, every patient with an outcome has features from the exam the outcome is attached to.

    python extract_outcome_exams.py
    python extract_outcome_exams.py --code-dir /path/to/hdf5 --features /path/to/features --model-dir /path/to/model

Check: exams.csv ships the network's predicted age for every exam (nn_predicted_age). The prediction made here from
the trace must match it, as it does for the saved files (to 0.0006 years). A larger difference means the wrong trace
was read, and the script stops. The agreement is written to results/outcome_extraction.json.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "germany"))   # the repository's copy of the extraction code
PARTS = range(18)
PREFIX = "code_outcome_part"
MAX_ABS_DIFF_YEARS = 0.01


def missing_outcome_exams(df, feat_dir):
    """Exam ids that carry the outcome (death and timey present) but are not in the saved code_part{k} files."""
    have = np.concatenate([np.load(f"{feat_dir}/code_part{k}_ids.npy") for k in PARTS])
    outcome = df.loc[df["death"].notna() & df["timey"].notna(), "exam_id"].to_numpy()
    return np.setdiff1d(outcome, have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "exams.csv"), help="path to exams.csv")
    ap.add_argument("--code-dir", default=os.environ.get("CODE15_DIR", os.path.expanduser("~/datasets/code15")),
                    help="folder holding exams_part{k}.hdf5")
    ap.add_argument("--features", default=os.path.join(HERE, "..", "..", "ecg-age-model", "features"),
                    help="folder holding the saved code_part{k} files; the new files are written here")
    ap.add_argument("--model-dir", default=None,
                    help="folder holding the network's model.pth and config.json (default: as in ../germany/common.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "outcome_extraction.json"))
    args = ap.parse_args()
    for p in (args.data, args.code_dir, args.features):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    import common  # needs torch; imported here so --help works without it
    if args.model_dir:
        common.MODEL_DIR = args.model_dir

    df = pd.read_csv(args.data, usecols=["exam_id", "patient_id", "death", "timey", "nn_predicted_age"])
    want = missing_outcome_exams(df, args.features)
    csv_pred = df.set_index("exam_id")["nn_predicted_age"]
    print(f"{len(want):,} outcome exams have no saved features", flush=True)

    dev = common.device()
    model = common.load_model(dev)
    t0, per_part, preds, ref = time.time(), {}, [], []
    for k in PARTS:
        path = f"{args.code_dir}/exams_part{k}.hdf5"
        if not os.path.exists(path):
            sys.exit(f"not found: {path}")
        with h5py.File(path, "r") as f:
            ids = np.array(f["exam_id"])
            tr = f["tracings"]
            sel = np.where(np.isin(ids, want))[0]   # increasing, as h5py needs
            F, P = common.run_batches(model, dev, lambda s, e: np.stack([tr[i] for i in sel[s:e]]), len(sel))
        got = ids[sel]
        if len(sel):
            diff = np.abs(P - csv_pred.loc[got].to_numpy())
            if diff.max() > MAX_ABS_DIFF_YEARS:
                sys.exit(f"STOP: part {k}: prediction differs from nn_predicted_age by {diff.max():.4f} years. "
                         "Nothing from this part was saved.")
        out = f"{args.features}/{PREFIX}{k}"
        np.save(out + "_feat.npy", F)
        np.save(out + "_ids.npy", got)
        np.save(out + "_pred.npy", P)
        per_part[k] = int(len(sel))
        preds.append(P)
        ref.append(csv_pred.loc[got].to_numpy())
        print(f"part {k}: {len(sel)} exams, {time.time() - t0:.0f}s", flush=True)

    P, R = np.concatenate(preds), np.concatenate(ref)
    found = int(sum(per_part.values()))
    report = {"outcome_exams_without_saved_features": int(len(want)), "extracted": found, "per_part": per_part,
              "agreement_with_nn_predicted_age": {"pearson_r": float(np.corrcoef(P, R)[0, 1]),
                                                  "mean_abs_diff_years": float(np.abs(P - R).mean()),
                                                  "max_abs_diff_years": float(np.abs(P - R).max())},
              "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    if found != len(want):
        sys.exit(f"STOP: found {found:,} of {len(want):,} exams in the HDF5 files")


if __name__ == "__main__":
    main()
