"""Extract final-layer inputs + prediction for the first exam (lowest exam_id) of every CODE-15% patient. Cached per part."""
import sys, os, time, h5py, numpy as np, pandas as pd
from common import *
parts = [int(a) for a in sys.argv[1:]] or range(18)
df = pd.read_csv(f"{CODE_DIR}/exams.csv")
first = df.sort_values("exam_id").drop_duplicates("patient_id")  # first exam per patient
first_ids = set(first.exam_id)
dev = device(); model = load_model(dev)
for k in parts:
    out = f"{FEAT_DIR}/code_part{k}"
    if os.path.exists(out + "_pred.npy"):
        print("cached", k); continue
    path = f"{CODE_DIR}/exams_part{k}.hdf5"
    if not os.path.exists(path):
        print("missing", k); continue
    t = time.time()
    with h5py.File(path, "r") as f:
        ids = np.array(f["exam_id"]); tr = f["tracings"]
        sel = np.where(np.isin(ids, list(first_ids)))[0]
        def gb(s, e):
            return np.stack([tr[i] for i in sel[s:e]])  # h5py needs increasing index; one by one is safe
        F, P = run_batches(model, dev, gb, len(sel))
    np.save(out + "_feat.npy", F); np.save(out + "_ids.npy", ids[sel]); np.save(out + "_pred.npy", P)
    print(f"part {k}: {len(sel)} exams, {time.time()-t:.0f}s", flush=True)
