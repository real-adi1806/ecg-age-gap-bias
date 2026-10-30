"""PTB-XL: read 500 Hz records (mV), resample to 400 Hz (polyphase 4/5), centre-pad 4000 -> 4096 with zeros, extract features.
Scale: README says signals are on a scale where V*1000 is fed to the network, i.e. mV * 1 (SCALE=1). Override with argv[1] for the sensitivity check."""
import sys, os, time, numpy as np, pandas as pd
from common import *
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
tag = "ptb" if SCALE == 1.0 and LIMIT is None else f"ptb_s{SCALE:g}_n{LIMIT}"
out = f"{FEAT_DIR}/{tag}"
if os.path.exists(out + "_pred.npy"): print("cached", tag); sys.exit()
d = pd.read_csv(f"{PTB_DIR}/ptbxl_database.csv")
d = d[(d.age >= PTB_AGE_RANGE[0]) & (d.age <= PTB_AGE_RANGE[1])].reset_index(drop=True)
if LIMIT: d = d.sample(LIMIT, random_state=0).reset_index(drop=True)
dev = device(); model = load_model(dev); t = time.time()
F, P = run_batches(model, dev, lambda a, b: np.stack([load_ptb_record(d.filename_hr[i], SCALE) for i in range(a, b)]), len(d))
np.save(out + "_feat.npy", F); np.save(out + "_pred.npy", P); d.to_csv(out + "_meta.csv", index=False)
print(tag, len(d), f"{time.time()-t:.0f}s")
r = np.corrcoef(P, d.age)[0, 1]; print("pearson r", r, "MAE", np.abs(P - d.age).mean())
