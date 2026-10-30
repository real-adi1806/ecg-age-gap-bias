import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")   # everything run.sh downloads or derives here; never committed (.gitignore)
sys.path.insert(0, os.path.join(DATA_DIR, "third_party", "ecg-age-prediction"))
from resnet import ResNet1d  # noqa: E402

# Data locations: override with environment variables, defaults are where run.sh downloads to.
CODE_DIR = os.environ.get("CODE15_DIR", os.path.expanduser("~/datasets/code15"))
PTB_DIR = os.environ.get("PTBXL_DIR", os.path.expanduser("~/datasets/ptb-xl"))
FEAT_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(DATA_DIR, "model", "model")
PTB_AGE_RANGE = (18, 89)   # PTB-XL codes ages above 89 as 300 (HIPAA); adults only, so under 18 is excluded
PTB_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # the network's lead order

def load_ptb_record(filename_hr, scale=1.0):
    """One PTB-XL 500 Hz record (mV) as the network's input: resampled to 400 Hz (polyphase 4/5),
    centre-padded 4000 -> 4096 with zeros as in CODE, mV fed unchanged (network README: volts x 1000)."""
    import wfdb
    from scipy.signal import resample_poly
    s, meta = wfdb.rdsamp(os.path.join(PTB_DIR, filename_hr))
    assert meta["fs"] == 500 and s.shape == (5000, 12) and list(meta["units"]) == ["mV"] * 12
    assert list(meta["sig_name"]) == PTB_LEADS, meta["sig_name"]
    s = np.nan_to_num(s)
    s = resample_poly(s, 4, 5, axis=0)              # 5000 -> 4000 samples at 400 Hz
    s = np.pad(s, ((48, 48), (0, 0)))               # 4000 -> 4096, zeros both sides
    return (s * scale).astype(np.float32)

def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(dev):
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    # weights_only=False: the checkpoint pickles a numpy scalar; file is the authors' Zenodo model.zip (md5 verified)
    ckpt = torch.load(os.path.join(MODEL_DIR, "model.pth"), map_location="cpu", weights_only=False)
    m = ResNet1d(input_dim=(12, cfg["seq_length"]),
                 blocks_dim=list(zip(cfg["net_filter_size"], cfg["net_seq_lengh"])),
                 n_classes=1, kernel_size=cfg["kernel_size"], dropout_rate=cfg["dropout_rate"])
    m.load_state_dict(ckpt["model"])
    return m.to(dev).eval()

@torch.no_grad()
def features_and_pred(model, x):
    """x: float32 tensor (N,12,4096). Returns (features N x 5120 = input of final linear layer, prediction N)."""
    x = model.bn1(model.conv1(x))
    y = x
    for blk in model.res_blocks:
        x, y = blk(x, y)
    f = x.reshape(x.size(0), -1)
    return f, model.lin(f).squeeze(-1)

def run_batches(model, dev, get_batch, n, bs=64):
    F = np.zeros((n, 5120), np.float32); P = np.zeros(n, np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        x = torch.from_numpy(get_batch(s, e)).transpose(1, 2).contiguous().to(dev, dtype=torch.float32)
        f, p = features_and_pred(model, x)
        F[s:e] = f.cpu().numpy(); P[s:e] = p.cpu().numpy()
    return F, P
