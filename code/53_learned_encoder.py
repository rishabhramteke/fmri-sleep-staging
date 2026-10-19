"""A learned temporal encoder on the raw within-window parcel series: is the REM null an artefact of hand-crafted
epoch features (30 s amplitude, 90 s network FC)?

Design, fixed before any result was seen (as 39_strong_physio.py did for the BiGRU):
  input    the derivative's percent-signal-change series of the 200 parcels in a 90 s window (30 frames) centred on
           the epoch, standardised per subject and parcel over that subject's uncensored frames (label-free);
           censored or missing frames set to 0, with the frame mask as a 201st channel
  encoder  Conv1d(201 -> 64, k=3) + GELU + dropout 0.3, Conv1d(64 -> 64, k=3, dilation 2) + GELU, masked mean and
           s.d. pooling over time (128 features); channel dropout 0.2 on the input parcels
  models   enc      encoder -> linear -> 5 stages (BOLD only)
           nb       the paper's non-BOLD features (motion, physiology, elapsed time) -> linear 48 + GELU + dropout 0.3 -> 5
           nb+enc   both branches concatenated (176) -> 5, trained jointly
  training leave-one-subject-out; class-weighted cross-entropy; AdamW (lr 1e-3, weight decay 1e-2); batches of 256
           epochs; early stopping on two held-out TRAINING subjects by the stage-averaged within-subject AUC
           (max 40 passes, patience 6); three seeds averaged. No sequence context: the HMM provides it afterwards.
  readout  the paper's statistic: within-subject one-vs-rest AUC gain of nb+enc over nb per stage (subject bootstrap
           CI, one-sided bounds, Wilcoxon), the N3 - REM contrast, enc-alone AUCs, and HMM kappa of enc and nb+enc.

    source <venv>/bin/activate
    python 53_learned_encoder.py --build      # windows_learned_encoder.npz (N x 201 x 30, ~370 MB)
    python 53_learned_encoder.py --fit        # results_learned_encoder.json, predictions_learned_encoder.npz
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).parent
WIN = 30                                  # frames (90 s at TR 3 s)
SEEDS = (0, 1, 2)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(2)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e = _load("e24", "24_experiments_v2.py")
ST = e.STAGES


# ------------------------------------------------------------------ data
def build():
    f22 = _load("f22", "22_features_v2.py")
    feat = f22.feat
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    want = set(np.load(BASE / "features_v2_afni.npz", allow_pickle=True)["run"].astype(str))
    keys, X, M, subj = [], [], [], []
    for stem in sorted(want):
        p = BASE / "derivatives" / "afni_ds005127" / raw_runs[stem].parent.name / f"{stem}_afni.npz"
        eps = feat.run_epochs(raw_runs[stem])
        if not p.exists() or not eps:
            continue
        z = np.load(p)
        ts = z["ts"].astype(np.float64)
        T = ts.shape[1]
        valid = ~z["censored"].astype(bool)
        x = f22.psc(ts, valid)                      # NaN where censored
        tr = float(z["tr"])
        for ep in eps:
            c = ep["start_vol"] + ep["n_vol"] / 2
            c0, c1 = f22.window(c, WIN * tr, tr, T)
            w = np.full((200, WIN), np.nan, np.float32)
            seg = x[:, c0:c1]
            o = c0 - int(round(c - WIN / 2))             # offset inside the window when the run edge clips it
            n = min(seg.shape[1], WIN - o)
            w[:, o:o + n] = seg[:, :n]
            keys.append((stem, ep["epoch"])); X.append(w); subj.append(raw_runs[stem].parent.name)
    X = np.array(X, np.float32)
    subj = np.array(subj)
    M = np.isfinite(X).mean(1) > 0.5                # frame present (uncensored) for most parcels
    # per-subject, per-parcel standardisation over valid frames (label-free)
    for u in np.unique(subj):
        m = subj == u
        v = X[m]
        mu = np.nanmean(v, axis=(0, 2), keepdims=True)
        sd = np.nanstd(v, axis=(0, 2), keepdims=True) + 1e-6
        X[m] = (v - mu) / sd
    X = np.nan_to_num(X, nan=0.0)
    print(f"windows {X.shape}, frames present {M.mean():.3f}, subjects {len(np.unique(subj))}", flush=True)
    np.savez_compressed(BASE / "windows_learned_encoder.npz", run=np.array([k[0] for k in keys]), epoch=np.array([k[1] for k in keys]),
                        X=X, M=M)


# ------------------------------------------------------------------ model
class Net(nn.Module):
    def __init__(self, d_nb, use_enc, use_nb, h=64):
        super().__init__()
        self.use_enc, self.use_nb = use_enc, use_nb
        if use_enc:
            self.cdrop = nn.Dropout1d(0.2)
            self.c1 = nn.Sequential(nn.Conv1d(201, h, 3, padding=1), nn.GELU(), nn.Dropout(0.3))
            self.c2 = nn.Sequential(nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.GELU())
        if use_nb:
            self.nb = nn.Sequential(nn.Linear(d_nb, 48), nn.GELU(), nn.Dropout(0.3))
        self.out = nn.Linear((2 * h if use_enc else 0) + (48 if use_nb else 0), len(ST))

    def forward(self, x, m, nb):
        parts = []
        if self.use_enc:
            z = torch.cat([self.cdrop(x), m[:, None, :].float()], 1)
            z = self.c2(self.c1(z))                                   # B x h x W
            mm = m[:, None, :].float()
            mean = (z * mm).sum(-1) / mm.sum(-1).clamp(min=1)
            sd = (((z - mean[..., None]) ** 2 * mm).sum(-1) / mm.sum(-1).clamp(min=1)).sqrt()
            parts += [mean, sd]
        if self.use_nb:
            parts.append(self.nb(nb))
        return self.out(torch.cat(parts, 1))


def ws_score(P, y, s):
    return float(np.mean([np.mean(list(e.ws_auc((y == st).astype(int), P[:, i], s).values())) for i, st in enumerate(ST)]))


def predict(net, X, M, NB, idx):
    net.eval()
    out = np.zeros((len(idx), len(ST)), np.float32)
    with torch.no_grad():
        for k in range(0, len(idx), 1024):
            j = torch.as_tensor(idx[k:k + 1024], device=DEV)
            out[k:k + 1024] = torch.softmax(net(X[j], M[j], NB[j]), -1).cpu().numpy()
    return out


def train(kind, X, M, NB, yi, y, s, tr_mask, seed, max_passes=40, patience=6):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    subs = np.unique(s[tr_mask])
    val_subs = rng.choice(subs, 2, replace=False)
    fit = np.flatnonzero(tr_mask & ~np.isin(s, val_subs))
    val = np.flatnonzero(tr_mask & np.isin(s, val_subs))
    counts = np.bincount(yi[fit], minlength=len(ST)).astype(np.float32)
    w = torch.tensor(counts.sum() / (len(ST) * np.maximum(counts, 1)), device=DEV)
    net = Net(NB.shape[1], use_enc=kind != "nb", use_nb=kind != "enc").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=w)
    Y = torch.from_numpy(yi.astype(np.int64)).to(DEV)
    best, best_state, bad = -1, None, 0
    for _ in range(max_passes):
        net.train()
        order = rng.permutation(fit)
        for k in range(0, len(order), 256):
            j = torch.as_tensor(order[k:k + 256], device=DEV)
            l = lossf(net(X[j], M[j], NB[j]), Y[j])
            opt.zero_grad(); l.backward(); opt.step()
        sc = ws_score(predict(net, X, M, NB, val), y[val], s[val])
        if sc > best:
            best, bad = sc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


def orbitofrontal_parcels():
    """Schaefer-200 parcels whose MNI centroid lies in orbitofrontal / frontal-pole cortex (y > 20 mm, z < -5 mm): the
    parcels nearest the eyes, where eye-movement artefacts enter the parcel series."""
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_schaefer_2018
    img = nib.load(fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)["maps"])
    lab = np.asanyarray(img.dataobj).astype(int)
    out = []
    for k in range(1, 201):
        ijk = np.argwhere(lab == k).mean(0)
        xyz = img.affine[:3, :3] @ ijk + img.affine[:3, 3]
        if xyz[1] > 20 and xyz[2] < -5:
            out.append(k - 1)
    return np.array(out)


def fit(variant="full"):
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    nb = np.hstack([B["motion"], B["physio"], B["clock"]]).astype(np.float32)
    nb = (nb - nb.mean(0)) / (nb.std(0) + 1e-6)                 # label-free global scaling of the clock columns etc.
    Z = np.load(BASE / "windows_learned_encoder.npz", allow_pickle=True)
    key = {(r, int(p)): i for i, (r, p) in enumerate(zip(Z["run"].astype(str), Z["epoch"]))}
    rows = np.array([key[(r, int(p))] for r, p in zip(rn, ep)])
    Xn = Z["X"][rows]
    dropped = []
    if variant == "mask_only":                       # leakage check: the censoring pattern alone, no BOLD
        Xn = np.zeros_like(Xn)
    elif variant == "no_orbitofrontal":              # leakage check: parcels nearest the eyes removed
        dropped = orbitofrontal_parcels()
        Xn = Xn.copy(); Xn[:, dropped, :] = 0.0
        print(f"  dropped {len(dropped)} orbitofrontal / frontal-pole parcels", flush=True)
    X = torch.from_numpy(Xn).to(DEV)
    M = torch.from_numpy(Z["M"][rows]).to(DEV)
    NB = torch.from_numpy(nb).to(DEV)
    yi = np.array([ST.index(v) for v in y])
    print(f"{len(y)} epochs, {len(np.unique(s))} subjects, device {DEV}; windows {tuple(X.shape)}", flush=True)
    kinds = ("enc", "nb", "nb+enc") if variant == "full" else ("enc", "nb+enc")
    P = {k: np.zeros((len(y), len(ST))) for k in kinds}
    for u in np.unique(s):
        te = np.flatnonzero(s == u)
        for kind in P:
            for sd in SEEDS:
                net = train(kind, X, M, NB, yi, y, s, s != u, seed=1000 * sd + int(u.split("-")[1]))
                P[kind][te] += predict(net, X, M, NB, te) / len(SEEDS)
        print(f"  fold {u} done ({time.time() - t0:.0f} s): " + " ".join(f"{k} N3 {e.auc((y[te] == 'N3').astype(int), P[k][te, 3]):.3f}"
                                                                   if (y[te] == 'N3').sum() else "" for k in P), flush=True)
    x = _load("x36b", "36b_extra_block.py")
    if "nb" not in P:                                # variants reuse the full run's non-BOLD network (identical design and seeds)
        P["nb"] = np.load(BASE / "predictions_learned_encoder.npz", allow_pickle=True)["proba__nb"]
    G = x.gains(P["nb+enc"], P["nb"], y, s)
    R = {"meta": dict(n_epochs=int(len(y)), n_subjects=int(len(np.unique(s))), window_frames=WIN, seeds=list(SEEDS), device=str(DEV),
                      variant=variant, dropped_parcels=[int(v) for v in dropped]),
         "gain_nb+enc_over_nb": {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST},
         "contrast": x.contrast(G),
         "auc_ws": {k: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[k][:, i], s).values()))) for i, st in enumerate(ST)} for k in P}}
    Gl = x.gains(P["nb+enc"], np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)["proba__nonbold"], y, s)
    R["gain_nb+enc_over_paper_nonbold_lr"] = {st: {k: v for k, v in Gl[st].items() if k != "boot_ws"} for st in ST}
    R["hmm"] = {}
    for k in P:
        pred = e.hmm_decode(P[k], y, s, rn, ep)
        met = e.staging(y, pred, s)
        met["kappa_ci"] = e.boot_kappa(y, pred, s)
        R["hmm"][k] = {kk: met[kk] for kk in ("kappa", "kappa_ci", "f1", "acc")}
    for st in ("N3", "REM"):
        g = R["gain_nb+enc_over_nb"][st]
        print(f"  gain {st}: {g['ws_mean']:+.3f} [{g['ws_ci'][0]:+.3f},{g['ws_ci'][1]:+.3f}] upper {g['ws_upper95']:+.3f} t {g['t_upper95']:+.3f} "
              f"{g['n_pos']}/{g['n_subj']}", flush=True)
    print(f"  N3-REM contrast {R['contrast']}", flush=True)
    print("  enc alone AUC", {st: round(v, 3) for st, v in R["auc_ws"]["enc"].items()})
    print("  HMM kappa", {k: round(v["kappa"], 3) for k, v in R["hmm"].items()}, flush=True)
    sfx = "" if variant == "full" else f"_{variant}"
    (BASE / f"results_learned_encoder{sfx}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_learned_encoder{sfx}.npz", y=y, subj=s, run=rn, epoch=ep, **{"proba__" + k: v.astype(np.float32) for k, v in P.items()})
    print(f"saved results_learned_encoder{sfx}.json, predictions_learned_encoder{sfx}.npz ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--variant", default="full", choices=["full", "mask_only", "no_orbitofrontal"])
    a = ap.parse_args()
    if a.build:
        build()
    if a.fit:
        fit(a.variant)


if __name__ == "__main__":
    main()
