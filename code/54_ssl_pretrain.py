"""Self-supervised pretraining on the whole-night corpus (JEPA-style), then a linear probe for sleep staging.

Why: no public sleep-fMRI foundation model exists, and whole-night EEG-fMRI is the ideal corpus for brain-state
dynamics. Hand-crafted epoch features (30 s amplitude, 90 s network FC) may miss REM-specific structure; a learned,
label-free representation is the cleanest test of that. This is follow-up work; the code is kept runnable so it can be
launched when compute is free (started 2026-10-19; results are used only if ready in time).

Corpus: every run of the dataset authors' derivative (185 runs, ~145 h at TR 3 s), percent signal change of the 200
Schaefer parcels, standardised per subject and parcel over uncensored frames (label-free), cut into 90 s windows
(30 frames) with a stride of 5 frames: about 34,000 windows. Censored frames are zeroed and masked in attention.

Model (small, for ~150 h of data): each frame's 200-parcel vector is a token (linear 200 -> d=128) plus a learned
position; a 4-layer transformer encoder (4 heads, FF 256, no dropout: MPS attention lacks it; EMA + weight decay regularise); a target encoder updated by EMA (0.996); a
2-layer transformer predictor. Objective (I-JEPA / Brain-JEPA style): predict the TARGET encoder's embeddings at
masked positions from the CONTEXT encoder's embeddings, smooth-L1, stop-gradient on targets. Two masking regimes
alternate per batch: (a) cross-time: two contiguous blocks covering ~40% of frames are removed from the context and
predicted; (b) cross-parcel: a random 30% of parcels are zeroed in every context frame and the full-frame target
embeddings of all frames are predicted. Nothing is reconstructed in signal space (BOLD is noisy; see Brain-JEPA).

Probe (the paper's readout, so results are comparable): the labelled 90 s windows of 53_learned_encoder.py are
embedded (mean of encoder tokens over valid frames, 128-d, frozen), then leave-one-subject-out class-balanced logistic
regression as in the paper: nb, nb + emb, nb + bold (hand-crafted), nb + bold + emb, emb alone; within-subject AUC
gains with subject bootstrap, the N3 - REM contrast, HMM kappa. Pretraining is label-free and transductive over
subjects by default; --exclude-subject SUB pretrains without that subject for a strict check.

    source <venv>/bin/activate
    python 54_ssl_pretrain.py --build                          # ssl_windows.npz
    python 54_ssl_pretrain.py --pretrain [--epochs 80]         # ssl_checkpoint.pt, ssl_log.json
    python 54_ssl_pretrain.py --probe                          # results_ssl_probe.json, predictions_ssl_probe.npz
    python 54_ssl_pretrain.py --build --pretrain --probe --smoke   # tiny end-to-end test on CPU
"""
import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).parent
WIN, STRIDE = 30, 5
D_MODEL, N_HEADS, N_LAYERS, FF = 128, 4, 4, 256
EMA = 0.996
torch.set_num_threads(2)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ corpus
def build(smoke=False):
    f22 = _load("f22", "22_features_v2.py")
    files = sorted((BASE / "derivatives" / "afni_ds005127").glob("sub-*/*_afni.npz"))
    if smoke:
        files = files[:6]
    X, M, subj, run = [], [], [], []
    per_subject = {}
    for p in files:
        z = np.load(p)
        ts = z["ts"].astype(np.float64)
        valid = ~z["censored"].astype(bool)
        x = f22.psc(ts, valid).astype(np.float32)          # NaN where censored
        per_subject.setdefault(p.parent.name, []).append(x)
        T = ts.shape[1]
        for a in range(0, T - WIN + 1, STRIDE):
            w = x[:, a:a + WIN]
            m = np.isfinite(w).mean(0) > 0.5
            if m.sum() < WIN // 2:
                continue
            X.append(w); M.append(m); subj.append(p.parent.name); run.append(p.name.replace("_afni.npz", ""))
    X, M, subj, run = np.array(X, np.float32), np.array(M), np.array(subj), np.array(run)
    for u, xs in per_subject.items():                      # label-free standardisation per subject and parcel
        cat = np.concatenate(xs, 1)
        mu, sd = np.nanmean(cat, 1), np.nanstd(cat, 1) + 1e-6
        X[subj == u] = (X[subj == u] - mu[None, :, None]) / sd[None, :, None]
    X = np.nan_to_num(X, nan=0.0)
    print(f"corpus: {len(X)} windows of {WIN} frames from {len(files)} runs, {len(per_subject)} subjects, "
          f"{len(X) * STRIDE * 3 / 3600:.0f} h covered (stride {STRIDE})", flush=True)
    np.savez_compressed(BASE / "ssl_windows.npz", X=X, M=M, subj=subj, run=run)


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(self, n_parcels=200, d=D_MODEL):
        super().__init__()
        self.inp = nn.Linear(n_parcels, d)
        self.pos = nn.Parameter(torch.zeros(1, WIN, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, N_HEADS, FF, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, keep):
        """x: B x P x W parcel series; keep: B x W bool (frames the encoder may attend to)."""
        h = self.inp(x.transpose(1, 2)) + self.pos
        h = self.enc(h, src_key_padding_mask=~keep)
        return self.norm(h)


class Predictor(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_tok, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, WIN, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, N_HEADS, FF, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, d)

    def forward(self, h_ctx, ctx_keep, predict_at, frame_ok):
        """h_ctx: context embeddings (B x W x d; positions not in ctx_keep are replaced by the mask token);
        predict_at: B x W bool positions to predict; frame_ok: B x W valid (uncensored) frames."""
        h = torch.where(ctx_keep[..., None], h_ctx, self.mask_tok.expand_as(h_ctx)) + self.pos
        h = self.dec(h, src_key_padding_mask=~frame_ok)
        return self.out(h)


def time_block_mask(B, rng, frac=0.4, n_blocks=2):
    m = np.zeros((B, WIN), bool)
    L = int(round(frac * WIN / n_blocks))
    for b in range(B):
        for _ in range(n_blocks):
            a = rng.integers(0, WIN - L + 1)
            m[b, a:a + L] = True
    return m


def pretrain(epochs=80, batch=128, lr=3e-4, exclude=None, smoke=False, save_every=0, tag=""):
    dev = torch.device("mps" if torch.backends.mps.is_available() and not smoke else "cpu")
    Z = np.load(BASE / "ssl_windows.npz", allow_pickle=True)
    X, M, subj = Z["X"], Z["M"], Z["subj"].astype(str)
    if exclude:
        keep = subj != exclude
        X, M = X[keep], M[keep]
    if smoke:
        X, M, epochs, batch = X[:512], M[:512], 2, 64
    X, M = torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev)
    enc, pred = Encoder().to(dev), Predictor().to(dev)
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(X) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    log, t0, step = [], time.time(), 0
    probe_idx = torch.arange(0, len(X), max(1, len(X) // 512), device=dev)[:512]      # fixed windows for the collapse check
    for epch in range(epochs):
        enc.train(); pred.train()
        order = rng.permutation(len(X))
        tot, n = 0.0, 0
        for k in range(0, len(order), batch):
            j = torch.as_tensor(order[k:k + batch], device=dev)
            x, m = X[j], M[j]
            B = len(j)
            with torch.no_grad():
                h_t = tgt(x, m)                                             # B x W x d, full-window targets
            if step % 2 == 0:                                                # (a) cross-time block masking
                blk = torch.from_numpy(time_block_mask(B, rng)).to(dev)
                ctx_keep = m & ~blk
                predict_at = blk & m
                h_c = enc(x, ctx_keep)
            else:                                                            # (b) cross-parcel masking
                pm = torch.from_numpy(rng.random((B, 200, 1)) < 0.3).to(dev)
                ctx_keep = m
                predict_at = m
                h_c = enc(x.masked_fill(pm, 0.0), ctx_keep)
            out = pred(h_c, ctx_keep, predict_at, m)
            loss = nn.functional.smooth_l1_loss(out[predict_at], h_t[predict_at])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            with torch.no_grad():                                            # EMA target encoder
                for pt, pe in zip(tgt.parameters(), enc.parameters()):
                    pt.mul_(EMA).add_(pe.detach(), alpha=1 - EMA)
            tot += float(loss) * B; n += B
        with torch.no_grad():                                            # collapse check: spread of pooled target embeddings
            tgt.eval()
            h = tgt(X[probe_idx], M[probe_idx])
            mm = M[probe_idx][..., None].float()
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            emb_std = float(pooled.std(0).mean())
        log.append(dict(epoch=epch + 1, loss=tot / n, emb_std=emb_std, seconds=time.time() - t0))
        print(f"  epoch {epch + 1}/{epochs} loss {tot / n:.4f} emb_std {emb_std:.3f} ({time.time() - t0:.0f} s)", flush=True)
        state = dict(enc=enc.state_dict(), tgt=tgt.state_dict(), pred=pred.state_dict(), epoch=epch + 1, exclude=exclude)
        torch.save(state, BASE / ("ssl_checkpoint_smoke.pt" if smoke else f"ssl_checkpoint{tag}.pt"))
        if save_every and (epch + 1) % save_every == 0 and not smoke:
            torch.save(state, BASE / f"ssl_checkpoint{tag}_ep{epch + 1}.pt")
        (BASE / ("ssl_log_smoke.json" if smoke else f"ssl_log{tag}.json")).write_text(json.dumps(log, indent=1))


# ------------------------------------------------------------------ probe
def embed(enc, X, M, dev, bs=512):
    enc.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            x = torch.from_numpy(X[k:k + bs]).to(dev)
            m = torch.from_numpy(M[k:k + bs]).to(dev)
            h = enc(x, m)
            mm = m[..., None].float()
            out.append(((h * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def probe(smoke=False, ckpt=None):
    e = _load("e24", "24_experiments_v2.py")
    x36 = _load("x36b", "36b_extra_block.py")
    ST = e.STAGES
    dev = torch.device("cpu")
    ckpt_path = Path(ckpt) if ckpt else BASE / ("ssl_checkpoint_smoke.pt" if smoke else "ssl_checkpoint.pt")
    ck = torch.load(ckpt_path, map_location=dev)
    enc = Encoder(); enc.load_state_dict(ck["tgt"])                      # the EMA target encoder is the representation
    D = e.load("afni")
    B = e.blocks(D)
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    W = np.load(BASE / "windows_learned_encoder.npz", allow_pickle=True)   # labelled 90 s windows (53_learned_encoder.py --build)
    key = {(r, int(p)): i for i, (r, p) in enumerate(zip(W["run"].astype(str), W["epoch"]))}
    rows = np.array([key[(r, int(p))] for r, p in zip(rn, ep)])
    if smoke:
        keep = np.isin(s, np.unique(s)[:3]); y, s, rn, ep, rows = y[keep], s[keep], rn[keep], ep[keep], rows[keep]
        B = {k: v[keep] for k, v in B.items()}
    emb = e.wz(embed(enc, W["X"][rows], W["M"][rows], dev), s)
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    sets = {"nb": nb, "nb+emb": np.hstack([nb, emb]), "nb+bold": np.hstack([nb, B["bold"]]), "nb+bold+emb": np.hstack([nb, B["bold"], emb]), "emb": emb, "bold": B["bold"]}
    P = {k: e.loso(v, y, s, ST) for k, v in sets.items()}
    R = {"meta": dict(n_epochs=int(len(y)), n_subjects=int(len(np.unique(s))), d_emb=int(emb.shape[1]), checkpoint_epoch=int(ck["epoch"]), exclude=ck.get("exclude")),
         "auc_ws": {k: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[k][:, i], s).values()) or [np.nan])) for i, st in enumerate(ST)} for k in P},
         "gain": {}}
    for a, b in (("nb+emb", "nb"), ("nb+bold+emb", "nb+bold"), ("nb+bold", "nb")):
        G = x36.gains(P[a], P[b], y, s)
        R["gain"][f"{a} - {b}"] = {"per_stage": {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}, "contrast": x36.contrast(G)}
        g3, gr = G["N3"], G["REM"]
        print(f"  {a} - {b}: N3 {g3['ws_mean']:+.3f} [{g3['ws_ci'][0]:+.3f},{g3['ws_ci'][1]:+.3f}] {g3['n_pos']}/{g3['n_subj']}  "
              f"REM {gr['ws_mean']:+.3f} up {gr['ws_upper95']:+.3f} t {gr['t_upper95']:+.3f}  contrast {R['gain'][f'{a} - {b}']['contrast']['value']:+.3f}", flush=True)
    R["hmm"] = {}
    HP = {}
    for k in ("emb", "bold", "nb+emb", "nb+bold+emb"):
        pred = e.hmm_decode(P[k], y, s, rn, ep)
        HP["hmm__" + k] = pred.astype(str)
        met = e.staging(y, pred, s)
        R["hmm"][k] = dict(kappa=met["kappa"], f1=met["f1"], kappa_ci=e.boot_kappa(y, pred, s) if not smoke else None)
        print(f"  HMM kappa {k:12s} {met['kappa']:.3f}", flush=True)
    print("  emb alone AUC", {st: round(v, 3) for st, v in R["auc_ws"]["emb"].items()}, flush=True)
    suffix = "_smoke" if smoke else (("_" + ckpt_path.stem.replace("ssl_checkpoint", "").strip("_")) if ckpt else "")
    R["meta"]["checkpoint"] = ckpt_path.name
    (BASE / f"results_ssl_probe{suffix}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_ssl_probe{suffix}.npz", y=y, subj=s, run=rn, epoch=ep, **{"proba__" + k: v.astype(np.float32) for k, v in P.items()}, **HP)
    print(f"saved results_ssl_probe{suffix}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--exclude-subject", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run on CPU (6 runs, 2 epochs, 3 probe subjects)")
    ap.add_argument("--save-every", type=int, default=0, help="also keep a checkpoint every K epochs (ssl_checkpoint<tag>_epK.pt)")
    ap.add_argument("--tag", default="", help="suffix for this pretraining run's files, e.g. _long")
    ap.add_argument("--probe-checkpoint", default=None, help="probe this checkpoint file instead of ssl_checkpoint.pt")
    a = ap.parse_args()
    if a.build:
        build(a.smoke)
    if a.pretrain:
        pretrain(epochs=a.epochs, exclude=a.exclude_subject, smoke=a.smoke, save_every=a.save_every, tag=a.tag)
    if a.probe:
        probe(a.smoke, a.probe_checkpoint)


if __name__ == "__main__":
    main()
