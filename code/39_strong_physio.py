"""A stronger non-BOLD baseline (post hoc): HRV-literature features and a sequence model on both sides.

The review's main objection: the N3 gain is measured against window summaries of physiology in a linear model, far
below what cardiorespiratory sleep stagers achieve (spectral HRV, breathing regularity, recurrent sequence models).
Here the non-BOLD model gets the 18 features of 12b_physio_spectral.py on top of its 92, and every estimator is given
to BOTH sides of the comparison, so the gain of BOLD is always measured within one model family:

  lr   early fusion, as the paper's primary estimator
  hgb  gradient-boosted trees, the paper's fixed setting
  gru  a bidirectional GRU over each run's epoch sequence (input dropout 0.2, linear 48 + GELU, BiGRU 48, dropout
       0.3), class-weighted cross-entropy, AdamW; early stopping on two held-out TRAINING subjects by the
       stage-agnostic mean within-subject AUC; three seeds averaged. Fixed in advance of seeing any GRU result.

Also reports the cardiorespiratory-only GRU (physiology + spectral, no motion, no time) with five- and four-class
(W, N1+N2, N3, REM) kappa from per-epoch argmax, the form in which HRV stagers report.

    source <venv>/bin/activate
    python 39_strong_physio.py                # writes results_strong_physio.json, predictions_strong_physio.npz
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e = _load("e24", "24_experiments_v2.py")
x = _load("x36b", "36b_extra_block.py")
e.N_JOBS = 6
ST = e.STAGES
torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, len(ST)))

    def forward(self, xx):
        return self.out(self.gru(self.inp(xx))[0])


def sequences(run, ep):
    return [np.flatnonzero(run == r)[np.argsort(ep[run == r])] for r in np.unique(run)]


def pad(X, seqs):
    L = max(len(q) for q in seqs)
    xb = np.zeros((len(seqs), L, X.shape[1]), np.float32)
    mb = np.zeros((len(seqs), L), bool)
    for i, q in enumerate(seqs):
        xb[i, :len(q)] = X[q]
        mb[i, :len(q)] = True
    return torch.from_numpy(xb), torch.from_numpy(mb)


def predict(net, X, seqs):
    net.eval()
    P = np.zeros((len(X), len(ST)), np.float32)
    with torch.no_grad():
        for k in range(0, len(seqs), 32):
            qs = seqs[k:k + 32]
            xb, mb = pad(X, qs)
            p = torch.softmax(net(xb), -1).numpy()
            for i, q in enumerate(qs):
                P[q] = p[i, :len(q)]
    return P


def score(P, y, s):
    return float(np.mean([np.mean(list(e.ws_auc((y == st).astype(int), P[:, i], s).values())) for i, st in enumerate(ST)]))


def train_gru(X, yi, s, run, ep, tr_mask, seed, max_passes=60, patience=10):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    subs = np.unique(s[tr_mask])
    val_subs = rng.choice(subs, 2, replace=False)
    fit = tr_mask & ~np.isin(s, val_subs)
    val = tr_mask & np.isin(s, val_subs)
    seq_fit = [q for q in sequences(run, ep) if fit[q[0]]]
    seq_val = [q for q in sequences(run, ep) if val[q[0]]]
    counts = np.bincount(yi[fit], minlength=len(ST)).astype(np.float32)
    w = torch.tensor(counts.sum() / (len(ST) * np.maximum(counts, 1)))
    net = Net(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=w, reduction="none")
    Y = torch.from_numpy(yi.astype(np.int64))
    best, best_state, bad = -1, None, 0
    yv, sv = np.array(ST)[yi[val]], s[val]
    for _ in range(max_passes):
        net.train()
        order = rng.permutation(len(seq_fit))
        for k in range(0, len(order), 16):
            qs = [seq_fit[j] for j in order[k:k + 16]]
            xb, mb = pad(X, qs)
            yb = torch.zeros(mb.shape, dtype=torch.int64)
            for i, q in enumerate(qs):
                yb[i, :len(q)] = Y[q]
            out = net(xb)
            l = (lossf(out.reshape(-1, len(ST)), yb.reshape(-1)) * mb.reshape(-1)).sum() / mb.sum()
            opt.zero_grad(); l.backward(); opt.step()
        Pv = predict(net, X, seq_val)[val]
        sc = score(Pv, yv, sv)
        if sc > best:
            best, bad = sc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


def loso_gru(X, y, s, run, ep, seeds=(0, 1, 2)):
    yi = np.array([ST.index(v) for v in y])
    P = np.zeros((len(y), len(ST)))
    seqs = sequences(run, ep)
    for u in np.unique(s):
        te = s == u
        seq_te = [q for q in seqs if te[q[0]]]
        for sd in seeds:
            net = train_gru(X, yi, s, run, ep, ~te, seed=1000 * sd + int(u.split("-")[1]))
            P[te] += predict(net, X, seq_te)[te] / len(seeds)
    return P


def kappas(P, y):
    pred = np.array(ST)[P.argmax(1)]
    four = lambda v: np.where(np.isin(v, ["N1", "N2"]), "L", v)
    P4 = np.column_stack([P[:, 0], P[:, 1] + P[:, 2], P[:, 3], P[:, 4]])
    pred4 = np.array(["W", "L", "N3", "REM"])[P4.argmax(1)]
    return dict(kappa5=float(cohen_kappa_score(y, pred)), kappa4=float(cohen_kappa_score(four(y), pred4)))


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    F = np.load(BASE / "features_physio_spectral.npz", allow_pickle=True)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(F["run"].astype(str), F["epoch"]))}
    rows = np.array([key[(r, int(t))] for r, t in zip(run, ep)])
    spec_raw = F["X"][rows].astype(np.float64)
    nan_frac = float(np.isnan(spec_raw).any(1).mean())
    spec = np.nan_to_num(e.wz(spec_raw, s))          # within-subject z; a missing window sits at the subject mean
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    nbs = np.hstack([nb, spec])
    bold = B["bold"]
    R = dict(n_spectral=int(spec.shape[1]), spectral_rows_with_nan=nan_frac, models={})
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all()
    P = {}
    # linear, early fusion
    P["lr_nb"], P["lr_all"] = e.loso(nbs, y, s, ST), e.loso(np.hstack([nbs, bold]), y, s, ST)
    print(f"  lr done ({time.time() - t0:.0f} s)", flush=True)
    P["gru_nb"] = loso_gru(nbs, y, s, run, ep)
    print(f"  gru non-BOLD done ({time.time() - t0:.0f} s)", flush=True)
    P["gru_all"] = loso_gru(np.hstack([nbs, bold]), y, s, run, ep)
    print(f"  gru with BOLD done ({time.time() - t0:.0f} s)", flush=True)
    phys = np.hstack([B["physio"], spec])
    P["gru_phys"] = loso_gru(phys, y, s, run, ep)
    print(f"  gru physiology only done ({time.time() - t0:.0f} s)", flush=True)
    P["lr_phys_spec"] = e.loso(phys, y, s, ST)
    P["hgb_nb"], P["hgb_all"] = e.loso(nbs, y, s, ST, model="hgb"), e.loso(np.hstack([nbs, bold]), y, s, ST, model="hgb")
    print(f"  hgb done ({time.time() - t0:.0f} s)", flush=True)
    for fam in ("lr", "hgb", "gru"):
        G = x.gains(P[f"{fam}_all"], P[f"{fam}_nb"], y, s)
        R["models"][fam] = dict(gain={st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST},
                                contrast=x.contrast(G),
                                auc_ws_nonbold={st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[f"{fam}_nb"][:, i], s).values())))
                                                for i, st in enumerate(ST)},
                                kappa_nonbold=kappas(P[f"{fam}_nb"], y), kappa_with_bold=kappas(P[f"{fam}_all"], y))
        g = G
        print(f"  {fam}: N3 {g['N3']['ws_mean']:+.3f} [{g['N3']['ws_ci'][0]:+.3f},{g['N3']['ws_ci'][1]:+.3f}] "
              f"{g['N3']['n_pos']}/{g['N3']['n_subj']}  REM {g['REM']['ws_mean']:+.3f} [{g['REM']['ws_ci'][0]:+.3f},"
              f"{g['REM']['ws_ci'][1]:+.3f}] up b{g['REM']['ws_upper95']:+.3f}/t{g['REM']['t_upper95']:+.3f} "
              f"{g['REM']['n_pos']}/{g['REM']['n_subj']}  N3>REM {R['models'][fam]['contrast']['n_pos']}/"
              f"{R['models'][fam]['contrast']['n']}  nonBOLD AUC N3 {R['models'][fam]['auc_ws_nonbold']['N3']:.3f} "
              f"REM {R['models'][fam]['auc_ws_nonbold']['REM']:.3f}  kappa {R['models'][fam]['kappa_nonbold']} -> "
              f"{R['models'][fam]['kappa_with_bold']}", flush=True)
    R["baseline_old_auc_ws_nonbold"] = {st: float(np.mean(list(e.ws_auc((y == st).astype(int), Z["proba__nonbold"][:, i], s).values())))
                                        for i, st in enumerate(ST)}
    R["cardioresp_only"] = dict(gru=kappas(P["gru_phys"], y), lr_spec=kappas(P["lr_phys_spec"], y),
                                lr_old=kappas(Z["proba__physio"], y),
                                auc_ws_gru={st: float(np.mean(list(e.ws_auc((y == st).astype(int), P["gru_phys"][:, i], s).values())))
                                            for i, st in enumerate(ST)})
    print(f"  cardiorespiratory only: {R['cardioresp_only']}", flush=True)
    (BASE / "results_strong_physio.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_strong_physio.npz", y=y, subj=s, **P)
    print(f"saved results_strong_physio.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
