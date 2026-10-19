"""Build epoch-aligned features from the parcel timeseries, for sleep-stage decoding.

THE ALIGNMENT, which is the part that silently breaks everything if got wrong:
the scoring files cover wall-clock time including periods when the scanner was not acquiring, so epoch
index does NOT map onto volume index. The MRIon column gives the fraction of each 30 s epoch with
concurrent BOLD (1.0 for most, 0.0 for 971 epochs, fractional for a handful). The cumulative sum of
MRIon converts epoch position into volume position. Verified across 229 runs: cumulative-MRIon gives a
median volumes-implied/volumes-actual ratio of 1.000 and agrees within 5% for 227 runs, against only 82
if epochs are assumed contiguous.

    source <venv>/bin/activate
    python 07_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018

BASE = Path(__file__).parent
DERIV = BASE / "derivatives" / "ds005127"
SCORING = BASE / "data" / "ds005127" / "derivatives" / "sleep_scoring_mr"
OUT = BASE / "features_ds005127.npz"

EPOCH_S = 30.0
SCORE_MAP = {6: "W", 4: "N1", 3: "N2", 2: "N3", 0: "unscorable"}
MIN_COVERAGE = 0.9        # epoch must be at least this fraction scanned
FC_WINDOW_S = 90.0        # Altmann 2016 found ~70 s is the minimum for usable FC


def network_ids(n_parcels=200):
    """Map each Schaefer parcel to its 7-network assignment, for network-level connectivity."""
    labels = fetch_atlas_schaefer_2018(n_rois=n_parcels, resolution_mm=2)["labels"]
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
    if len(labels) == n_parcels + 1:      # some versions include a background label
        labels = labels[1:]
    nets, ids = [], []
    for l in labels:
        parts = l.split("_")
        net = parts[2] if len(parts) > 2 else "unknown"
        if net not in nets:
            nets.append(net)
        ids.append(nets.index(net))
    return np.array(ids), nets


def run_epochs(npz_path):
    """Yield per-epoch features for one run, aligned via cumulative MRIon."""
    stem = npz_path.name.replace("_parcels.npz", "")
    cand = list(SCORING.rglob(stem + ".txt"))
    if not cand:
        return None
    z = np.load(npz_path)
    ts = z["ts"]                              # 200 x T
    tr = float(z["tr"])
    fd, dv = z["fd"], z["dvars"]
    T = ts.shape[1]

    d = pd.read_csv(cand[0], sep="\t")
    on = d["MRIon"].to_numpy(dtype=float)
    # volume index at the START of each epoch = cumulative scanned time before it, divided by TR
    start_vol = np.concatenate([[0.0], np.cumsum(on)[:-1]]) * EPOCH_S / tr
    dur_vol = on * EPOCH_S / tr

    # normalise each parcel within the run: removes scanner scale and drift differences between runs
    good = np.isfinite(ts).all(axis=1)
    tsz = np.full_like(ts, np.nan)
    m = ts[good].mean(axis=1, keepdims=True)
    s = ts[good].std(axis=1, keepdims=True) + 1e-9
    tsz[good] = (ts[good] - m) / s

    out = []
    for i in range(len(d)):
        if on[i] < MIN_COVERAGE:
            continue
        a, b = int(round(start_vol[i])), int(round(start_vol[i] + dur_vol[i]))
        if b > T or b - a < 5:
            continue
        stage = SCORE_MAP.get(int(d["Score"].iloc[i]))
        if int(d["Score"].iloc[i]) == 4 and int(d["REM"].iloc[i]) == 1:
            stage = "REM"
        if stage in (None, "unscorable"):
            continue
        seg = tsz[:, a:b]
        # wider symmetric window for connectivity, since 30 s is too short to estimate it
        half = int(round(FC_WINDOW_S / 2 / tr))
        c0, c1 = max(0, (a + b) // 2 - half), min(T, (a + b) // 2 + half)
        win = tsz[:, c0:c1]
        out.append(dict(stem=stem, epoch=i, start_vol=a, stage=stage, n_vol=b - a,
                        seg=seg, win=win, good=good,
                        fd=float(np.nanmean(fd[a:b])), dvars=float(np.nanmean(dv[a:b])),
                        fd_max=float(np.nanmax(fd[a:b])), dvars_sd=float(np.nanstd(dv[a:b]))))
    return out


def edge_fc(win, common):
    """Parcel-level connectivity on the common-parcel set, Fisher-z.

    The 7-network summary (28 features) discards almost all spatial detail: 84 common parcels give
    3,486 edges. With ~13.6k sleep epochs and L2-regularised LOSO this is a workable width, and it is
    the representation the published fMRI stagers actually use (Tagliazucchi 2012, Altmann 2016).
    """
    w = win[common]
    if w.shape[1] < 8 or not np.isfinite(w).all():
        n = common.sum()
        return np.full(n * (n - 1) // 2, np.nan, dtype=np.float32)
    cc = np.corrcoef(w)
    iu = np.triu_indices(cc.shape[0], 1)
    r = np.clip(cc[iu], -0.999, 0.999)
    return np.arctanh(r).astype(np.float32)


def featurise(ep, net_ids, n_nets):
    """Amplitude per parcel, plus network-by-network connectivity in the wider window."""
    seg, win, good = ep["seg"], ep["win"], ep["good"]
    amp = np.nanstd(seg, axis=1)                       # 200: per-parcel fluctuation amplitude
    mean = np.nanmean(seg, axis=1)                     # 200: per-parcel mean level

    # network-level connectivity: average parcel timeseries within network, then correlate.
    # 200x200 edges would be 19,900 features for 19k epochs; networks give an interpretable 28.
    netts = np.full((n_nets, win.shape[1]), np.nan)
    for k in range(n_nets):
        sel = (net_ids == k) & good
        if sel.sum():
            netts[k] = np.nanmean(win[sel], axis=0)
    if np.isfinite(netts).all() and win.shape[1] > 8:
        cc = np.corrcoef(netts)
    else:
        cc = np.full((n_nets, n_nets), np.nan)
    iu = np.triu_indices(n_nets, 0)
    fc = cc[iu]
    return amp, mean, fc


def main():
    net_ids, nets = network_ids()
    n_nets = len(nets)
    print(f"networks ({n_nets}): {nets}")

    runs = sorted(DERIV.rglob("*_parcels.npz"))
    print(f"{len(runs)} runs")

    # parcels present in every run, computed once so edge features have a fixed width
    common = None
    for r in runs:
        g = np.isfinite(np.load(r)["ts"]).all(axis=1)
        common = g if common is None else (common & g)
    print(f"common parcels across all runs: {common.sum()}/200 "
          f"-> {common.sum()*(common.sum()-1)//2} edges")

    A, M, F, N, E, meta = [], [], [], [], [], []
    for i, r in enumerate(runs, 1):
        eps = run_epochs(r)
        if not eps:
            continue
        for ep in eps:
            amp, mean, fc = featurise(ep, net_ids, n_nets)
            A.append(amp); M.append(mean); F.append(fc); E.append(edge_fc(ep["win"], common))
            N.append([ep["fd"], ep["dvars"], ep["fd_max"], ep["dvars_sd"]])
            meta.append((r.parent.name, ep["stem"], ep["epoch"], ep["stage"], ep["n_vol"]))
        if i % 40 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] epochs so far: {len(A)}", flush=True)

    A = np.array(A, dtype=np.float32); M = np.array(M, dtype=np.float32)
    F = np.array(F, dtype=np.float32); N = np.array(N, dtype=np.float32)
    E = np.array(E, dtype=np.float32)
    md = pd.DataFrame(meta, columns=["subject", "run", "epoch", "stage", "n_vol"])
    print(f"\n{len(md)} usable epochs from {md.subject.nunique()} subjects")
    print(md.stage.value_counts().to_string())
    print(f"\nfeature blocks: amplitude {A.shape}, mean {M.shape}, network-FC {F.shape}, "
          f"edge-FC {E.shape}, nuisance {N.shape}")
    print(f"NaN rate: amp {np.isnan(A).mean():.3f}  fc {np.isnan(F).mean():.3f}")

    np.savez_compressed(OUT, amp=A, mean=M, fc=F, edge=E, nuis=N, common=common,
                        subject=md.subject.values, run=md.run.values,
                        stage=md.stage.values, epoch=md.epoch.values, n_vol=md.n_vol.values,
                        networks=np.array(nets))
    print(f"\nsaved -> {OUT.name} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
