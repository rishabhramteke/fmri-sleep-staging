"""Epoch features, version 2, computed identically for two preprocessing streams.

  --source raw   our minimal pipeline (06_preprocess.py: atlas carried to native EPI, no realignment)
  --source rv2   our pipeline, version 2 (06b_raw_v2.py: realigned, despiked, FD-censored; no physiological
                 correction), with RETROICOR-only (*_ric) and RETROICOR + slow-regressor (*_phys) variants and the eye block
  --source afni  the dataset authors' AFNI derivative (20_fetch_afni_deriv.py: despike, slice-wise
                 RETROICOR, slice timing, motion correction + nonlinear TT_N27 warp, censoring,
                 nuisance regression); censored frames are treated as missing

Per epoch (alignment exactly as 07_features.py, cumulative MRIon):
  amp       parcel s.d. over the 30 s epoch, on the parcels present in every subject of the stream
  netfc     the 21 between-network correlations of the 7 Schaefer networks in a 90 s window, network
            means taken over the SAME fixed parcel set (07_features.py used each run's own parcels)
  gs        global-signal mean and s.d. in five windows (30-300 s): a separate block, so that "BOLD
            beyond the global signal" can be tested
  *_hrf     the same features with every window shifted 6 s (2 TR) later, for haemodynamic delay
  raw only:   *_phys   after regressing physiological regressors from every parcel (12_confounds model)
              *_physnull  after regressing the SAME regressors circularly shifted by half a run,
                          which keeps their spectra and dimensionality and destroys their timing
  afni only:  *_acc    after regressing 5 white-matter PCs + CSF mean (aCompCor-style)
              *_gsr    after regressing the global signal
              cens_*   fraction of censored frames in five windows (a motion indicator)

    source <venv>/bin/activate
    python 22_features_v2.py --source raw|afni     # writes features_v2_<source>.npz
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
WINDOWS = (30, 60, 90, 150, 300)
FC_WIN = 90.0
HRF_SHIFT = 2                         # volumes = 6 s


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


feat = _load("feat", "07_features.py")
conf = _load("conf", "12_confounds.py")


def zrun(ts, valid):
    """z-score each parcel within run over valid (uncensored) frames; invalid frames -> NaN."""
    out = np.full(ts.shape, np.nan)
    v = ts[:, valid]
    out[:, valid] = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-9)
    return out


def psc(ts, valid):
    """Percent signal change about each parcel's temporal mean (valid frames); invalid frames -> NaN.

    Within-run z-scoring forces every run to unit variance, which discards between-run differences in
    fluctuation amplitude that track stage (a run spent mostly in N3 fluctuates more). PSC keeps them;
    between-subject scale is still removed later by within-subject z-scoring of the epoch features.
    """
    out = np.full(ts.shape, np.nan)
    v = ts[:, valid]
    mu = v.mean(1, keepdims=True)
    out[:, valid] = 100.0 * (v - mu) / np.where(np.abs(mu) > 1e-6, mu, np.nan)
    return out


def regress(ts, R, valid):
    """Residualise every parcel on [1, trend, R] using valid frames (same units as the input)."""
    T = ts.shape[1]
    X = np.column_stack([np.ones(T), np.linspace(-1, 1, T), R])
    out = np.full(ts.shape, np.nan)
    good = np.isfinite(ts[:, valid]).all(1)
    Y = ts[good][:, valid].T
    beta, *_ = np.linalg.lstsq(X[valid], Y, rcond=None)
    out[np.ix_(good, valid)] = (Y - X[valid] @ beta).T
    return out


def window(c, w, tr, T, shift=0):
    h = w / tr / 2
    return int(max(0, round(c + shift - h))), int(min(T, round(c + shift + h)))


def nstat(x, fn, min_frac=0.6):
    x = x[np.isfinite(x)] if x.ndim == 1 else x
    return fn(x) if x.size else np.nan


def epoch_block(ts, gs, ep, common, net_ids, n_nets, tr, T, shift=0):
    a, n = ep["start_vol"] + shift, ep["n_vol"]
    b = min(T, a + n)
    seg = ts[common, a:b]
    ok = np.isfinite(seg).all(0)
    amp = np.nanstd(seg[:, ok], axis=1) if ok.sum() >= 5 else np.full(common.sum(), np.nan)
    c = (ep["start_vol"] + ep["start_vol"] + n) / 2
    c0, c1 = window(c, FC_WIN, tr, T, shift)
    win = ts[common, c0:c1]
    okw = np.isfinite(win).all(0)
    nets_ok = [k for k in range(n_nets) if (net_ids[common] == k).sum() >= 2]
    iu = np.triu_indices(len(nets_ok), 1)
    if okw.sum() >= 15:
        nets = np.vstack([win[net_ids[common] == k][:, okw].mean(0) for k in nets_ok])
        fc = np.arctanh(np.clip(np.corrcoef(nets)[iu], -0.999, 0.999))
    else:
        fc = np.full(len(iu[0]), np.nan)
    g = []
    for w in WINDOWS:
        w0, w1 = window(c, w, tr, T, shift)
        x = gs[w0:w1]
        x = x[np.isfinite(x)]
        g += [x.mean(), x.std()] if x.size >= 5 else [np.nan, np.nan]
    return amp, fc, np.array(g)


def eye_block(eye, valid, ep, tr, T):
    """Eye signals (mean, PC1, PC2, left-right; 06b_raw_v2.py) per epoch: s.d. and mean |first difference| over the
    epoch's uncensored frames, and s.d. over a 90 s window: 4 x 3 = 12 features."""
    a, n = ep["start_vol"], ep["n_vol"]
    c = a + n / 2
    w0, w1 = window(c, FC_WIN, tr, T)
    out = []
    for x in eye:
        seg = np.where(valid[a:a + n], x[a:a + n], np.nan)
        out += [nstat(seg, np.std), nstat(np.abs(np.diff(seg)), np.mean),
                nstat(np.where(valid[w0:w1], x[w0:w1], np.nan), np.std)]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raw", "afni", "rv2"], required=True)
    args = ap.parse_args()
    net_ids, nets = feat.network_ids()
    n_nets = len(nets)
    F = np.load(BASE / "features_ds005127.npz", allow_pickle=True)
    want = set(F["run"].astype(str))
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    src = {}
    for stem, r in raw_runs.items():
        if stem not in want:
            continue
        if args.source == "raw":
            src[stem] = r
        elif args.source == "rv2":
            p = BASE / "derivatives" / "rv2_ds005127" / r.parent.name / f"{stem}_rv2.npz"
            if p.exists():
                src[stem] = p
        else:
            p = BASE / "derivatives" / "afni_ds005127" / r.parent.name / f"{stem}_afni.npz"
            if p.exists():
                src[stem] = p
    print(f"{args.source}: {len(src)} runs", flush=True)

    # fixed parcel set: present (finite, non-trivial) in every run of the stream
    common = np.ones(200, bool)
    for stem, p in src.items():
        z = np.load(p)
        ts = z["ts"]
        common &= np.isfinite(ts).all(1) & (np.nanstd(ts, 1) > 0)
        if args.source == "afni":
            common &= z["parcel_sizes"] >= 10
    if args.source == "rv2":   # same parcels as the minimal pipeline, so only realignment/despiking/censoring differ
        common &= np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    per_net = [int((net_ids[common] == k).sum()) for k in range(n_nets)]
    n_fc = len([k for k in per_net if k >= 2]) * (len([k for k in per_net if k >= 2]) - 1) // 2
    print(f"common parcels: {common.sum()}/200; per network {dict(zip(nets, per_net))}; "
          f"{n_fc} between-network pairs (networks with >=2 common parcels)", flush=True)

    keys, blocks = [], {}

    def put(name, v):
        blocks.setdefault(name, []).append(v)

    for i, (stem, p) in enumerate(sorted(src.items()), 1):
        eps = feat.run_epochs(raw_runs[stem])
        if not eps:
            continue
        z = np.load(p)
        ts_raw = z["ts"].astype(np.float64)
        T = ts_raw.shape[1]
        if T != np.load(raw_runs[stem])["ts"].shape[1]:
            print(f"  skip {stem}: volume count differs from raw", flush=True)
            continue
        valid = ~z["censored"] if "censored" in z.files else np.ones(T, bool)
        ts = psc(ts_raw, valid)
        ts_z = zrun(ts_raw, valid)
        gs = (z["gs"] if "gs" in z.files else np.nanmean(ts_raw[common], 0)).astype(np.float64)
        gs = np.where(valid, (gs - gs[valid].mean()) / (gs[valid].std() + 1e-9), np.nan)
        variants = {"": ts, "_z": ts_z}
        if args.source in ("raw", "rv2"):
            pf = conf.PHYS / raw_runs[stem].parent.name / f"{stem}_physio.npz"
            Pphys = None
            if pf.exists():
                P = dict(np.load(pf, allow_pickle=True))
                qc, qr = conf.run_quality(P)
                grid, S = conf.physio_signals(P, T, 3.0)
                R1, R2 = conf.retroicor(P, T, 3.0), conf.slow_regressors(grid, S, T, 3.0)
                if (qc >= 0.7 and qr >= 0.7 and R1 is not None and R2 is not None
                        and conf.trigger_consistent(P, T, 3.0)):
                    Pphys = np.hstack([R1, R2])
            if Pphys is not None:
                variants["_phys"] = regress(ts, Pphys, valid)
                if args.source == "raw":
                    variants["_physnull"] = regress(ts, np.roll(Pphys, T // 2, axis=0), valid)
                else:
                    variants["_ric"] = regress(ts, R1, valid)
            if args.source == "rv2":   # 24-parameter motion regression (Friston): realignment parameters, derivatives, squares
                mp = np.load(BASE / "derivatives" / "motion_ds005127" / raw_runs[stem].parent.name / f"{stem}_motion.npz")["params"].astype(np.float64)
                dm = np.vstack([np.zeros((1, 6)), np.diff(mp, axis=0)])
                Rm = np.hstack([mp, dm, mp ** 2, dm ** 2])
                Rm = (Rm - Rm[valid].mean(0)) / (Rm[valid].std(0) + 1e-9)
                variants["_mot"] = regress(ts, Rm, valid)
        else:
            Racc = np.column_stack([z["wm_pcs"].T, z["csf"]])
            Racc = (Racc - Racc[valid].mean(0)) / (Racc[valid].std(0) + 1e-9)
            variants["_acc"] = regress(ts, Racc, valid)
            variants["_gsr"] = regress(ts, np.nan_to_num(gs)[:, None], valid)
        dt_r = float(z["tr"]) if "tr" in z.files else 3.0
        for ep in eps:
            keys.append((stem, ep["epoch"]))
            for tag, x in variants.items():
                amp, fc, g = epoch_block(x, gs, ep, common, net_ids, n_nets, dt_r, T)
                put("amp" + tag, amp); put("netfc" + tag, fc)
                if tag == "":
                    put("gs", g)
                    amp2, fc2, g2 = epoch_block(x, gs, ep, common, net_ids, n_nets, dt_r, T, HRF_SHIFT)
                    put("amp_hrf", amp2); put("netfc_hrf", fc2); put("gs_hrf", g2)
            for tag in {"raw": ["_phys", "_physnull"], "rv2": ["_phys", "_ric"]}.get(args.source, []):
                if tag not in variants:
                    put("amp" + tag, np.full(common.sum(), np.nan)); put("netfc" + tag, np.full(n_fc, np.nan))
            if args.source == "rv2":
                put("eye", eye_block(z["eye"], valid, ep, dt_r, T) if int(z["eye_nvox"]) >= 15
                    else np.full(12, np.nan))
            if args.source in ("afni", "rv2"):
                c = ep["start_vol"] + ep["n_vol"] / 2
                cf = []
                for w in WINDOWS:
                    w0, w1 = window(c, w, dt_r, T)
                    cf.append(1 - valid[w0:w1].mean())
                put("cens", np.array(cf))
        if i % 25 == 0 or i == len(src):
            print(f"  [{i}/{len(src)}] epochs {len(keys)}", flush=True)
    out = {k: np.array(v, np.float32) for k, v in blocks.items()}
    np.savez_compressed(BASE / f"features_v2_{args.source}.npz", run=np.array([k[0] for k in keys]),
                        epoch=np.array([k[1] for k in keys]), common=common, **out)
    print({k: v.shape for k, v in out.items()})
    print({k: round(float(np.isnan(v).any(1).mean()), 3) for k, v in out.items()})


if __name__ == "__main__":
    main()
