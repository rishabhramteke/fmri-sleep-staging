"""Checks asked by the panel on draft 14 (post hoc, exploratory).

  coverage  parcel coverage in the derivative: in-mask voxels per run relative to the atlas parcel in TT_N27; the
            main gains with amplitude restricted to parcels covered >= 80% in every run (network FC unchanged)
  bands     the N3 amplitude signature by frequency band: per-parcel within-subject N3 AUC of the epoch s.d. of the
            0.01-0.1 Hz and 0.1-0.16 Hz band-passed derivative series
  tsnr      Spearman correlation across parcels between the N3 amplitude AUC and the parcel's temporal SNR

    source <venv>/bin/activate
    python 48_panel_checks3.py                 # writes results_panel_checks3.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


x = _load("x36b", "36b_extra_block.py")
e = x.e
f22 = _load("f22", "22_features_v2.py")
feat = f22.feat
ST = e.STAGES


def main():
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    Pnb = Z["proba__nonbold"]
    common = np.load(BASE / "features_v2_afni.npz", allow_pickle=True)["common"].astype(bool)
    lab = np.load(BASE / "derivatives" / "afni_ds005127" / "atlas_tt.npz", allow_pickle=True)
    key = [k for k in lab.files if "label" in k or "atlas" in k][0]
    full = np.bincount(np.asarray(lab[key]).astype(int).ravel(), minlength=201)[1:201].astype(float)
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    cov, tsnr = [], []
    band_feats = {}
    sos_lo = butter(4, [0.01, 0.1], btype="band", fs=1 / 3.0, output="sos")
    sos_hi = butter(4, [0.1, 0.16], btype="band", fs=1 / 3.0, output="sos")
    used = set(run)
    for f in sorted((BASE / "derivatives" / "afni_ds005127").glob("sub-*/*_afni.npz")):
        stem = f.name.replace("_afni.npz", "")
        if stem not in used:
            continue
        z = np.load(f)
        cov.append(z["parcel_sizes"] / np.maximum(full, 1))
        ts = z["ts"].astype(np.float64)
        valid = ~z["censored"].astype(bool)
        mu, sd = ts[:, valid].mean(1), ts[:, valid].std(1)
        tsnr.append(np.where(sd > 0, np.abs(mu) / (sd + 1e-9), np.nan))
        psc = f22.psc(ts, valid)
        filled = np.where(np.isfinite(psc), psc, 0.0)
        lo, hi = sosfiltfilt(sos_lo, filled, axis=1), sosfiltfilt(sos_hi, filled, axis=1)
        for k in feat.run_epochs(raw_runs[stem]) or []:
            a, n = k["start_vol"], k["n_vol"]
            okf = valid[a:a + n]
            if okf.sum() >= 5:
                band_feats[(stem, k["epoch"])] = (lo[:, a:a + n][:, okf].std(1), hi[:, a:a + n][:, okf].std(1))
    cov, tsnr = np.array(cov), np.nanmean(np.array(tsnr), 0)
    mincov = cov.min(0)
    R = {"coverage": dict(n_parcels=200, n_ge80_all_runs=int((mincov >= 0.8).sum()), median_min_cov=float(np.median(mincov)))}
    # main gains with amplitude restricted to well-covered parcels
    keep = (mincov >= 0.8)[common]
    n_amp = int(common.sum())
    Xb = np.hstack([B["bold"][:, :n_amp][:, keep], B["bold"][:, n_amp:]])
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Pa = e.loso(np.hstack([nb, Xb]), y, s, ST)
    G = x.gains(Pa, Pnb, y, s)
    R["coverage"]["gain"] = {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}
    R["coverage"]["contrast"] = x.contrast(G)
    R["coverage"]["n_amp_kept"] = int(keep.sum())
    print(f"  well-covered parcels: {keep.sum()}/{n_amp} amplitudes; N3 {G['N3']['ws_mean']:+.3f} REM {G['REM']['ws_mean']:+.3f} "
          f"t up {G['REM']['t_upper95']:+.3f} N3>REM {R['coverage']['contrast']['n_pos']}/{R['coverage']['contrast']['n']}", flush=True)
    # band-limited N3 amplitude AUC per parcel
    m = np.array([(r, int(t)) in band_feats for r, t in zip(run, ep)])
    R["bands"] = {}
    for bi, name in ((0, "0.01-0.1 Hz"), (1, "0.1-0.16 Hz")):
        A = np.array([band_feats[(r, int(t))][bi] for r, t, mm in zip(run, ep, m) if mm])
        yy, ss = y[m], s[m]
        per = []
        for u in np.unique(ss):
            mu_ = ss == u
            pos = yy[mu_] == "N3"
            if pos.sum() >= 20 and (~pos).sum() >= 20:
                per.append([e.auc(pos, A[mu_][:, j]) for j in range(A.shape[1])])
        per = np.nanmean(np.array(per), 0)
        R["bands"][name] = dict(frac_below=float(np.mean(per < 0.5)), median_auc=float(np.median(per)))
    print("  N3 amplitude AUC by band:", R["bands"], flush=True)
    # tSNR vs the N3 amplitude AUC map
    NP = np.load(BASE / "n3_parcel_map.npz", allow_pickle=True)
    rho, p = spearmanr(NP["N3"], tsnr[:len(NP["N3"])], nan_policy="omit")
    R["tsnr"] = dict(rho=float(rho), p=float(p))
    print(f"  N3 AUC vs tSNR across parcels: rho {rho:.2f} (p {p:.2g})", flush=True)
    (BASE / "results_panel_checks3.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
