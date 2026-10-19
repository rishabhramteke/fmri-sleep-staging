"""Compact multi-window BOLD block (AFNI derivative), aligned row for row with features_v2_afni.npz.

The main BOLD block sees 30 s (amplitude) and 90 s (network FC); the non-BOLD rival sees windows up to
300 s. This block gives BOLD the rival's context at low dimension, so the comparison is not a
context-length artefact:
  mw_amp  s.d. of percent signal change of the 7 network means and the global (common-parcel) mean in
          each of the five non-BOLD windows (30, 60, 90, 150, 300 s): 40 features
  mw_fc   the 21 between-network Fisher-z correlations over 300 s: 21 features

    source <venv>/bin/activate
    python 22c_multiwindow_bold.py            # writes features_v2_afni_mw.npz
"""

import importlib.util
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f22", BASE / "22_features_v2.py")
f22 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f22)


def main():
    V = np.load(BASE / "features_v2_afni.npz", allow_pickle=True)
    want = list(zip(V["run"].astype(str), V["epoch"]))
    common = V["common"].astype(bool)
    net_ids, nets = f22.feat.network_ids()
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    iu = np.triu_indices(len(nets), 1)
    rows = {}
    stems = sorted(set(r for r, _ in want))
    for i, stem in enumerate(stems, 1):
        z = np.load(BASE / "derivatives" / "afni_ds005127" / stem.split("_")[0] / f"{stem}_afni.npz")
        valid = ~z["censored"]
        T = int(z["n_vol"])
        ts = f22.psc(z["ts"].astype(np.float64), valid)
        sig = np.vstack([ts[common & (net_ids == k)].mean(0) for k in range(len(nets))] + [ts[common].mean(0)])
        for ep in f22.feat.run_epochs(raw_runs[stem]):
            c = (2 * ep["start_vol"] + ep["n_vol"]) / 2
            amp = []
            for w in f22.WINDOWS:
                w0, w1 = f22.window(c, w, 3.0, T)
                x = sig[:, w0:w1]
                x = x[:, np.isfinite(x).all(0)]
                amp += list(x.std(1)) if x.shape[1] >= 5 else [np.nan] * sig.shape[0]
            w0, w1 = f22.window(c, 300.0, 3.0, T)
            x = sig[:len(nets), w0:w1]
            x = x[:, np.isfinite(x).all(0)]
            fc = (np.arctanh(np.clip(np.corrcoef(x)[iu], -0.999, 0.999)) if x.shape[1] >= 15
                  else np.full(len(iu[0]), np.nan))
            rows[(stem, ep["epoch"])] = (np.array(amp), fc)
        if i % 25 == 0 or i == len(stems):
            print(f"  [{i}/{len(stems)}]", flush=True)
    amp = np.array([rows[k][0] for k in want], np.float32)
    fc = np.array([rows[k][1] for k in want], np.float32)
    np.savez_compressed(BASE / "features_v2_afni_mw.npz", run=V["run"], epoch=V["epoch"], mw_amp=amp, mw_fc=fc)
    print("mw_amp", amp.shape, "mw_fc", fc.shape, "NaN rows", float(np.isnan(np.hstack([amp, fc])).any(1).mean()))


if __name__ == "__main__":
    main()
