"""Subcortical BOLD features per epoch (AFNI derivative), aligned row for row with features_v2_afni.npz.

  subc_amp  s.d. of percent signal change over the epoch's uncensored frames, 15 Harvard-Oxford
            structures (20b_fetch_subcortex.py)
  subc_fc   Fisher-z correlation, over the same 90 s window as 22's network FC, between 5 subcortical
            groups (thalamus, brainstem, hippocampus, amygdala, striatum = caudate + putamen + pallidum +
            accumbens; left and right averaged) and the 7 cortical Schaefer networks (common parcels)

    source <venv>/bin/activate
    python 22b_subcortex_features.py          # writes features_v2_afni_subc.npz
"""

import importlib.util
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f22", BASE / "22_features_v2.py")
f22 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f22)
GROUPS = {"thalamus": ["Thalamus"], "brainstem": ["Brain-Stem"], "hippocampus": ["Hippocampus"],
          "amygdala": ["Amygdala"], "striatum": ["Caudate", "Putamen", "Pallidum", "Accumbens"]}


def main():
    V = np.load(BASE / "features_v2_afni.npz", allow_pickle=True)
    want = list(zip(V["run"].astype(str), V["epoch"]))
    common = V["common"].astype(bool)
    net_ids, nets = f22.feat.network_ids()
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    rows = {}
    stems = sorted(set(r for r, _ in want))
    for i, stem in enumerate(stems, 1):
        sub = stem.split("_")[0]
        zc = np.load(BASE / "derivatives" / "afni_ds005127" / sub / f"{stem}_afni.npz")
        zs = np.load(BASE / "derivatives" / "afni_ds005127_subc" / sub / f"{stem}_subc.npz")
        names = [str(n) for n in zs["names"]]
        valid = ~zc["censored"]
        T = int(zc["n_vol"])
        cort = f22.psc(zc["ts"].astype(np.float64), valid)
        subc = f22.psc(zs["ts"].astype(np.float64), valid)
        netm = np.vstack([cort[common & (net_ids == k)].mean(0) for k in range(len(nets))])
        grp = np.vstack([subc[[j for j, n in enumerate(names) if any(n.endswith(g) for g in gs)]].mean(0)
                         for gs in GROUPS.values()])
        for ep in f22.feat.run_epochs(raw_runs[stem]):
            a, n = ep["start_vol"], ep["n_vol"]
            seg = subc[:, a:min(T, a + n)]
            ok = np.isfinite(seg).all(0)
            amp = np.nanstd(seg[:, ok], axis=1) if ok.sum() >= 5 else np.full(len(names), np.nan)
            c = (2 * a + n) / 2
            c0, c1 = f22.window(c, f22.FC_WIN, 3.0, T)
            okw = np.isfinite(netm[:, c0:c1]).all(0) & np.isfinite(grp[:, c0:c1]).all(0)
            if okw.sum() >= 15:
                C = np.corrcoef(np.vstack([grp[:, c0:c1][:, okw], netm[:, c0:c1][:, okw]]))[:len(GROUPS), len(GROUPS):]
                fc = np.arctanh(np.clip(C, -0.999, 0.999)).ravel()
            else:
                fc = np.full(len(GROUPS) * len(nets), np.nan)
            rows[(stem, ep["epoch"])] = (amp, fc)
        if i % 25 == 0 or i == len(stems):
            print(f"  [{i}/{len(stems)}]", flush=True)
    amp = np.array([rows[k][0] for k in want], np.float32)
    fc = np.array([rows[k][1] for k in want], np.float32)
    np.savez_compressed(BASE / "features_v2_afni_subc.npz", run=V["run"], epoch=V["epoch"], subc_amp=amp, subc_fc=fc,
                        names=np.array(names), groups=np.array(list(GROUPS)))
    print("subc_amp", amp.shape, "subc_fc", fc.shape, "NaN rows", float(np.isnan(np.hstack([amp, fc])).any(1).mean()))


if __name__ == "__main__":
    main()
