"""Regional BOLD level per epoch (AFNI derivative), aligned row for row with features_v2_afni.npz.

The main BOLD block describes each epoch by fluctuation amplitude (s.d.) and network coupling, not by
signal level, the contrast in which PET and fMRI studies of REM report regional activation. This adds,
over the same uncensored frames as the amplitude feature:
  lvl_parcel  mean percent signal change of each of the 200 Schaefer parcels (common parcels)
  lvl_subc    mean percent signal change of the 15 Harvard-Oxford subcortical structures (19b/20b)
PSC is about each parcel's run mean, so a level is relative to the rest of its run.

    source <venv>/bin/activate
    python 22d_level_features.py              # writes features_v2_afni_lvl.npz
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
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    rows = {}
    stems = sorted(set(r for r, _ in want))
    for i, stem in enumerate(stems, 1):
        sub = stem.split("_")[0]
        zc = np.load(BASE / "derivatives" / "afni_ds005127" / sub / f"{stem}_afni.npz")
        zs = np.load(BASE / "derivatives" / "afni_ds005127_subc" / sub / f"{stem}_subc.npz")
        valid = ~zc["censored"]
        T = int(zc["n_vol"])
        cort = f22.psc(zc["ts"].astype(np.float64), valid)[common]
        subc = f22.psc(zs["ts"].astype(np.float64), valid)
        for ep in f22.feat.run_epochs(raw_runs[stem]):
            a, n = ep["start_vol"], ep["n_vol"]
            seg_c, seg_s = cort[:, a:min(T, a + n)], subc[:, a:min(T, a + n)]
            ok = np.isfinite(seg_c).all(0) & np.isfinite(seg_s).all(0)
            if ok.sum() >= 5:
                rows[(stem, ep["epoch"])] = (seg_c[:, ok].mean(1), seg_s[:, ok].mean(1))
            else:
                rows[(stem, ep["epoch"])] = (np.full(len(cort), np.nan), np.full(len(subc), np.nan))
        if i % 25 == 0 or i == len(stems):
            print(f"  [{i}/{len(stems)}]", flush=True)
    lp = np.array([rows[k][0] for k in want], np.float32)
    ls = np.array([rows[k][1] for k in want], np.float32)
    np.savez_compressed(BASE / "features_v2_afni_lvl.npz", run=V["run"], epoch=V["epoch"], lvl_parcel=lp, lvl_subc=ls)
    print("lvl_parcel", lp.shape, "lvl_subc", ls.shape, "NaN rows", float(np.isnan(np.hstack([lp, ls])).any(1).mean()))


if __name__ == "__main__":
    main()
