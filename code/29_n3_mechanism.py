"""Where does the N3 (and the absent REM) gain live? AFNI derivative, main-analysis epochs.

1. Block ablation: gain from adding parcel amplitude only, or network FC only, to the non-BOLD model.
2. Parcel map: per parcel, the within-subject AUC of PSC amplitude alone for N3 vs the other stages
   (and REM vs the others), averaged over eligible subjects, with the sign kept (AUC < 0.5 = amplitude
   lower in that stage). Summarised per Schaefer network; saved for the figure.

    source <venv>/bin/activate
    python 29_n3_mechanism.py                 # writes results_n3_mechanism.json, n3_parcel_map.npz
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
from nilearn import datasets

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def main():
    D = e.load("afni")
    B = e.blocks(D)
    y, s = D["stage"], D["subj"]
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    amp, fc = e.wz(D["amp"], s), e.wz(D["netfc"], s)
    P = {"nonbold": e.loso(nb, y, s, STAGES), "nonbold+amp": e.loso(np.hstack([nb, amp]), y, s, STAGES),
         "nonbold+fc": e.loso(np.hstack([nb, fc]), y, s, STAGES)}
    R = {"increment": {}}
    for k in ("nonbold+amp", "nonbold+fc"):
        inc = {st: e.increment((y == st).astype(int), P[k][:, i], P["nonbold"][:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R["increment"][k] = inc
        print(f"  {k:12s} " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},{inc[st]['ws_ci'][1]:+.3f}]"
                                       for st in STAGES), flush=True)

    # per-parcel univariate within-subject AUC (no fitting: the feature itself is the score)
    maps = {}
    for st in ("N3", "REM"):
        y01 = (y == st).astype(int)
        maps[st] = np.array([np.mean(list(e.ws_auc(y01, amp[:, j], s).values())) for j in range(amp.shape[1])])
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas["labels"]]
    labels = [l for l in labels if l != "Background"]
    nets = np.array([l.split("_")[2] for l in labels])
    R["parcel_auc"] = {st: dict(median=float(np.median(m)), frac_above=float((m > 0.5).mean()),
                                min=float(m.min()), max=float(m.max()),
                                per_network={n: float(m[nets == n].mean()) for n in dict.fromkeys(nets)})
                       for st, m in maps.items()}
    for st in maps:
        print(f"  parcel AUC {st}: {json.dumps(R['parcel_auc'][st])}", flush=True)
    (BASE / "results_n3_mechanism.json").write_text(json.dumps(R, indent=1))
    np.savez(BASE / "n3_parcel_map.npz", N3=maps["N3"], REM=maps["REM"], labels=np.array(labels), nets=nets)
    print("saved results_n3_mechanism.json, n3_parcel_map.npz")


if __name__ == "__main__":
    main()
