"""Sensitivity: the headline gain on the runs the dataset authors list as best aligned (FAQ whitelist).

The FAQ names 73 runs whose EEG-fMRI alignment the authors found best. Training and testing are both
restricted to those runs, then the non-BOLD and non-BOLD + BOLD models are refitted exactly as in
24_experiments_v2.py and the per-stage within-subject gains recomputed.

    source <venv>/bin/activate
    python 31_whitelist.py                    # writes results_whitelist.json
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def whitelist():
    txt = (BASE / "data" / "ds005127" / "FAQ").read_text()
    return set(re.findall(r"^\s+(\d{8}_\d{4})\s*$", txt, re.M))


def main():
    wl = whitelist()
    R = {"n_whitelist": len(wl)}
    for src in ("afni", "raw"):
        D = e.load(src)
        B = e.blocks(D)
        key = np.array(["_".join(re.search(r"acq-(\d{8})T(\d{4})", r).groups()) for r in D["run"]])
        m = np.isin(key, list(wl))
        y, s = D["stage"][m], D["subj"][m]
        nb = np.hstack([B[b][m] for b in ("motion", "physio", "clock")])
        P0 = e.loso(nb, y, s, STAGES)
        P1 = e.loso(np.hstack([nb, B["bold"][m]]), y, s, STAGES)
        inc = {st: e.increment((y == st).astype(int), P1[:, i], P0[:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R[src] = dict(n_runs=int(len(np.unique(D["run"][m]))), n_epochs=int(m.sum()),
                      n_subjects=int(len(np.unique(s))), increment=inc)
        print(f"{src}: {R[src]['n_runs']} runs, {m.sum()} epochs, {len(np.unique(s))} subjects", flush=True)
        print("  " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},{inc[st]['ws_ci'][1]:+.3f}]"
                              f" {inc[st]['n_pos']}/{inc[st]['n_subj']}" for st in STAGES), flush=True)
    (BASE / "results_whitelist.json").write_text(json.dumps(R, indent=1))
    print("saved results_whitelist.json")


if __name__ == "__main__":
    main()
