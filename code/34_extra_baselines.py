"""Which non-BOLD signal absorbs BOLD's REM information, and is the REM null an overfitting artefact?

Derivative, refitted exactly as in 24_experiments_v2.py:
  * BOLD over elapsed time alone, and over head motion + elapsed time (physiology alone and
    physiology + time are already in results_v2_afni.json)
  * the main comparison with ten-fold stronger regularisation, C = 0.05 instead of 1

    source <venv>/bin/activate
    python 34_extra_baselines.py              # writes results_extra_baselines.json
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def main():
    D = e.load("afni")
    B = e.blocks(D)
    y, s = D["stage"], D["subj"]
    fit = lambda blocks, C=1.0: e.loso(np.hstack([B[b] for b in blocks]), y, s, STAGES, C=C)
    comps = {"clock+bold - clock": (["clock", "bold"], ["clock"], 1.0),
             "motion+clock+bold - motion+clock": (["motion", "clock", "bold"], ["motion", "clock"], 1.0),
             "nonbold+bold - nonbold (C=0.05)": (["motion", "physio", "clock", "bold"], ["motion", "physio", "clock"], 0.05)}
    R = {}
    for name, (a, b, C) in comps.items():
        Pa, Pb = fit(a, C), fit(b, C)
        inc = {st: e.increment((y == st).astype(int), Pa[:, i], Pb[:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R[name] = inc
        print(f"{name:34s} " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},{inc[st]['ws_ci'][1]:+.3f}]"
                                         f" up{inc[st]['ws_upper95']:+.3f} {inc[st]['n_pos']}/{inc[st]['n_subj']}"
                                         for st in ("N3", "REM")), flush=True)
    (BASE / "results_extra_baselines.json").write_text(json.dumps(R, indent=1))
    print("saved results_extra_baselines.json")


if __name__ == "__main__":
    main()
