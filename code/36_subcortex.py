"""Does subcortical BOLD change the REM result? (AFNI derivative, main-analysis epochs.)

The main BOLD block is cortical (Schaefer-200). This adds the 50 subcortical features of
22b_subcortex_features.py (15 structure amplitudes, 5 x 7 subcortico-cortical correlations) and asks,
with the same models and statistics as 24_experiments_v2.py:
  * subcortex alone, per-stage within-subject AUC
  * gain of non-BOLD + cortical + subcortical BOLD over non-BOLD     (all of BOLD over non-BOLD)
  * gain of non-BOLD + cortical + subcortical BOLD over non-BOLD + cortical BOLD   (subcortex beyond cortex)

    source <venv>/bin/activate
    python 36_subcortex.py                    # writes results_subcortex.json
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
    S = np.load(BASE / "features_v2_afni_subc.npz", allow_pickle=True)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(S["run"].astype(str), S["epoch"]))}
    rows = np.array([key[(r, int(t))] for r, t in zip(D["run"], D["epoch"])])
    X = np.hstack([S["subc_amp"][rows], S["subc_fc"][rows]])
    ok = np.isfinite(X).all(1)
    print(f"{ok.mean():.3f} of epochs with complete subcortical features", flush=True)
    X = np.nan_to_num(e.wz(X, s))
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    P = {"subc": e.loso(X, y, s, STAGES), "nonbold": e.loso(nb, y, s, STAGES),
         "nonbold+bold": e.loso(np.hstack([nb, B["bold"]]), y, s, STAGES),
         "nonbold+bold+subc": e.loso(np.hstack([nb, B["bold"], X]), y, s, STAGES),
         "nonbold+subc": e.loso(np.hstack([nb, X]), y, s, STAGES)}
    R = {"n_features": int(X.shape[1]), "frac_complete": float(ok.mean()),
         "auc_ws": {k: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), p[:, i], s).values())))
                        for i, st in enumerate(STAGES)} for k, p in P.items()}, "increment": {}}
    for a, b in [("nonbold+bold+subc", "nonbold"), ("nonbold+bold+subc", "nonbold+bold"), ("nonbold+subc", "nonbold")]:
        inc = {st: e.increment((y == st).astype(int), P[a][:, i], P[b][:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R["increment"][f"{a} - {b}"] = inc
        print(f"  {a} - {b}: " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},{inc[st]['ws_ci'][1]:+.3f}]"
                                          for st in STAGES), flush=True)
    print("  subcortex alone AUC " + " ".join(f"{st}:{R['auc_ws']['subc'][st]:.3f}" for st in STAGES))
    (BASE / "results_subcortex.json").write_text(json.dumps(R, indent=1))
    print("saved results_subcortex.json")


if __name__ == "__main__":
    main()
