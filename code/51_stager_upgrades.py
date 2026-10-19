"""Quick stager upgrades from saved out-of-fold posteriors: does any richer model beat BOLD alone (kappa 0.45)?

Candidates (all leave-one-subject-out posteriors already on disk, main-analysis derivative epochs):
  predictions_v2_afni.npz        LR and boosted-tree posteriors of bold / nonbold+bold, and their average (ensemble)
  predictions_block_max.npz      all BOLD blocks (cortex, global signal, subcortex, multi-window context, level):
                                 early fusion, late fusion, nested stacking
  predictions_strong_physio.npz  BiGRU over each run's epochs (gru_all = stronger non-BOLD + BOLD), LR and trees
Each is scored as argmax (per-epoch) and after HMM decoding with emission P^tau, tau in a sweep (not nested: an upper
bound; the nested choice is in 50_tempered_hmm.py for the main models).

    source <venv>/bin/activate
    python 51_stager_upgrades.py                # writes results_stager_upgrades.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

BASE = Path(__file__).parent
TAUS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
ST = e.STAGES


def score(name, P, y, s, rn, ep, R):
    P = np.clip(P, 1e-9, None)
    P = P / P.sum(1, keepdims=True)
    arg = np.array(ST)[P.argmax(1)]
    row = {"kappa_argmax": float(cohen_kappa_score(y, arg)), "hmm": {}}
    for tau in TAUS:
        pred = e.hmm_decode(P ** tau, y, s, rn, ep)
        row["hmm"][str(tau)] = dict(kappa=float(cohen_kappa_score(y, pred)),
                                    f1=dict(zip(ST, map(float, f1_score(y, pred, labels=ST, average=None, zero_division=0)))))
    best = max(row["hmm"], key=lambda t: row["hmm"][t]["kappa"])
    R[name] = row
    print(f"  {name:36s} argmax {row['kappa_argmax']:.3f}  HMM " + " ".join(f"tau{t}:{v['kappa']:.3f}" for t, v in row["hmm"].items())
          + f"  best REM F1 {row['hmm'][best]['f1']['REM']:.2f} N3 F1 {row['hmm'][best]['f1']['N3']:.2f}", flush=True)


def main():
    D = e.load("afni")
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    R = {"meta": dict(n_epochs=int(len(y)), taus=list(TAUS))}
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all() and (Z["subj"] == s).all(), "predictions_v2_afni rows differ from load('afni')"
    for k in ("bold", "nonbold+bold", "hgb:bold", "hgb:nonbold+bold", "latefusion(nonbold,bold)"):
        score(k, Z["proba__" + k], y, s, rn, ep, R)
    score("ens(lr,hgb) bold", 0.5 * (Z["proba__bold"] + Z["proba__hgb:bold"]), y, s, rn, ep, R)
    score("ens(lr,hgb) nonbold+bold", 0.5 * (Z["proba__nonbold+bold"] + Z["proba__hgb:nonbold+bold"]), y, s, rn, ep, R)
    score("geo(lr,hgb) nonbold+bold", np.sqrt(Z["proba__nonbold+bold"] * Z["proba__hgb:nonbold+bold"]), y, s, rn, ep, R)
    for f, keys in (("predictions_block_max.npz", ("early3", "late2", "late3", "stack1", "stack2", "stack3")),
                    ("predictions_strong_physio.npz", ("lr_all", "hgb_all", "gru_all", "gru_nb", "lr_nb"))):
        W = np.load(BASE / f, allow_pickle=True)
        if len(W["y"]) != len(y) or not ((W["y"] == y).all() and (W["subj"] == s).all()):
            print(f"  {f}: rows differ from load('afni') ({len(W['y'])} vs {len(y)}); argmax kappa only")
            for k in keys:
                if k in W.files:
                    R[f"{f}:{k}"] = {"kappa_argmax": float(cohen_kappa_score(W["y"], np.array(ST)[W[k].argmax(1)]))}
                    print(f"  {f}:{k:12s} argmax {R[f'{f}:{k}']['kappa_argmax']:.3f}")
            continue
        for k in keys:
            if k in W.files:
                score(f"{f.split('_')[1].split('.')[0]}:{k}", W[k], y, s, rn, ep, R)
    (BASE / "results_stager_upgrades.json").write_text(json.dumps(R, indent=1))
    print("saved results_stager_upgrades.json")


if __name__ == "__main__":
    main()
