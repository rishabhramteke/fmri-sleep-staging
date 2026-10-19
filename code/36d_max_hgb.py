"""Gradient-boosted trees on the richest BOLD model the paper tests (exploratory upper envelope).

36b_extra_block.py --block max adds every extra BOLD block (subcortex, multi-window context, regional
level of parcels and structures, global signal) with early, late and stacked fusion. Boosted trees gave
the largest REM gain with subcortex (36c), so this fits them on non-BOLD + cortical BOLD + all extra
blocks, against boosted trees on non-BOLD (predictions_v2_afni.npz), with 36b's statistics.

    source <venv>/bin/activate
    python 36d_max_hgb.py                     # writes results_max_hgb.json, predictions_max_hgb.npz
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("b36", BASE / "36b_extra_block.py")
b36 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b36)
e = b36.e
ST = e.STAGES


def main():
    t0 = time.time()
    D = e.load("afni"); B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    parts = []
    for fname, cols in [("features_v2_afni_subc.npz", ("subc_amp", "subc_fc")), ("features_v2_afni_mw.npz", ("mw_amp", "mw_fc")),
                        ("features_v2_afni_lvl.npz", ("lvl_parcel", "lvl_subc"))]:
        S = np.load(BASE / fname, allow_pickle=True)
        key = {(r, int(t)): i for i, (r, t) in enumerate(zip(S["run"].astype(str), S["epoch"]))}
        rows = np.array([key[(r, int(t))] for r, t in zip(run, ep)])
        parts.append(np.hstack([S[c][rows] for c in cols]))
    X = np.hstack([e.wz(np.hstack(parts), s), B["gs"]])
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"].astype(str) == y).all() and (Z["subj"].astype(str) == s).all()
    P = e.loso(np.hstack([nb, B["bold"], X]), y, s, ST, model="hgb")
    R = {"n_features_extra": int(X.shape[1]), "n_features_total": int(nb.shape[1] + B["bold"].shape[1] + X.shape[1])}
    for name, ref in (("all BOLD over non-BOLD", Z["proba__hgb:nonbold"]), ("extra blocks beyond cortical BOLD", Z["proba__hgb:nonbold+bold"])):
        G = b36.gains(P, ref, y, s)
        R[name] = dict(gain=G, contrast=b36.contrast(G))
        g = G["REM"]
        print(f"  hgb {name:34s} REM {g['ws_mean']:+.4f} [{g['ws_ci'][0]:+.3f},{g['ws_ci'][1]:+.3f}] up t{g['t_upper95']:+.3f} "
              f"{g['n_pos']}/{g['n_subj']} | N3 {G['N3']['ws_mean']:+.3f} | N3>REM {R[name]['contrast']['n_pos']}/{R[name]['contrast']['n']}", flush=True)
    (BASE / "results_max_hgb.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_max_hgb.npz", y=y, subj=s, hgb_max=P)
    print(f"saved results_max_hgb.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
