"""Nested stacked fusion: a learned combination of the non-BOLD and BOLD models (derivative).

Early fusion concatenates the blocks; late fusion multiplies the two posteriors with no learned weights.
Stacking learns the combination, nested so that the held-out subject never informs it: for each outer
subject u, the non-BOLD and BOLD models are refitted leave-one-subject-out over the other 11 subjects to
give out-of-fold probabilities there; a class-balanced multinomial logistic regression is fitted on those
out-of-fold log-probabilities (10 inputs for "both", the 5 non-BOLD ones for the reference), and applied
to u's outer probabilities (predictions_v2_afni.npz, models trained on all 11). The gain is
AUC(stack of both) - AUC(stack of non-BOLD), with 24's increment().

    source <venv>/bin/activate
    python 35_stacked.py                      # writes results_stacked.json
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s = D["stage"], D["subj"]
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all() and (Z["subj"] == s).all()
    Xnb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Xb = B["bold"]
    lg = lambda p: np.log(np.clip(p, 1e-9, 1))
    out_nb, out_both = np.zeros((len(y), 5)), np.zeros((len(y), 5))
    for u in np.unique(s):
        tr, te = s != u, s == u
        inner_nb = e.loso(Xnb[tr], y[tr], s[tr], STAGES)
        inner_b = e.loso(Xb[tr], y[tr], s[tr], STAGES)
        for inp_tr, inp_te, dest in [(lg(inner_nb), lg(Z["proba__nonbold"][te]), out_nb),
                                     (np.hstack([lg(inner_nb), lg(inner_b)]),
                                      np.hstack([lg(Z["proba__nonbold"][te]), lg(Z["proba__bold"][te])]), out_both)]:
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(inp_tr, y[tr])
            dest[te] = clf.predict_proba(inp_te)[:, [list(clf.classes_).index(c) for c in STAGES]]
        print(f"  outer {u} done ({time.time() - t0:.0f} s)", flush=True)
    inc = {st: e.increment((y == st).astype(int), out_both[:, i], out_nb[:, i], s) for i, st in enumerate(STAGES)}
    for st in STAGES:
        inc[st].pop("boot_ws")
    print("  stacked gain " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},{inc[st]['ws_ci'][1]:+.3f}]"
                                       f" up{inc[st]['ws_upper95']:+.3f} {inc[st]['n_pos']}/{inc[st]['n_subj']}"
                                       for st in STAGES), flush=True)
    (BASE / "results_stacked.json").write_text(json.dumps({"increment": inc}, indent=1))
    np.savez_compressed(BASE / "predictions_stacked.npz", y=y, subj=s, stack_nonbold=out_nb, stack_both=out_both)
    print(f"saved results_stacked.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
