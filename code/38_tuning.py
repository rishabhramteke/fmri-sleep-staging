"""Nested tuning of the classifiers (post hoc, answers "no setting was tuned at all").

Outer loop: leave one subject out, as everywhere in the paper. Inner loop, on the training subjects only:
subject-grouped 4-fold CV over a small grid; the setting with the highest inner score is refitted on all training
subjects and predicts the held-out subject. The inner score is stage-agnostic: the mean over the five stages of the
within-subject one-vs-rest AUC (the paper's statistic), so the tuning cannot target REM. Each model is tuned on its
own (the non-BOLD reference and the model with BOLD may pick different settings).

  lr   L2 multinomial logistic regression, C in {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10}
  hgb  gradient-boosted trees, (learning rate, leaves, iterations) in {(0.05, 15, 300) = the paper's fixed setting,
       (0.1, 7, 150), (0.05, 31, 300)}
  feature sets: main (non-BOLD vs non-BOLD + the 221 cortical features) and max (vs non-BOLD + all 557 BOLD features)

    source <venv>/bin/activate
    OMP_NUM_THREADS=2 python 38_tuning.py --model lr  --set main   # writes results_tuned_lr_main.json
    OMP_NUM_THREADS=2 python 38_tuning.py --model hgb --set max
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e = _load("e24", "24_experiments_v2.py")
x = _load("x36b", "36b_extra_block.py")
ST = e.STAGES
GRID = {"lr": [dict(C=c) for c in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)],
        "hgb": [dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=300),
                dict(learning_rate=0.1, max_leaf_nodes=7, max_iter=150),
                dict(learning_rate=0.05, max_leaf_nodes=31, max_iter=300)]}


def make(model, hp):
    if model == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=hp["C"], class_weight="balanced"))
    return HistGradientBoostingClassifier(l2_regularization=1.0, class_weight="balanced", random_state=0, **hp)


def proba(clf, X):
    return clf.predict_proba(X)[:, [list(clf.classes_).index(c) for c in ST]]


def score(P, y, s):
    """Stage-agnostic inner score: mean over stages of the mean within-subject one-vs-rest AUC."""
    v = []
    for i, st in enumerate(ST):
        w = e.ws_auc((y == st).astype(int), P[:, i], s)
        if w:
            v.append(np.mean(list(w.values())))
    return float(np.mean(v))


def outer_fold(X, y, s, u, model):
    tr, te = s != u, s == u
    Xt, yt, st_ = X[tr], y[tr], s[tr]
    scores = []
    for hp in GRID[model]:
        P = np.zeros((tr.sum(), len(ST)))
        for a, b in GroupKFold(n_splits=4).split(Xt, yt, st_):
            P[b] = proba(make(model, hp).fit(Xt[a], yt[a]), Xt[b])
        scores.append(score(P, yt, st_))
    best = int(np.argmax(scores))
    clf = make(model, GRID[model][best]).fit(Xt, yt)
    return u, te, proba(clf, X[te]), best, scores


def nested(X, y, s, model, n_jobs):
    P = np.zeros((len(y), len(ST)))
    picks = {}
    for u, te, p, best, sc in Parallel(n_jobs=n_jobs)(delayed(outer_fold)(X, y, s, u, model) for u in np.unique(s)):
        P[te] = p
        picks[u] = dict(choice=GRID[model][best], inner_scores=sc)
    return P, picks


def extra_blocks(D, B):
    run, ep, s = D["run"], D["epoch"], D["subj"]
    parts = []
    for fname, cols in [("features_v2_afni_subc.npz", ("subc_amp", "subc_fc")),
                        ("features_v2_afni_mw.npz", ("mw_amp", "mw_fc")),
                        ("features_v2_afni_lvl.npz", ("lvl_parcel", "lvl_subc"))]:
        S = np.load(BASE / fname, allow_pickle=True)
        key = {(r, int(t)): i for i, (r, t) in enumerate(zip(S["run"].astype(str), S["epoch"]))}
        rows = np.array([key[(r, int(t))] for r, t in zip(run, ep)])
        parts.append(np.hstack([S[c][rows] for c in cols]))
    return np.hstack([e.wz(np.hstack(parts), s), B["gs"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "hgb"], required=True)
    ap.add_argument("--set", choices=["main", "max"], required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s = D["stage"], D["subj"]
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    bold = B["bold"] if a.set == "main" else np.hstack([B["bold"], extra_blocks(D, B)])
    assert np.isfinite(bold).all()
    Pnb, pick_nb = nested(nb, y, s, a.model, a.jobs)
    print(f"  non-BOLD tuned ({time.time() - t0:.0f} s): {[v['choice'] for v in pick_nb.values()]}", flush=True)
    Pall, pick_all = nested(np.hstack([nb, bold]), y, s, a.model, a.jobs)
    print(f"  with BOLD tuned ({time.time() - t0:.0f} s): {[v['choice'] for v in pick_all.values()]}", flush=True)
    G = x.gains(Pall, Pnb, y, s)
    R = dict(model=a.model, set=a.set, n_bold=int(bold.shape[1]), grid=GRID[a.model],
             picks_nonbold=pick_nb, picks_with_bold=pick_all,
             gain={st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}, contrast=x.contrast(G),
             auc_ws={nm: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[:, i], s).values())))
                          for i, st in enumerate(ST)} for nm, P in (("nonbold", Pnb), ("with_bold", Pall))})
    for st in ST:
        g = G[st]
        print(f"  {st}: {g['ws_mean']:+.3f} [{g['ws_ci'][0]:+.3f},{g['ws_ci'][1]:+.3f}] up b{g['ws_upper95']:+.3f}"
              f"/t{g['t_upper95']:+.3f} {g['n_pos']}/{g['n_subj']}", flush=True)
    print(f"  N3-REM {R['contrast']}", flush=True)
    out = BASE / f"results_tuned_{a.model}_{a.set}.json"
    out.write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_tuned_{a.model}_{a.set}.npz", y=y, subj=s, nonbold=Pnb, with_bold=Pall)
    print(f"saved {out.name} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
