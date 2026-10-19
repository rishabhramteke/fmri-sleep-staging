"""The full baseline ladder for 5-class sleep staging from fMRI, all under leave-one-subject-out.

The point of the ladder is to isolate what the brain features actually buy. Several of these baselines
use no imaging at all, and in this dataset they are strong: sleep is highly autocorrelated and REM is
back-loaded in the night, so persistence and clock-position alone go a long way. A brain model that
does not clear them is not a result.

Ladder:
  1  majority class / stratified random
  2  persistence (previous epoch's stage)
  3  clock-only (position in the night, from training-subject marginals)
  4  prior-only Viterbi (transitions + priors, no features at all)
  5  motion-only, multi-window
  6  motion + clock
  7  Tagliazucchi 2012 style: 60 s window network FC, one-vs-one linear SVM
  8  Altmann 2016 style: 70 s window parcel-pair FC, linear SVM, extended to 5 classes with REM
  9  ours: within-subject z amplitude + network FC
 10  ours + HMM smoothing (transition matrix from training subjects)
 11  ours + motion
 12  brain-shuffled control (must collapse to the prior-only row)

    source <venv>/bin/activate
    python 09_baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

BASE = Path(__file__).parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
RNG = np.random.default_rng(0)


def load():
    F = np.load(BASE / "features_ds005127.npz", allow_pickle=True)
    d = dict(amp=F["amp"], fc=np.nan_to_num(F["fc"]), edge=np.nan_to_num(F["edge"]),
             nuis=np.nan_to_num(F["nuis"]), subj=F["subject"].astype(str),
             run=F["run"].astype(str), stage=F["stage"].astype(str), epoch=F["epoch"])
    ok = np.ones(d["amp"].shape[1], bool)
    for s in np.unique(d["subj"]):
        ok &= np.isfinite(d["amp"][d["subj"] == s]).all(axis=0)
    d["amp84"] = np.nan_to_num(d["amp"][:, ok])
    # night position: order runs within subject+session by acquisition timestamp in the filename,
    # then accumulate epochs. This is "time since the night started", not clock time.
    night, pos = [], []
    for s in np.unique(d["subj"]):
        m = d["subj"] == s
        for ses in ["ses-1", "ses-2"]:
            sel = m & np.array([ses in r for r in d["run"]])
            if not sel.any():
                continue
            runs = sorted(set(d["run"][sel]), key=lambda r: r.split("acq-")[-1])
            base = 0
            for r in runs:
                rm = sel & (d["run"] == r)
                order = np.argsort(d["epoch"][rm])
                p = np.empty(rm.sum())
                p[order] = np.arange(rm.sum()) + base
                idx = np.flatnonzero(rm)
                night += list(idx)
                pos += list(p)
                base += rm.sum()
    npos = np.full(len(d["stage"]), np.nan)
    npos[np.array(night)] = np.array(pos)
    # fraction through that night, which is what carries the REM information
    d["night_frac"] = np.zeros(len(npos))
    for s in np.unique(d["subj"]):
        m = d["subj"] == s
        mx = np.nanmax(npos[m]) or 1
        d["night_frac"][m] = npos[m] / mx
    return d


def wz(X, subj):
    X = X.copy().astype(np.float32)
    for s in np.unique(subj):
        m = subj == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return X


def transition_matrix(stage, run, eps=1.0):
    """5x5 transition counts from consecutive epochs within a run, Laplace-smoothed."""
    T = np.full((5, 5), eps)
    for r in np.unique(run):
        m = run == r
        s = stage[m]
        for a, b in zip(s[:-1], s[1:]):
            T[STAGES.index(a), STAGES.index(b)] += 1
    return T / T.sum(axis=1, keepdims=True)


def viterbi(logemit, logT, logpi):
    n, k = logemit.shape
    dp = np.full((n, k), -np.inf)
    bk = np.zeros((n, k), int)
    dp[0] = logpi + logemit[0]
    for t in range(1, n):
        sc = dp[t - 1][:, None] + logT
        bk[t] = sc.argmax(0)
        dp[t] = sc.max(0) + logemit[t]
    path = [dp[-1].argmax()]
    for t in range(n - 1, 0, -1):
        path.append(bk[t, path[-1]])
    return np.array(path[::-1])


def report(name, true, pred, rows):
    m = pred != None  # noqa: E711
    acc = (pred[m] == true[m]).mean()
    kap = cohen_kappa_score(true[m], pred[m])
    f1 = f1_score(true[m], pred[m], average="macro", labels=STAGES, zero_division=0)
    frem = f1_score(true[m], pred[m], labels=["REM"], average="macro", zero_division=0)
    rows.append((name, acc, kap, f1, frem))
    print(f"  {name:38s} {acc:7.3f} {kap:7.3f} {f1:7.3f} {frem:7.3f}", flush=True)


def loso_predict(X, y, subj, model="lr"):
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), 5))
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        if model == "svm":
            clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight="balanced",
                                                            max_iter=5000, dual="auto"))
            clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
            df = clf.decision_function(X[te])
            e = np.exp(df - df.max(1, keepdims=True)); proba[te] = e / e.sum(1, keepdims=True)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                                                                     class_weight="balanced"))
            clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
            p = clf.predict_proba(X[te])
            cols = [list(clf.classes_).index(s_) for s_ in STAGES]
            proba[te] = p[:, cols]
    return pred, proba


def smooth_viterbi(proba, stage, subj, run):
    """Decode each run with a transition matrix estimated on the TRAINING subjects only."""
    out = np.empty(len(stage), dtype=object)
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        T = transition_matrix(stage[tr], run[tr])
        pi = np.array([(stage[tr] == x).mean() for x in STAGES])
        for r in np.unique(run[te]):
            m = run == r
            idx = np.flatnonzero(m)
            p = np.clip(proba[m], 1e-9, None)
            path = viterbi(np.log(p), np.log(T), np.log(pi + 1e-9))
            out[idx] = np.array(STAGES)[path]
    return out


def main():
    d = load()
    stage, subj, run = d["stage"], d["subj"], d["run"]
    A, N, U = wz(d["amp84"], subj), wz(d["fc"], subj), wz(d["nuis"], subj)
    BRAIN = np.hstack([A, N])
    CLOCK = np.column_stack([d["night_frac"], d["night_frac"] ** 2])
    print(f"{len(stage)} epochs, {len(np.unique(subj))} subjects, {len(np.unique(run))} runs")
    print(f"\n  {'baseline':38s} {'acc':>7s} {'kappa':>7s} {'macroF1':>7s} {'F1 REM':>7s}")
    rows = []

    # 1 trivial
    report("1 majority class (N2)", stage, np.array(["N2"] * len(stage), dtype=object), rows)
    prior = np.array([(stage == s).mean() for s in STAGES])
    rnd = RNG.choice(STAGES, size=len(stage), p=prior)
    report("1b stratified random", stage, rnd.astype(object), rows)

    # 2 persistence
    per = np.empty(len(stage), dtype=object)
    for r in np.unique(run):
        m = np.flatnonzero(run == r)
        o = m[np.argsort(d["epoch"][m])]
        per[o[1:]] = stage[o[:-1]]; per[o[0]] = stage[o[0]]
    report("2 persistence (previous epoch)", stage, per, rows)

    # 3-6 no-imaging models
    for name, X in [("3 clock only (night position)", CLOCK),
                    ("5 motion only (multi-window proxy)", U),
                    ("6 motion + clock", np.hstack([U, CLOCK]))]:
        p, _ = loso_predict(X, stage, subj)
        report(name, stage, p, rows)

    # 4 prior-only Viterbi: flat emissions, transitions do all the work
    flat = np.tile(prior, (len(stage), 1))
    report("4 prior-only Viterbi (no features)", stage,
           smooth_viterbi(flat, stage, subj, run), rows)

    # 7-8 published reimplementations
    p, _ = loso_predict(N, stage, subj, model="svm")
    report("7 Tagliazucchi-style (netFC, SVM)", stage, p, rows)
    p, _ = loso_predict(wz(d["edge"], subj), stage, subj, model="svm")
    report("8 Altmann-style (parcel-pair FC, SVM)", stage, p, rows)

    # 9-11 ours
    p9, pr9 = loso_predict(BRAIN, stage, subj)
    report("9 ours: amp + netFC", stage, p9, rows)
    report("10 ours + Viterbi smoothing", stage, smooth_viterbi(pr9, stage, subj, run), rows)
    p11, pr11 = loso_predict(np.hstack([BRAIN, U]), stage, subj)
    report("11 ours + motion", stage, p11, rows)

    # 12 control: shuffle brain features across epochs within subject
    Bs = BRAIN.copy()
    for s in np.unique(subj):
        m = np.flatnonzero(subj == s)
        Bs[m] = Bs[RNG.permutation(m)]
    p, _ = loso_predict(Bs, stage, subj)
    report("12 brain-shuffled control", stage, p, rows)

    pd.DataFrame(rows, columns=["baseline", "acc", "kappa", "macroF1", "F1_REM"]).to_csv(
        BASE / "results_baselines.csv", index=False)
    print(f"\nsaved -> results_baselines.csv")


if __name__ == "__main__":
    main()
