"""Version-2 experiments: what BOLD adds to the non-BOLD signals of a whole-night fMRI session.

Changes from 13/15, each answering a committee point:
  * BOLD source is selectable: --source afni (the dataset authors' AFNI derivative, main analysis)
    or --source raw (our minimal pipeline, sensitivity); features from 22_features_v2.py, amplitude
    as percent signal change, censored frames excluded.
  * NON-BOLD confounds are genuinely not computed from BOLD voxels: realignment framewise
    displacement (23_realign.py; the old centre-of-mass proxy and DVARS are image-derived and move to
    a separate "image-derived" row), pulse-oximeter and respiratory-belt physiology with trigger QC
    (12_confounds.py), and time since lights-out (21_night_position.py; causal), plus the AFNI censor
    fraction when the derivative is the source.
  * Primary statistic is the WITHIN-SUBJECT one-vs-rest AUC (subjects with >=20 epochs of the stage
    and >=20 of the rest), averaged over subjects; pooled AUC is secondary. Increments get a
    subject-bootstrap 95% CI, a one-sided 95% upper bound, Holm correction across the five stages for
    "BOLD adds", and a pre-stated non-superiority margin delta = 0.03 for "BOLD adds no more than".
  * Late fusion (product of the two models' posteriors) alongside early fusion; gradient boosting;
    the N3-minus-REM interaction; a time-matched REM test (second half of the night only).
  * Published-stager reimplementations use L2 logistic regression on their feature sets, so the HMM
    receives calibrated probabilities for every model (softmaxed SVM scores were near-flat).

    source <venv>/bin/activate
    python 24_experiments_v2.py --source afni      # writes results_v2_afni.json, predictions_v2_afni.npz
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
N_BOOT = 2000
DELTA = 0.03
MIN_EP = 20
WINDOWS = (30, 60, 90, 150, 300)
RNG = np.random.default_rng(0)
N_JOBS = 6

_s = importlib.util.spec_from_file_location("bl", BASE / "09_baselines.py")
bl = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bl)


# ------------------------------------------------------------------ data
def wz(X, subj):
    X = np.asarray(X, np.float64).copy()
    for s in np.unique(subj):
        m = subj == s
        X[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-6)
    return np.nan_to_num(X)


def motion_features(run, epoch, n_vol_run):
    """Realignment FD summarised in five windows x {mean, sd, max, p90}, aligned like 07_features."""
    feat = importlib.util.module_from_spec(importlib.util.spec_from_file_location("feat", BASE / "07_features.py"))
    feat.__spec__.loader.exec_module(feat)
    out = np.full((len(run), 20), np.nan)
    cache = {}
    index = {(r, e): i for i, (r, e) in enumerate(zip(run, epoch))}
    for stem in np.unique(run):
        sub = stem.split("_")[0]
        mf = BASE / "derivatives" / "motion_ds005127" / sub / f"{stem}_motion.npz"
        pf = BASE / "derivatives" / "ds005127" / sub / f"{stem}_parcels.npz"
        if not mf.exists():
            continue
        fd = np.load(mf)["fd"].astype(float)
        eps = feat.run_epochs(pf)
        T = len(fd)
        for ep in eps or []:
            i = index.get((stem, ep["epoch"]))
            if i is None:
                continue
            c = ep["start_vol"] + ep["n_vol"] / 2
            v = []
            for w in WINDOWS:
                h = w / 3.0 / 2
                a, b = int(max(0, round(c - h))), int(min(T, round(c + h)))
                x = fd[a:b]
                v += [x.mean(), x.std(), x.max(), np.percentile(x, 90)] if len(x) else [np.nan] * 4
            out[i] = v
    return out


def load(source):
    d = bl.load()                                   # stage, subj, run, epoch, edge (raw 84-parcel FC)
    V = np.load(BASE / f"features_v2_{source}.npz", allow_pickle=True)
    C = np.load(BASE / "confounds_ds005127.npz", allow_pickle=True)
    N = np.load(BASE / "night_position_ds005127.npz", allow_pickle=True)
    assert (C["run"].astype(str) == d["run"]).all() and (N["run"].astype(str) == d["run"]).all()
    key = {(r, e): i for i, (r, e) in enumerate(zip(d["run"], d["epoch"]))}
    rows = np.array([key[(r, e)] for r, e in zip(V["run"].astype(str), V["epoch"])])
    D = {k: d[k][rows] for k in ("stage", "subj", "run", "epoch")}
    D["edge"] = d["edge"][rows] if source == "raw" else None
    for k in V.files:
        if k not in ("run", "epoch", "common"):
            D[k] = V[k]
    D["physio"] = C["physio"][rows]
    D["imgmotion"] = C["motion"][rows]                     # centre-of-mass FD + DVARS (image-derived)
    D["motion"] = motion_features(D["run"], D["epoch"], None)
    if source == "afni":
        D["motion"] = np.hstack([D["motion"], D["cens"]])
    start = {}
    for r, cm in zip(N["run"].astype(str), N["clock_min"]):
        night = "_".join(r.split("_")[:2])
        start[night] = min(start.get(night, np.inf), cm)
    t_since = np.array([cm - start["_".join(r.split("_")[:2])] for r, cm in zip(N["run"].astype(str), N["clock_min"])])
    D["clock"] = np.column_stack([t_since, t_since ** 2])[rows] / np.array([60.0, 3600.0])   # hours since lights-out
    D["night_frac"] = N["night_frac"][rows]
    need = ["amp", "netfc", "gs", "physio", "motion"]
    ok = np.ones(len(rows), bool)
    for k in need:
        ok &= np.isfinite(D[k]).all(1)
    if source in ("afni", "rv2"):
        ok &= D["cens"][:, 0] <= 0.5                     # epoch itself at most half censored
    D = {k: (v[ok] if v is not None else None) for k, v in D.items()}
    return D


def blocks(D):
    s = D["subj"]
    B = dict(bold=np.hstack([wz(D["amp"], s), wz(D["netfc"], s)]),
             gs=wz(D["gs"], s), motion=wz(D["motion"], s), physio=wz(D["physio"], s),
             clock=D["clock"], netfc=wz(D["netfc"], s), imgmotion=wz(D["imgmotion"], s),
             bold_hrf=np.hstack([wz(D["amp_hrf"], s), wz(D["netfc_hrf"], s)]),
             bold_z=np.hstack([wz(D["amp_z"], s), wz(D["netfc_z"], s)]),
             night_frac=np.column_stack([D["night_frac"], D["night_frac"] ** 2]))
    for tag in ("_phys", "_physnull", "_acc", "_gsr", "_ric"):
        if "amp" + tag in D and np.isfinite(D["amp" + tag]).all():
            B["bold" + tag] = np.hstack([wz(D["amp" + tag], s), wz(D["netfc" + tag], s)])
    if "eye" in D:                                      # 06b_raw_v2.py eye signals (NaN where the FOV misses the eyes)
        B["eye"] = wz(D["eye"], s)
    if D.get("edge") is not None:
        B["edge"] = wz(D["edge"], s)
    return B


# ------------------------------------------------------------------ models
def _fold(X, y, subj, s, classes, model, C):
    te, tr = subj == s, subj != s
    if model == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return te, clf.predict_proba(X[te])[:, [list(clf.classes_).index(c) for c in classes]]


def loso(X, y, subj, classes, model="lr", C=1.0):
    """Leave-one-subject-out probabilities; folds are independent, so they are fitted in parallel."""
    P = np.zeros((len(y), len(classes)))
    for te, p in Parallel(n_jobs=N_JOBS)(delayed(_fold)(X, y, subj, s, classes, model, C) for s in np.unique(subj)):
        P[te] = p
    return P


def late_fusion(Pa, Pb, y, subj):
    """Product of experts over two class-balanced models (no refitting, no extra parameters)."""
    # class-balanced models already return posteriors under a uniform prior, so their product is the
    # conditionally-independent combination; no prior is divided out (that would double-count it)
    p = Pa * Pb
    return p / p.sum(1, keepdims=True)


def hmm_decode(P, y, subj, run, epoch):
    out = np.empty(len(y), dtype=object)
    for s in np.unique(subj):
        te, tr = subj == s, subj != s
        T = bl.transition_matrix(y[tr], run[tr])
        pi = np.array([(y[tr] == x).mean() for x in STAGES])
        for r in np.unique(run[te]):
            idx = np.flatnonzero(run == r)
            idx = idx[np.argsort(epoch[idx])]
            path = bl.viterbi(np.log(np.clip(P[idx], 1e-9, None)), np.log(T), np.log(pi + 1e-9))
            out[idx] = np.array(STAGES)[path]
    return out


# ------------------------------------------------------------------ statistics
def auc(y01, p):
    """Mann-Whitney AUC with mid-ranks for ties; equal to roc_auc_score, without its per-call overhead."""
    y01 = np.asarray(y01).astype(bool)
    n1 = y01.sum()
    n0 = len(y01) - n1
    return float((rankdata(p)[y01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ws_auc(y01, p, subj):
    """Within-subject AUC per eligible subject."""
    out = {}
    for s in np.unique(subj):
        m = subj == s
        if y01[m].sum() >= MIN_EP and (~y01[m].astype(bool)).sum() >= MIN_EP:
            out[s] = auc(y01[m], p[m])
    return out


def increment(y01, pa, pb, subj):
    """Pooled and within-subject AUC gain of model a over model b, with subject-bootstrap CIs."""
    subs = np.unique(subj)
    idx = {s: np.flatnonzero(subj == s) for s in subs}
    A, Bw = ws_auc(y01, pa, subj), ws_auc(y01, pb, subj)
    keys = sorted(A)
    dws = np.array([A[k] - Bw[k] for k in keys])
    pooled = auc(y01, pa) - auc(y01, pb)
    bp, bw = [], []
    for _ in range(N_BOOT):
        pick = RNG.choice(subs, len(subs))
        ii = np.concatenate([idx[s] for s in pick])
        if 0 < y01[ii].sum() < len(ii):
            bp.append(auc(y01[ii], pa[ii]) - auc(y01[ii], pb[ii]))
        kk = [k for k in pick if k in A]
        if kk:
            bw.append(np.mean([A[k] - Bw[k] for k in kk]))
    w = wilcoxon(dws) if len(dws) >= 6 else None
    return dict(pooled=float(pooled), pooled_ci=[float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
                ws_mean=float(dws.mean()) if len(dws) else np.nan,
                ws_ci=[float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                ws_upper95=float(np.percentile(bw, 95)), ws_lower95=float(np.percentile(bw, 5)),
                n_subj=len(dws), n_pos=int((dws > 0).sum()),
                wilcoxon_p=float(w.pvalue) if w is not None else np.nan,
                noninferior_upper_below_delta=bool(np.percentile(bw, 95) < DELTA),
                ws_values={k: float(A[k] - Bw[k]) for k in keys}, boot_ws=bw)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run_max)
    return adj


def staging(y, pred, subj):
    per = {s: float(cohen_kappa_score(y[subj == s], pred[subj == s])) for s in np.unique(subj)
           if len(np.unique(y[subj == s])) > 1}
    f1 = f1_score(y, pred, labels=STAGES, average=None, zero_division=0)
    return dict(acc=float((y == pred).mean()), kappa=float(cohen_kappa_score(y, pred)),
                macroF1=float(f1.mean()), f1=dict(zip(STAGES, map(float, f1))), per_subject_kappa=per,
                confusion=confusion_matrix(y, pred, labels=STAGES).tolist())


def boot_kappa(y, pred, subj, other=None):
    subs = np.unique(subj)
    idx = {s: np.flatnonzero(subj == s) for s in subs}
    v = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[s] for s in RNG.choice(subs, len(subs))])
        k = cohen_kappa_score(y[ii], pred[ii])
        if other is not None:
            k -= cohen_kappa_score(y[ii], other[ii])
        v.append(k)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raw", "afni"], required=True)
    ap.add_argument("--quick", action="store_true", help="skip gradient boosting and the 5-class ladder")
    args = ap.parse_args()
    t0 = time.time()
    D = load(args.source)
    B = blocks(D)
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    R = {"meta": dict(source=args.source, n_epochs=int(len(y)), n_subjects=int(len(np.unique(s))),
                      n_runs=int(len(np.unique(rn))),
                      stage_counts={st: int((y == st).sum()) for st in STAGES},
                      subjects_with_stage={st: int(len(np.unique(s[y == st]))) for st in STAGES},
                      subjects_eligible_ws={st: int(sum(((s == u) & (y == st)).sum() >= MIN_EP
                                                        and ((s == u) & (y != st)).sum() >= MIN_EP
                                                        for u in np.unique(s))) for st in STAGES},
                      n_bold_features=int(B["bold"].shape[1]), n_motion=int(B["motion"].shape[1]),
                      delta=DELTA, min_epochs=MIN_EP)}
    print(json.dumps(R["meta"]), flush=True)
    NB = ["motion", "physio", "clock"]
    sets = {"bold": ["bold"], "gs": ["gs"], "motion": ["motion"], "physio": ["physio"], "clock": ["clock"],
            "physio+clock": ["physio", "clock"], "motion+physio": ["motion", "physio"],
            "nonbold": NB, "nonbold+bold": NB + ["bold"], "nonbold+gs": NB + ["gs"],
            "nonbold+gs+bold": NB + ["gs", "bold"], "physio+bold": ["physio", "bold"],
            "physio+clock+bold": ["physio", "clock", "bold"], "motion+physio+bold": ["motion", "physio", "bold"],
            "nonbold_img": NB + ["imgmotion"], "nonbold_img+bold": NB + ["imgmotion", "bold"],
            "nonbold_nightfrac": ["motion", "physio", "night_frac"],
            "nonbold_nightfrac+bold": ["motion", "physio", "night_frac", "bold"],
            "nonbold+bold_hrf": NB + ["bold_hrf"], "nonbold+bold_z": NB + ["bold_z"]}
    for tag in ("_phys", "_physnull", "_acc", "_gsr"):
        if "bold" + tag in B:
            sets["bold" + tag] = ["bold" + tag]
            sets["nonbold+bold" + tag] = NB + ["bold" + tag]
    P = {}
    for k, v in sets.items():
        P[k] = loso(np.hstack([B[b] for b in v]), y, s, STAGES)
        print(f"  {k:24s} " + " ".join(f"{st}:{roc_auc_score(y == st, P[k][:, i]):.3f}"
                                       for i, st in enumerate(STAGES)), flush=True)
    P["latefusion(nonbold,bold)"] = late_fusion(P["nonbold"], P["bold"], y, s)
    R["auc_pooled"] = {k: {st: float(roc_auc_score(y == st, p[:, i])) for i, st in enumerate(STAGES)} for k, p in P.items()}
    R["auc_ws"] = {k: {st: dict(mean=float(np.mean(list(ws_auc((y == st).astype(int), p[:, i], s).values()))),
                                values=ws_auc((y == st).astype(int), p[:, i], s))
                       for i, st in enumerate(STAGES)} for k, p in P.items()}

    # ---------------- increments
    comps = [("nonbold+bold", "nonbold"), ("latefusion(nonbold,bold)", "nonbold"),
             ("motion+physio+bold", "motion+physio"), ("physio+bold", "physio"),
             ("physio+clock+bold", "physio+clock"), ("nonbold+gs", "nonbold"),
             ("nonbold+gs+bold", "nonbold+gs"), ("nonbold_img+bold", "nonbold_img"),
             ("nonbold_nightfrac+bold", "nonbold_nightfrac"), ("nonbold+bold_hrf", "nonbold"),
             ("nonbold+bold_z", "nonbold"), ("bold", "gs")]
    for tag in ("_phys", "_physnull", "_acc", "_gsr"):
        if "nonbold+bold" + tag in P:
            comps.append(("nonbold+bold" + tag, "nonbold"))
    R["increment"] = {}
    for a, b in comps:
        R["increment"][f"{a} - {b}"] = {st: increment((y == st).astype(int), P[a][:, i], P[b][:, i], s)
                                       for i, st in enumerate(STAGES)}
    for key, inc in R["increment"].items():
        adj = holm(np.array([inc[st]["wilcoxon_p"] for st in STAGES]))
        for st, a_ in zip(STAGES, adj):
            inc[st]["wilcoxon_p_holm"] = float(a_)
    head = R["increment"]["nonbold+bold - nonbold"]
    for st in STAGES:
        r = head[st]
        print(f"  BOLD over non-BOLD [{st}]: ws {r['ws_mean']:+.3f} [{r['ws_ci'][0]:+.3f},{r['ws_ci'][1]:+.3f}] "
              f"upper95 {r['ws_upper95']:+.3f} | pooled {r['pooled']:+.3f} [{r['pooled_ci'][0]:+.3f},{r['pooled_ci'][1]:+.3f}] "
              f"+{r['n_pos']}/{r['n_subj']} p_holm={r['wilcoxon_p_holm']:.3g}", flush=True)
    # paired: per-subject N3 gain minus REM gain in the subjects eligible for both, bootstrapped over
    # them with its own generator (same statistic and seed as 33_contrast.py)
    both = sorted(set(head["N3"]["ws_values"]) & set(head["REM"]["ws_values"]))
    dd = np.array([head["N3"]["ws_values"][k] - head["REM"]["ws_values"][k] for k in both])
    rng_c = np.random.default_rng(0)
    bs = [dd[rng_c.integers(0, len(dd), len(dd))].mean() for _ in range(N_BOOT)]
    R["interaction_N3_minus_REM"] = dict(value=float(dd.mean()), n=len(dd), n_n3_above_rem=int((dd > 0).sum()),
                                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    print(f"  interaction dN3 - dREM: {R['interaction_N3_minus_REM']}", flush=True)
    for inc in R["increment"].values():
        for st in STAGES:
            inc[st].pop("boot_ws")

    # ---------------- time-matched REM test: REM vs NREM, second half of each night only
    late = (D["night_frac"] >= 0.5) & np.isin(y, ["N1", "N2", "N3", "REM"])
    yl, sl = (y[late] == "REM").astype(int), s[late]
    PL = {k: loso(np.hstack([B[b][late] for b in sets[k]]), yl, sl, [0, 1])[:, 1]
          for k in ["bold", "physio", "physio+bold", "nonbold", "nonbold+bold"]}
    R["time_matched_rem"] = dict(
        n_epochs=int(late.sum()), n_rem=int(yl.sum()),
        auc_ws={k: float(np.mean(list(ws_auc(yl, v, sl).values()))) for k, v in PL.items()},
        auc_pooled={k: float(roc_auc_score(yl, v)) for k, v in PL.items()},
        inc_physio=increment(yl, PL["physio+bold"], PL["physio"], sl),
        inc_nonbold=increment(yl, PL["nonbold+bold"], PL["nonbold"], sl))
    for k in ("inc_physio", "inc_nonbold"):
        R["time_matched_rem"][k].pop("boot_ws")
    print(f"  time-matched REM: {json.dumps(R['time_matched_rem']['auc_ws'])}", flush=True)

    # ---------------- pairwise NREM separations, for comparison with the literature
    R["pairwise_auc_bold"] = {}
    for a, b in [("W", "N2"), ("N1", "N2"), ("N2", "N3"), ("W", "N1")]:
        mm = np.isin(y, [a, b])
        pp = loso(B["bold"][mm], (y[mm] == b).astype(int), s[mm], [0, 1])[:, 1]
        R["pairwise_auc_bold"][f"{a}/{b}"] = dict(pooled=float(roc_auc_score(y[mm] == b, pp)),
                                                  ws=float(np.mean(list(ws_auc((y[mm] == b).astype(int), pp, s[mm]).values()))))
    print(f"  pairwise BOLD AUC: {R['pairwise_auc_bold']}", flush=True)

    if not args.quick:
        # ---------------- gradient boosting
        R["hgb"] = {}
        for k in ["bold", "nonbold", "nonbold+bold"]:
            P["hgb:" + k] = loso(np.hstack([B[b] for b in sets[k]]), y, s, STAGES, model="hgb")
        R["hgb"]["increment"] = {st: increment((y == st).astype(int), P["hgb:nonbold+bold"][:, i],
                                               P["hgb:nonbold"][:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            R["hgb"]["increment"][st].pop("boot_ws")
        R["hgb"]["auc_ws"] = {k: {st: float(np.mean(list(ws_auc((y == st).astype(int), P["hgb:" + k][:, i], s).values())))
                                  for i, st in enumerate(STAGES)} for k in ["bold", "nonbold", "nonbold+bold"]}
        print(f"  HGB increments ws: " + " ".join(f"{st}:{R['hgb']['increment'][st]['ws_mean']:+.3f}" for st in STAGES), flush=True)

        # ---------------- five-class ladder, identical HMM
        ladder = {"prior": None, "clock": "clock", "motion": "motion", "physio": "physio", "nonbold": "nonbold",
                  "gs": "gs", "netFC-LR (after Tagliazucchi 2012)": "netfc", "bold": "bold",
                  "nonbold+gs": "nonbold+gs", "nonbold+bold": "nonbold+bold",
                  "latefusion(nonbold,bold)": "latefusion(nonbold,bold)"}
        if "edge" in B:
            P["edgeFC-LR"] = loso(B["edge"], y, s, STAGES, C=0.05)
            ladder["edgeFC-LR (after Altmann 2016)"] = "edgeFC-LR"
        if "netfc" not in P:
            P["netfc"] = loso(B["netfc"], y, s, STAGES)
        R["ladder"], PRED = {}, {}
        prior = np.array([(y == c).mean() for c in STAGES])
        for name, key in ladder.items():
            Pm = np.tile(prior, (len(y), 1)) if key is None else P[key]
            pred = hmm_decode(Pm, y, s, rn, ep)
            PRED[name] = pred
            met = staging(y, pred, s)
            met["kappa_ci"] = boot_kappa(y, pred, s)
            R["ladder"][name] = met
            print(f"  {name:36s} kappa {met['kappa']:.3f} [{met['kappa_ci'][0]:.3f},{met['kappa_ci'][1]:.3f}] "
                  f"mF1 {met['macroF1']:.3f} F1 N3 {met['f1']['N3']:.2f} REM {met['f1']['REM']:.2f}", flush=True)
        R["ladder_paired"] = {}
        pairs = [("nonbold+bold", "nonbold"), ("bold", "nonbold"), ("bold", "netFC-LR (after Tagliazucchi 2012)")]
        if "edge" in B:
            pairs.append(("bold", "edgeFC-LR (after Altmann 2016)"))
        for a, b in pairs:
            pa_, pb_ = R["ladder"][a]["per_subject_kappa"], R["ladder"][b]["per_subject_kappa"]
            kk = sorted(set(pa_) & set(pb_))
            dd = np.array([pa_[k] - pb_[k] for k in kk])
            R["ladder_paired"][f"{a} vs {b}"] = dict(diff=R["ladder"][a]["kappa"] - R["ladder"][b]["kappa"],
                                                     ci=boot_kappa(y, PRED[a], s, other=PRED[b]),
                                                     n_better=int((dd > 0).sum()), n=len(dd),
                                                     p=float(wilcoxon(dd).pvalue))
            print(f"  paired {a} vs {b}: {R['ladder_paired'][f'{a} vs {b}']}", flush=True)
        per = np.empty(len(y), dtype=object)
        for r in np.unique(rn):
            ii = np.flatnonzero(rn == r); ii = ii[np.argsort(ep[ii])]
            per[ii[1:]] = y[ii[:-1]]; per[ii[0]] = y[ii[0]]
        R["oracle_previous_label_kappa"] = float(cohen_kappa_score(y, per))
        np.savez_compressed(BASE / f"predictions_v2_{args.source}.npz", y=y, subj=s, run=rn, epoch=ep,
                            night_frac=D["night_frac"], **{f"hmm__{k}": v for k, v in PRED.items()},
                            **{f"proba__{k}": v for k, v in P.items()})

    R["meta"]["runtime_s"] = time.time() - t0
    json.dump(R, open(BASE / f"results_v2_{args.source}.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved results_v2_{args.source}.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
