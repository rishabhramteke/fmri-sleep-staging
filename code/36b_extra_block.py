"""An extra BOLD block tested with every estimator the paper uses (subcortex, or multi-window BOLD).

Adapted from the round-3 reviewer's r5_subc_stack.py (dreams/review_committee_2026-10-15_round3.md).
--block subc : the 50 subcortical features of 22b_subcortex_features.py (pre-registered test, see
               next_steps.md "PRE-REGISTERED DECISION RULE")
--block mw   : the 61 multi-window BOLD features of 22c_multiwindow_bold.py (context-length check)
--block lvl  : the 200 regional-level features (epoch-mean PSC per parcel) of 22d_level_features.py
--block max  : every extra BOLD block at once (subcortex, multi-window, regional level of parcels and
               structures, global signal): the richest BOLD model the paper tests

36_subcortex.py tests the subcortical block with early fusion only. The paper's own Fig. 1c shows early
fusion can LOSE REM AUC when a weak block is added, so an early-fusion null for 50 extra features would
not be interpretable. This script adds, on the same epochs and with 24's functions:
  * early fusion (as 36), late fusion (product of posteriors), and nested stacking (as 35_stacked.py)
    with three blocks (non-BOLD, cortical BOLD, subcortex)
  * one-sided 95% bounds by subject bootstrap AND t (the paper's dagger rule), N3-REM paired contrast
  * REM vs each competing stage; HMM kappa of cortical+subcortical BOLD vs cortical BOLD
  * per-structure within-subject amplitude AUC for REM and N3 (descriptive)

    source <venv>/bin/activate
    python 36b_extra_block.py --block subc        # writes results_block_subc.json
    python 36b_extra_block.py --block mw          # writes results_block_mw.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
e.N_JOBS = 6
ST = e.STAGES
DELTA = 0.03
lg = lambda p: np.log(np.clip(p, 1e-9, 1))


def tb(v, upper=True):
    v = np.asarray(v)
    h = tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() + h) if upper else float(v.mean() - h)


def summarise(inc):
    r = {k: v for k, v in inc.items() if k != "boot_ws"}
    v = list(inc["ws_values"].values())
    r["t_upper95"], r["t_lower95"] = tb(v), tb(v, False)
    r["dagger"] = bool(inc["ws_upper95"] < DELTA and r["t_upper95"] < DELTA)
    r["lower_above_zero"] = bool(inc["ws_lower95"] > 0 and r["t_lower95"] > 0)
    return r


def gains(pa, pb, y, s):
    return {st: summarise(e.increment((y == st).astype(int), pa[:, i], pb[:, i], s)) for i, st in enumerate(ST)}


def contrast(G):
    a, b = G["N3"]["ws_values"], G["REM"]["ws_values"]
    both = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in both])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return dict(value=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=len(d), n_pos=int((d > 0).sum()))


def rem_vs(pa, pb, y, s):
    i, out = ST.index("REM"), {}
    for other in ("W", "N1", "N2", "N3"):
        g = []
        for u in np.unique(s):
            m = (s == u) & np.isin(y, ["REM", other]); y01 = y[m] == "REM"
            if y01.sum() >= 20 and (~y01).sum() >= 20:
                g.append(e.auc(y01, pa[m, i]) - e.auc(y01, pb[m, i]))
        out[other] = dict(mean=float(np.mean(g)), n=len(g), n_pos=int((np.array(g) > 0).sum()))
    return out


def stack(blocks_tr_oof, blocks_te, y_tr):
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(np.hstack(blocks_tr_oof), y_tr)
    return clf.predict_proba(np.hstack(blocks_te))[:, [list(clf.classes_).index(c) for c in ST]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["subc", "mw", "lvl", "max"], required=True)
    ap.add_argument("--standin", default=None, help="use a block from 24's blocks() instead (code-path check)")
    a = ap.parse_args()
    a.out = str(BASE / f"results_block_{a.standin or a.block}.json")
    t0 = time.time()
    D = e.load("afni"); B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    if a.standin:
        X, names = B[a.standin], [f"{a.standin}{j}" for j in range(B[a.standin].shape[1])]
        amp_cols = []
    else:
        SRC = {"subc": [("features_v2_afni_subc.npz", ("subc_amp", "subc_fc"))],
               "mw": [("features_v2_afni_mw.npz", ("mw_amp", "mw_fc"))],
               "lvl": [("features_v2_afni_lvl.npz", ("lvl_parcel",))],
               "max": [("features_v2_afni_subc.npz", ("subc_amp", "subc_fc")),
                       ("features_v2_afni_mw.npz", ("mw_amp", "mw_fc")),
                       ("features_v2_afni_lvl.npz", ("lvl_parcel", "lvl_subc"))]}[a.block]
        parts = []
        for fname, cols in SRC:
            S = np.load(BASE / fname, allow_pickle=True)
            key = {(r, int(t)): i for i, (r, t) in enumerate(zip(S["run"].astype(str), S["epoch"]))}
            rows = np.array([key[(r, int(t))] for r, t in zip(run, ep)])
            parts.append(np.hstack([S[c][rows] for c in cols]))
        raw = np.hstack(parts)
        print(f"complete {a.block} rows: {np.isfinite(raw).all(1).mean():.3f}", flush=True)
        X = e.wz(raw, s)
        if a.block == "max":   # everything BOLD the paper tests: also the global-signal block
            X = np.hstack([X, B["gs"]])
        if a.block == "subc":
            names = [str(n) for n in S["names"]]
            amp_cols = list(range(len(names)))
        else:
            names, amp_cols = [f"{a.block}{j}" for j in range(X.shape[1])], []
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"].astype(str) == y).all() and (Z["subj"].astype(str) == s).all()
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    P = {"nonbold": Z["proba__nonbold"], "bold": Z["proba__bold"], "nonbold+bold": Z["proba__nonbold+bold"]}
    P["subc"] = e.loso(X, y, s, ST)
    P["bold+subc"] = e.loso(np.hstack([B["bold"], X]), y, s, ST)
    P["nonbold+bold+subc"] = e.loso(np.hstack([nb, B["bold"], X]), y, s, ST)
    print(f"  LOSO done ({time.time() - t0:.0f} s)", flush=True)
    lf2 = P["nonbold"] * P["bold"]; lf2 /= lf2.sum(1, keepdims=True)
    lf3 = lf2 * P["subc"]; lf3 /= lf3.sum(1, keepdims=True)
    # nested stacking with three blocks; reference stacks use the same inner out-of-fold predictions
    S1, S2, S3 = (np.zeros((len(y), 5)) for _ in range(3))
    for u in np.unique(s):
        tr, te = s != u, s == u
        o_nb = lg(e.loso(nb[tr], y[tr], s[tr], ST)); o_b = lg(e.loso(B["bold"][tr], y[tr], s[tr], ST))
        o_sc = lg(e.loso(X[tr], y[tr], s[tr], ST))
        t_nb, t_b, t_sc = lg(P["nonbold"][te]), lg(P["bold"][te]), lg(P["subc"][te])
        S1[te] = stack([o_nb], [t_nb], y[tr])
        S2[te] = stack([o_nb, o_b], [t_nb, t_b], y[tr])
        S3[te] = stack([o_nb, o_b, o_sc], [t_nb, t_b, t_sc], y[tr])
        print(f"  outer {u} ({time.time() - t0:.0f} s)", flush=True)
    R = {"block": a.block, "standin": a.standin, "n_features": int(X.shape[1]), "auc_ws": {}, "gain": {}, "contrast": {}, "rem_vs": {}}
    for k in ("subc", "bold", "bold+subc", "nonbold", "nonbold+bold", "nonbold+bold+subc"):
        R["auc_ws"][k] = {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[k][:, i], s).values())))
                          for i, st in enumerate(ST)}
    comps = {  # (estimator, question): (model with subcortex, reference)
        "early: all BOLD over non-BOLD": (P["nonbold+bold+subc"], P["nonbold"]),
        "early: block beyond cortical BOLD": (P["nonbold+bold+subc"], P["nonbold+bold"]),
        "late: all BOLD over non-BOLD": (lf3, P["nonbold"]),
        "late: block beyond cortical BOLD": (lf3, lf2),
        "stack: all BOLD over non-BOLD": (S3, S1),
        "stack: block beyond cortical BOLD": (S3, S2),
        "stack: cortical BOLD over non-BOLD (check vs 35)": (S2, S1),
    }
    for name, (pa, pb) in comps.items():
        G = gains(pa, pb, y, s)
        R["gain"][name], R["contrast"][name], R["rem_vs"][name] = G, contrast(G), rem_vs(pa, pb, y, s)
        g = G["REM"]
        print(f"  {name:48s} REM {g['ws_mean']:+.3f} [{g['ws_ci'][0]:+.3f},{g['ws_ci'][1]:+.3f}] up b{g['ws_upper95']:+.3f}"
              f"/t{g['t_upper95']:+.3f} lo t{g['t_lower95']:+.3f} {g['n_pos']}/{g['n_subj']} dagger={g['dagger']} | "
              f"N3 {G['N3']['ws_mean']:+.3f} | N3-REM {R['contrast'][name]['value']:+.3f}", flush=True)
    # five-class staging: does subcortex improve the BOLD stager?
    kb = e.hmm_decode(P["bold"], y, s, run, ep); kbs = e.hmm_decode(P["bold+subc"], y, s, run, ep)
    per = lambda pr: np.array([cohen_kappa_score(y[s == u], pr[s == u]) for u in np.unique(s)])
    d = per(kbs) - per(kb)
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    R["kappa"] = dict(bold=float(cohen_kappa_score(y, kb)), bold_subc=float(cohen_kappa_score(y, kbs)),
                      diff_mean_subject=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      n_better=int((d > 0).sum()))
    print(f"  kappa BOLD {R['kappa']['bold']:.3f} -> +subcortex {R['kappa']['bold_subc']:.3f}", flush=True)
    if amp_cols:  # descriptive: amplitude of each structure, within-subject AUC for REM and N3 vs rest
        R["structure_auc"] = {}
        for j in amp_cols:
            R["structure_auc"][names[j]] = {st: float(np.nanmean(list(e.ws_auc((y == st).astype(int), X[:, j], s).values())))
                                            for st in ("N3", "REM")}
    Path(a.out).write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_block_{a.standin or a.block}.npz", y=y, subj=s, block=P["subc"],
                        early3=P["nonbold+bold+subc"], late2=lf2, late3=lf3, stack1=S1, stack2=S2, stack3=S3)
    print(f"saved {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
