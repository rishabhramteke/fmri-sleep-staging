"""Own pipeline v2 (realigned, no physiological correction) and the eye signal (post hoc; 06b_raw_v2.py, 22 --source rv2).

A. Nuisance regression. On the epochs both own streams share, with one non-BOLD model (realignment FD, physiology,
   elapsed time; the minimal pipeline's), early fusion: BOLD-alone AUC and gain over non-BOLD for
     minimal (no realignment), v2 (realigned, despiked, FD-censored, NO physiological correction),
     v2 + RETROICOR (what the derivative removed), v2 + RETROICOR + slow HR/RV/pulse/respiration regressors.
   The informative control is a stream that detects REM without any physiological regression.
B. Eyes. The eye block (12 features) where the field of view includes the eyes: alone, over non-BOLD, and over
   non-BOLD + BOLD, on (i) the derivative cohort and models of the main analysis, (ii) the v2 stream.

    source <venv>/bin/activate
    python 41_rv2_eyes.py                      # writes results_rv2_eyes.json, predictions_rv2_eyes.npz
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e = _load("e24", "24_experiments_v2.py")
x = _load("x36b", "36b_extra_block.py")
e.N_JOBS = 6
ST = e.STAGES


def aucs(P, y, s):
    return {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[:, i], s).values()))) for i, st in enumerate(ST)}


def strip(G):
    return {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}


def report(tag, G, C):
    print(f"  {tag:34s} N3 {G['N3']['ws_mean']:+.3f} ({G['N3']['n_pos']}/{G['N3']['n_subj']})  REM {G['REM']['ws_mean']:+.3f} "
          f"[{G['REM']['ws_ci'][0]:+.3f},{G['REM']['ws_ci'][1]:+.3f}] t up {G['REM']['t_upper95']:+.3f} "
          f"({G['REM']['n_pos']}/{G['REM']['n_subj']})  N3>REM {C['n_pos']}/{C['n']}", flush=True)


def join(Dsrc, Ddst, key):
    idx = {(r, int(t)): i for i, (r, t) in enumerate(zip(Dsrc["run"], Dsrc["epoch"]))}
    return np.array([idx.get((r, int(t)), -1) for r, t in zip(Ddst["run"], Ddst["epoch"])])


def main():
    t0 = time.time()
    R = {"A": {}, "B": {}}
    # ---------------- A: own streams on shared epochs
    V = e.load("rv2"); BV = e.blocks(V)
    W = e.load("raw"); BW = e.blocks(W)
    j = join(W, V, None)
    keep = j >= 0
    y, s = V["stage"][keep], V["subj"][keep]
    jw = j[keep]
    nb = np.hstack([BW["motion"], BW["physio"], BW["clock"]])[jw]
    streams = {"minimal": BW["bold"][jw], "v2": BV["bold"][keep], "v2_ric": BV["bold_ric"][keep], "v2_phys": BV["bold_phys"][keep]}
    R["A"]["n_epochs"], R["A"]["n_subjects"] = int(keep.sum()), int(len(np.unique(s)))
    Pnb = e.loso(nb, y, s, ST)
    R["A"]["auc_nonbold"] = aucs(Pnb, y, s)
    P = {"nonbold": Pnb}
    for k, Xb in streams.items():
        Pb = e.loso(Xb, y, s, ST)
        Pa = e.loso(np.hstack([nb, Xb]), y, s, ST)
        lf = Pnb * Pb; lf /= lf.sum(1, keepdims=True)
        G, Gl = x.gains(Pa, Pnb, y, s), x.gains(lf, Pnb, y, s)
        R["A"][k] = dict(auc_bold=aucs(Pb, y, s), gain=strip(G), contrast=x.contrast(G),
                         gain_late=strip(Gl), contrast_late=x.contrast(Gl))
        P[f"bold_{k}"], P[f"all_{k}"] = Pb, Pa
        print(f"  [{k}] BOLD-alone AUC N3 {R['A'][k]['auc_bold']['N3']:.3f} REM {R['A'][k]['auc_bold']['REM']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        report(f"{k} early", G, R["A"][k]["contrast"])
        report(f"{k} late", Gl, R["A"][k]["contrast_late"])
    # ---------------- B: eyes
    eye_v = BV["eye"]
    cover = np.isfinite(eye_v).all(1)
    R["B"]["coverage_by_subject_v2"] = {u: float(cover[V["subj"] == u].mean()) for u in np.unique(V["subj"])}
    # (i) derivative cohort, main-analysis models
    D = e.load("afni"); B = e.blocks(D)
    jd = join(V, D, None)
    eye_d = np.full((len(D["stage"]), eye_v.shape[1]), np.nan)
    eye_d[jd >= 0] = eye_v[jd[jd >= 0]]
    m = np.isfinite(eye_d).all(1)
    yd, sd = D["stage"][m], D["subj"][m]
    nbd = np.hstack([B["motion"], B["physio"], B["clock"]])[m]
    Xe = e.wz(eye_d[m], sd)
    R["B"]["deriv_n_epochs"], R["B"]["deriv_subjects"] = int(m.sum()), sorted(map(str, np.unique(sd)))
    R["B"]["deriv_frac_epochs_covered"] = float(m.mean())
    Pe = e.loso(Xe, yd, sd, ST)
    Pn = e.loso(nbd, yd, sd, ST)
    Pnb_e = e.loso(np.hstack([nbd, Xe]), yd, sd, ST)
    Pnb_b = e.loso(np.hstack([nbd, B["bold"][m]]), yd, sd, ST)
    Pnb_b_e = e.loso(np.hstack([nbd, B["bold"][m], Xe]), yd, sd, ST)
    R["B"]["deriv"] = dict(auc_eye=aucs(Pe, yd, sd), auc_nonbold=aucs(Pn, yd, sd),
                           eye_over_nonbold=strip(x.gains(Pnb_e, Pn, yd, sd)),
                           bold_over_nonbold=strip(x.gains(Pnb_b, Pn, yd, sd)),
                           eye_over_nonbold_bold=strip(x.gains(Pnb_b_e, Pnb_b, yd, sd)),
                           bold_over_nonbold_eye=strip(x.gains(Pnb_b_e, Pnb_e, yd, sd)))
    for k in ("eye_over_nonbold", "bold_over_nonbold", "eye_over_nonbold_bold", "bold_over_nonbold_eye"):
        G = R["B"]["deriv"][k]
        print(f"  eyes (derivative, {m.sum()} epochs, {len(np.unique(sd))} subj) {k:24s} N3 {G['N3']['ws_mean']:+.3f} "
              f"REM {G['REM']['ws_mean']:+.3f} [{G['REM']['ws_ci'][0]:+.3f},{G['REM']['ws_ci'][1]:+.3f}] "
              f"({G['REM']['n_pos']}/{G['REM']['n_subj']}) W {G['W']['ws_mean']:+.3f}", flush=True)
    print(f"  eye alone AUC (derivative subset): {R['B']['deriv']['auc_eye']}", flush=True)
    # (ii) v2 stream
    mv = cover & np.isin(np.arange(len(V["stage"])), np.flatnonzero(keep))
    yv, sv = V["stage"][mv], V["subj"][mv]
    jw2 = join(W, V, None)[mv]
    nbv = np.hstack([BW["motion"], BW["physio"], BW["clock"]])[jw2]
    Xev = e.wz(eye_v[mv], sv)
    Pn2 = e.loso(nbv, yv, sv, ST)
    Pe2 = e.loso(np.hstack([nbv, Xev]), yv, sv, ST)
    R["B"]["v2"] = dict(n_epochs=int(mv.sum()), n_subjects=int(len(np.unique(sv))),
                        auc_eye=aucs(e.loso(Xev, yv, sv, ST), yv, sv), eye_over_nonbold=strip(x.gains(Pe2, Pn2, yv, sv)))
    G = R["B"]["v2"]["eye_over_nonbold"]
    print(f"  eyes (v2, {mv.sum()} epochs): eye over non-BOLD REM {G['REM']['ws_mean']:+.3f} "
          f"[{G['REM']['ws_ci'][0]:+.3f},{G['REM']['ws_ci'][1]:+.3f}] ({G['REM']['n_pos']}/{G['REM']['n_subj']})", flush=True)
    (BASE / "results_rv2_eyes.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_rv2_eyes.npz", y=y, subj=s, **P)
    print(f"saved results_rv2_eyes.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
