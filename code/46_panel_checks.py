"""Checks asked by the three-reviewer panel on draft 13 (post hoc, exploratory).

  lagged    a physiology baseline in BOLD's own time course: heart rate, RV, pulse amplitude and respiratory rate sampled
            at every volume and lagged 0-15 s (12_confounds.slow_regressors), averaged over the epoch (24 features), added
            to the non-BOLD model; BOLD's gain over it
  tones5    the tone proxy with a wider window: no epochs within 5 min before or after an awakening from N2 or N3
  f1ci      subject-bootstrap 95% CIs of the decoded per-stage F1 change (non-BOLD -> non-BOLD + BOLD, HMM paths)

    source <venv>/bin/activate
    python 46_panel_checks.py                  # writes results_panel_checks.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


x = _load("x36b", "36b_extra_block.py")
e = x.e
f22 = _load("f22", "22_features_v2.py")
conf, feat = f22.conf, f22.feat
ST = e.STAGES


def main():
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all()
    strip = lambda G: {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}
    R = {}
    # ---------------- lagged physiology baseline
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    lag = {}
    for stem in np.unique(run):
        r = raw_runs[stem]
        T, tr = np.load(r)["ts"].shape[1], 3.0
        P = dict(np.load(conf.PHYS / r.parent.name / f"{stem}_physio.npz", allow_pickle=True))
        grid, S = conf.physio_signals(P, T, tr)
        R2 = conf.slow_regressors(grid, S, T, tr)
        for k in feat.run_epochs(r) or []:
            a, n = k["start_vol"], k["n_vol"]
            lag[(stem, k["epoch"])] = R2[a:a + n].mean(0)
    XL = np.array([lag[(r_, t_)] for r_, t_ in zip(run, ep)])
    XL = e.wz(XL, s)
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Pl = e.loso(np.hstack([nb, XL]), y, s, ST)
    Plb = e.loso(np.hstack([nb, XL, B["bold"]]), y, s, ST)
    G = x.gains(Plb, Pl, y, s)
    R["lagged"] = dict(gain=strip(G), contrast=x.contrast(G), n_features=int(XL.shape[1]),
                       auc_nonbold_lag={st: float(np.mean(list(e.ws_auc((y == st).astype(int), Pl[:, i], s).values())))
                                        for i, st in enumerate(ST)})
    print(f"  lagged physiology: N3 {G['N3']['ws_mean']:+.3f} ({G['N3']['n_pos']}/{G['N3']['n_subj']}) REM {G['REM']['ws_mean']:+.3f} "
          f"t up {G['REM']['t_upper95']:+.3f}  N3>REM {R['lagged']['contrast']['n_pos']}/{R['lagged']['contrast']['n']}", flush=True)
    # ---------------- tones, +-5 min
    near = np.zeros(len(y), bool)
    n_aw = 0
    for r_ in np.unique(run):
        idx = np.flatnonzero(run == r_)
        idx = idx[np.argsort(ep[idx])]
        for k in range(1, len(idx)):
            if y[idx[k]] == "W" and y[idx[k - 1]] in ("N2", "N3") and ep[idx[k]] - ep[idx[k - 1]] <= 2:
                n_aw += 1
                t = ep[idx[k]]
                near |= (run == r_) & (ep >= t - 10) & (ep <= t + 10)
    keep = ~near
    G = x.gains(Z["proba__nonbold+bold"][keep], Z["proba__nonbold"][keep], y[keep], s[keep])
    R["tones5"] = dict(gain=strip(G), contrast=x.contrast(G), n_removed=int(near.sum()), frac_removed=float(near.mean()))
    print(f"  tones +-5 min ({near.mean():.0%} removed): N3 {G['N3']['ws_mean']:+.3f} REM {G['REM']['ws_mean']:+.3f} "
          f"t up {G['REM']['t_upper95']:+.3f}", flush=True)
    # ---------------- decoded F1 change CIs
    ha, hb = Z["hmm__nonbold+bold"].astype(str), Z["hmm__nonbold"].astype(str)
    yy = y.astype(str)
    subs = np.unique(s)
    idx = {u: np.flatnonzero(s == u) for u in subs}
    rng = np.random.default_rng(0)
    R["f1ci"] = {}
    for st in ST:
        d0 = f1_score(yy == st, ha == st) - f1_score(yy == st, hb == st)
        bs = []
        for _ in range(2000):
            ii = np.concatenate([idx[u] for u in rng.choice(subs, len(subs))])
            bs.append(f1_score(yy[ii] == st, ha[ii] == st) - f1_score(yy[ii] == st, hb[ii] == st))
        R["f1ci"][st] = dict(diff=float(d0), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    print("  decoded F1 change:", {st: (round(v["diff"], 2), [round(c, 2) for c in v["ci"]]) for st, v in R["f1ci"].items()})
    (BASE / "results_panel_checks.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
