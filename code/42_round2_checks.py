"""Checks asked by the PDF-only review of draft 10 (all post hoc, exploratory).

  slow      derivative BOLD with heart-rate, RV, pulse-amplitude and respiratory-rate regressors (0-15 s lags,
            12_confounds.slow_regressors) regressed out of every parcel: does the N3 gain survive the removal of the
            slow physiological variance the derivative retains (40_derivative_physio_r2.py)?
  detrend   derivative BOLD with a cubic within-run trend removed: is BOLD's REM-time overlap scanner drift?
  clock     how well BOLD predicts elapsed time (LOSO ridge; within-subject correlation), before and after detrending
  nights    cross-night generalisation: train on one night of the training subjects, test on the other night of the
            held-out subject (both directions)
  logloss   balanced one-vs-rest log-loss gain per stage (within subject), a more sensitive measure than AUC
  drift     the main gains without the runs whose labels drift by more than an epoch
  parcels   parcels whose 30 s amplitude is higher in N3 (AUC > 0.5), by network

    source <venv>/bin/activate
    python 42_round2_checks.py                 # writes results_round2_checks.json (~20 min)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e = _load("e24", "24_experiments_v2.py")
x = _load("x36b", "36b_extra_block.py")
f22 = _load("f22", "22_features_v2.py")
conf, feat = f22.conf, f22.feat
e.N_JOBS = 6
ST = e.STAGES


def strip(G):
    return {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}


def variant_features():
    """Cortical BOLD block (amplitude on the common parcels + network FC) for two derivative variants."""
    common = np.load(BASE / "features_v2_afni.npz", allow_pickle=True)["common"].astype(bool)
    net_ids, nets = feat.network_ids()
    n_nets = len(nets)
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    keys, out = [], {"slow": ([], []), "dt": ([], [])}
    for f in sorted((BASE / "derivatives" / "afni_ds005127").glob("sub-*/*_afni.npz")):
        stem = f.name.replace("_afni.npz", "")
        if stem not in raw_runs:
            continue
        eps = feat.run_epochs(raw_runs[stem])
        if not eps:
            continue
        z = np.load(f)
        ts_raw = z["ts"].astype(np.float64)
        T, tr = ts_raw.shape[1], float(z["tr"])
        valid = ~z["censored"].astype(bool)
        ts = f22.psc(ts_raw, valid)
        gs = np.zeros(T)
        pf = conf.PHYS / raw_runs[stem].parent.name / f"{stem}_physio.npz"
        R2 = None
        if pf.exists():
            P = dict(np.load(pf, allow_pickle=True))
            grid, S = conf.physio_signals(P, T, tr)
            R2 = conf.slow_regressors(grid, S, T, tr)
        tt = np.linspace(-1, 1, T)
        v_slow = f22.regress(ts, R2, valid) if R2 is not None else np.full(ts.shape, np.nan)
        v_dt = f22.regress(ts, np.column_stack([tt ** 2, tt ** 3]), valid)     # with the intercept + linear trend
        for ep in eps:
            keys.append((stem, ep["epoch"]))
            for tag, v in (("slow", v_slow), ("dt", v_dt)):
                amp, fc, _ = f22.epoch_block(v, gs, ep, common, net_ids, n_nets, tr, T)
                out[tag][0].append(amp); out[tag][1].append(fc)
    return keys, {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def balanced_logloss_gain(pa, pb, y, s, st):
    i = ST.index(st)
    g = []
    for u in np.unique(s):
        m = s == u
        pos = y[m] == st
        if pos.sum() >= 20 and (~pos).sum() >= 20:
            def bll(p):
                q = np.clip(p[m, i], 1e-6, 1 - 1e-6)
                return -0.5 * (np.log(q[pos]).mean() + np.log(1 - q[~pos]).mean())
            g.append(bll(pb) - bll(pa))            # positive = BOLD lowers the loss
    g = np.array(g)
    rng = np.random.default_rng(0)
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
    return dict(mean=float(g.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_pos=int((g > 0).sum()), n=len(g))


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all()
    Pnb, Pall = Z["proba__nonbold"], Z["proba__nonbold+bold"]
    R = {}
    # ---------------- derivative variants
    keys, V = variant_features()
    idx = {k: i for i, k in enumerate(keys)}
    rows = np.array([idx[(r, int(t))] for r, t in zip(run, ep)])
    for tag in ("slow", "dt"):
        amp, fc = V[tag][0][rows], V[tag][1][rows]
        ok = np.isfinite(amp).all(1) & np.isfinite(fc).all(1)
        Xb = np.hstack([e.wz(amp, s), e.wz(fc, s)])
        Pa = e.loso(np.hstack([nb, Xb])[ok], y[ok], s[ok], ST)
        Pb = Pnb[ok] if ok.all() else e.loso(nb[ok], y[ok], s[ok], ST)
        G = x.gains(Pa, Pb, y[ok], s[ok])
        R[tag] = dict(gain=strip(G), contrast=x.contrast(G), n_epochs=int(ok.sum()))
        R[tag]["bold"] = Xb
        print(f"  {tag}: N3 {G['N3']['ws_mean']:+.3f} [{G['N3']['ws_ci'][0]:+.3f},{G['N3']['ws_ci'][1]:+.3f}] "
              f"({G['N3']['n_pos']}/{G['N3']['n_subj']})  REM {G['REM']['ws_mean']:+.3f} t up {G['REM']['t_upper95']:+.3f} "
              f"N3>REM {R[tag]['contrast']['n_pos']}/{R[tag]['contrast']['n']}  W {G['W']['ws_mean']:+.3f} ({time.time()-t0:.0f} s)", flush=True)
    # ---------------- does BOLD encode elapsed time?
    hours = D["clock"][:, 0]
    R["clock_from_bold"] = {}
    for tag, Xb in (("bold", B["bold"]), ("bold_detrended", R["dt"]["bold"])):
        pred = np.zeros(len(y))
        for u in np.unique(s):
            tr_, te_ = s != u, s == u
            pred[te_] = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Xb[tr_], hours[tr_]).predict(Xb[te_])
        rr = [np.corrcoef(pred[s == u], hours[s == u])[0, 1] for u in np.unique(s)]
        R["clock_from_bold"][tag] = dict(r_mean=float(np.mean(rr)), r_min=float(np.min(rr)), r_max=float(np.max(rr)))
        print(f"  elapsed time from {tag}: within-subject r {np.mean(rr):.2f} ({np.min(rr):.2f}..{np.max(rr):.2f})", flush=True)
    for tag in ("slow", "dt"):
        R[tag].pop("bold")
    # ---------------- cross-night generalisation
    ses = np.array([r.split("_")[1] for r in run])
    Xall = np.hstack([nb, B["bold"]])
    Pc_nb, Pc_all = np.full(Pnb.shape, np.nan), np.full(Pnb.shape, np.nan)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for u in np.unique(s):
        for a, b in (("ses-1", "ses-2"), ("ses-2", "ses-1")):
            tr_, te_ = (s != u) & (ses == a), (s == u) & (ses == b)
            if te_.sum() == 0:
                continue
            for X_, P_ in ((nb, Pc_nb), (Xall, Pc_all)):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
                clf.fit(X_[tr_], y[tr_])
                P_[te_] = clf.predict_proba(X_[te_])[:, [list(clf.classes_).index(c) for c in ST]]
    ok = np.isfinite(Pc_nb).all(1)
    G = x.gains(Pc_all[ok], Pc_nb[ok], y[ok], s[ok])
    R["cross_night"] = dict(gain=strip(G), contrast=x.contrast(G), n_epochs=int(ok.sum()))
    print(f"  cross-night: N3 {G['N3']['ws_mean']:+.3f} ({G['N3']['n_pos']}/{G['N3']['n_subj']})  REM {G['REM']['ws_mean']:+.3f} "
          f"[{G['REM']['ws_ci'][0]:+.3f},{G['REM']['ws_ci'][1]:+.3f}] t up {G['REM']['t_upper95']:+.3f}  "
          f"N3>REM {R['cross_night']['contrast']['n_pos']}/{R['cross_night']['contrast']['n']} ({time.time()-t0:.0f} s)", flush=True)
    # ---------------- balanced log-loss gain
    R["logloss"] = {st: balanced_logloss_gain(Pall, Pnb, y, s, st) for st in ST}
    print("  balanced log-loss gain:", {st: round(v["mean"], 3) for st, v in R["logloss"].items()}, flush=True)
    # ---------------- without the drifting runs
    sc = BASE / "data" / "ds005127" / "derivatives" / "sleep_scoring_mr"
    drift = []
    for f in sorted((BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")):
        stem = f.name.replace("_parcels.npz", "")
        c = list(sc.rglob(stem + ".txt"))
        if c and stem in set(run):
            T_ = np.load(f)["ts"].shape[1]
            on = pd.read_csv(c[0], sep="\t")["MRIon"].to_numpy(float)
            if abs(on.sum() * 30 / 3.0 - T_) > 10:
                drift.append(stem)
    m = ~np.isin(run, drift)
    G = x.gains(Pall[m], Pnb[m], y[m], s[m])
    R["no_drift_runs"] = dict(gain=strip(G), contrast=x.contrast(G), n_runs_removed=len(drift), n_epochs_removed=int((~m).sum()))
    print(f"  without {len(drift)} drifting runs: N3 {G['N3']['ws_mean']:+.3f}  REM {G['REM']['ws_mean']:+.3f}", flush=True)
    # ---------------- parcels with higher amplitude in N3
    NP = np.load(BASE / "n3_parcel_map.npz", allow_pickle=True)
    nets_ = np.array([str(n) for n in NP["nets"]])
    above = NP["N3"] > 0.5
    R["n3_parcels_above"] = dict(n=int(above.sum()), by_network={n: int((above & (nets_ == n)).sum()) for n in np.unique(nets_)})
    print("  N3 parcels above 0.5:", R["n3_parcels_above"], flush=True)
    (BASE / "results_round2_checks.json").write_text(json.dumps(R, indent=1))
    print(f"saved results_round2_checks.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
