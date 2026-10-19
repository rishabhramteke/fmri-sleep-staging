"""Cross-site transfer of the BOLD stager: trained on ds005127 (NIH, whole night), tested on ds003768 (Penn State naps).

ds003768 (Gu et al., Data in Brief 2023): 33 healthy adults, up to 8 in-scanner sleep runs of 15 min (3 T, TR 2.1 s,
3 x 3 x 4 mm), EEG-scored in 30 s epochs as W, N1, N2 or N3 by a registered technologist; almost no N3 and no REM, no
physiological recordings. So it cannot test the N3 or REM claims, nor the confound question (no pulse or breathing),
but it can test whether the stager itself leaves its site: the same minimal pipeline (06_preprocess.py), the same 84
parcels, the same features (30 s amplitude, 90 s network FC), within-subject z-scoring, a class-balanced logistic
regression trained on all 16 ds005127 subjects, applied unchanged to ds003768 and scored on W, N1 and N2.

Reported: 3-class kappa and per-class within-subject AUC of the transferred model (argmax, and Viterbi with the
ds005127 transition matrix restricted to the three stages), the 5-class prediction distribution (how often N3 or REM is
called in naps), and within-ds003768 leave-one-subject-out references with the same features: BOLD, time since the
run started (nap runs drift from W into N2), and both.

    source <venv>/bin/activate
    python 52_cross_dataset.py --features        # features_ds003768_transfer.npz
    python 52_cross_dataset.py --test            # results_cross_dataset.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
PSU = BASE / "data" / "ds003768"
DER = BASE / "derivatives" / "ds003768"
EPOCH_S = 30.0
PSU_MAP = {"W": "W", "1": "N1", "2": "N2", "3": "N3"}
THREE = ["W", "N1", "N2"]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def hypnograms():
    """{(sub, run): DataFrame(t, stage)} for the sleep runs; '(uncertain)' and other labels are dropped."""
    out = {}
    for f in sorted((PSU / "sourcedata").glob("*-sleep-stage.tsv")):
        sub = f.stem.split("-sleep")[0]
        rows = []
        for line in f.read_text().splitlines()[1:]:
            p = [x.strip() for x in line.split("\t")]
            if len(p) == 2 and " " in p[1]:                 # one merged cell ("810 W") in sub-32
                p = [p[0]] + p[1].split(" ", 1)
            if len(p) < 3 or "sleep" not in p[0]:
                continue
            try:
                t = float(p[1])
            except ValueError:
                continue
            rows.append((p[0].replace("task-sleep_", ""), t, PSU_MAP.get(p[2], "other")))
        d = pd.DataFrame(rows, columns=["run", "t", "stage"])
        for r, g in d.groupby("run"):
            out[(sub, r)] = g.sort_values("t")
    return out


def features():
    f22 = _load("f22", "22_features_v2.py")
    net_ids, nets = f22.feat.network_ids()
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    H = hypnograms()
    files = sorted(DER.glob("sub-*/*task-sleep*_parcels.npz"))
    present = np.ones(200, bool)
    for p in files:
        z = np.load(p)
        present &= np.isfinite(z["ts"]).all(1) & (np.nanstd(z["ts"], 1) > 0) & (z["parcel_sizes"] >= 10)
    use = common & present
    print(f"{len(files)} runs; ds005127 shared parcels {common.sum()}, present in every ds003768 run {present.sum()}, used {use.sum()}", flush=True)
    keys, A, F, G, Y, Tt, n_other = [], [], [], [], [], [], 0
    for p in files:
        sub = p.parent.name
        run = p.name.split("_task-sleep_")[1].replace("_parcels.npz", "")
        if (sub, run) not in H:
            continue
        z = np.load(p)
        ts, tr = z["ts"].astype(np.float64), float(z["tr"])
        T = ts.shape[1]
        valid = np.ones(T, bool)
        x = f22.psc(ts, valid)
        gs = np.nanmean(ts[use], 0)
        gs = (gs - gs.mean()) / (gs.std() + 1e-9)
        n_vol = int(round(EPOCH_S / tr))
        for _, row in H[(sub, run)].iterrows():
            a = int(round(row.t / tr))
            if a + n_vol > T:
                continue
            if row.stage == "other":
                n_other += 1
                continue
            ep = dict(start_vol=a, n_vol=n_vol, epoch=int(round(row.t / EPOCH_S)))
            amp, fc, g = f22.epoch_block(x, gs, ep, use, net_ids, len(nets), tr, T)
            keys.append((sub, run, ep["epoch"])); A.append(amp); F.append(fc); G.append(g); Y.append(row.stage); Tt.append(row.t)
    A, F, G = np.array(A, np.float32), np.array(F, np.float32), np.array(G, np.float32)
    print(f"epochs {len(Y)} (dropped {n_other} unscorable/uncertain); stages {pd.Series(Y).value_counts().to_dict()}; "
          f"subjects {len(set(k[0] for k in keys))}; NaN rows amp {np.isnan(A).any(1).mean():.3f} fc {np.isnan(F).any(1).mean():.3f}", flush=True)
    np.savez_compressed(BASE / "features_ds003768_transfer.npz", subj=np.array([k[0] for k in keys]), run=np.array([k[0] + "_" + k[1] for k in keys]),
                        epoch=np.array([k[2] for k in keys]), t=np.array(Tt, np.float32), stage=np.array(Y), amp=A, netfc=F, gs=G, use=use)


def kappa3(y, pred):
    m = np.isin(y, THREE)
    return float(cohen_kappa_score(y[m], pred[m], labels=THREE))


def decode(P, classes, run, epoch, Tm, pi):
    """Viterbi per run with a fixed transition matrix (no target labels used)."""
    bl = _load("bl", "09_baselines.py")
    out = np.empty(len(run), dtype=object)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        idx = idx[np.argsort(epoch[idx])]
        path = bl.viterbi(np.log(np.clip(P[idx], 1e-9, None)), np.log(Tm), np.log(pi + 1e-9))
        out[idx] = np.array(classes)[path]
    return out


def transitions(stage, run, classes):
    T = np.full((len(classes), len(classes)), 1.0)
    for r in np.unique(run):
        s = stage[run == r]
        for a, b in zip(s[:-1], s[1:]):
            if a in classes and b in classes:
                T[classes.index(a), classes.index(b)] += 1
    return T / T.sum(1, keepdims=True), np.array([(stage == c).mean() for c in classes]) / max(1e-9, sum((stage == c).mean() for c in classes))


def test():
    e = _load("e24", "24_experiments_v2.py")
    ST = e.STAGES
    D = e.load("raw")
    use = np.load(BASE / "features_ds003768_transfer.npz", allow_pickle=True)["use"].astype(bool)
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    sel = use[common]                                      # columns of the ds005127 amplitude block to keep
    assert sel.all(), "ds003768 misses some of the 84 parcels; the FC block would differ, recompute ds005127 features on the subset"
    Z = np.load(BASE / "features_ds003768_transfer.npz", allow_pickle=True)
    y5, s5 = D["stage"], D["subj"]
    X5 = np.hstack([e.wz(D["amp"], s5), e.wz(D["netfc"], s5)])
    yp, sp, rp, epp, tp = Z["stage"].astype(str), Z["subj"].astype(str), Z["run"].astype(str), Z["epoch"].astype(int), Z["t"].astype(float)
    ok = np.isfinite(Z["amp"]).all(1) & np.isfinite(Z["netfc"]).all(1)
    yp, sp, rp, epp, tp = yp[ok], sp[ok], rp[ok], epp[ok], tp[ok]
    Xp = np.hstack([e.wz(Z["amp"][ok].astype(np.float64), sp), e.wz(Z["netfc"][ok].astype(np.float64), sp)])
    assert X5.shape[1] == Xp.shape[1], (X5.shape, Xp.shape)
    R = {"meta": dict(n_train=int(len(y5)), n_train_subjects=int(len(np.unique(s5))), n_test=int(len(yp)), n_test_subjects=int(len(np.unique(sp))),
                      n_test_runs=int(len(np.unique(rp))), test_stage_counts={k: int(v) for k, v in zip(*np.unique(yp, return_counts=True))},
                      n_features=int(Xp.shape[1]))}
    print(json.dumps(R["meta"]), flush=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
    clf.fit(X5, y5)
    P5 = clf.predict_proba(Xp)[:, [list(clf.classes_).index(c) for c in ST]]
    arg5 = np.array(ST)[P5.argmax(1)]
    R["transfer"] = {"pred5_distribution": {c: float((arg5 == c).mean()) for c in ST},
                     "confusion5_rows_true_W_N1_N2_N3": confusion_matrix(yp, arg5, labels=["W", "N1", "N2", "N3", "REM"]).tolist()}
    i3 = [ST.index(c) for c in THREE]
    P3 = P5[:, i3] / P5[:, i3].sum(1, keepdims=True)
    arg3 = np.array(THREE)[P3.argmax(1)]
    Tm, pi = transitions(y5, D["run"], THREE)              # ds005127 transitions among W, N1, N2 only
    hmm3 = decode(P3, THREE, rp, epp, Tm, pi)
    m3 = np.isin(yp, THREE)

    def metrics(pred, P):
        out = dict(kappa=kappa3(yp, pred), kappa_ci=e.boot_kappa(yp[m3], pred[m3], sp[m3]), acc=float((pred[m3] == yp[m3]).mean()),
                   f1=dict(zip(THREE, map(float, f1_score(yp[m3], pred[m3], labels=THREE, average=None, zero_division=0)))))
        out["auc_ws"] = {c: float(np.mean(list(e.ws_auc((yp[m3] == c).astype(int), P[m3][:, j], sp[m3]).values()) or [np.nan])) for j, c in enumerate(THREE)}
        out["auc_ws_n"] = {c: len(e.ws_auc((yp[m3] == c).astype(int), P[m3][:, j], sp[m3])) for j, c in enumerate(THREE)}
        return out
    R["transfer"]["argmax"] = metrics(arg3, P3)
    R["transfer"]["hmm_ds005127_transitions"] = metrics(hmm3, P3)
    for k, v in R["transfer"].items():
        if isinstance(v, dict) and "kappa" in v:
            print(f"  transfer {k:28s} kappa {v['kappa']:.3f} [{v['kappa_ci'][0]:.3f},{v['kappa_ci'][1]:.3f}] acc {v['acc']:.3f} "
                  f"AUC W {v['auc_ws']['W']:.3f} N1 {v['auc_ws']['N1']:.3f} N2 {v['auc_ws']['N2']:.3f} (n subj {v['auc_ws_n']})", flush=True)
    print(f"  5-class calls on naps: {R['transfer']['pred5_distribution']}", flush=True)
    # within-ds003768 references, leave-one-subject-out, three classes
    tt = np.column_stack([tp / 60.0, (tp / 60.0) ** 2])
    refs = {"bold": Xp, "time": tt, "bold+time": np.hstack([Xp, tt])}
    R["within"] = {}
    for name, X in refs.items():
        P = np.zeros((len(yp), 3))
        for u in np.unique(sp):
            tr_, te_ = (sp != u) & m3, sp == u
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
            c.fit(X[tr_], yp[tr_])
            P[te_] = c.predict_proba(X[te_])[:, [list(c.classes_).index(k) for k in THREE]]
        arg = np.array(THREE)[P.argmax(1)]
        hmm = np.empty(len(yp), dtype=object)
        for u in np.unique(sp):
            Tm_, pi_ = transitions(yp[(sp != u) & m3], rp[(sp != u) & m3], THREE)
            te_ = sp == u
            hmm[te_] = decode(P[te_], THREE, rp[te_], epp[te_], Tm_, pi_)
        R["within"][name] = {"argmax": metrics(arg, P), "hmm": metrics(hmm, P)}
        for k in ("argmax", "hmm"):
            v = R["within"][name][k]
            print(f"  within   {name:12s} {k:6s} kappa {v['kappa']:.3f} [{v['kappa_ci'][0]:.3f},{v['kappa_ci'][1]:.3f}] "
                  f"AUC W {v['auc_ws']['W']:.3f} N1 {v['auc_ws']['N1']:.3f} N2 {v['auc_ws']['N2']:.3f}", flush=True)
    (BASE / "results_cross_dataset.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_cross_dataset.npz", y=yp, subj=sp, run=rp, epoch=epp, t=tp, proba5_transfer=P5.astype(np.float32),
                        pred3_transfer_argmax=arg3, pred3_transfer_hmm=hmm3)
    print("saved results_cross_dataset.json, predictions_cross_dataset.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.features:
        features()
    if a.test:
        test()


if __name__ == "__main__":
    main()
