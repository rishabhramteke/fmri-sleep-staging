"""Cross-site follow-up: the Penn State nap dataset (ds003768) through the same realigned, SyN-aligned, nuisance-regressed
chain as ds005127 (49_preproc_ladder.py), then (i) within-site staging references, (ii) cross-site transfer of the hand
features with the identical pipeline on both sides, (iii) a joint self-supervised corpus (54_ssl_pretrain.py) for the
cross-site transfer of the embedding and for the "does a second site's unlabelled data help" question on ds005127.

ds003768 (Gu et al. 2023): 33 subjects, 189 sleep runs and 66 rest runs of 15 min (TR 2.1 s, 3 x 3 x 4 mm), EEG-scored in
30 s epochs as W/N1/N2 (N3 in 42 epochs, no REM), no physiological recordings. 52_cross_dataset.py tested the minimal
pipeline's features (within-site kappa 0.07, W-vs-N2 AUC 0.62; a label-shift scan peaks at zero lag, so the labels are
aligned). The ladder on ds005127 showed the minimal pipeline's amplitude is the weak signal, so the question is reopened
with the nuisance-regressed chain. The other public sleep-scored fMRI dataset (NIH sleep2, ds006163) is embargoed.

    source <venv>/bin/activate
    python 55_cross_site.py --motion [--workers 3]       # derivatives/motion_ds003768/<sub>/<run>_motion.npz (23_realign.py, sleep + rest)
    python 55_cross_site.py --sample [--workers 4 --runs-per-task 3]   # derivatives/rv3_ds003768/<sub>/<run>_rv3.npz (49 chain)
    python 55_cross_site.py --features                   # features_ds003768_rv3.npz (84 shared parcels; amp and network FC per variant/regression)
    python 55_cross_site.py --transfer                   # results_cross_site_hand.json (within-site, NIH->PSU, PSU->NIH; identical pipeline)
    python 55_cross_site.py --ssl-build --psu-source rv3|minimal   # ssl_windows_joint_<src>.npz (= ssl_windows.npz + PSU windows at TR 3 s),
                                                                   # windows_psu_<src>.npz (labelled 90 s windows)
    python 55_cross_site.py --ssl-eval --checkpoint ssl_checkpoint.pt --psu-source rv3 --tag nih_rv3   # results_cross_site_ssl_<tag>.json
"""
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
PSU = BASE / "data" / "ds003768"
MOT = BASE / "derivatives" / "motion_ds003768"
OUT = BASE / "derivatives" / "rv3_ds003768"
MIN = BASE / "derivatives" / "ds003768"
EPOCH_S = 30.0
TR_TARGET = 3.0
FD_CENSOR = 0.5
THREE = ["W", "N1", "N2"]
PSU_MAP = {"W": "W", "1": "N1", "2": "N2", "3": "N3"}
VARIANTS = ("aff", "aff_gm", "syn", "syn_gm")
REGRESSIONS = ("", "_acc", "_accmot")
WIN, STRIDE = 30, 5


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def hypnograms():
    """{run stem: DataFrame(t, stage)} for sleep AND rest runs. Files have 3 columns (session, t, stage) or 4 (subject
    first); one file has a merged '810 W' cell. '(uncertain)' and anything unmapped -> 'other' (dropped later)."""
    out = {}
    for f in sorted((PSU / "sourcedata").glob("sub-*-sleep-stage.tsv")):
        sub = f.name.split("-sleep")[0]
        rows = []
        for line in f.read_text().splitlines()[1:]:
            p = [x.strip() for x in line.split("\t") if x.strip() != ""]
            if p and " " in p[-1] and p[-1].split(" ")[0].replace(".", "").isdigit():
                a, b = p[-1].split(" ", 1)
                p = p[:-1] + [a, b]
            if len(p) < 3 or "task-" not in p[-3]:
                continue
            try:
                t = float(p[-2])
            except ValueError:
                continue
            rows.append((f"{sub}_{p[-3]}", t, PSU_MAP.get(p[-1], "other")))
        d = pd.DataFrame(rows, columns=["stem", "t", "stage"])
        for stem, g in d.groupby("stem"):
            out[stem] = g.sort_values("t")
    return out


# ------------------------------------------------------------------ motion (23_realign.py on ds003768)
def _motion_one(path):
    ra = _load("ra", "23_realign.py")
    ra.OUT = MOT
    return ra.one(path)


def motion(workers):
    paths = sorted(p for p in PSU.rglob("sub-*/func/*_bold.nii.gz"))
    print(f"{len(paths)} runs (sleep + rest)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_motion_one, str(p)): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                stem, status, dt = f.result()
                print(f"[{i}/{len(paths)}] {status} {stem} {dt:.0f}s ({time.time() - t0:.0f} s)", flush=True)
            except Exception as err:
                print(f"[{i}/{len(paths)}] FAIL {futs[f].name}: {err!r}", flush=True)


# ------------------------------------------------------------------ sample (49 chain on ds003768, sleep + rest runs)
def _ladder():
    L = _load("ladder", "49_preproc_ladder.py")
    L.RAW, L.MOT, L.OUT = PSU, MOT, OUT
    L.TASK_GLOB = "*task-*_bold.nii.gz"
    return L


def _sample_one(sub, shard):
    return _ladder().process_subject(sub, 0, shard)


def sample(workers, runs_per_task, subject=None):
    L = _ladder()
    subs = [subject] if subject else sorted(p.name for p in PSU.glob("sub-*") if p.is_dir())
    tasks = []
    for s in subs:
        n = len(L.remaining(s))
        k = max(1, -(-n // runs_per_task)) if runs_per_task else 1
        tasks += [(s, (i, k), n) for i in range(k)]
    tasks.sort(key=lambda x: -x[2])
    print(f"{len(tasks)} tasks over {len(subs)} subjects, {sum(x[2] for x in tasks if x[1][0] == 0)} runs remaining, {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_sample_one, s, sh): (s, sh) for s, sh, _ in tasks}
        for fu in as_completed(futs):
            try:
                s, msg = fu.result()
                print(f"[done {s} shard {futs[fu][1][0] + 1}/{futs[fu][1][1]}] {msg} ({time.time() - t0:.0f} s)", flush=True)
            except Exception as err:
                print(f"[FAILED {futs[fu]}] {err!r}", flush=True)


# ------------------------------------------------------------------ features (84 shared parcels, as 49 --features)
def _nuisance(z, stem, valid):
    """aCompCor-like (5 WM PCs + CSF) and the 24 motion parameters, standardised over valid frames."""
    Racc = np.column_stack([z["wm_pcs"].T, z["csf"]]).astype(np.float64)
    Racc = (Racc - Racc[valid].mean(0)) / (Racc[valid].std(0) + 1e-9)
    mp = np.load(MOT / stem.split("_")[0] / f"{stem}_motion.npz")["params"].astype(np.float64)
    dm = np.vstack([np.zeros((1, 6)), np.diff(mp, axis=0)])
    Rm = np.hstack([mp, dm, mp ** 2, dm ** 2])
    Rm = (Rm - Rm[valid].mean(0)) / (Rm[valid].std(0) + 1e-9)
    regs = {"": None, "_acc": Racc, "_accmot": np.hstack([Racc, Rm])}
    if not np.isfinite(Racc).all():
        regs = {"": None}
    return regs


def psu_epochs(H, stem, tr, T):
    n_vol = int(round(EPOCH_S / tr))
    eps = []
    for _, row in H[stem].iterrows():
        a = int(round(row.t / tr))
        if a + n_vol > T or row.stage == "other":
            continue
        eps.append(dict(start_vol=a, n_vol=n_vol, epoch=int(round(row.t / EPOCH_S)), stage=row.stage, t=float(row.t)))
    return eps


def features():
    f22 = _load("f22", "22_features_v2.py")
    net_ids, nets = f22.feat.network_ids()
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    H = hypnograms()
    files = sorted(OUT.glob("sub-*/*_rv3.npz"))
    print(f"rv3 ds003768: {len(files)} runs, {sum(1 for p in files if p.name.replace('_rv3.npz', '') in H)} with hypnograms; {common.sum()} shared parcels", flush=True)
    keys, blocks, qcs, n_other = [], {}, [], 0
    for i, p in enumerate(sorted(files), 1):
        stem = p.name.replace("_rv3.npz", "")
        if stem not in H:
            continue
        z = np.load(p)
        T, tr = int(z["n_vol"]), float(z["tr"])
        valid = ~z["censored"]
        if valid.sum() < 20:
            continue
        qcs.append(json.loads(str(z["qc"])))
        gs = z["gs"].astype(np.float64)
        gs = np.where(valid, (gs - gs[valid].mean()) / (gs[valid].std() + 1e-9), np.nan)
        regs = _nuisance(z, stem, valid)
        eps = psu_epochs(H, stem, tr, T)
        n_other += int((H[stem]["stage"] == "other").sum())
        for v in VARIANTS:
            ts = f22.psc(z["ts_" + v].astype(np.float64), valid)
            for tag in REGRESSIONS:
                R = regs.get(tag, None)
                x = ts if tag == "" else (f22.regress(ts, R, valid) if R is not None else None)
                for ep in eps:
                    if x is None:
                        amp, fc = np.full(common.sum(), np.nan), np.full(15, np.nan)
                    else:
                        amp, fc, _ = f22.epoch_block(x, gs, ep, common, net_ids, len(nets), tr, T)
                    blocks.setdefault(f"amp_{v}{tag}", []).append(amp)
                    blocks.setdefault(f"netfc_{v}{tag}", []).append(fc)
        for ep in eps:
            keys.append((stem.split("_")[0], stem, ep["epoch"], ep["t"], ep["stage"], "rest" if "task-rest" in stem else "sleep"))
        if i % 25 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] epochs {len(keys)}", flush=True)
    out = {k: np.array(v, np.float32) for k, v in blocks.items()}
    k = np.array(keys, dtype=object)
    print(f"epochs {len(keys)} (dropped {n_other} unscorable); stages {pd.Series(k[:, 4]).value_counts().to_dict()}; "
          f"subjects {len(set(k[:, 0]))}; NaN rows " + ", ".join(f"{n} {np.isnan(v).any(1).mean():.3f}" for n, v in out.items() if n.startswith("amp")), flush=True)
    np.savez_compressed(BASE / "features_ds003768_rv3.npz", subj=k[:, 0].astype(str), run=k[:, 1].astype(str), epoch=k[:, 2].astype(int),
                        t=k[:, 3].astype(np.float32), stage=k[:, 4].astype(str), task=k[:, 5].astype(str), common=common, qc=json.dumps(qcs), **out)
    print("saved features_ds003768_rv3.npz")


# ------------------------------------------------------------------ staging metrics shared by the hand-feature and embedding tests
def transitions(stage, run, classes):
    T = np.full((len(classes), len(classes)), 1.0)
    for r in np.unique(run):
        s = stage[run == r]
        for a, b in zip(s[:-1], s[1:]):
            if a in classes and b in classes:
                T[classes.index(a), classes.index(b)] += 1
    pi = np.array([(stage == c).mean() for c in classes])
    return T / T.sum(1, keepdims=True), pi / max(1e-9, pi.sum())


def decode(P, classes, run, epoch, Tm, pi):
    bl = _load("bl", "09_baselines.py")
    out = np.empty(len(run), dtype=object)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        idx = idx[np.argsort(epoch[idx])]
        path = bl.viterbi(np.log(np.clip(P[idx], 1e-9, None)), np.log(Tm), np.log(pi + 1e-9))
        out[idx] = np.array(classes)[path]
    return out


def metrics3(e, y, pred, P, subj):
    m = np.isin(y, THREE)
    out = dict(kappa=float(cohen_kappa_score(y[m], pred[m], labels=THREE)), kappa_ci=e.boot_kappa(y[m], pred[m], subj[m]),
               acc=float((pred[m] == y[m]).mean()),
               f1=dict(zip(THREE, map(float, f1_score(y[m], pred[m], labels=THREE, average=None, zero_division=0)))),
               confusion_rows_true=confusion_matrix(y[m], pred[m], labels=THREE).tolist())
    out["auc_ws"] = {c: float(np.mean(list(e.ws_auc((y[m] == c).astype(int), P[m][:, j], subj[m]).values()) or [np.nan])) for j, c in enumerate(THREE)}
    out["auc_ws_n"] = {c: len(e.ws_auc((y[m] == c).astype(int), P[m][:, j], subj[m])) for j, c in enumerate(THREE)}
    return out


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))


def three_class_tests(e, Xn, yn, sn, rn, epn, Xp, yp, sp, rp, epp, ST):
    """NIH -> PSU (5-class model, scored on W/N1/N2), within-PSU LOSO, PSU -> NIH (W/N1/N2 epochs), on matched feature spaces."""
    R = {}
    # NIH -> PSU
    c = lr().fit(Xn, yn)
    P5 = c.predict_proba(Xp)[:, [list(c.classes_).index(v) for v in ST]]
    arg5 = np.array(ST)[P5.argmax(1)]
    i3 = [ST.index(v) for v in THREE]
    P3 = P5[:, i3] / P5[:, i3].sum(1, keepdims=True)
    Tm, pi = transitions(yn, rn, THREE)
    R["nih_to_psu"] = {"pred5_distribution": {v: float((arg5 == v).mean()) for v in ST},
                       "argmax": metrics3(e, yp, np.array(THREE)[P3.argmax(1)], P3, sp),
                       "hmm_nih_transitions": metrics3(e, yp, decode(P3, THREE, rp, epp, Tm, pi), P3, sp)}
    # NIH -> PSU with a three-class NIH model (trained on the NIH W/N1/N2 epochs only; mirrors the PSU -> NIH direction)
    mn3 = np.isin(yn, THREE)
    c3 = lr().fit(Xn[mn3], yn[mn3])
    P3b = c3.predict_proba(Xp)[:, [list(c3.classes_).index(v) for v in THREE]]
    R["nih3_to_psu"] = {"argmax": metrics3(e, yp, np.array(THREE)[P3b.argmax(1)], P3b, sp),
                        "hmm_nih_transitions": metrics3(e, yp, decode(P3b, THREE, rp, epp, Tm, pi), P3b, sp)}
    # within PSU, LOSO on W/N1/N2
    m3 = np.isin(yp, THREE)
    P = np.zeros((len(yp), 3))
    hmm = np.empty(len(yp), dtype=object)
    for u in np.unique(sp):
        tr_, te_ = (sp != u) & m3, sp == u
        c = lr().fit(Xp[tr_], yp[tr_])
        P[te_] = c.predict_proba(Xp[te_])[:, [list(c.classes_).index(v) for v in THREE]]
        Tm_, pi_ = transitions(yp[tr_], rp[tr_], THREE)
        hmm[te_] = decode(P[te_], THREE, rp[te_], epp[te_], Tm_, pi_)
    R["psu_within"] = {"argmax": metrics3(e, yp, np.array(THREE)[P.argmax(1)], P, sp), "hmm": metrics3(e, yp, hmm, P, sp)}
    # PSU -> NIH on the NIH epochs scored W/N1/N2
    c = lr().fit(Xp[m3], yp[m3])
    mn = np.isin(yn, THREE)
    Pn = c.predict_proba(Xn[mn])[:, [list(c.classes_).index(v) for v in THREE]]
    R["psu_to_nih"] = metrics3(e, yn[mn], np.array(THREE)[Pn.argmax(1)], Pn, sn[mn])
    # within NIH, LOSO on W/N1/N2: the reference the PSU -> NIH direction is measured against
    yn3, sn3, Xn3 = yn[mn], sn[mn], Xn[mn]
    Pw = np.zeros((mn.sum(), 3))
    for u in np.unique(sn3):
        tr_, te_ = sn3 != u, sn3 == u
        c = lr().fit(Xn3[tr_], yn3[tr_])
        Pw[te_] = c.predict_proba(Xn3[te_])[:, [list(c.classes_).index(v) for v in THREE]]
    R["nih_within"] = metrics3(e, yn3, np.array(THREE)[Pw.argmax(1)], Pw, sn3)
    preds = dict(psu_y=yp, psu_subj=sp, psu_run=rp, psu_epoch=epp, nih3_y=yn3, nih3_subj=sn3, nih3_run=rn[mn], nih3_epoch=epn[mn],
                 proba3__nih_to_psu=P3.astype(np.float32), proba3__nih3_to_psu=P3b.astype(np.float32), proba3__psu_within=P.astype(np.float32),
                 proba3__psu_to_nih=Pn.astype(np.float32), proba3__nih_within=Pw.astype(np.float32))
    return R, preds


def _print(R, name):
    for k in ("nih_to_psu", "nih3_to_psu", "psu_within"):
        for kk, v in R[k].items():
            if isinstance(v, dict) and "kappa" in v:
                print(f"  {name:24s} {k:11s} {kk:20s} kappa {v['kappa']:.3f} [{v['kappa_ci'][0]:.3f},{v['kappa_ci'][1]:.3f}] "
                      f"AUC W {v['auc_ws']['W']:.3f} N1 {v['auc_ws']['N1']:.3f} N2 {v['auc_ws']['N2']:.3f}", flush=True)
    for k in ("psu_to_nih", "nih_within"):
        v = R[k]
        print(f"  {name:24s} {k:11s} {'argmax':20s} kappa {v['kappa']:.3f} [{v['kappa_ci'][0]:.3f},{v['kappa_ci'][1]:.3f}] "
              f"AUC W {v['auc_ws']['W']:.3f} N1 {v['auc_ws']['N1']:.3f} N2 {v['auc_ws']['N2']:.3f}", flush=True)


# ------------------------------------------------------------------ hand features, identical pipeline on both sites
def nih_rv3(e):
    """ds005127 rv3 features (49) with labels, as 49 gains(): rows finite for every variant used here."""
    Dr = e.load("raw")
    V3 = np.load(BASE / "features_v2_rv3.npz", allow_pickle=True)
    k3 = {(r, int(p)): i for i, (r, p) in enumerate(zip(V3["run"].astype(str), V3["epoch"]))}
    kr = {(r, int(p)): i for i, (r, p) in enumerate(zip(Dr["run"], Dr["epoch"]))}
    names = [f"amp_{v}{t}" for v in VARIANTS for t in REGRESSIONS]
    fin = np.ones(len(V3["run"]), bool)
    for n in names:
        fin &= np.isfinite(V3[n]).all(1)
    keys = sorted(k for k in set(kr) & set(k3) if fin[k3[k]])
    ir, i3 = (np.array([d[k] for k in keys]) for d in (kr, k3))
    D = dict(stage=Dr["stage"][ir], subj=Dr["subj"][ir], run=Dr["run"][ir], epoch=Dr["epoch"][ir])
    for n in names:
        D[n] = V3[n][i3].astype(np.float64)
    return D


def transfer():
    e = _load("e24", "24_experiments_v2.py")
    ST = e.STAGES
    N = nih_rv3(e)
    Z = np.load(BASE / "features_ds003768_rv3.npz", allow_pickle=True)
    names = [f"amp_{v}{t}" for v in VARIANTS for t in REGRESSIONS]
    ok = np.ones(len(Z["stage"]), bool)
    for n in names:
        ok &= np.isfinite(Z[n]).all(1)
    yp, sp, rp, epp = Z["stage"].astype(str)[ok], Z["subj"].astype(str)[ok], Z["run"].astype(str)[ok], Z["epoch"].astype(int)[ok]
    R = {"meta": dict(nih=dict(n_epochs=int(len(N["stage"])), n_subjects=int(len(np.unique(N["subj"]))),
                               stage_counts={v: int((N["stage"] == v).sum()) for v in ST}),
                      psu=dict(n_epochs=int(ok.sum()), n_subjects=int(len(np.unique(sp))), n_runs=int(len(np.unique(rp))),
                               n_rest_runs=int(len(np.unique(rp[Z["task"].astype(str)[ok] == "rest"]))),
                               stage_counts={k: int(v) for k, v in zip(*np.unique(yp, return_counts=True))}),
                      variants=names, note="identical chain on both sites (49): realigned, affine or SyN atlas, GM prior, "
                                            "aCompCor-like WM PCs + CSF (_acc), + 24 motion parameters (_accmot)")}
    print(json.dumps(R["meta"]), flush=True)
    R["hand"], allp = {}, {}
    for n in names:
        Xn = e.wz(N[n], N["subj"])
        Xp = e.wz(Z[n][ok].astype(np.float64), sp)
        R["hand"][n], preds = three_class_tests(e, Xn, N["stage"], N["subj"], N["run"], N["epoch"], Xp, yp, sp, rp, epp, ST)
        _print(R["hand"][n], n)
        allp.update({k: v for k, v in preds.items() if not k.startswith("proba3__")})
        allp.update({f"{n}__{k}": v for k, v in preds.items() if k.startswith("proba3__")})
    np.savez_compressed(BASE / "predictions_cross_site_hand.npz", **allp)
    # time-in-run reference within PSU (nap runs drift from W into N2), as 52
    tp = Z["t"].astype(float)[ok]
    tt = np.column_stack([tp / 60.0, (tp / 60.0) ** 2])
    m3 = np.isin(yp, THREE)
    P = np.zeros((len(yp), 3))
    for u in np.unique(sp):
        tr_, te_ = (sp != u) & m3, sp == u
        c = lr().fit(tt[tr_], yp[tr_])
        P[te_] = c.predict_proba(tt[te_])[:, [list(c.classes_).index(v) for v in THREE]]
    R["psu_within_time"] = metrics3(e, yp, np.array(THREE)[P.argmax(1)], P, sp)
    print(f"  psu within, time in run only: kappa {R['psu_within_time']['kappa']:.3f} AUC {R['psu_within_time']['auc_ws']}", flush=True)
    (BASE / "results_cross_site_hand.json").write_text(json.dumps(R, indent=1))
    print("saved results_cross_site_hand.json")


# ------------------------------------------------------------------ self-supervised corpus and labelled windows at TR 3 s
def resample(x, tr, tr_target=TR_TARGET):
    """Linear interpolation of P x T series (NaN = censored) onto a tr_target grid; NaN wherever a neighbour is NaN."""
    T = x.shape[1]
    t_src = np.arange(T) * tr
    t_new = np.arange(0.0, t_src[-1] + 1e-6, tr_target)
    idx = np.clip(np.searchsorted(t_src, t_new, side="right") - 1, 0, T - 2)
    w = ((t_new - t_src[idx]) / tr)[None, :]
    return x[:, idx] * (1 - w) + x[:, idx + 1] * w


def psu_series(source):
    """{run stem: (x at TR 3 s, 200 x T', percent signal change, NaN where censored)}; rv3 = SyN + GM prior + aCompCor + motion."""
    f22 = _load("f22", "22_features_v2.py")
    out = {}
    if source == "rv3":
        for p in sorted(OUT.glob("sub-*/*_rv3.npz")):
            stem = p.name.replace("_rv3.npz", "")
            z = np.load(p)
            valid = ~z["censored"]
            if valid.sum() < 20:
                continue
            regs = _nuisance(z, stem, valid)
            ts = f22.psc(z["ts_syn_gm"].astype(np.float64), valid)
            x = f22.regress(ts, regs["_accmot"], valid) if "_accmot" in regs else ts
            out[stem] = resample(x, float(z["tr"]))
    else:
        for p in sorted(MIN.glob("sub-*/*_parcels.npz")):
            stem = p.name.replace("_parcels.npz", "")
            z = np.load(p)
            fd = z["fd"].astype(np.float64)
            cens = fd > FD_CENSOR
            valid = ~(cens | np.r_[cens[1:], False])
            x = f22.psc(z["ts"].astype(np.float64), valid)
            out[stem] = resample(x, float(z["tr"]))
    return out


def ssl_build(source):
    S = psu_series(source)
    H = hypnograms()
    X, M, subj, run = [], [], [], []
    LX, Lkeys = [], []
    per_subject = {}
    for stem, x in S.items():
        sub = "psu-" + stem.split("_")[0]
        x = x.astype(np.float32)
        per_subject.setdefault(sub, []).append(x)
        T = x.shape[1]
        for a in range(0, T - WIN + 1, STRIDE):
            w = x[:, a:a + WIN]
            m = np.isfinite(w).mean(0) > 0.5
            if m.sum() < WIN // 2:
                continue
            X.append(w); M.append(m); subj.append(sub); run.append(stem)
        if stem in H:                                        # labelled 90 s windows centred on each 30 s epoch (as 53 --build)
            for _, row in H[stem].iterrows():
                if row.stage == "other" or row.t + EPOCH_S > T * TR_TARGET:
                    continue
                c = (row.t + EPOCH_S / 2) / TR_TARGET
                c0, c1 = int(max(0, round(c - WIN / 2))), int(min(T, round(c + WIN / 2)))
                w = np.full((200, WIN), np.nan, np.float32)
                o = c0 - int(round(c - WIN / 2))
                n = min(c1 - c0, WIN - o)
                w[:, o:o + n] = x[:, c0:c0 + n]
                LX.append(w); Lkeys.append((sub, stem, int(round(row.t / EPOCH_S)), float(row.t), row.stage, "rest" if "task-rest" in stem else "sleep"))
    X, M, subj, run = np.array(X, np.float32), np.array(M), np.array(subj), np.array(run)
    for u, xs in per_subject.items():
        cat = np.concatenate(xs, 1)
        mu, sd = np.nanmean(cat, 1), np.nanstd(cat, 1) + 1e-6
        X[subj == u] = (X[subj == u] - mu[None, :, None]) / sd[None, :, None]
    X = np.nan_to_num(X, nan=0.0)
    Z = np.load(BASE / "ssl_windows.npz", allow_pickle=True)
    print(f"PSU ({source}): {len(X)} windows from {len(S)} runs, {len(per_subject)} subjects, {len(X) * STRIDE * 3 / 3600:.0f} h; "
          f"NIH corpus {len(Z['X'])} windows; joint {len(X) + len(Z['X'])}", flush=True)
    np.savez_compressed(BASE / f"ssl_windows_joint_{source}.npz", X=np.concatenate([Z["X"], X]), M=np.concatenate([Z["M"], M]),
                        subj=np.concatenate([Z["subj"].astype(str), subj]), run=np.concatenate([Z["run"].astype(str), run]))
    LX = np.array(LX, np.float32)
    k = np.array(Lkeys, dtype=object)
    Lsub = k[:, 0].astype(str)
    LM = np.isfinite(LX).mean(1) > 0.5
    for u in np.unique(Lsub):
        m = Lsub == u
        v = LX[m]
        mu = np.nanmean(v, axis=(0, 2), keepdims=True)
        sd = np.nanstd(v, axis=(0, 2), keepdims=True) + 1e-6
        LX[m] = (v - mu) / sd
    LX = np.nan_to_num(LX, nan=0.0)
    print(f"PSU labelled windows {LX.shape}, frames present {LM.mean():.3f}, stages {pd.Series(k[:, 4]).value_counts().to_dict()}", flush=True)
    np.savez_compressed(BASE / f"windows_psu_{source}.npz", X=LX, M=LM, subj=Lsub, run=k[:, 1].astype(str), epoch=k[:, 2].astype(int),
                        t=k[:, 3].astype(np.float32), stage=k[:, 4].astype(str), task=k[:, 5].astype(str))
    print(f"saved ssl_windows_joint_{source}.npz, windows_psu_{source}.npz")


def ssl_eval(ckpt, source, tag):
    import torch
    s54 = _load("s54", "54_ssl_pretrain.py")
    e = _load("e24", "24_experiments_v2.py")
    ST = e.STAGES
    dev = torch.device("cpu")
    ck = torch.load(BASE / ckpt, map_location=dev)
    enc = s54.Encoder(); enc.load_state_dict(ck["tgt"])
    D = e.load("afni")
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    W = np.load(BASE / "windows_learned_encoder.npz", allow_pickle=True)
    key = {(r, int(p)): i for i, (r, p) in enumerate(zip(W["run"].astype(str), W["epoch"]))}
    rows = np.array([key[(r, int(p))] for r, p in zip(rn, ep)])
    emb_n = e.wz(s54.embed(enc, W["X"][rows], W["M"][rows], dev), s)
    Wp = np.load(BASE / f"windows_psu_{source}.npz", allow_pickle=True)
    yp, sp, rp, epp = Wp["stage"].astype(str), Wp["subj"].astype(str), Wp["run"].astype(str), Wp["epoch"].astype(int)
    emb_p = e.wz(s54.embed(enc, Wp["X"], Wp["M"], dev), sp)
    win = ck.get("windows") or "ssl_windows.npz"               # checkpoints before the --windows option were NIH-only
    pre_subj = set(np.load(BASE / win, allow_pickle=True)["subj"].astype(str))
    R = {"meta": dict(checkpoint=ckpt, checkpoint_epoch=int(ck["epoch"]), exclude=ck.get("exclude"), windows=win,
                      psu_source=source, d_emb=int(emb_n.shape[1]),
                      nih=dict(n_epochs=int(len(y)), n_subjects=int(len(np.unique(s)))),
                      psu=dict(n_epochs=int(len(yp)), n_subjects=int(len(np.unique(sp))), n_runs=int(len(np.unique(rp))),
                               stage_counts={k: int(v) for k, v in zip(*np.unique(yp, return_counts=True))}),
                      psu_in_pretraining=any(u.startswith("psu-") for u in pre_subj))}
    print(json.dumps(R["meta"]), flush=True)
    R["embedding"], preds = three_class_tests(e, emb_n, y, s, rn, ep, emb_p, yp, sp, rp, epp, ST)
    _print(R["embedding"], f"emb[{tag}]")
    (BASE / f"results_cross_site_ssl_{tag}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_cross_site_ssl_{tag}.npz", **preds)
    print(f"saved results_cross_site_ssl_{tag}.json")


def ssl_eval_strict(tag, source):
    """Strict PSU -> NIH: for each NIH subject, the joint-corpus encoder pretrained WITHOUT that subject
    (ssl_checkpoint<tag><sub>.pt) embeds both sites; the W/N1/N2 readout is fitted on the Penn State labels and
    predicts that NIH subject. Same encoders give the strict within-NIH W/N1/N2 reference (LOSO on the other subjects)."""
    import torch
    s54 = _load("s54", "54_ssl_pretrain.py")
    e = _load("e24", "24_experiments_v2.py")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    D = e.load("afni")
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    W = np.load(BASE / "windows_learned_encoder.npz", allow_pickle=True)
    key = {(r, int(p)): i for i, (r, p) in enumerate(zip(W["run"].astype(str), W["epoch"]))}
    rows = np.array([key[(r, int(p))] for r, p in zip(rn, ep)])
    Wp = np.load(BASE / f"windows_psu_{source}.npz", allow_pickle=True)
    yp, sp = Wp["stage"].astype(str), Wp["subj"].astype(str)
    m3p = np.isin(yp, THREE)
    mn = np.isin(y, THREE)
    y3, s3 = y[mn], s[mn]
    Pn, Pw = np.zeros((mn.sum(), 3)), np.zeros((mn.sum(), 3))
    for u in np.unique(s):
        ck = torch.load(BASE / f"ssl_checkpoint{tag}{u}.pt", map_location="cpu")
        assert ck.get("exclude") == u, (u, ck.get("exclude"))
        enc = s54.Encoder(); enc.load_state_dict(ck["tgt"]); enc.to(dev)
        emb_n = e.wz(s54.embed(enc, W["X"][rows], W["M"][rows], dev), s)[mn]
        emb_p = e.wz(s54.embed(enc, Wp["X"], Wp["M"], dev), sp)
        te = s3 == u
        c = lr().fit(emb_p[m3p], yp[m3p])
        Pn[te] = c.predict_proba(emb_n[te])[:, [list(c.classes_).index(v) for v in THREE]]
        c2 = lr().fit(emb_n[~te], y3[~te])
        Pw[te] = c2.predict_proba(emb_n[te])[:, [list(c2.classes_).index(v) for v in THREE]]
        print(f"  fold {u} done", flush=True)
    R = {"meta": dict(strict=True, checkpoint_tag=tag, psu_source=source, n_nih_epochs3=int(mn.sum()), n_nih_subjects=int(len(np.unique(s))),
                      psu=dict(n_epochs3=int(m3p.sum()), n_subjects=int(len(np.unique(sp))))),
         "psu_to_nih": metrics3(e, y3, np.array(THREE)[Pn.argmax(1)], Pn, s3),
         "nih_within": metrics3(e, y3, np.array(THREE)[Pw.argmax(1)], Pw, s3)}
    for k in ("psu_to_nih", "nih_within"):
        v = R[k]
        print(f"  strict {k:11s} kappa {v['kappa']:.3f} [{v['kappa_ci'][0]:.3f},{v['kappa_ci'][1]:.3f}] AUC W {v['auc_ws']['W']:.3f} N1 {v['auc_ws']['N1']:.3f} N2 {v['auc_ws']['N2']:.3f}", flush=True)
    suf = tag.replace("_ex_", "")
    (BASE / f"results_cross_site_ssl_strict{suf}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_cross_site_ssl_strict{suf}.npz", nih3_y=y3, nih3_subj=s3, nih3_run=rn[mn], nih3_epoch=ep[mn],
                        proba3__psu_to_nih=Pn.astype(np.float32), proba3__nih_within=Pw.astype(np.float32))
    print(f"saved results_cross_site_ssl_strict{suf}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--ssl-build", action="store_true")
    ap.add_argument("--ssl-eval", action="store_true")
    ap.add_argument("--ssl-eval-strict", action="store_true", help="per-fold encoders ssl_checkpoint<--strict-tag><sub>.pt")
    ap.add_argument("--strict-tag", default="_joint_ex_")
    ap.add_argument("--psu-source", default="rv3", choices=("rv3", "minimal"))
    ap.add_argument("--checkpoint", default="ssl_checkpoint.pt")
    ap.add_argument("--tag", default="nih")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--runs-per-task", type=int, default=3)
    ap.add_argument("--subject", default=None)
    a = ap.parse_args()
    if a.motion:
        motion(a.workers)
    if a.sample:
        sample(a.workers, a.runs_per_task, a.subject)
    if a.features:
        features()
    if a.transfer:
        transfer()
    if a.ssl_build:
        ssl_build(a.psu_source)
    if a.ssl_eval:
        ssl_eval(a.checkpoint, a.psu_source, a.tag)
    if a.ssl_eval_strict:
        ssl_eval_strict(a.strict_tag, a.psu_source)


if __name__ == "__main__":
    main()
