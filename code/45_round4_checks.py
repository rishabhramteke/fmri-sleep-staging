"""Checks asked by the PDF-only review of draft 12 (post hoc, exploratory).

  stratified  REM (and N3) AUC with time of night controlled by design: within each subject, AUC within 1 h bins of
              elapsed time (bins with >= 5 epochs of the stage and >= 5 of the rest), pooled over bins by number of
              pairs; gain of the main non-BOLD + BOLD model over non-BOLD
  ap          within-subject average-precision gain (REM is 9% of epochs; ROC AUC can hide low-false-positive gains)
  tissue      negative control: white-matter and CSF signals of the derivative (mean and s.d. over the epoch, s.d. over
              90 s) as a block; its gain over non-BOLD, and BOLD's gain over non-BOLD + tissue
  eligible    the N3 gain on the 10 REM-eligible subjects only; the N3-REM asymmetry on the logit(AUC) scale

    source <venv>/bin/activate
    python 45_round4_checks.py                 # writes results_round4_checks.json (~5 min)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


x = _load("x36b", "36b_extra_block.py")
e = x.e
f22 = _load("f22", "22_features_v2.py")
feat = f22.feat
ST = e.STAGES


def boot(v, n=2000):
    v = np.asarray(v)
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                t_upper95=x.tb(v), n_pos=int((v > 0).sum()), n=len(v))


def stratified_auc(y01, p, hours, min_n=5):
    num = den = 0.0
    for h in np.unique(np.floor(hours)):
        m = np.floor(hours) == h
        a, b = y01[m].sum(), (~y01[m]).sum()
        if a >= min_n and b >= min_n:
            num += e.auc(y01[m], p[m]) * a * b
            den += a * b
    return num / den if den else np.nan


def main():
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all()
    Pnb, Pall = Z["proba__nonbold"], Z["proba__nonbold+bold"]
    hours = D["clock"][:, 0]
    R = {}
    # ---------------- time-stratified AUC
    R["stratified"] = {}
    for st in ("N3", "REM"):
        i = ST.index(st)
        g = []
        for u in np.unique(s):
            m = s == u
            y01 = y[m] == st
            if y01.sum() >= 20 and (~y01).sum() >= 20:
                a, b = stratified_auc(y01, Pall[m, i], hours[m]), stratified_auc(y01, Pnb[m, i], hours[m])
                if np.isfinite(a) and np.isfinite(b):
                    g.append(a - b)
        R["stratified"][st] = boot(g)
    # ---------------- average precision
    R["ap"] = {}
    for st in ("N3", "REM"):
        i = ST.index(st)
        g = [average_precision_score(y[s == u] == st, Pall[s == u, i]) - average_precision_score(y[s == u] == st, Pnb[s == u, i])
             for u in np.unique(s) if (y[s == u] == st).sum() >= 20]
        R["ap"][st] = boot(g)
    # ---------------- tissue negative control (derivative WM / CSF signals)
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    feats = {}
    for f in sorted((BASE / "derivatives" / "afni_ds005127").glob("sub-*/*_afni.npz")):
        stem = f.name.replace("_afni.npz", "")
        if stem not in raw_runs:
            continue
        eps = feat.run_epochs(raw_runs[stem])
        if not eps:
            continue
        z = np.load(f)
        T, tr = int(z["n_vol"]), float(z["tr"])
        valid = ~z["censored"].astype(bool)
        sig = f22.psc(np.vstack([z["wm"], z["csf"]]).astype(np.float64), valid)
        for k in eps:
            a, n = k["start_vol"], k["n_vol"]
            c = a + n / 2
            w0, w1 = f22.window(c, 90.0, tr, T)
            row = []
            for v in sig:
                seg = v[a:a + n][np.isfinite(v[a:a + n])]
                win = v[w0:w1][np.isfinite(v[w0:w1])]
                row += [seg.std() if seg.size >= 5 else np.nan, seg.mean() if seg.size >= 5 else np.nan,
                        win.std() if win.size >= 15 else np.nan]
            feats[(stem, k["epoch"])] = row
    Xt = np.array([feats.get((r, t), [np.nan] * 6) for r, t in zip(run, ep)])
    ok = np.isfinite(Xt).all(1)
    Xt = e.wz(Xt, s)
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Pn = e.loso(nb[ok], y[ok], s[ok], ST)
    Pt = e.loso(np.hstack([nb, Xt])[ok], y[ok], s[ok], ST)
    Ptb = e.loso(np.hstack([nb, Xt, B["bold"]])[ok], y[ok], s[ok], ST)
    strip = lambda G: {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}
    R["tissue"] = dict(n_epochs=int(ok.sum()), tissue_over_nonbold=strip(x.gains(Pt, Pn, y[ok], s[ok])),
                       bold_over_nonbold_tissue=strip(x.gains(Ptb, Pt, y[ok], s[ok])))
    # ---------------- N3 on REM-eligible subjects; logit-scale asymmetry
    J = json.load(open(BASE / "results_v2_afni.json"))["increment"]["nonbold+bold - nonbold"]
    a3, ar = J["N3"]["ws_values"], J["REM"]["ws_values"]
    both = sorted(set(a3) & set(ar))
    R["n3_on_rem_eligible"] = boot([a3[u] for u in both])
    lg = lambda p: np.log(p / (1 - p))
    lo = {}
    for st in ("N3", "REM"):
        i = ST.index(st)
        lo[st] = {u: lg(e.auc(y[s == u] == st, Pall[s == u, i])) - lg(e.auc(y[s == u] == st, Pnb[s == u, i]))
                  for u in both}
    R["logit_contrast"] = boot([lo["N3"][u] - lo["REM"][u] for u in both])
    R["logit_gain"] = {st: boot(list(lo[st].values())) for st in lo}
    for k in ("stratified", "ap", "logit_gain"):
        print(k, {st: (round(v["mean"], 3), [round(c, 3) for c in v["ci"]], v["n_pos"], v["n"]) for st, v in R[k].items()})
    for k in ("tissue_over_nonbold", "bold_over_nonbold_tissue"):
        G = R["tissue"][k]
        print(k, {st: round(G[st]["ws_mean"], 3) for st in ST}, "N3 CI", [round(c, 3) for c in G["N3"]["ws_ci"]])
    print("N3 on REM-eligible:", R["n3_on_rem_eligible"]["mean"], " logit contrast:", R["logit_contrast"])
    (BASE / "results_round4_checks.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
