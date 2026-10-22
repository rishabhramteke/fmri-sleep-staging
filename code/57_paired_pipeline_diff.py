"""Paired tests behind "preprocessing-dependent" (post hoc; asked by the author on 2026-10-22).

The paper compares BOLD's N3 gain across pipelines on the same subjects and epochs (Table 3b: plain non-BOLD baseline,
planned LR; Table 4: stronger + eyes baseline, LR / boosted trees / BiGRU). Both models of a comparison share one
baseline, so the per-subject difference of two gains equals the difference of the two with-BOLD AUCs. This script
gives, for every pipeline pair, the mean per-subject difference, a percentile subject-bootstrap 95% CI (2,000
resamples, seed 0, as elsewhere), the one-sided 95% t bounds, a Wilcoxon signed-rank p and the number of subjects
with a positive difference; and, for Table 3b, the fraction of the derivative's advantage that rebuilding recovers,
with a bootstrap CI. Nothing is refitted: only saved predictions are read.

    source <venv>/bin/activate
    python 57_paired_pipeline_diff.py          # results_paired_pipeline.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
MIN_EP = 20
N_BOOT = 2000


def ws_auc(y, p, s, st):
    """Within-subject one-vs-rest AUC per assessable subject (>= 20 epochs of the stage and of the rest)."""
    i, out = STAGES.index(st), {}
    for u in np.unique(s):
        m = s == u
        pos = y[m] == st
        if pos.sum() >= MIN_EP and (~pos).sum() >= MIN_EP:
            out[str(u)] = float(roc_auc_score(pos, p[m, i]))
    return out


def summarise(d):
    """d: per-subject differences (dict). Mean, bootstrap CI, one-sided t bounds, Wilcoxon p, n_pos."""
    v = np.array(list(d.values()))
    rng = np.random.default_rng(0)
    bs = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
    h = tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                t_lower95=float(v.mean() - h), t_upper95=float(v.mean() + h),
                wilcoxon_p=float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0,
                n_pos=int((v > 0).sum()), n=int(len(v)), values=d)


def pair(y, s, pa, pb, st):
    a, b = ws_auc(y, pa, s, st), ws_auc(y, pb, s, st)
    return summarise({u: a[u] - b[u] for u in a if u in b})


def main():
    R = {"meta": dict(note="per-subject difference of with-BOLD AUCs = difference of gains (shared baseline)",
                      n_boot=N_BOOT, min_epochs=MIN_EP)}
    # ---- Table 4: stronger + eyes baseline, three estimators, 12,509 epochs
    R["allcontrols"] = {}
    for est in ("lr", "hgb", "gru"):
        if est == "gru":
            P = {}
            for k in ("deriv84", "rebuilt_aff", "rebuilt_syngm", "minimal84"):
                Z = np.load(BASE / f"predictions_combined_controls_gru_{k}.npz", allow_pickle=True)
                P[k] = Z["proba__" + k]; y, s = Z["y"].astype(str), Z["subj"].astype(str)
        else:
            Z = np.load(BASE / f"predictions_combined_controls_{est}.npz", allow_pickle=True)
            P = {k: Z["proba__" + k] for k in ("deriv84", "rebuilt_aff", "rebuilt_syngm", "minimal84")}
            y, s = Z["y"].astype(str), Z["subj"].astype(str)
        R["allcontrols"][est] = {}
        for a, b in (("deriv84", "rebuilt_aff"), ("deriv84", "rebuilt_syngm"), ("deriv84", "minimal84"), ("rebuilt_aff", "minimal84")):
            R["allcontrols"][est][f"{a}-{b}"] = {st: pair(y, s, P[a], P[b], st) for st in ("N3", "REM")}
    # ---- Table 3b: plain non-BOLD baseline, planned LR, 12,694 epochs
    Z = np.load(BASE / "predictions_preproc_ladder.npz", allow_pickle=True)
    y, s = Z["y"].astype(str), Z["subj"].astype(str)
    W = {k: Z["proba__nonbold+" + k] for k in ("afni84", "amp_aff", "amp_aff_accmotphys", "amp_syn_gm_accmotphys")}
    nb = Z["proba__nonbold"]
    R["ladder"] = {}
    for a, b in (("afni84", "amp_aff"), ("afni84", "amp_aff_accmotphys"), ("afni84", "amp_syn_gm_accmotphys"),
                 ("amp_aff_accmotphys", "amp_aff"), ("amp_syn_gm_accmotphys", "amp_aff_accmotphys")):
        R["ladder"][f"{a}-{b}"] = {st: pair(y, s, W[a], W[b], st) for st in ("N3", "REM")}
    # fraction of the derivative's N3 advantage that the rebuild recovers: mean(g1 - g0) / mean(gd - g0), in gains
    g = {k: ws_auc(y, W[k], s, "N3") for k in ("afni84", "amp_aff", "amp_aff_accmotphys")}
    subs = sorted(set(g["afni84"]) & set(g["amp_aff"]) & set(g["amp_aff_accmotphys"]))
    num = np.array([g["amp_aff_accmotphys"][u] - g["amp_aff"][u] for u in subs])
    den = np.array([g["afni84"][u] - g["amp_aff"][u] for u in subs])
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, len(subs), len(subs))
        fr.append(num[ii].mean() / den[ii].mean())
    R["ladder"]["recovered_fraction_N3"] = dict(value=float(num.mean() / den.mean()),
                                               ci=[float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))], n=len(subs))
    # sanity: the per-subject gain differences reproduce the ladder's reported gain differences
    LD = json.load(open(BASE / "results_preproc_ladder.json"))
    for a, b in (("afni84", "amp_aff_accmotphys"),):
        want = LD["increment"][a]["N3"]["ws_mean"] - LD["increment"][b]["N3"]["ws_mean"]
        got = R["ladder"][f"{a}-{b}"]["N3"]["mean"]
        assert abs(want - got) < 2e-3, (want, got)
    (BASE / "results_paired_pipeline.json").write_text(json.dumps(R, indent=1))
    for grp in ("allcontrols",):
        for est, D in R[grp].items():
            for k, v in D.items():
                n3 = v["N3"]
                print(f"{grp:11s} {est:3s} {k:24s} N3 diff {n3['mean']:+.3f} [{n3['ci'][0]:+.3f},{n3['ci'][1]:+.3f}] "
                      f"t lo {n3['t_lower95']:+.3f} p {n3['wilcoxon_p']:.3f} {n3['n_pos']}/{n3['n']}")
    for k, v in R["ladder"].items():
        if k.startswith("recovered"):
            print(f"ladder recovered fraction {v['value']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}] n={v['n']}")
            continue
        n3 = v["N3"]
        print(f"ladder      lr  {k:40s} N3 diff {n3['mean']:+.3f} [{n3['ci'][0]:+.3f},{n3['ci'][1]:+.3f}] "
              f"t lo {n3['t_lower95']:+.3f} p {n3['wilcoxon_p']:.3f} {n3['n_pos']}/{n3['n']}")


if __name__ == "__main__":
    main()
