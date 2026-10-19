"""Paired N3-minus-REM contrast of the BOLD gain, and t-based bounds, from the saved probabilities.

The gain is a within-subject AUC difference per stage; the contrast is formed PER SUBJECT, in the
subjects eligible for both N3 and REM, and bootstrapped over those subjects (2,000 resamples, seed 0).
(24_experiments_v2.py before 2026-10-15 subtracted replicates of two separate bootstraps over 12 and 10
subjects, which is not a paired statistic.) Estimators: early fusion (logistic regression), late fusion
(product of the two models' posteriors) and gradient-boosted trees, on both streams. Also the one-sided
95% t-based upper bound of the REM gain, beside the bootstrap one. On the derivative, also:
  * headroom-scaled gain per subject, dAUC / (1 - AUC_nonBOLD), for N3 and REM
  * REM against each competing stage separately (epochs of REM and that stage only, REM probability)
  * a clock check: within N2 epochs, how well each model's REM probability separates second-half from
    first-half epochs (within-subject AUC) -- a REM score that were only a disguised clock would do so well

    source <venv>/bin/activate
    python 33_contrast.py                     # writes results_contrast.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as tdist

BASE = Path(__file__).parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
MIN_EP, N_BOOT = 20, 2000


def auc(y01, p):
    n1 = y01.sum()
    n0 = len(y01) - n1
    return (rankdata(p)[y01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def gains(y, s, pa, pb, st):
    i = STAGES.index(st)
    out = {}
    for u in np.unique(s):
        m = s == u
        y01 = y[m] == st
        if y01.sum() >= MIN_EP and (~y01).sum() >= MIN_EP:
            out[u] = auc(y01, pa[m, i]) - auc(y01, pb[m, i])
    return out


def main():
    R = {}
    for src in ("afni", "raw"):
        Z = np.load(BASE / f"predictions_v2_{src}.npz", allow_pickle=True)
        y, s = Z["y"].astype(str), Z["subj"].astype(str)
        lf = Z["proba__nonbold"] * Z["proba__bold"]
        lf = lf / lf.sum(1, keepdims=True)
        est = {"lr": (Z["proba__nonbold+bold"], Z["proba__nonbold"]), "latefusion": (lf, Z["proba__nonbold"]),
               "hgb": (Z["proba__hgb:nonbold+bold"], Z["proba__hgb:nonbold"]),
               "physio+clock": (Z["proba__physio+clock+bold"], Z["proba__physio+clock"])}
        if src == "afni":
            S2 = np.load(BASE / "predictions_stacked.npz", allow_pickle=True)
            assert (S2["y"].astype(str) == y).all()
            est["stack"] = (S2["stack_both"], S2["stack_nonbold"])
        R[src] = {}
        for k, (pa, pb) in est.items():
            g3, gr = gains(y, s, pa, pb, "N3"), gains(y, s, pa, pb, "REM")
            both = sorted(set(g3) & set(gr))
            d = np.array([g3[u] - gr[u] for u in both])
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            rem = np.array(list(gr.values()))
            R[src][k] = dict(contrast=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             n=len(d), n_n3_above_rem=int((d > 0).sum()),
                             rem_mean=float(rem.mean()),
                             rem_upper95_t=float(rem.mean() + tdist.ppf(0.95, len(rem) - 1) * rem.std(ddof=1) / np.sqrt(len(rem))))
            r = R[src][k]
            print(f"{src:4s} {k:12s} N3-REM {r['contrast']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                  f"N3>REM {r['n_n3_above_rem']}/{r['n']} | REM gain {r['rem_mean']:+.4f}, t-upper95 {r['rem_upper95_t']:+.4f}")
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, nf = Z["y"].astype(str), Z["subj"].astype(str), Z["night_frac"]
    pa, pb = Z["proba__nonbold+bold"], Z["proba__nonbold"]

    def boot_ci(v):
        rng = np.random.default_rng(0)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    lf = Z["proba__nonbold"] * Z["proba__bold"]
    lf = lf / lf.sum(1, keepdims=True)
    S2 = np.load(BASE / "predictions_stacked.npz", allow_pickle=True)
    ests = {"lr": (pa, pb), "latefusion": (lf, pb), "hgb": (Z["proba__hgb:nonbold+bold"], Z["proba__hgb:nonbold"]),
            "stack": (S2["stack_both"], S2["stack_nonbold"])}
    # headroom as a ratio of means (a mean of per-subject ratios explodes near AUC 1), paired bootstrap
    R["headroom"] = {}
    for st in ("N3", "REM"):
        i = STAGES.index(st)
        d, h = [], []
        for u in np.unique(s):
            m = s == u
            y01 = y[m] == st
            if y01.sum() >= MIN_EP and (~y01).sum() >= MIN_EP:
                a0_, a1_ = auc(y01, pb[m, i]), auc(y01, pa[m, i])
                d.append(a1_ - a0_); h.append(1 - a0_)
        d, h = np.array(d), np.array(h)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(N_BOOT):
            j = rng.integers(0, len(d), len(d))
            bs.append(d[j].mean() / h[j].mean())
        R["headroom"][st] = dict(mean=float(d.mean() / h.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 n=len(d))
    # REM against each competing stage, under each of the four estimators
    R["rem_vs"] = {}
    i = STAGES.index("REM")
    for k, (qa, qb) in ests.items():
        R["rem_vs"][k] = {}
        for other in ("W", "N1", "N2", "N3"):
            g = []
            for u in np.unique(s):
                m = (s == u) & np.isin(y, ["REM", other])
                y01 = y[m] == "REM"
                if y01.sum() >= MIN_EP and (~y01).sum() >= MIN_EP:
                    g.append(auc(y01, qa[m, i]) - auc(y01, qb[m, i]))
            g = np.array(g)
            R["rem_vs"][k][other] = dict(mean=float(g.mean()), ci=boot_ci(g), n=len(g), n_pos=int((g > 0).sum()))
    R["clock_check_n2"] = {}
    PB = np.load(BASE / "predictions_block_subc.npz", allow_pickle=True)
    assert (PB["y"].astype(str) == y).all()
    for k in ("bold", "nonbold", "subcortex"):
        p = (PB["block"] if k == "subcortex" else Z[f"proba__{k}"])[:, i]
        v = []
        for u in np.unique(s):
            m = (s == u) & (y == "N2")
            late = nf[m] >= 0.5
            if late.sum() >= MIN_EP and (~late).sum() >= MIN_EP:
                v.append(auc(late, p[m]))
        R["clock_check_n2"][k] = dict(mean=float(np.mean(v)), n=len(v))
    print("headroom", {k: round(v["mean"], 3) for k, v in R["headroom"].items()},
          "| REM vs", {k: {o: round(v["mean"], 3) for o, v in d.items()} for k, d in R["rem_vs"].items()},
          "| clock check", {k: round(v["mean"], 3) for k, v in R["clock_check_n2"].items()})
    (BASE / "results_contrast.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
