"""Why adding the non-BOLD signals to BOLD lowers five-class kappa (derivative, saved HMM paths).

  * paired subject bootstrap of kappa(non-BOLD + BOLD) - kappa(BOLD), and per-subject direction
  * REM precision / recall / F1 after HMM decoding, and REM recall in the first vs second half of the
    night (night_frac < 0.5 vs >= 0.5), for BOLD, non-BOLD and both
  * kappa of the per-epoch argmax without the HMM, to separate the classifier from the decoder
  * the same kappa loss WITHOUT elapsed time (motion + physiology + BOLD, physiology + BOLD), decoded by
    24's HMM, and two emission diagnostics: lag-1 correlation of centred log-probabilities between
    adjacent epochs of the same true stage, and mean maximum probability (confidence)
  * an emission-weight sweep: decoding P^alpha (renormalised) for alpha in 0.5..3, which tests whether
    the HMM over- or under-weights a model's emissions; and the kappa the HMM adds over the argmax

    source <venv>/bin/activate
    python 32_staging_checks.py               # writes results_staging_checks.json
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

BASE = Path(__file__).parent
STAGES = np.array(["W", "N1", "N2", "N3", "REM"])
N_BOOT = 2000
ALPHAS = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0)


def main():
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, nf = Z["y"].astype(str), Z["subj"].astype(str), Z["night_frac"]
    H = {k: Z[f"hmm__{k}"].astype(str) for k in ("bold", "nonbold", "nonbold+bold")}
    subs = np.unique(s)
    idx = {u: np.flatnonzero(s == u) for u in subs}
    rng = np.random.default_rng(0)
    R = {}

    a, b = H["nonbold+bold"], H["bold"]
    boot = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[u] for u in rng.choice(subs, len(subs))])
        boot.append(cohen_kappa_score(y[ii], a[ii]) - cohen_kappa_score(y[ii], b[ii]))
    per = [cohen_kappa_score(y[idx[u]], a[idx[u]]) - cohen_kappa_score(y[idx[u]], b[idx[u]]) for u in subs]
    R["kappa_all_minus_bold"] = dict(diff=float(cohen_kappa_score(y, a) - cohen_kappa_score(y, b)),
                                     ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                     n_worse=int(sum(d < 0 for d in per)), n=len(per))

    R["rem"] = {}
    for k, p in H.items():
        pr, rc, f1, _ = precision_recall_fscore_support(y == "REM", p == "REM", average="binary", zero_division=0)
        early, late = (nf < 0.5) & (y == "REM"), (nf >= 0.5) & (y == "REM")
        R["rem"][k] = dict(precision=float(pr), recall=float(rc), f1=float(f1),
                           recall_early=float(np.mean(p[early] == "REM")), recall_late=float(np.mean(p[late] == "REM")),
                           n_early=int(early.sum()), n_late=int(late.sum()),
                           false_rem_late=float(np.mean(p[(nf >= 0.5) & (y != "REM")] == "REM")))
    R["argmax_kappa"] = {k: float(cohen_kappa_score(y, STAGES[Z[f"proba__{k}"].argmax(1)]))
                         for k in ("bold", "nonbold", "nonbold+bold")}

    # kappa loss without elapsed time: decode the saved probabilities with the identical HMM
    spec = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    run, epoch = Z["run"].astype(str), Z["epoch"]
    R["no_time"] = {}
    for k in ("motion+physio+bold", "physio+bold"):
        h = e.hmm_decode(Z[f"proba__{k}"], y, s, run, epoch).astype(str)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(N_BOOT):
            ii = np.concatenate([idx[u] for u in rng.choice(subs, len(subs))])
            boot.append(cohen_kappa_score(y[ii], h[ii]) - cohen_kappa_score(y[ii], b[ii]))
        early = (nf < 0.5) & (y == "REM")
        _, _, f1, _ = precision_recall_fscore_support(y == "REM", h == "REM", average="binary", zero_division=0)
        R["no_time"][k] = dict(kappa=float(cohen_kappa_score(y, h)),
                               diff_vs_bold=float(cohen_kappa_score(y, h) - cohen_kappa_score(y, b)),
                               ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                               f1_rem=float(f1), recall_early_rem=float(np.mean(h[early] == "REM")),
                               argmax_kappa=float(cohen_kappa_score(y, STAGES[Z[f"proba__{k}"].argmax(1)])))

    # emission diagnostics
    key = {(r_, int(t)): i for i, (r_, t) in enumerate(zip(run, epoch))}
    nxt = np.array([key.get((r_, int(t) + 1), -1) for r_, t in zip(run, epoch)])
    same = (nxt >= 0) & (y == y[np.where(nxt >= 0, nxt, 0)])
    R["emission"] = {}
    for k in ("bold", "nonbold", "nonbold+bold", "motion+physio+bold"):
        p = Z[f"proba__{k}"]
        lp = np.log(np.clip(p, 1e-9, 1))
        lp -= lp.mean(1, keepdims=True)
        a_, b_ = lp[same], lp[nxt[same]]
        R["emission"][k] = dict(lag1_r=float(np.mean([np.corrcoef(a_[:, j], b_[:, j])[0, 1] for j in range(5)])),
                                mean_max_p=float(p.max(1).mean()))
    R["alpha_sweep"] = {}
    for k in ("bold", "nonbold+bold", "motion+physio+bold"):
        R["alpha_sweep"][k] = {}
        for alpha in ALPHAS:
            q = np.clip(Z[f"proba__{k}"], 1e-12, 1) ** alpha
            q = q / q.sum(1, keepdims=True)
            R["alpha_sweep"][k][str(alpha)] = float(cohen_kappa_score(y, e.hmm_decode(q, y, s, run, epoch).astype(str)))
    R["hmm_gain"] = {k: float(cohen_kappa_score(y, H[k]) - R["argmax_kappa"][k]) for k in ("bold", "nonbold+bold")}
    R["bold_above_combined_all_alpha"] = bool(all(R["alpha_sweep"]["bold"][a_] > R["alpha_sweep"]["nonbold+bold"][a_]
                                                  for a_ in R["alpha_sweep"]["bold"]))
    print(json.dumps(R, indent=1))
    (BASE / "results_staging_checks.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
