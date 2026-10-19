"""Tempered emissions for the HMM stager: does the combined model stop losing kappa to BOLD alone?

Draft 17 reports that adding the non-BOLD signals to BOLD lowers decoded kappa (0.41 against 0.45) although the
per-epoch kappa is higher, because the fused posteriors are more confident and smoother, so the HMM has less to add.
The standard remedy is to weight emissions against transitions: the emission is P^tau (tau < 1 flattens overconfident
posteriors). tau is chosen by nested cross-validation inside each outer training fold (inner leave-one-subject-out
over the training subjects, decoded with transitions from the inner training subjects, kappa over the pooled inner
predictions), so the held-out subject never informs it. Models: bold, nonbold, nonbold+bold (early fusion), late
fusion (product of the two posteriors); the main-analysis derivative features.

    source <venv>/bin/activate
    python 50_tempered_hmm.py                  # writes results_tempered_hmm.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parent
TAUS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
bl, ST = e.bl, e.STAGES


def decode(P, y, subj, run, epoch, train_mask, test_subj, tau):
    """Viterbi decoding of one subject's runs with emissions P^tau and transitions from train_mask."""
    T = bl.transition_matrix(y[train_mask], run[train_mask])
    pi = np.array([(y[train_mask] == x).mean() for x in ST])
    out = {}
    te = subj == test_subj
    for r in np.unique(run[te]):
        idx = np.flatnonzero(run == r)
        idx = idx[np.argsort(epoch[idx])]
        path = bl.viterbi(tau * np.log(np.clip(P[idx], 1e-9, None)), np.log(T), np.log(pi + 1e-9))
        out.update(zip(idx.tolist(), np.array(ST)[path]))
    return out


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, rn, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    X = {"bold": B["bold"], "nonbold": np.hstack([B["motion"], B["physio"], B["clock"]])}
    X["nonbold+bold"] = np.hstack([X["nonbold"], X["bold"]])
    subs = np.unique(s)
    # outer LOSO posteriors (as the paper) and, per outer fold, inner LOSO posteriors over the training subjects
    P = {k: e.loso(v, y, s, ST) for k, v in X.items()}
    P["latefusion"] = e.late_fusion(P["nonbold"], P["bold"], y, s)
    print(f"outer posteriors in {time.time() - t0:.0f} s", flush=True)
    inner = {}
    for u in subs:
        tr = s != u
        Pi = {k: np.full((len(y), len(ST)), np.nan) for k in X}
        for k, v in X.items():
            Pi[k][tr] = e.loso(v[tr], y[tr], s[tr], ST)
        Pi["latefusion"] = Pi["nonbold"] * Pi["bold"]
        Pi["latefusion"] /= Pi["latefusion"].sum(1, keepdims=True)
        inner[u] = Pi
        print(f"  inner posteriors for fold {u} done ({time.time() - t0:.0f} s)", flush=True)
    R = {"meta": dict(n_epochs=int(len(y)), n_subjects=int(len(subs)), taus=list(TAUS)), "models": {}}
    pred_nested = {}
    for k in P:
        # diagnostic sweep: one tau for all folds (not nested)
        sweep = {}
        for tau in TAUS:
            pred = np.empty(len(y), dtype=object)
            for u in subs:
                for i, lab in decode(P[k], y, s, rn, ep, s != u, u, tau).items():
                    pred[i] = lab
            sweep[str(tau)] = float(cohen_kappa_score(y, pred))
        # nested choice of tau per outer fold
        pred = np.empty(len(y), dtype=object)
        chosen = {}
        for u in subs:
            tr = s != u
            best, best_k = None, -np.inf
            for tau in TAUS:
                pi_ = np.empty(len(y), dtype=object)
                for v in subs[subs != u]:
                    for i, lab in decode(inner[u][k], y, s, rn, ep, tr & (s != v), v, tau).items():
                        pi_[i] = lab
                kk = cohen_kappa_score(y[tr], pi_[tr])
                if kk > best_k:
                    best, best_k = tau, kk
            chosen[str(u)] = best
            for i, lab in decode(P[k], y, s, rn, ep, tr, u, best).items():
                pred[i] = lab
        pred_nested[k] = pred
        met = e.staging(y, pred, s)
        met["kappa_ci"] = e.boot_kappa(y, pred, s)
        met["tau_per_fold"] = chosen
        met["sweep_kappa"] = sweep
        met["kappa_tau1"] = sweep["1.0"]
        met["kappa_argmax"] = float(cohen_kappa_score(y, np.array(ST)[P[k].argmax(1)]))
        R["models"][k] = met
        print(f"  {k:14s} kappa tau=1 {sweep['1.0']:.3f} -> nested {met['kappa']:.3f} [{met['kappa_ci'][0]:.3f},{met['kappa_ci'][1]:.3f}] "
              f"REM F1 {met['f1']['REM']:.2f} N3 F1 {met['f1']['N3']:.2f}; tau {sorted(set(chosen.values()))}; "
              f"sweep {' '.join(f'{t}:{v:.3f}' for t, v in sweep.items())}", flush=True)
    R["paired"] = {}
    for a, b in (("nonbold+bold", "bold"), ("latefusion", "bold"), ("nonbold+bold", "nonbold")):
        pa, pb = R["models"][a]["per_subject_kappa"], R["models"][b]["per_subject_kappa"]
        R["paired"][f"{a} vs {b}"] = dict(diff=R["models"][a]["kappa"] - R["models"][b]["kappa"],
                                          ci=e.boot_kappa(y, pred_nested[a], s, other=pred_nested[b]),
                                          n_pos=int(sum(pa[u] > pb[u] for u in pa)), n=len(pa))
        print(f"  {a} - {b}: {R['paired'][f'{a} vs {b}']}", flush=True)
    (BASE / "results_tempered_hmm.json").write_text(json.dumps(R, indent=1))
    print(f"saved results_tempered_hmm.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
