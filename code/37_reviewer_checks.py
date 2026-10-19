"""Checks raised by the PDF-only review of draft 9 and the audit of earlier reviews (all post hoc, exploratory).

  bouts       N3 and REM gains on epochs inside long bouts (>= 5 and >= 10 min), and bout- vs epoch-weighted
              bout medians: is the N3 > REM asymmetry a bout-geometry artefact (short REM bouts vs 90 s FC windows)?
  uncensored  refit on epochs with no censored frame in the epoch, so every amplitude uses all 10 frames
  night       night index added to the non-BOLD model; elapsed-time REM AUC within each night; REM gain on night 2
  C sweep     the main early-fusion gains for C in 0.01, 0.1, 1, 10 (untuned; nested tuning is 38_tuning.py)
  stats       exact Holm p per stage, paired N3-REM Wilcoxon p, second-half t bound, smallest margin for the REM claim
  staging     per-stage F1 and confusion of BOLD and non-BOLD + BOLD; stacked fusion decoded by the same HMM;
              N3 gains of the two subjects without enough REM

    source <venv>/bin/activate
    python 37_reviewer_checks.py                 # writes results_reviewer_checks.json (~15 min)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

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


def bout_lengths(y, run, ep):
    """Length (epochs) of the same-stage bout each epoch sits in: consecutive analysed epochs of one stage in one run."""
    L = np.zeros(len(y), int)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        idx = idx[np.argsort(ep[idx])]
        a = 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or y[idx[k]] != y[idx[a]] or ep[idx[k]] != ep[idx[k - 1]] + 1:
                L[idx[a:k]] = k - a
                a = k
    return L


def masked_gains(pa, pb, y, s, mask):
    G = x.gains(pa[mask], pb[mask], y[mask], s[mask])
    return dict(N3=G["N3"], REM=G["REM"], contrast=x.contrast(G))


def short(G):
    """Drop per-subject bootstrap arrays; keep what the paper and 27_verify need."""
    out = {}
    for k, v in G.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != "boot_ws"} if isinstance(v, dict) else v
    return out


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all() and (Z["subj"] == s).all()
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Pnb, Pall = Z["proba__nonbold"], Z["proba__nonbold+bold"]
    R = {}

    # ---------------- bouts
    L = bout_lengths(y, run, ep)
    R["bout_median_epochs"] = {st: dict(epoch_weighted=float(np.median(L[y == st])),
                                        bout_weighted=float(np.median([l for r in np.unique(run) for l in
                                                                       _bouts_in_run(y, run, ep, r, st)])))
                               for st in ST}
    R["rem_epochs_in_bouts_ge5min"] = float((L[y == "REM"] >= 10).mean())
    R["n3_epochs_in_bouts_ge5min"] = float((L[y == "N3"] >= 10).mean())
    R["bouts"] = {}
    for lab, Lmin in (("ge5min", 10), ("ge10min", 20)):
        R["bouts"][lab] = short(masked_gains(Pall, Pnb, y, s, L >= Lmin))
        R["bouts"][lab]["n_epochs"] = int((L >= Lmin).sum())
        g = R["bouts"][lab]
        print(f"  bouts {lab}: N3 {g['N3']['ws_mean']:+.3f} ({g['N3']['n_pos']}/{g['N3']['n_subj']}) "
              f"REM {g['REM']['ws_mean']:+.3f} [{g['REM']['ws_ci'][0]:+.3f},{g['REM']['ws_ci'][1]:+.3f}] "
              f"t up {g['REM']['t_upper95']:+.3f} ({g['REM']['n_pos']}/{g['REM']['n_subj']}) "
              f"N3>REM {g['contrast']['n_pos']}/{g['contrast']['n']}", flush=True)

    # ---------------- fully uncensored epochs (refit on them)
    m = D["cens"][:, 0] == 0
    Pu_nb = e.loso(nb[m], y[m], s[m], ST)
    Pu_all = e.loso(np.hstack([nb[m], B["bold"][m]]), y[m], s[m], ST)
    R["uncensored"] = short(masked_gains(Pu_all, Pu_nb, y[m], s[m], np.ones(m.sum(), bool)))
    R["uncensored"].update(n_epochs=int(m.sum()), frac_epochs=float(m.mean()),
                           frac_by_stage={st: float(m[y == st].mean()) for st in ST})
    g = R["uncensored"]
    print(f"  uncensored ({m.sum()} epochs): N3 {g['N3']['ws_mean']:+.3f} REM {g['REM']['ws_mean']:+.3f} "
          f"t up {g['REM']['t_upper95']:+.3f} N3>REM {g['contrast']['n_pos']}/{g['contrast']['n']} ({time.time()-t0:.0f} s)",
          flush=True)

    # ---------------- night index, within-night time, night 2 only
    ses = np.array([r.split("_")[1] for r in run])
    night = e.wz((ses == "ses-2").astype(float)[:, None], s)
    Pn_nb = e.loso(np.hstack([nb, night]), y, s, ST)
    Pn_all = e.loso(np.hstack([nb, night, B["bold"]]), y, s, ST)
    R["night_index"] = short(masked_gains(Pn_all, Pn_nb, y, s, np.ones(len(y), bool)))
    R["night_index"]["auc_ws_nonbold_REM"] = float(np.mean(list(e.ws_auc((y == "REM").astype(int), Pn_nb[:, 4], s).values())))
    pc = Z["proba__clock"][:, ST.index("REM")]
    R["time_auc_rem_within_night"] = {}
    for lab in ("ses-1", "ses-2"):
        k = ses == lab
        v = e.ws_auc((y[k] == "REM").astype(int), pc[k], s[k])
        R["time_auc_rem_within_night"][lab] = dict(mean=float(np.mean(list(v.values()))), n=len(v))
    R["night2_only"] = short(masked_gains(Pall, Pnb, y, s, ses == "ses-2"))
    for k in ("night_index", "night2_only"):
        g = R[k]
        print(f"  {k}: N3 {g['N3']['ws_mean']:+.3f} REM {g['REM']['ws_mean']:+.3f} [{g['REM']['ws_ci'][0]:+.3f},"
              f"{g['REM']['ws_ci'][1]:+.3f}] t up {g['REM']['t_upper95']:+.3f} ({g['REM']['n_pos']}/{g['REM']['n_subj']}) "
              f"N3>REM {g['contrast']['n_pos']}/{g['contrast']['n']}", flush=True)
    print(f"  time-alone REM AUC within night: {R['time_auc_rem_within_night']} ({time.time()-t0:.0f} s)", flush=True)

    # ---------------- C sweep (early fusion, untuned)
    R["C_sweep"] = {}
    for C in (0.01, 0.1, 1.0, 10.0):
        if C == 1.0:
            pa, pb = Pall, Pnb
        else:
            pb = e.loso(nb, y, s, ST, C=C)
            pa = e.loso(np.hstack([nb, B["bold"]]), y, s, ST, C=C)
        g = short(masked_gains(pa, pb, y, s, np.ones(len(y), bool)))
        g["kappa_hmm_all"] = float(cohen_kappa_score(y, e.hmm_decode(pa, y, s, run, D["epoch"])))
        R["C_sweep"][str(C)] = g
        print(f"  C={C}: N3 {g['N3']['ws_mean']:+.3f} REM {g['REM']['ws_mean']:+.3f} t up {g['REM']['t_upper95']:+.3f} "
              f"N3>REM {g['contrast']['n_pos']}/{g['contrast']['n']} ({time.time()-t0:.0f} s)", flush=True)

    # ---------------- statistics from the main result
    J = json.load(open(BASE / "results_v2_afni.json"))
    inc = J["increment"]["nonbold+bold - nonbold"]
    R["holm_p"] = {st: inc[st]["wilcoxon_p_holm"] for st in ST}
    a, b = inc["N3"]["ws_values"], inc["REM"]["ws_values"]
    both = sorted(set(a) & set(b))
    R["contrast_wilcoxon_p"] = float(wilcoxon([a[u] - b[u] for u in both]).pvalue)
    tm = J["time_matched_rem"]["inc_nonbold"]
    R["second_half"] = dict(mean=tm["ws_mean"], ci=tm["ws_ci"], boot_upper=tm["ws_upper95"],
                            t_upper=x.tb(list(tm["ws_values"].values())), n=tm["n_subj"], n_pos=tm["n_pos"])
    rem_vals = list(b.values())
    R["smallest_margin_rem"] = dict(boot=inc["REM"]["ws_upper95"], t=x.tb(rem_vals))
    R["rem_ineligible_n3"] = {u: a[u] for u in a if u not in b}
    print(f"  holm {R['holm_p']}  contrast p {R['contrast_wilcoxon_p']:.4f}  second half t up "
          f"{R['second_half']['t_upper']:+.3f}  margin {R['smallest_margin_rem']}", flush=True)

    # ---------------- staging detail
    R["staging"] = {}
    for k in ("bold", "nonbold", "nonbold+bold"):
        L_ = J["ladder"][k]
        R["staging"][k] = dict(kappa=L_["kappa"], f1=L_["f1"], confusion=L_["confusion"])
    S = np.load(BASE / "predictions_stacked.npz", allow_pickle=True)
    assert (S["y"] == y).all()
    path = e.hmm_decode(S["stack_both"], y, s, run, D["epoch"])
    f1 = f1_score(y, path, labels=ST, average=None, zero_division=0)
    R["staging"]["stacked"] = dict(kappa=float(cohen_kappa_score(y, path)), f1=dict(zip(ST, map(float, f1))),
                                   confusion=confusion_matrix(y, path, labels=ST).tolist())
    print(f"  stacked decoded: kappa {R['staging']['stacked']['kappa']:.3f} F1 REM {f1[4]:.2f}", flush=True)

    (BASE / "results_reviewer_checks.json").write_text(json.dumps(R, indent=1))
    print(f"saved results_reviewer_checks.json ({time.time() - t0:.0f} s)")


def _bouts_in_run(y, run, ep, r, st):
    idx = np.flatnonzero(run == r)
    idx = idx[np.argsort(ep[idx])]
    out, n = [], 0
    for k in range(len(idx)):
        if y[idx[k]] == st and (n == 0 or ep[idx[k]] == ep[idx[k - 1]] + 1):
            n += 1
        else:
            if n:
                out.append(n)
            n = 1 if y[idx[k]] == st else 0
    if n:
        out.append(n)
    return out


if __name__ == "__main__":
    main()
