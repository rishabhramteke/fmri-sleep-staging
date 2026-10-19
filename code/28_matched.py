"""Why does BOLD add less in our minimal pipeline than in the dataset authors' AFNI derivative?

The two streams differ in subjects (16 vs the 12 the derivative covers), in epochs (censoring, physio
QC) and in parcel coverage (84 parcels survive the raw field of view in every subject, 200 in the
derivative). This puts both on IDENTICAL epochs with an IDENTICAL non-BOLD block (realignment FD,
physiology, time since lights-out; no AFNI censor fraction), so only the BOLD features differ:

  raw          our minimal pipeline, 84 parcels + its network FC
  afni84       AFNI derivative amplitude on the same 84 parcels (no network FC)
  raw84amp     raw amplitude only, 84 parcels (the like-for-like partner of afni84)
  rv84amp      our realigned stream (06b_raw_v2.py), amplitude on the same 84 parcels, no physiological correction
  rv84phys     the same after RETROICOR + slow heart-rate/RV/pulse/respiration regressors
  afni         AFNI derivative, 200 parcels + network FC (the main-analysis BOLD block)

For each: within-subject AUC of BOLD alone, and the gain from adding it to the non-BOLD model, by early
fusion (one model on both blocks) and by late fusion (product of the two models' posteriors).

    source <venv>/bin/activate
    python 28_matched.py                     # writes results_matched.json
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def main():
    Dr, Da, Dv = e.load("raw"), e.load("afni"), e.load("rv2")
    kr = {(r, int(p)): i for i, (r, p) in enumerate(zip(Dr["run"], Dr["epoch"]))}
    ka = {(r, int(p)): i for i, (r, p) in enumerate(zip(Da["run"], Da["epoch"]))}
    kv = {(r, int(p)): i for i, (r, p) in enumerate(zip(Dv["run"], Dv["epoch"]))}
    keys = sorted(set(kr) & set(ka) & set(kv))       # draft 11: also our realigned stream (06b_raw_v2.py)
    ir, ia, iv = np.array([kr[k] for k in keys]), np.array([ka[k] for k in keys]), np.array([kv[k] for k in keys])
    y, s = Dr["stage"][ir], Dr["subj"][ir]
    assert (y == Da["stage"][ia]).all() and (s == Da["subj"][ia]).all()
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    wz = e.wz
    nb = np.hstack([wz(Dr["motion"][ir], s), wz(Dr["physio"][ir], s), Dr["clock"][ir]])
    bold = {"raw": np.hstack([wz(Dr["amp"][ir], s), wz(Dr["netfc"][ir], s)]),
            "raw84amp": wz(Dr["amp"][ir], s),
            "rv84amp": wz(Dv["amp"][iv], s),            # realigned, despiked, FD-censored; no physiological correction
            "rv84phys": wz(Dv["amp_phys"][iv], s),       # + RETROICOR and slow physiological regressors
            "rv84mot": wz(Dv["amp_mot"][iv], s),         # realigned + 24-parameter motion regression
            "afni84": wz(Da["amp"][ia][:, common], s),
            "afni": np.hstack([wz(Da["amp"][ia], s), wz(Da["netfc"][ia], s)])}
    R = {"meta": dict(n_epochs=len(keys), n_subjects=int(len(np.unique(s))),
                      n_raw=int(len(Dr["stage"])), n_afni=int(len(Da["stage"])),
                      n_common_parcels=int(common.sum()),
                      stage_counts={st: int((y == st).sum()) for st in STAGES})}
    print(json.dumps(R["meta"]), flush=True)
    P = {"nonbold": e.loso(nb, y, s, STAGES)}
    for k, X in bold.items():
        P[k] = e.loso(X, y, s, STAGES)
        P["nonbold+" + k] = e.loso(np.hstack([nb, X]), y, s, STAGES)
    R["auc_ws"] = {k: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), p[:, i], s).values())))
                       for i, st in enumerate(STAGES)} for k, p in P.items()}
    R["increment"] = {}
    for k in bold:
        inc = {st: e.increment((y == st).astype(int), P["nonbold+" + k][:, i], P["nonbold"][:, i], s)
               for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R["increment"][k] = inc
        print(f"  {k:9s} BOLD alone " + " ".join(f"{st}:{R['auc_ws'][k][st]:.3f}" for st in STAGES), flush=True)
        print(f"  {'':9s} gain       " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}[{inc[st]['ws_ci'][0]:+.3f},"
                                                  f"{inc[st]['ws_ci'][1]:+.3f}]" for st in STAGES), flush=True)
    print("  nonbold    " + " ".join(f"{st}:{R['auc_ws']['nonbold'][st]:.3f}" for st in STAGES))
    # the same ladder under late fusion (product of the non-BOLD and BOLD-alone posteriors): early fusion can
    # lose AUC when a weak block is added, so the REM half of the ladder is reported under both fusion rules
    R["increment_late"] = {}
    for k in bold:
        lf = e.late_fusion(P["nonbold"], P[k], y, s)
        inc = {st: e.increment((y == st).astype(int), lf[:, i], P["nonbold"][:, i], s) for i, st in enumerate(STAGES)}
        for st in STAGES:
            inc[st].pop("boot_ws")
        R["increment_late"][k] = inc
        print(f"  {k:9s} late gain  " + " ".join(f"{st}:{inc[st]['ws_mean']:+.3f}" for st in STAGES), flush=True)
    (BASE / "results_matched.json").write_text(json.dumps(R, indent=1))
    print("saved results_matched.json")


if __name__ == "__main__":
    main()
