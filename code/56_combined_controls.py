"""All controls at once (post hoc, exploratory; asked after draft 29). Each control so far was applied on its own:
the stronger cardiorespiratory baseline, the eye block, our own preprocessing, a sequence model. Here they are
combined: the baseline is the combined-conservative one of 47_panel_checks2.py (non-BOLD + 18 cardiorespiratory
staging features + the 12-feature eye block), and the BOLD features come from the derivative or from our own
pipelines, all as the same 84 shared parcels' 30 s amplitude (the parcels every pipeline covers), on the epochs every
source covers. Estimators: the planned LR, boosted trees and the BiGRU of 39_strong_physio.py, on both sides.

    source <venv>/bin/activate
    python 56_combined_controls.py --est lr,hgb          # minutes
    python 56_combined_controls.py --est gru --bold KEY  # one BiGRU comparison per call (parallel-friendly)
    python 56_combined_controls.py --merge               # results_combined_controls.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


x = _load("x36b", "36b_extra_block.py")
e = x.e
ST = e.STAGES
BOLD_SETS = {                                    # name -> (source, column or key)
    "deriv221": "derivative, 200 parcels + FC (the main BOLD features)",
    "deriv84": "derivative, 84 shared parcels",
    "rebuilt_aff": "rebuilt pipeline, affine atlas, all regressors",
    "rebuilt_syngm": "rebuilt pipeline, nonlinear registration + grey matter, all regressors",
    "minimal84": "minimal pipeline, 84 shared parcels",
}


def data():
    D = e.load("afni"); B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    key = [(r, int(t)) for r, t in zip(run, ep)]
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    F = np.load(BASE / "features_physio_spectral.npz", allow_pickle=True)
    kf = {(r, int(t)): i for i, (r, t) in enumerate(zip(F["run"].astype(str), F["epoch"]))}
    spec = np.nan_to_num(e.wz(F["X"][np.array([kf[k] for k in key])].astype(np.float64), s))
    V = e.load("rv2"); kv = {(r, int(t)): i for i, (r, t) in enumerate(zip(V["run"], V["epoch"]))}
    Dr = e.load("raw"); kr = {(r, int(t)): i for i, (r, t) in enumerate(zip(Dr["run"], Dr["epoch"]))}
    V3 = np.load(BASE / "features_v2_rv3.npz", allow_pickle=True)
    k3 = {(r, int(t)): i for i, (r, t) in enumerate(zip(V3["run"].astype(str), V3["epoch"]))}
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    jv = np.array([kv.get(k, -1) for k in key]); jr = np.array([kr.get(k, -1) for k in key]); j3 = np.array([k3.get(k, -1) for k in key])
    eye = np.full((len(y), 12), np.nan); eye[jv >= 0] = V["eye"][jv[jv >= 0]]
    raw = np.full((len(y), int(common.sum())), np.nan); raw[jr >= 0] = Dr["amp"][jr[jr >= 0]]
    rb = {}
    for nm, col in (("rebuilt_aff", "amp_aff_accmotphys"), ("rebuilt_syngm", "amp_syn_gm_accmotphys")):
        a = np.full((len(y), int(common.sum())), np.nan); a[j3 >= 0] = V3[col][j3[j3 >= 0]]; rb[nm] = a
    m = np.isfinite(eye).all(1) & np.isfinite(raw).all(1) & np.isfinite(rb["rebuilt_aff"]).all(1) & np.isfinite(rb["rebuilt_syngm"]).all(1)
    ym, sm, rm, em = y[m], s[m], run[m], ep[m]
    base = np.hstack([nb, spec, e.wz(eye, s)])[m]
    bold = {"deriv221": B["bold"][m], "deriv84": e.wz(D["amp"][:, common][m], sm), "minimal84": e.wz(raw[m], sm),
            "rebuilt_aff": e.wz(rb["rebuilt_aff"][m], sm), "rebuilt_syngm": e.wz(rb["rebuilt_syngm"][m], sm)}
    return ym, sm, rm, em, base, bold


def run(ests, only=None):
    t0 = time.time()
    y, s, r, ep, base, bold = data()
    print(f"{len(y)} epochs, {len(np.unique(s))} subjects; baseline {base.shape[1]} features ({time.time() - t0:.0f} s)", flush=True)
    strip = lambda G: {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}
    for est in ests:
        if est == "gru":
            g39 = _load("g39", "39_strong_physio.py")
            fit = lambda X: g39.loso_gru(X, y, s, r, ep)
        else:
            fit = lambda X, est=est: e.loso(X, y, s, ST, model=est)
        part = BASE / f"results_combined_controls_part_{est}{'_' + only if only else ''}.json"
        R = {"meta": dict(n_epochs=int(len(y)), n_subjects=int(len(np.unique(s))), n_base=int(base.shape[1]), estimator=est)}
        cache = BASE / f"predictions_combined_controls_base_{est}.npz"
        if cache.exists():
            Pb = np.load(cache)["P"]
        else:
            Pb = fit(base); np.savez_compressed(cache, P=Pb.astype(np.float32))
        R["auc_base"] = {st: float(np.mean(list(e.ws_auc((y == st).astype(int), Pb[:, i], s).values()))) for i, st in enumerate(ST)}
        P = {"base": Pb}
        for nm, X in bold.items():
            if only and nm != only:
                continue
            Pa = fit(np.hstack([base, X])); P[nm] = Pa
            G = x.gains(Pa, Pb, y, s)
            R[nm] = dict(label=BOLD_SETS[nm], n_bold=int(X.shape[1]), gain=strip(G), contrast=x.contrast(G))
            print(f"  {est} {nm:14s} N3 {G['N3']['ws_mean']:+.3f} [{G['N3']['ws_ci'][0]:+.3f},{G['N3']['ws_ci'][1]:+.3f}] "
                  f"{G['N3']['n_pos']}/{G['N3']['n_subj']} | REM {G['REM']['ws_mean']:+.3f} [{G['REM']['ws_ci'][0]:+.3f},{G['REM']['ws_ci'][1]:+.3f}] "
                  f"up {G['REM']['ws_upper95']:+.3f} t {G['REM']['t_upper95']:+.3f} | N3-REM {R[nm]['contrast']['value']:+.3f} "
                  f"{R[nm]['contrast']['n_pos']}/{R[nm]['contrast']['n']} ({time.time() - t0:.0f} s)", flush=True)
            part.write_text(json.dumps(R, indent=1))
        np.savez_compressed(BASE / f"predictions_combined_controls_{est}{'_' + only if only else ''}.npz", y=y, subj=s, run=r, epoch=ep,
                            **{"proba__" + k: v.astype(np.float32) for k, v in P.items()})
        part.write_text(json.dumps(R, indent=1))
    print(f"done ({time.time() - t0:.0f} s)")


def merge():
    R = {}
    for f in sorted(BASE.glob("results_combined_controls_part_*.json")):
        P = json.loads(f.read_text()); est = P["meta"]["estimator"]
        R.setdefault(est, {"meta": P["meta"], "auc_base": P["auc_base"]})
        R[est].update({k: v for k, v in P.items() if k not in ("meta", "auc_base")})
    (BASE / "results_combined_controls.json").write_text(json.dumps(R, indent=1))
    print("merged:", {k: sorted(x_ for x_ in v if x_ not in ("meta", "auc_base")) for k, v in R.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", default="lr,hgb")
    ap.add_argument("--bold", default=None, help="only this BOLD set (for parallel BiGRU runs)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    run(a.est.split(","), a.bold)


if __name__ == "__main__":
    main()
