"""More checks asked by the three-reviewer panel on draft 13 (post hoc, exploratory).

  pairwise   BOLD's AUC gain for the physiologically informative pairs: REM vs W, REM vs N1, N3 vs N2 (main models'
             saved probabilities; within subject, >= 20 epochs of each class)
  noaud      tone confound, second check: BOLD without the somatomotor-auditory network's parcels and FC
  conserv    one combined-conservative specification: non-BOLD + 18 cardiorespiratory features + the eye block as the
             baseline, on eye-covered epochs, with LR, boosted trees and the BiGRU (39_strong_physio.py) on both sides
  hmm30      the decoded kappa when the non-BOLD emissions use only the 30 s window (no overlapping 60-300 s
             windows): does the fusion drop in kappa come from autocorrelated emissions?

    source <venv>/bin/activate
    python 47_panel_checks2.py                 # writes results_panel_checks2.json (~45 min, the BiGRU dominates)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


x = _load("x36b", "36b_extra_block.py")
e = x.e
g39 = _load("g39", "39_strong_physio.py")
ST = e.STAGES


def pair_gain(pa, pb, y, s, pos, neg):
    i = ST.index(pos)
    g = []
    for u in np.unique(s):
        m = (s == u) & np.isin(y, [pos, neg])
        y01 = y[m] == pos
        if y01.sum() >= 20 and (~y01).sum() >= 20:
            g.append(e.auc(y01, pa[m, i]) - e.auc(y01, pb[m, i]))
    g = np.array(g)
    rng = np.random.default_rng(0)
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
    return dict(mean=float(g.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                t_upper95=x.tb(g), n_pos=int((g > 0).sum()), n=len(g))


def main():
    t0 = time.time()
    D = e.load("afni")
    B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"].astype(int)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"] == y).all()
    Pnb, Pall = Z["proba__nonbold"], Z["proba__nonbold+bold"]
    strip = lambda G: {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}
    R = {"pairwise": {f"{a}_vs_{b}": pair_gain(Pall, Pnb, y, s, a, b) for a, b in (("REM", "W"), ("REM", "N1"), ("REM", "N2"), ("N3", "N2"))}}
    print("  pairwise:", {k: (round(v["mean"], 3), [round(c, 3) for c in v["ci"]], f"{v['n_pos']}/{v['n']}") for k, v in R["pairwise"].items()}, flush=True)
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    # ---------------- without the somatomotor (incl. auditory) network
    from nilearn import datasets
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch["labels"]]
    names = [n for n in names if n != "Background"][:200]
    common = np.load(BASE / "features_v2_afni.npz", allow_pickle=True)["common"].astype(bool)
    keep_par = np.array(["SomMot" not in n for n in np.array(names)[common]])
    nets = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    pairs = [(a, b) for ia, a in enumerate(nets) for b in nets[ia + 1:]]
    keep_fc = np.array(["SomMot" not in (a, b) for a, b in pairs])
    n_amp = int(common.sum())
    bold_noaud = np.hstack([B["bold"][:, :n_amp][:, keep_par], B["bold"][:, n_amp:][:, keep_fc]])
    Pna = e.loso(np.hstack([nb, bold_noaud]), y, s, ST)
    G = x.gains(Pna, Pnb, y, s)
    R["noaud"] = dict(gain=strip(G), contrast=x.contrast(G), n_features=int(bold_noaud.shape[1]))
    print(f"  without somatomotor/auditory ({bold_noaud.shape[1]} features): N3 {G['N3']['ws_mean']:+.3f} REM {G['REM']['ws_mean']:+.3f} ({time.time()-t0:.0f} s)", flush=True)
    # ---------------- hmm with 30 s non-BOLD emissions
    nb30 = np.hstack([B["motion"][:, :4], B["motion"][:, 20:21], B["physio"][:, :13], B["clock"]])
    P30 = e.loso(np.hstack([nb30, B["bold"]]), y, s, ST)
    k30 = cohen_kappa_score(y, e.hmm_decode(P30, y, s, run, D["epoch"]))
    kb = cohen_kappa_score(y, Z["hmm__bold"].astype(str))
    ka = cohen_kappa_score(y, Z["hmm__nonbold+bold"].astype(str))
    R["hmm30"] = dict(kappa_nb30_bold=float(k30), kappa_bold=float(kb), kappa_all=float(ka))
    print(f"  kappa: BOLD {kb:.3f}, all {ka:.3f}, 30 s non-BOLD + BOLD {k30:.3f}", flush=True)
    # ---------------- combined-conservative specification
    F = np.load(BASE / "features_physio_spectral.npz", allow_pickle=True)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(F["run"].astype(str), F["epoch"]))}
    spec = np.nan_to_num(e.wz(F["X"][np.array([key[(r, int(t))] for r, t in zip(run, ep)])].astype(np.float64), s))
    V = e.load("rv2")
    kv = {(r, int(t)): i for i, (r, t) in enumerate(zip(V["run"], V["epoch"]))}
    jv = np.array([kv.get((r, int(t)), -1) for r, t in zip(run, ep)])
    eye = np.full((len(y), 12), np.nan)
    eye[jv >= 0] = V["eye"][jv[jv >= 0]]
    m = np.isfinite(eye).all(1)
    base = np.hstack([nb, spec, e.wz(eye, s)])[m]
    full = np.hstack([base, B["bold"][m]])
    ym, sm, rm, em = y[m], s[m], run[m], ep[m]
    R["conserv"] = {"n_epochs": int(m.sum())}
    for fam in ("lr", "hgb", "gru"):
        if fam == "gru":
            Pb, Pa = g39.loso_gru(base, ym, sm, rm, em), g39.loso_gru(full, ym, sm, rm, em)
        else:
            mdl = "lr" if fam == "lr" else "hgb"
            Pb, Pa = e.loso(base, ym, sm, ST, model=mdl), e.loso(full, ym, sm, ST, model=mdl)
        G = x.gains(Pa, Pb, ym, sm)
        R["conserv"][fam] = dict(gain=strip(G), contrast=x.contrast(G),
                                 auc_base={st: float(np.mean(list(e.ws_auc((ym == st).astype(int), Pb[:, i], sm).values())))
                                           for i, st in enumerate(ST)})
        print(f"  conservative {fam}: N3 {G['N3']['ws_mean']:+.3f} [{G['N3']['ws_ci'][0]:+.3f},{G['N3']['ws_ci'][1]:+.3f}] "
              f"({G['N3']['n_pos']}/{G['N3']['n_subj']}) REM {G['REM']['ws_mean']:+.3f} t up {G['REM']['t_upper95']:+.3f} "
              f"N3>REM {R['conserv'][fam]['contrast']['n_pos']}/{R['conserv'][fam]['contrast']['n']} ({time.time()-t0:.0f} s)", flush=True)
        (BASE / "results_panel_checks2.json").write_text(json.dumps(R, indent=1))
    print(f"saved results_panel_checks2.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
