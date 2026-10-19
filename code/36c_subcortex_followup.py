"""Follow-up of the subcortical test (EXPLORATORY, after the result; the decision itself is pre-registered).

36b_extra_block.py --block subc gave case C of the rule in next_steps.md: early fusion keeps the REM gain
inside the margin, nested stacking does not. This script asks where that stacked REM gain comes from,
with 36b's functions and the same epochs. All gains are "all BOLD over non-BOLD" for REM (and N3) unless
stated, with the paper's bootstrap and t bounds:
  * the fourth estimator: gradient-boosted trees on non-BOLD + cortical + subcortical BOLD
  * stacking with the subcortical block reduced: each of the five groups left out in turn (its
    amplitudes and its 7 network correlations), amplitudes only, correlations only
  * stacking beyond cortex AND the global signal (GS as a fourth stacked block, in the reference too):
    a whole-brain physiological fluctuation would sit in the GS
  * the best-aligned runs (FAQ whitelist, as 31_whitelist.py), early fusion and stacking refitted there

    source <venv>/bin/activate
    python 36c_subcortex_followup.py          # writes results_subcortex_followup.json
"""
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent


def _mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, BASE / fn)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


b36 = _mod("b36", "36b_extra_block.py")
w31 = _mod("w31", "31_whitelist.py")
e = b36.e
ST, lg = e.STAGES, b36.lg


def brief(G):
    return {st: {k: G[st][k] for k in ("ws_mean", "ws_ci", "ws_lower95", "ws_upper95", "t_lower95", "t_upper95",
                                       "n_pos", "n_subj", "dagger", "lower_above_zero", "ws_values")} for st in ("N3", "REM")}


def line(name, G):
    g = G["REM"]
    return (f"  {name:42s} REM {g['ws_mean']:+.4f} [{g['ws_ci'][0]:+.3f},{g['ws_ci'][1]:+.3f}] lo t{g['t_lower95']:+.3f}"
            f" up b{g['ws_upper95']:+.3f}/t{g['t_upper95']:+.3f} {g['n_pos']}/{g['n_subj']} | N3 {G['N3']['ws_mean']:+.3f}")


def nested_stacks(blocks, y, s, P_outer):
    """blocks: name -> feature matrix; returns name-tuple -> stacked probabilities for each requested set."""
    def run(sets):
        out = {k: np.zeros((len(y), 5)) for k in sets}
        for u in np.unique(s):
            tr, te = s != u, s == u
            inner = {b: lg(e.loso(X[tr], y[tr], s[tr], ST)) for b, X in blocks.items()}
            for k in sets:
                out[k][te] = b36.stack([inner[b] for b in k], [lg(P_outer[b][te]) for b in k], y[tr])
        return out
    return run


def main():
    t0 = time.time()
    D = e.load("afni"); B = e.blocks(D)
    y, s, run, ep = D["stage"], D["subj"], D["run"], D["epoch"]
    S = np.load(BASE / "features_v2_afni_subc.npz", allow_pickle=True)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(S["run"].astype(str), S["epoch"]))}
    rows = np.array([key[(r, int(t))] for r, t in zip(run, ep)])
    X = e.wz(np.hstack([S["subc_amp"][rows], S["subc_fc"][rows]]), s)
    names, groups = [str(n) for n in S["names"]], [str(g) for g in S["groups"]]
    gmembers = {"thalamus": ["Thalamus"], "brainstem": ["Brain-Stem"], "hippocampus": ["Hippocampus"],
                "amygdala": ["Amygdala"], "striatum": ["Caudate", "Putamen", "Pallidum", "Accumbens"]}
    assert list(gmembers) == groups
    n_amp, n_net = len(names), 7
    cols = {g: [j for j, n in enumerate(names) if any(n.endswith(m) for m in gmembers[g])]
               + list(range(n_amp + gi * n_net, n_amp + (gi + 1) * n_net)) for gi, g in enumerate(groups)}
    assert sorted(sum(cols.values(), [])) == list(range(X.shape[1]))
    nb = np.hstack([B["motion"], B["physio"], B["clock"]])
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (Z["y"].astype(str) == y).all()
    R = {"note": "exploratory follow-up of the pre-registered case-C result", "n_features": int(X.shape[1])}

    # 1. gradient-boosted trees, the fourth estimator
    Ph = e.loso(np.hstack([nb, B["bold"], X]), y, s, ST, model="hgb")
    G = b36.gains(Ph, Z["proba__hgb:nonbold"], y, s)
    R["hgb"] = dict(gain=brief(G), contrast=b36.contrast(G))
    print(line("hgb: all BOLD over non-BOLD", G), f"N3-REM {R['hgb']['contrast']['value']:+.3f}", flush=True)
    G = b36.gains(Ph, Z["proba__hgb:nonbold+bold"], y, s)
    R["hgb_beyond_cortex"] = dict(gain=brief(G))
    print(line("hgb: subcortex beyond cortex", G), f"({time.time() - t0:.0f} s)", flush=True)

    # 2. stacking with reduced subcortical blocks, and with the global signal
    variants = {"all": list(range(X.shape[1])), "amplitudes only": list(range(n_amp)),
                "correlations only": list(range(n_amp, X.shape[1]))}
    for g in groups:
        variants[f"without {g}"] = [j for j in range(X.shape[1]) if j not in cols[g]]
    for g in groups:
        variants[f"only {g}"] = cols[g]
    blocks = {"nb": nb, "b": B["bold"], "gs": B["gs"]}
    P_outer = {"nb": Z["proba__nonbold"], "b": Z["proba__bold"]}
    P_outer["gs"] = e.loso(B["gs"], y, s, ST)
    for v, c in variants.items():
        blocks[v] = X[:, c]
        P_outer[v] = e.loso(X[:, c], y, s, ST)
    sets = [("nb",), ("nb", "b"), ("nb", "b", "gs")] + [("nb", "b", v) for v in variants] + [("nb", "b", "gs", "all")]
    SP = nested_stacks(blocks, y, s, P_outer)(sets)
    print(f"  nested stacks done ({time.time() - t0:.0f} s)", flush=True)
    R["stack"] = {}
    for v in variants:
        G = b36.gains(SP[("nb", "b", v)], SP[("nb",)], y, s)
        R["stack"][v] = dict(gain=brief(G), contrast=b36.contrast(G), n_features=len(variants[v]))
        print(line(f"stack: {v} ({len(variants[v])})", G), flush=True)
    G = b36.gains(SP[("nb", "b", "gs", "all")], SP[("nb", "b", "gs")], y, s)
    R["stack_beyond_cortex_and_gs"] = dict(gain=brief(G))
    print(line("stack: subcortex beyond cortex + GS", G), flush=True)
    G = b36.gains(SP[("nb", "b", "gs")], SP[("nb",)], y, s)
    R["stack_cortex_gs_over_nonbold"] = dict(gain=brief(G))
    print(line("stack: cortex + GS over non-BOLD", G), flush=True)

    # 3. best-aligned runs only: everything refitted on the subset
    wl = w31.whitelist()
    kk = np.array(["_".join(re.search(r"acq-(\d{8})T(\d{4})", r).groups()) for r in run])
    m = np.isin(kk, list(wl))
    ym, sm = y[m], s[m]
    bm = {"nb": nb[m], "b": B["bold"][m], "sc": X[m]}
    Pm = {k: e.loso(v, ym, sm, ST) for k, v in bm.items()}
    Pe3 = e.loso(np.hstack([bm["nb"], bm["b"], bm["sc"]]), ym, sm, ST)
    SPm = nested_stacks(bm, ym, sm, Pm)([("nb",), ("nb", "b"), ("nb", "b", "sc")])
    R["best_aligned"] = dict(n_runs=int(len(np.unique(run[m]))), n_epochs=int(m.sum()), n_subjects=int(len(np.unique(sm))))
    for name, (pa, pb) in {"early: all BOLD over non-BOLD": (Pe3, Pm["nb"]),
                           "stack: all BOLD over non-BOLD": (SPm[("nb", "b", "sc")], SPm[("nb",)]),
                           "stack: cortical BOLD over non-BOLD": (SPm[("nb", "b")], SPm[("nb",)]),
                           "stack: subcortex beyond cortex": (SPm[("nb", "b", "sc")], SPm[("nb", "b")])}.items():
        G = b36.gains(pa, pb, ym, sm)
        R["best_aligned"][name] = dict(gain=brief(G))
        print(line(f"best-aligned {name}", G), flush=True)
    (BASE / "results_subcortex_followup.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_subcortex_followup.npz", y=y, subj=s, hgb3=Ph,
                        stack_gs3=SP[("nb", "b", "gs")], stack_gs4=SP[("nb", "b", "gs", "all")],
                        wl_mask=m, wl_stack1=SPm[("nb",)], wl_stack2=SPm[("nb", "b")], wl_stack3=SPm[("nb", "b", "sc")])
    print(f"saved results_subcortex_followup.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
