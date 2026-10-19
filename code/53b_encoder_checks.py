"""Follow-up checks on the learned encoder (53_learned_encoder.py): gains against the paper's LR non-BOLD baseline,
late fusion of the encoder with the LR non-BOLD posteriors, 1 h time-stratified gains, and REM pairwise gains.

    source <venv>/bin/activate
    python 53b_encoder_checks.py        # writes results_learned_encoder_extra.json
"""
import importlib.util, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


e = _load("e24", "24_experiments_v2.py"); x = _load("x36b", "36b_extra_block.py"); r45 = _load("r45", "45_round4_checks.py")
ST = e.STAGES
RNG = np.random.default_rng(0)


def strat_gain(pa, pb, y, s, hours, st):
    i = ST.index(st); v = []
    for u in np.unique(s):
        m = s == u; y01 = (y[m] == st)
        if y01.sum() >= 20 and (~y01).sum() >= 20:
            v.append(r45.stratified_auc(y01, pa[m, i], hours[m]) - r45.stratified_auc(y01, pb[m, i], hours[m]))
    v = np.array(v); v = v[np.isfinite(v)]
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], upper95=float(np.percentile(bs, 95)),
                t_upper95=float(x.tb(v)), n=len(v), n_pos=int((v > 0).sum()))


def main():
    D = e.load("afni"); y, s = D["stage"], D["subj"]; hours = D["clock"][:, 0]
    L = np.load(BASE / "predictions_learned_encoder.npz", allow_pickle=True); Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    assert (L["y"] == y).all() and (Z["y"] == y).all()
    P = {"enc": L["proba__enc"], "nb_nn": L["proba__nb"], "nb+enc_nn": L["proba__nb+enc"],
         "nb_lr": Z["proba__nonbold"], "nb+bold_lr": Z["proba__nonbold+bold"], "bold_lr": Z["proba__bold"]}
    P["late(nb_lr,enc)"] = e.late_fusion(P["nb_lr"], P["enc"], y, s)
    comps = [("nb+enc_nn", "nb_nn"), ("nb+enc_nn", "nb_lr"), ("late(nb_lr,enc)", "nb_lr"), ("nb+bold_lr", "nb_lr")]
    R = {"auc_ws": {k: {st: float(np.mean(list(e.ws_auc((y == st).astype(int), p[:, i], s).values()))) for i, st in enumerate(ST)} for k, p in P.items()},
         "gain": {}, "stratified_1h": {}, "rem_pairwise": {}}
    for a, b in comps:
        G = x.gains(P[a], P[b], y, s)
        R["gain"][f"{a} - {b}"] = {"per_stage": {st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in ST}, "contrast": x.contrast(G)}
        R["stratified_1h"][f"{a} - {b}"] = {st: strat_gain(P[a], P[b], y, s, hours, st) for st in ("N3", "REM")}
        R["rem_pairwise"][f"{a} - {b}"] = x.rem_vs(P[a], P[b], y, s)
        g3, gr, c = G["N3"], G["REM"], R["gain"][f"{a} - {b}"]["contrast"]
        s3, sr = R["stratified_1h"][f"{a} - {b}"]["N3"], R["stratified_1h"][f"{a} - {b}"]["REM"]
        print(f"{a} - {b}:\n   N3 {g3['ws_mean']:+.3f} [{g3['ws_ci'][0]:+.3f},{g3['ws_ci'][1]:+.3f}] {g3['n_pos']}/{g3['n_subj']} | "
              f"REM {gr['ws_mean']:+.3f} [{gr['ws_ci'][0]:+.3f},{gr['ws_ci'][1]:+.3f}] up {gr['ws_upper95']:+.3f} t {gr['t_upper95']:+.3f} {gr['n_pos']}/{gr['n_subj']} p {gr['wilcoxon_p']:.3g} | "
              f"N3-REM {c['value']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] {c['n_pos']}/{c['n']}\n   within 1 h bins: N3 {s3['mean']:+.3f} | REM {sr['mean']:+.3f} [{sr['ci'][0]:+.3f},{sr['ci'][1]:+.3f}] up {sr['upper95']:+.3f} t {sr['t_upper95']:+.3f}\n"
              f"   REM pairwise: " + " ".join(f"{k}:{v['mean']:+.3f}" if isinstance(v, dict) and 'mean' in v else f"{k}:{v}" for k, v in R["rem_pairwise"][f"{a} - {b}"].items()), flush=True)
    print("AUC alone: enc", {st: round(v, 3) for st, v in R["auc_ws"]["enc"].items()}, "| bold_lr", {st: round(v, 3) for st, v in R["auc_ws"]["bold_lr"].items()},
          "| nb_lr", {st: round(v, 3) for st, v in R["auc_ws"]["nb_lr"].items()}, "| nb_nn", {st: round(v, 3) for st, v in R["auc_ws"]["nb_nn"].items()})
    (BASE / "results_learned_encoder_extra.json").write_text(json.dumps(R, indent=1)); print("saved results_learned_encoder_extra.json")


if __name__ == "__main__":
    main()
