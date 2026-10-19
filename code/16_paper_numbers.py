"""Generate paper_isbi/numbers.tex and paper_isbi/tab_ladder.tex from the results files.

The paper contains no hand-typed numbers: every value is a macro written here from
results_paper.json (13_experiments.py), results_followups.json (15_followups.py),
results_robustness.json (15b_robustness.py) and a few dataset facts recomputed here from disk
(alignment check, physiological-model R^2, demographics). 18_verify_paper.py re-derives the
load-bearing ones independently and checks the built PDF text against them.

    source <venv>/bin/activate
    python 16_paper_numbers.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
PAPER = BASE / "paper_isbi"
STAGES = ["W", "N1", "N2", "N3", "REM"]
SNAME = {"W": "W", "N1": "None", "N2": "Ntwo", "N3": "Nthree", "REM": "REM"}
NN = "nonneural(motion+physio+clock)"


def f3(x):
    return f"{x:.3f}"


def sgn(x):
    return f"{x:+.3f}".replace("-", "$-$") if x < 0 else f"{x:+.3f}"


def ci(c):
    return f"{sgn(c[0])}, {sgn(c[1])}"


def ci_plain(c):
    return f"{c[0]:.3f}--{c[1]:.3f}"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset_facts():
    """Alignment check, physiological R^2 and demographics, recomputed from the files on disk."""
    feat = _load("feat", "07_features.py")
    conf = _load("conf", "12_confounds.py")
    runs = sorted((BASE / "derivatives" / "ds005127").rglob("*_parcels.npz"))
    ok = naive_ok = n = 0
    r2 = []
    for r in runs:
        stem = r.name.replace("_parcels.npz", "")
        cand = list(feat.SCORING.rglob(stem + ".txt"))
        if not cand:
            continue
        z = np.load(r)
        T, tr = z["ts"].shape[1], float(z["tr"])
        on = pd.read_csv(cand[0], sep="\t")["MRIon"].to_numpy(float)
        implied = on.sum() * feat.EPOCH_S / tr
        idx = np.flatnonzero(on > 0)
        naive = (idx[-1] - idx[0] + 1) * feat.EPOCH_S / tr if len(idx) else 0
        n += 1
        ok += abs(implied / T - 1) <= 0.05
        naive_ok += abs(naive / T - 1) <= 0.05
        pf = conf.PHYS / r.parent.name / f"{stem}_physio.npz"
        if pf.exists():
            P = dict(np.load(pf, allow_pickle=True))
            qc, qr = conf.run_quality(P)
            grid, S = conf.physio_signals(P, T, tr)
            R1, R2 = conf.retroicor(P, T, tr), conf.slow_regressors(grid, S, T, tr)
            if qc >= 0.7 and qr >= 0.7 and R1 is not None and R2 is not None:
                ts = z["ts"]
                good = np.isfinite(ts).all(1)
                Y = ts[good]
                Y = ((Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-9)).T
                X = np.column_stack([np.ones(T), np.linspace(-1, 1, T), R1, R2])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                # R^2 of the physiological regressors beyond intercept + trend
                X0 = X[:, :2]
                b0, *_ = np.linalg.lstsq(X0, Y, rcond=None)
                ss0 = ((Y - X0 @ b0) ** 2).sum(0)
                ss1 = ((Y - X @ beta) ** 2).sum(0)
                r2.append(np.median(1 - ss1 / ss0))
    part = pd.read_csv(BASE / "data" / "ds005127" / "participants.tsv", sep="\t")
    return dict(align_ok=int(ok), align_runs=int(n), align_naive=int(naive_ok),
                phys_r2=float(np.median(r2)), n_r2_runs=len(r2),
                age_min=float(part.age.min()), age_max=float(part.age.max()),
                n_female=int((part.sex == "F").sum()))


def main():
    R = json.load(open(BASE / "results_paper.json"))
    Fu = json.load(open(BASE / "results_followups.json"))
    rob_path = BASE / "results_robustness.json"
    Rb = json.load(open(rob_path)) if rob_path.exists() else None
    facts = dataset_facts()
    json.dump(facts, open(BASE / "results_dataset_facts.json", "w"), indent=1)
    meta = R["meta"]
    M = {}
    M["nSubj"] = str(meta["n_subjects_all"])
    M["hoursBold"] = f"{meta['n_epochs_all'] * 30 / 3600:.0f}"
    M["nEpochsAll"] = f"{meta['n_epochs_all']:,}"
    M["nEpochsPhys"] = f"{meta['n_epochs_phys']:,}"
    M["physFrac"] = f"{100 * meta['n_epochs_phys'] / meta['n_epochs_all']:.1f}"
    for st in STAGES:
        M["n" + SNAME[st]] = f"{meta['stage_counts_phys'][st]:,}"
    M["nParcels"] = str(meta["n_parcels"])
    M["nRuns"] = str(len(np.unique(np.load(BASE / "features_ds005127.npz", allow_pickle=True)["run"])))
    pred = np.load(BASE / "predictions_paper.npz", allow_pickle=True)
    y, sj = pred["y"].astype(str), pred["subj"].astype(str)
    M["nSubjREMphys"] = str(len(np.unique(sj[y == "REM"])))
    M["ageRange"] = f"{facts['age_min']:.0f}--{facts['age_max']:.0f}"
    M["nFemale"] = str(facts["n_female"])
    M["alignOK"], M["alignRuns"], M["alignNaive"] = (str(facts["align_ok"]), str(facts["align_runs"]),
                                                     str(facts["align_naive"]))
    M["physRsq"] = f"{100 * facts['phys_r2']:.0f}"

    A = R["per_stage_auc"]
    for st in STAGES:
        M["aucBrain" + SNAME[st]] = f3(A["brain"][st])
        M["aucNn" + SNAME[st]] = f3(A[NN][st])
        M["aucPhys" + SNAME[st]] = f3(A["physio"][st])
        M["aucClean" + SNAME[st]] = f3(A["brain_physio_cleaned"][st])
        M["aucCleanOrth" + SNAME[st]] = f3(A["brain_cleaned_orth_nonneural"][st])
    M["aucGsREM"] = f3(A["global_signal"]["REM"])
    for st in ("REM", "N3"):
        keep = (A["brain_cleaned_orth_nonneural"][st] - 0.5) / (A["brain"][st] - 0.5)
        M["retain" + SNAME[st]] = f"{100 * keep:.0f}"
    inc = Fu["increment"]
    for key, tag in [("nonneural+brain - nonneural", "Nn"), ("motion+physio+brain - motion+physio", "Mp")]:
        for st in STAGES:
            r = inc[key][st]
            M[f"inc{tag}{SNAME[st]}"] = sgn(r["delta"])
            M[f"inc{tag}{SNAME[st]}ci"] = ci(r["ci"])
            M[f"inc{tag}{SNAME[st]}Subj"] = str(r["subjects"]["n_positive"])
            M[f"inc{tag}{SNAME[st]}SubjN"] = str(r["subjects"]["n"])
            M[f"inc{tag}{SNAME[st]}P"] = f"{r['subjects']['wilcoxon_p']:.2g}"
    rb = R["rem_vs_nrem_auc"]
    M["remBinBrain"], M["remBinNn"], M["remBinAll"] = f3(rb["brain"]), f3(rb[NN]), f3(rb["nonneural+brain"])
    M["aucClockREMbin"] = f3(rb["clock"])
    M["remBinPhys"] = f3(rb["physio"])

    L = R["ladder"]
    M["kappaBrain"] = f3(L["brain (ours)"]["hmm"]["kappa"])
    M["kappaBrainCI"] = ci_plain(L["brain (ours)"]["hmm"]["kappa_ci"])
    M["kappaMotion"] = f3(L["motion"]["hmm"]["kappa"])
    M["kappaNn"] = f3(L[NN]["hmm"]["kappa"])
    M["kappaAll"] = f3(L["brain + nonneural"]["hmm"]["kappa"])
    M["kappaAllCI"] = ci_plain(L["brain + nonneural"]["hmm"]["kappa_ci"])
    pb = R["paired"][f"brain (ours) vs {NN}"]
    M["kappaDiffBrainNn"] = sgn(pb["kappa_diff"])
    M["kappaDiffBrainNnCI"] = ci(pb["kappa_diff_ci"])
    pa = R["paired"][f"brain + nonneural vs {NN}"]
    M["kappaAllDiff"] = sgn(pa["kappa_diff"])
    M["kappaAllDiffCI"] = ci(pa["kappa_diff_ci"])
    M["kappaAllSubj"] = str(pa["wilcoxon"]["n_better"])
    M["kappaAllP"] = f"{pa['wilcoxon']['p']:.2g}"
    M["nullMax"] = f3(R["brain_shuffled_null_hmm_kappa"]["max"])

    H = Fu["hgb_auc"]
    M["hgbIncNthree"] = sgn(H["nonneural+brain"]["N3"] - H["nonneural"]["N3"])
    M["hgbIncREM"] = sgn(H["nonneural+brain"]["REM"] - H["nonneural"]["REM"])
    M["hgbNnREM"] = f3(H["nonneural"]["REM"])
    M["hgbAllREM"] = f3(H["nonneural+brain"]["REM"])
    # within-subject z profiles (descriptive)
    Pf = Fu["profiles"]
    nets = [k for k in Pf if not k.startswith("physio:")]
    M["profREMamp"] = sgn(np.mean([Pf[k]["REM"]["mean"] for k in nets])).replace("+", "")
    M["profREMampMin"] = f"{min(Pf[k]['REM']['mean'] for k in nets):.2f}".replace("-", "$-$")
    M["profREMampMax"] = f"{max(Pf[k]['REM']['mean'] for k in nets):.2f}".replace("-", "$-$")
    for key, mac in [("physio:resp_rate_sd_w90", "RespSd"), ("physio:hr_mean_w30", "Hr"),
                     ("physio:pulse_amp_mean_w30", "Pulse")]:
        for st in ("REM", "N3"):
            v = Pf[key][st]["mean"]
            M[f"prof{SNAME[st]}{mac}"] = (f"{v:+.2f}").replace("-", "$-$")
    M["kappaBrainMotionDiff"] = sgn(R["paired"]["brain (ours) vs motion"]["kappa_diff"])
    M["kappaBrainMotionDiffCI"] = ci(R["paired"]["brain (ours) vs motion"]["kappa_diff_ci"])
    M["kappaBrainTagDiff"] = sgn(R["paired"]["brain (ours) vs Tagliazucchi-style netFC SVM"]["kappa_diff"])
    M["kappaBrainTagSubj"] = str(R["paired"]["brain (ours) vs Tagliazucchi-style netFC SVM"]["wilcoxon"]["n_better"])
    M["kappaBrainAltDiff"] = sgn(R["paired"]["brain (ours) vs Altmann-style edge FC SVM"]["kappa_diff"])
    M["kappaAllEpochsBrain"] = f3(R["all_epochs"]["brain"]["hmm"]["kappa"])
    M["incNnREMbin"] = sgn(Fu["rem_vs_nrem_increment"]["nonneural+brain - nonneural"]["delta"])
    M["incNnREMbinci"] = ci(Fu["rem_vs_nrem_increment"]["nonneural+brain - nonneural"]["ci"])
    M["aucBrainMwREM"] = f3(Fu["auc"]["brain_mw"]["REM"])
    if Rb:
        M["robNthree"] = sgn(Rb["increment"]["N3"]["delta"])
        M["robREM"] = sgn(Rb["increment"]["REM"]["delta"])
    M["aucAllBrainREM"] = f3(R["all_epochs"]["brain"]["auc"]["REM"])
    ab = json.load(open(BASE / "results_ablation_amp_fc.json"))
    M["ablFcNthree"] = sgn(ab["netFC"]["N3"]["delta"])
    M["ablFcNthreeci"] = ci(ab["netFC"]["N3"]["ci"])
    M["ablAmpNthree"] = sgn(ab["amplitude"]["N3"]["delta"])
    M["ablAmpNthreeci"] = ci(ab["amplitude"]["N3"]["ci"])
    M["searchDates"] = "on 29 July 2026 and re-run on 13 October 2026"

    with open(PAPER / "numbers.tex", "w") as f:
        f.write("% generated by 16_paper_numbers.py -- do not edit\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")

    # ---------------- Table 1
    rows = [("Prior only (no features)", "prior-only"), ("Night position", "clock"),
            ("Motion (40)", "motion"), ("Physiology (65)", "physio"),
            ("Non-neural: all three (107)", NN), ("Global signal (10)", "global_signal"),
            ("Network-FC SVM~\\cite{tagliazucchi2012}", "Tagliazucchi-style netFC SVM"),
            ("Edge-FC SVM~\\cite{altmann2016}", "Altmann-style edge FC SVM"),
            ("BOLD (105)", "brain (ours)"), ("BOLD, physiology-cleaned", "brain physio-cleaned"),
            ("Non-neural + BOLD", "brain + nonneural")]
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}",
           "\\caption{Five-class staging, leave-one-subject-out, every model decoded by the same HMM. "
           "$\\kappa$ with 95\\% subject-bootstrap CI; F1 per stage. "
           f"{M['nEpochsPhys']} epochs, {M['nSubj']} subjects.}}",
           "\\label{tab:ladder}",
           "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
           "Model (features) & $\\kappa$ [95\\% CI] & Acc & F1$_\\mathrm{N3}$ & F1$_\\mathrm{REM}$ & mF1 \\\\",
           "\\midrule"]
    best = max(L[k]["hmm"]["kappa"] for _, k in rows)
    for label, key in rows:
        h = L[key]["hmm"]
        kap = f"{h['kappa']:.3f}"
        if abs(h["kappa"] - best) < 1e-12:
            kap = f"\\textbf{{{kap}}}"
        c = h.get("kappa_ci", [np.nan, np.nan])
        cis = f" [{c[0]:.2f}, {c[1]:.2f}]" if key != "prior-only" else ""
        out.append(f"{label} & {kap}{cis} & {h['acc']:.2f} & {h['f1']['N3']:.2f} & "
                   f"{h['f1']['REM']:.2f} & {h['macroF1']:.2f} \\\\")
        if key in ("prior-only", "global_signal", "Altmann-style edge FC SVM"):
            out.append("\\midrule")
    per = L["persistence (oracle previous label)"]["hmm"]
    out += ["\\midrule",
            f"\\emph{{Previous true label (oracle)}} & {per['kappa']:.3f} & {per['acc']:.2f} & "
            f"{per['f1']['N3']:.2f} & {per['f1']['REM']:.2f} & {per['macroF1']:.2f} \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tab_ladder.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {len(M)} macros and Table 1; facts: {facts}")


if __name__ == "__main__":
    main()
