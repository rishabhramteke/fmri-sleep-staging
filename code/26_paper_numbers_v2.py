"""Generate paper_isbi/numbers.tex and paper_isbi/tab_ladder.tex for the v2 paper.

Sources: results_v2_afni.json (main analysis, dataset authors' AFNI derivative), results_v2_raw.json
(our minimal pipeline, sensitivity), results_dataset_facts.json (16_paper_numbers.py: alignment,
demographics). No number in the paper is typed by hand; 27_verify_v2.py re-derives the headline ones
from the saved predictions.

    source <venv>/bin/activate
    python 26_paper_numbers_v2.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent
PAPER = BASE / "paper_isbi"
STAGES = ["W", "N1", "N2", "N3", "REM"]
SN = {"W": "W", "N1": "None", "N2": "Ntwo", "N3": "Nthree", "REM": "REM"}


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def sg(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def ci(c, d=3):
    return f"{sg(c[0], d)}, {sg(c[1], d)}"


def tbound(r, upper=True):
    """One-sided 95% t-based bound of the within-subject gain, from its per-subject values."""
    from scipy.stats import t as tdist
    v = np.array(list(r["ws_values"].values()))
    h = tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() + h) if upper else float(v.mean() - h)


def pfmt(p):
    return f"{p:.1g}" if p < 0.01 else f"{p:.2f}"


def gainmac(M, pre, G, contrast=None, stages=("N3", "REM")):
    """Macros for a gain dict {stage: summary}: mean, CI, bounds, n_pos/n_subj (+ N3-REM contrast counts)."""
    for st in stages:
        r = G[st]
        b = f"{pre}{SN[st]}"
        M[b], M[b + "ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
        M[b + "up"] = sg(r["ws_upper95"])
        M[b + "upT"] = sg(r["t_upper95"] if "t_upper95" in r else tbound(r))
        M[b + "loT"] = sg(r["t_lower95"] if "t_lower95" in r else tbound(r, upper=False))
        M[b + "Subj"], M[b + "SubjN"] = str(r["n_pos"]), str(r["n_subj"])
    if contrast is not None:
        M[pre + "Ctr"], M[pre + "Ctrci"] = sg(contrast["value"]), ci(contrast["ci"])
        M[pre + "CtrSubj"], M[pre + "CtrN"] = str(contrast["n_pos"]), str(contrast["n"])


def draft10(M, A, W, ma):
    """Macros for the checks added after the PDF-only review of draft 9 (all exploratory)."""
    d = ma["delta"]
    k2 = lambda x: f"{x:.2f}"
    # two-decimal kappa for the text and Table 1 (one precision throughout)
    L = A["ladder"]
    for name, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All"),
                      ("netFC-LR (after Tagliazucchi 2012)", "Tag")]:
        M[f"kk{tag}"] = k2(L[name]["kappa"])
    M["kkBoldRaw"], M["kkNbRaw"] = k2(W["ladder"]["bold"]["kappa"]), k2(W["ladder"]["nonbold"]["kappa"])
    pp = A["ladder_paired"]
    M["kkdBoldNb"], M["kkdBoldTag"] = sg(pp["bold vs nonbold"]["diff"], 2), sg(pp["bold vs netFC-LR (after Tagliazucchi 2012)"]["diff"], 2)
    for st in STAGES:
        M[f"fone{SN[st]}Bold"] = k2(L["bold"]["f1"][st])
    M["fREMTime"] = k2(L["clock"]["f1"]["REM"])
    assert L["bold"]["f1"]["REM"] == max(L[k]["f1"]["REM"] for k in ("bold", "nonbold", "nonbold+bold", "clock", "physio", "motion")), "text: BOLD alone is the best REM stager"
    # draft 22: subject-bootstrap 95% CI of BOLD-alone decoded REM F1 (HMM paths saved by 24_experiments_v2.py)
    from sklearn.metrics import f1_score
    Zp = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    yb, sb, hb = Zp["y"].astype(str), Zp["subj"].astype(str), Zp["hmm__bold"].astype(str)
    assert abs(f1_score(yb, hb, labels=["REM"], average=None)[0] - L["bold"]["f1"]["REM"]) < 1e-9
    rng_ = np.random.default_rng(0)
    subs_ = np.unique(sb); idx_ = {u: np.flatnonzero(sb == u) for u in subs_}
    bs_ = []
    for _ in range(2000):
        ii = np.concatenate([idx_[u] for u in rng_.choice(subs_, len(subs_))])
        bs_.append(f1_score(yb[ii], hb[ii], labels=["REM"], average=None, zero_division=0)[0])
    M["foneREMBoldci"] = f"{np.percentile(bs_, 2.5):.2f}, {np.percentile(bs_, 97.5):.2f}"   # an F1, not a gain: no signs
    SCk = json.load(open(BASE / "results_staging_checks.json"))
    r_ = SCk["kappa_all_minus_bold"]
    M["kkdAllBold"], M["kkdAllBoldci"] = sg(r_["diff"], 2), ci(r_["ci"], 2)
    M["kkArgAll"], M["kkArgBold"] = k2(SCk["argmax_kappa"]["nonbold+bold"]), k2(SCk["argmax_kappa"]["bold"])
    M["kkHmmAll"], M["kkHmmBold"] = sg(SCk["hmm_gain"]["nonbold+bold"], 2), sg(SCk["hmm_gain"]["bold"], 2)
    # N3 amplitude drop in our minimal pipeline (no realignment, no nuisance regression): per-parcel within-subject
    # AUC of 30 s amplitude, N3 vs the rest, averaged over subjects
    Zr = np.load(BASE / "predictions_v2_raw.npz", allow_pickle=True)
    Fr = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(Fr["run"].astype(str), Fr["epoch"]))}
    rows = np.array([key[(r, int(t))] for r, t in zip(Zr["run"].astype(str), Zr["epoch"])])
    amp, yr, sr = Fr["amp"][rows], Zr["y"].astype(str), Zr["subj"].astype(str)
    per = []
    for u in np.unique(sr):
        m = sr == u
        pos = yr[m] == "N3"
        if pos.sum() >= ma["min_epochs"] and (~pos).sum() >= ma["min_epochs"]:
            per.append([roc_auc_score(pos, amp[m][:, j]) for j in range(amp.shape[1])])
    M["parcelNthreeBelowRaw"] = f"{100 * np.mean(np.mean(per, 0) < 0.5):.0f}"
    assert np.mean(np.mean(per, 0) < 0.5) > 0.8, "text: the N3 amplitude drop is also present in the minimal pipeline"
    # ---------------- 37_reviewer_checks.py
    RC = json.load(open(BASE / "results_reviewer_checks.json"))
    for st in ("W", "N1", "N2", "N3", "REM"):
        M[f"pHolm{SN[st]}"] = pfmt(RC["holm_p"][st])
    f1b = L["bold"]["f1"]
    assert min(f1b, key=f1b.get) == "N1" and max(f1b, key=f1b.get) == "N3", "text: F1 from N1 (lowest) to N3 (highest)"
    CTp = json.load(open(BASE / "results_contrast.json"))["rem_vs"]["lr"]
    assert CTp["W"]["mean"] <= 0 and CTp["N1"]["mean"] <= 0, "text: REM gains against W and N1 are <= 0"
    M["ctrP"] = pfmt(RC["contrast_wilcoxon_p"])
    _n = int(M["ctrN"]) if "ctrN" in M else None
    if _n is not None:
        assert abs(RC["contrast_wilcoxon_p"] - 2 / 2 ** _n) < 1e-6, "text: the N3-REM p is the smallest a two-sided Wilcoxon test can give"
    M["shREMupT"] = sg(RC["second_half"]["t_upper"])
    assert RC["second_half"]["t_upper"] > d, "text: the second-half test is too imprecise to place inside the margin"
    M["marginT"] = sg(RC["smallest_margin_rem"]["t"])
    head_rem = 1 - A["auc_ws"]["nonbold"]["REM"]["mean"]
    assert abs(d / head_rem - 0.2) < 0.025, "text: delta is a fifth of the non-BOLD model's remaining REM 1-AUC"
    bm = RC["bout_median_epochs"]
    M["boutRemEp"], M["boutNthreeEp"] = f"{bm['REM']['epoch_weighted'] / 2:.1f}", f"{bm['N3']['epoch_weighted'] / 2:.1f}"
    assert bm["REM"]["epoch_weighted"] > bm["N3"]["epoch_weighted"], "text: REM epochs sit in longer bouts than N3 epochs"
    M["remInLong"] = f"{100 * RC['rem_epochs_in_bouts_ge5min']:.0f}"
    b5 = RC["bouts"]["ge5min"]
    gainmac(M, "bl", b5, b5["contrast"])
    assert b5["REM"]["t_upper95"] < d and b5["REM"]["ws_upper95"] < d
    uc = RC["uncensored"]
    gainmac(M, "uc", uc, uc["contrast"])
    M["ucFrac"] = f"{100 * uc['frac_epochs']:.0f}"
    ni = RC["night_index"]
    gainmac(M, "ni", ni, ni["contrast"])
    n2 = RC["night2_only"]
    gainmac(M, "nt", n2, n2["contrast"])
    assert n2["REM"]["t_upper95"] < d and n2["REM"]["ws_upper95"] < d, "text: night 2 alone stays inside the margin"
    tw = RC["time_auc_rem_within_night"]
    M["tnOne"], M["tnTwo"] = f2(tw["ses-1"]["mean"]), f2(tw["ses-2"]["mean"])
    cs = RC["C_sweep"]
    crem = [cs[c]["REM"]["ws_mean"] for c in cs]; cn3 = [cs[c]["N3"]["ws_mean"] for c in cs]
    M["cREMmin"], M["cREMmax"], M["cNthreeMin"], M["cNthreeMax"] = sg(min(crem)), sg(max(crem)), sg(min(cn3)), sg(max(cn3))
    assert all(cs[c]["REM"]["t_upper95"] < d and cs[c]["REM"]["ws_upper95"] < d for c in cs), "C sweep inside margin"
    ine = RC["rem_ineligible_n3"]
    M["inelNthreeMin"], M["inelNthreeMax"] = sg(min(ine.values())), sg(max(ine.values()))
    stg = RC["staging"]["stacked"]
    M["kkStHmm"], M["fREMStHmm"] = k2(stg["kappa"]), k2(stg["f1"]["REM"])
    # ---------------- 38_tuning.py (nested tuning)
    for mdl, sset, tag in (("lr", "main", "tuLr"), ("hgb", "main", "tuHgb"), ("lr", "max", "tuLrMx"), ("hgb", "max", "tuHgbMx")):
        f = BASE / f"results_tuned_{mdl}_{sset}.json"
        if not f.exists():
            continue
        TU = json.load(open(f))
        gainmac(M, tag, TU["gain"], TU["contrast"])
        picks = [str(v["choice"]) for v in TU["picks_with_bold"].values()]
        if mdl == "lr":
            M["tuneCmin"], M["tuneCmax"] = f"{min(g['C'] for g in TU['grid']):g}", f"{max(g['C'] for g in TU['grid']):g}"
            M[tag + "CmaxPick"] = f"{max(v['choice']['C'] for v in TU['picks_with_bold'].values()):g}"
        if mdl == "lr" and sset == "main":   # text: "Nested tuning picks strong regularisation (C <= 0.003)"
            assert all(v["choice"]["C"] <= 0.003 for v in TU["picks_with_bold"].values())
        M[tag + "Picks"] = str(len(set(picks)))
    # ---------------- 39_strong_physio.py (stronger non-BOLD baseline; one model family on both sides)
    f = BASE / "results_strong_physio.json"
    if f.exists():
        SP = json.load(open(f))
        M["nSpecFeat"] = str(SP["n_spectral"])
        for fam, tag in (("lr", "spLr"), ("hgb", "spHgb"), ("gru", "spGru")):
            if fam not in SP["models"]:
                continue
            m_ = SP["models"][fam]
            gainmac(M, tag, m_["gain"], m_["contrast"], stages=STAGES)
            for st in ("N3", "REM"):
                M[f"{tag}AucNb{SN[st]}"] = f3(m_["auc_ws_nonbold"][st])
            M[f"{tag}KkNb"], M[f"{tag}KkAll"] = k2(m_["kappa_nonbold"]["kappa5"]), k2(m_["kappa_with_bold"]["kappa5"])
        for fam in ("lr", "hgb", "gru"):   # "all three inside the margin"
            g_ = SP["models"][fam]["gain"]["REM"]
            assert g_["ws_upper95"] < d and g_["t_upper95"] < d, fam
        gg = SP["models"]["gru"]["gain"]
        assert all(gg[st]["ws_ci"][0] > 0 for st in ("W", "N1", "N2", "N3")), "W, N1, N2 and N3 gains stay positive under the BiGRU"
        assert gg["N3"]["ws_mean"] < 0.5 * A["increment"]["nonbold+bold - nonbold"]["N3"]["ws_mean"], "more than halves"
        cr = SP["cardioresp_only"]
        M["crGruKfive"], M["crGruKfour"] = k2(cr["gru"]["kappa5"]), k2(cr["gru"]["kappa4"])
        M["crOldKfive"], M["crOldKfour"] = k2(cr["lr_old"]["kappa5"]), k2(cr["lr_old"]["kappa4"])
    # ---------------- 40_derivative_physio_r2.py (what the derivative removed)
    PR = json.load(open(BASE / "results_derivative_physio_r2.json"))["summary"]
    for k, tag in (("deriv_retroicor", "prDerivRic"), ("deriv_slow", "prDerivSlow"), ("raw_retroicor", "prRawRic"), ("raw_slow", "prRawSlow")):
        M[tag], M[tag + "Null"] = (f3(PR[k]["mean"]).replace("-", "$-$"), f3(PR[k + "_null"]["mean"]).replace("-", "$-$"))
    M["prSlowAbove"], M["prSlowN"] = str(PR["deriv_slow_excess"]["n_above"]), str(PR["deriv_slow_excess"]["n"])
    M["prRicAbove"] = str(PR["deriv_retroicor_excess"]["n_above"])
    M["prSlowExcess"], M["prRawSlowExcess"] = f2(PR["deriv_slow_excess"]["mean"]), f2(PR["raw_slow_excess"]["mean"])
    assert PR["deriv_retroicor_excess"]["n_above"] == 0 and PR["deriv_slow_excess"]["n_above"] == PR["deriv_slow_excess"]["n"]
    # ---------------- 42_round2_checks.py (review of draft 10)
    R2c = json.load(open(BASE / "results_round2_checks.json"))
    gainmac(M, "slw", R2c["slow"]["gain"], R2c["slow"]["contrast"], stages=STAGES)
    gainmac(M, "dtr", R2c["dt"]["gain"], R2c["dt"]["contrast"])
    gainmac(M, "xn", R2c["cross_night"]["gain"], R2c["cross_night"]["contrast"])
    gainmac(M, "nod", R2c["no_drift_runs"]["gain"], R2c["no_drift_runs"]["contrast"])
    cf = R2c["clock_from_bold"]
    M["clkR"], M["clkRmin"], M["clkRmax"], M["clkRdt"] = (f2(cf["bold"]["r_mean"]), f2(cf["bold"]["r_min"]), f2(cf["bold"]["r_max"]),
                                                          f2(cf["bold_detrended"]["r_mean"]))
    for st in ("N3", "REM"):
        M[f"ll{SN[st]}"] = sg(R2c["logloss"][st]["mean"], 2)
    M["nNthreeUp"], M["nNthreeUpVis"] = str(R2c["n3_parcels_above"]["n"]), str(R2c["n3_parcels_above"]["by_network"]["Vis"])
    # guards for the sentences
    assert R2c["slow"]["gain"]["N3"]["ws_ci"][0] > 0 and R2c["slow"]["gain"]["N3"]["n_pos"] == 12
    assert R2c["slow"]["gain"]["REM"]["t_upper95"] < d and R2c["dt"]["gain"]["REM"]["t_upper95"] < d
    assert abs(cf["bold"]["r_mean"] - cf["bold_detrended"]["r_mean"]) < 0.02, "detrending leaves BOLD's time information"
    assert R2c["logloss"]["N3"]["mean"] > 0 > R2c["logloss"]["REM"]["mean"]
    assert R2c["n3_parcels_above"]["by_network"]["Vis"] > R2c["n3_parcels_above"]["n"] / 2
    M["retREMd"] = f"{100 * json.load(open(BASE / 'results_data_checks.json'))['retention']['afni']['REM']['frac']:.0f}"
    # ---------------- review of draft 11: decoded F1 gains, kappa vs regularisation, tones, EOG, spectrum
    nbf, alf = L["nonbold"]["f1"], L["nonbold+bold"]["f1"]
    for st in STAGES:
        M[f"dfone{SN[st]}"] = sg(alf[st] - nbf[st], 3)          # three decimals: Table 1 shows rounded F1
    assert max(alf[st] - nbf[st] for st in STAGES) == alf["N3"] - nbf["N3"], "text: decoded F1 gain largest for N3"
    M["kkAllCmin"] = k2(RC["C_sweep"]["0.01"]["kappa_hmm_all"])
    assert RC["C_sweep"]["0.01"]["kappa_hmm_all"] > RC["C_sweep"]["1.0"]["kappa_hmm_all"]
    TP = json.load(open(BASE / "results_tone_proxy.json"))
    gainmac(M, "tp", TP["gain"], TP["contrast"])
    M["tpAwake"] = str(TP["n_awakenings"])
    EO = json.load(open(BASE / "results_eog_phasic.json"))
    M["eogAucREM"] = f2(EO["eog_auc_rem"])
    gainmac(M, "eogPh", {"REM": EO["phasic"]}, stages=("REM",))
    gainmac(M, "eogTo", {"REM": EO["tonic"]}, stages=("REM",))
    assert EO["eog_auc_rem"] < 0.65 and EO["phasic"]["t_upper95"] < d, "text: weak EOG marker; no phasic gain"
    SPc = json.load(open(BASE / "results_derivative_spectrum.json"))
    M["specHiDeriv"], M["specHiRaw"] = f"{100 * SPc['deriv'][2]:.0f}", f"{100 * SPc['raw'][2]:.0f}"
    M["specLoDeriv"], M["specLoRaw"] = f"{100 * SPc['deriv'][0]:.0f}", f"{100 * SPc['raw'][0]:.0f}"
    assert SPc["deriv"][2] > 0.05, "text: the derivative is not band-passed"
    M["tuLrREMupB"] = sg(json.load(open(BASE / "results_tuned_lr_main.json"))["gain"]["REM"]["ws_upper95"])
    # ---------------- review of draft 12 (45_round4_checks.py)
    R4 = json.load(open(BASE / "results_round4_checks.json"))
    for k, tag in (("stratified", "str"), ("ap", "ap"), ("logit_gain", "lgt")):
        for st in ("N3", "REM"):
            r_ = R4[k][st]
            M[f"{tag}{SN[st]}"], M[f"{tag}{SN[st]}ci"] = sg(r_["mean"], 2 if k == "logit_gain" else 3), ci(r_["ci"], 2 if k == "logit_gain" else 3)
            M[f"{tag}{SN[st]}Subj"], M[f"{tag}{SN[st]}SubjN"] = str(r_["n_pos"]), str(r_["n"])
    r_ = R4["stratified_phys_rem"]
    M["strPhREM"], M["strPhREMci"] = sg(r_["mean"]), ci(r_["ci"])
    assert r_["ci"][0] < 0, "text: with time controlled, BOLD's REM gain over physiology alone loses its CI above zero"
    assert R4["stratified"]["REM"]["ci"][0] < 0 < R4["stratified"]["REM"]["ci"][1] and R4["stratified"]["N3"]["ci"][0] > 0
    lc = R4["logit_contrast"]
    M["lgtCtr"], M["lgtCtrci"], M["lgtCtrSubj"] = sg(lc["mean"], 2), ci(lc["ci"], 2), str(lc["n_pos"])
    tis = R4["tissue"]
    M["tisNthree"], M["tisNthreeci"] = sg(tis["tissue_over_nonbold"]["N3"]["ws_mean"]), ci(tis["tissue_over_nonbold"]["N3"]["ws_ci"])
    M["tisBoldNthree"], M["tisBoldNthreeci"] = sg(tis["bold_over_nonbold_tissue"]["N3"]["ws_mean"]), ci(tis["bold_over_nonbold_tissue"]["N3"]["ws_ci"])
    M["tisBoldREM"] = sg(tis["bold_over_nonbold_tissue"]["REM"]["ws_mean"])
    assert tis["bold_over_nonbold_tissue"]["N3"]["ws_ci"][0] > 0 and tis["tissue_over_nonbold"]["N3"]["ws_mean"] < 0.03
    M["nthreeRemElig"] = sg(R4["n3_on_rem_eligible"]["mean"])
    # ---------------- panel on draft 13 (46_panel_checks.py, 47_panel_checks2.py)
    P1 = json.load(open(BASE / "results_panel_checks.json"))
    gainmac(M, "lag", P1["lagged"]["gain"], P1["lagged"]["contrast"])
    gainmac(M, "tpf", P1["tones5"]["gain"], P1["tones5"]["contrast"])
    M["tpfFrac"] = f"{100 * P1['tones5']['frac_removed']:.0f}"
    for st in STAGES:
        f_ = P1["f1ci"][st]
        M[f"dfone{SN[st]}ci"] = ci(f_["ci"], 2)
    M["strREMupT"] = sg(R4["stratified"]["REM"]["t_upper95"])
    assert R4["stratified"]["REM"]["t_upper95"] > ma["delta"], "abstract: within 1 h bins the REM gain is not inside the margin"
    f = BASE / "results_panel_checks2.json"
    if f.exists():
        P2 = json.load(open(f))
        for k, tag in (("REM_vs_W", "pwRW"), ("REM_vs_N1", "pwRN"), ("REM_vs_N2", "pwRNt"), ("N3_vs_N2", "pwNN")):
            r_ = P2["pairwise"][k]
            M[tag], M[tag + "ci"], M[tag + "Subj"], M[tag + "N"] = sg(r_["mean"]), ci(r_["ci"]), str(r_["n_pos"]), str(r_["n"])
        if "noaud" in P2:
            gainmac(M, "noa", P2["noaud"]["gain"], P2["noaud"]["contrast"])
            tones = [json.load(open(BASE / "results_tone_proxy.json"))["gain"], P1["tones5"]["gain"], P2["noaud"]["gain"]]
            M["toneNthreeMin"], M["toneNthreeMax"] = sg(min(g["N3"]["ws_mean"] for g in tones)), sg(max(g["N3"]["ws_mean"] for g in tones))
            M["toneREMmin"], M["toneREMmax"] = sg(min(g["REM"]["ws_mean"] for g in tones)), sg(max(g["REM"]["ws_mean"] for g in tones))
        if "hmm30" in P2:
            M["kkNbThirty"] = k2(P2["hmm30"]["kappa_nb30_bold"])
        if "conserv" in P2:
            M["cvN"] = f"{P2['conserv']['n_epochs']:,}"
        for fam, tag in (("lr", "cvLr"), ("hgb", "cvHgb"), ("gru", "cvGru")):
            if fam in P2.get("conserv", {}):
                gainmac(M, tag, P2["conserv"][fam]["gain"], P2["conserv"][fam]["contrast"])
                M[tag + "AucNbNthree"] = f3(P2["conserv"][fam]["auc_base"]["N3"])
    # ---------------- panel on draft 14 (48_panel_checks3.py) and the pilot that fixed delta (results_paper.json, 13 Oct)
    P3 = json.load(open(BASE / "results_panel_checks3.json"))
    M["covAll"] = str(P3["coverage"]["n_ge80_all_runs"])
    assert P3["coverage"]["n_ge80_all_runs"] == 200, "text: all 200 parcels covered >= 80% in every run"
    bl_, bh_ = P3["bands"]["0.01-0.1 Hz"], P3["bands"]["0.1-0.16 Hz"]
    M["bandLoFrac"], M["bandHiFrac"] = f"{100 * bl_['frac_below']:.0f}", f"{100 * bh_['frac_below']:.0f}"
    M["bandLoAuc"], M["bandHiAuc"] = f2(bl_["median_auc"]), f2(bh_["median_auc"])
    assert bl_["frac_below"] > bh_["frac_below"] and bl_["median_auc"] < bh_["median_auc"]
    M["tsnrRho"] = sg(P3["tsnr"]["rho"], 2)
    assert P3["tsnr"]["rho"] < 0 and P3["tsnr"]["p"] < 0.01, "text: the drop is larger where tSNR is high"
    pil = json.load(open(BASE / "results_paper.json"))["per_stage_margin_ci"]["nonneural+brain - nonneural(motion+physio+clock)"]
    M["pilotNthree"], M["pilotREM"] = sg(pil["N3"]["delta"]), sg(pil["REM"]["delta"])
    M["confAllS"], M["confBoldS"] = f2(SCk["emission"]["nonbold+bold"]["mean_max_p"]), f2(SCk["emission"]["bold"]["mean_max_p"])
    M["acAllS"], M["acBoldS"] = f2(SCk["emission"]["nonbold+bold"]["lag1_r"]), f2(SCk["emission"]["bold"]["lag1_r"])
    # ---------------- 41_rv2_eyes.py (realigned own pipeline without physiological correction; eyes)
    f = BASE / "results_rv2_eyes.json"
    if f.exists():
        RV = json.load(open(f))
        a_ = RV["A"]
        M["rvN"], M["rvNsubj"] = f"{a_['n_epochs']:,}", str(a_["n_subjects"])
        for k, tag in (("minimal", "rvMin"), ("v2", "rvTwo"), ("v2_ric", "rvRic"), ("v2_phys", "rvPhys")):
            gainmac(M, tag, a_[k]["gain"], a_[k]["contrast"])
            gainmac(M, tag + "Lf", a_[k]["gain_late"], a_[k]["contrast_late"])
            for st in ("N3", "REM"):
                M[f"{tag}Auc{SN[st]}"] = f3(a_[k]["auc_bold"][st])
        # text: "improves its N3 detection but not its REM detection"; "physiological regression does not make the REM null"
        assert a_["v2"]["auc_bold"]["N3"] > a_["minimal"]["auc_bold"]["N3"] + 0.02
        assert abs(a_["v2"]["auc_bold"]["REM"] - a_["minimal"]["auc_bold"]["REM"]) < 0.02
        assert a_["v2"]["gain"]["REM"]["ws_mean"] < 0 and a_["v2_phys"]["gain"]["REM"]["ws_mean"] < 0
        b_ = RV["B"]
        M["eyeNsubj"], M["eyeFrac"] = str(len(b_["deriv_subjects"])), f"{100 * b_['deriv_frac_epochs_covered']:.0f}"
        dv = b_["deriv"]
        # text: "The field of view reaches the eyes in all subjects"; eyes add nothing to REM; BOLD still adds to N3
        assert len(b_["deriv_subjects"]) == ma["n_subjects"]
        for k in ("eye_over_nonbold", "eye_over_nonbold_bold"):
            assert dv[k]["REM"]["ws_ci"][0] < 0 < dv[k]["REM"]["ws_ci"][1], k
        assert dv["bold_over_nonbold_eye"]["N3"]["ws_ci"][0] > 0
        assert dv["bold_over_nonbold_eye"]["N3"]["n_pos"] == dv["bold_over_nonbold_eye"]["N3"]["n_subj"]
        for st in ("N3", "REM", "W"):
            M[f"eyeAuc{SN[st]}"] = f3(dv["auc_eye"][st])
        for k, tag in (("eye_over_nonbold", "eyeNb"), ("bold_over_nonbold", "eyeRefBold"),
                       ("eye_over_nonbold_bold", "eyeNbBold"), ("bold_over_nonbold_eye", "eyeBoldOver")):
            gainmac(M, tag, dv[k], stages=STAGES)


def table3(M, ma):
    """Table 3 (draft 29): the secondary comparisons the Results discuss, each with its pipeline, baseline, epoch set
    and estimator, so the prose can cite rows. Every cell comes from a results_*.json file."""
    MT = json.load(open(BASE / "results_matched.json")); LD = json.load(open(BASE / "results_preproc_ladder.json"))
    EY = json.load(open(BASE / "results_rv2_eyes.json"))["B"]; R4 = json.load(open(BASE / "results_round4_checks.json"))
    LX = json.load(open(BASE / "results_learned_encoder_extra.json")); SC = json.load(open(BASE / "results_block_subc.json"))
    M["eyN"] = f"{EY['deriv_n_epochs']:,}"
    f2_ = lambda x: f"{x:.2f}"
    f3_ = lambda x: f"{x:+.3f}".replace("-", "$-$")

    def gcell(r):
        txt = f3_(r["ws_mean"] if "ws_mean" in r else r["mean"])
        lo, hi = r["ws_ci"] if "ws_ci" in r else r["ci"]
        return f"\\textbf{{{txt}}}" if lo > 0 else txt

    def rcell(r):
        lo, hi = r["ws_ci"] if "ws_ci" in r else r["ci"]
        return f"{gcell(r)} [{f3_(lo)},\\,{f3_(hi)}]"

    rows = []
    add = lambda lab, auc, inc: rows.append(f"{lab} & {auc} & {gcell(inc['N3'])} & {rcell(inc['REM'])} \\\\")
    # draft 38: the old group (a) (minimal and realigned pipelines on the 13,448 shared epochs) moved to the text
    dv = EY["deriv"]
    assert abs(dv["bold_over_nonbold"]["N3"]["ws_mean"] - MT["increment"]["afni"]["N3"]["ws_mean"]) < 6e-4
    assert LD["meta"]["n_epochs"] <= MT["meta"]["n_epochs"]
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{Rebuilt pipeline, before and after regression ({LD['meta']['n_epochs']:,} epochs)}}}} \\\\")
    for k, lab in (("amp_aff", "Rebuilt, affine, before regression"), ("amp_aff_accmotphys", "Rebuilt, affine"), ("amp_syn_gm_accmotphys", "Rebuilt, nonlinear, GM"),
                   ("afni84", "Derivative, 84 parcels")):
        add(lab, f2_(LD["auc_ws"][k]["N3"]), LD["increment"][k])
    # (e) all controls at once (56_combined_controls.py): baseline = non-BOLD + 18 cardiorespiratory + eye block
    CC = json.load(open(BASE / "results_combined_controls.json"))
    # one row per estimator for the derivative, the rebuilt pipeline and the minimal pipeline, so every number the text
    # quotes for this comparison is a cell
    cc_sets = (("deriv221", "Derivative, 200 parcels $+$ FC, LR"), ("rebuilt_aff", "Rebuilt, all regressors, LR"),
               ("minimal84", "Minimal, 84 parcels, LR"))
    cc_rows = [f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{(c) All controls: over stronger baseline $+$ eye block ({CC['lr']['meta']['n_epochs']:,} epochs)}}}} \\\\"]
    for k, lab in cc_sets:
        for est, elab in (("lr", lab), ("hgb", "\\quad boosted trees"), ("gru", "\\quad BiGRU")):
            if est in CC and k in CC[est]:
                r_ = CC[est][k]["gain"]
                cc_rows.append(f"{elab} & -- & {gcell(r_['N3'])} & {rcell(r_['REM'])} \\\\")
    for est in [k for k in ("lr", "hgb", "gru") if k in CC]:
        tag = {"lr": "Lr", "hgb": "Hgb", "gru": "Gru"}[est]
        for k, kt in (("deriv221", "DerivAll"), ("deriv84", "DerivEf"), ("rebuilt_aff", "RebAff"), ("rebuilt_syngm", "RebSyn"), ("minimal84", "Min")):
            if k in CC[est]:
                M[f"cc{tag}{kt}Nthree"] = sg(CC[est][k]["gain"]["N3"]["ws_mean"])
                M[f"cc{tag}{kt}REM"] = sg(CC[est][k]["gain"]["REM"]["ws_mean"])
                M[f"cc{tag}{kt}NthreeSubj"] = f"{CC[est][k]['gain']['N3']['n_pos']}/{CC[est][k]['gain']['N3']['n_subj']}"
        own = [CC[est][k]["gain"]["N3"]["ws_mean"] for k in ("rebuilt_aff", "minimal84") if k in CC[est]]     # the rows Table 3c shows
        if own:
            M[f"cc{tag}OwnMin"], M[f"cc{tag}OwnMax"] = sg(min(own)), sg(max(own))
    M["ccN"] = f"{CC['lr']['meta']['n_epochs']:,}"
    M["ccOwnMaxAll"] = sg(max(CC[est][k]["gain"]["N3"]["ws_mean"] for est in CC for k in ("rebuilt_aff", "minimal84") if k in CC[est]))
    M["ccLrDerivAllNthreeN"] = str(CC["lr"]["deriv221"]["gain"]["N3"]["n_subj"])
    M["ccDerivMinAll"] = sg(min(CC[est]["deriv221"]["gain"]["N3"]["ws_mean"] for est in CC if "deriv221" in CC[est]))
    M["ccDerivMaxAll"] = sg(max(CC[est]["deriv221"]["gain"]["N3"]["ws_mean"] for est in CC if "deriv221" in CC[est]))
    assert all(CC[est]["deriv221"]["gain"]["N3"]["ws_ci"][0] > 0 for est in CC if "deriv221" in CC[est]), "text: with the derivative the N3 gain survives all controls"
    assert CC["lr"]["minimal84"]["gain"]["N3"]["ws_ci"][0] < 0 and CC["lr"]["rebuilt_aff"]["gain"]["N3"]["ws_ci"][0] < 0, \
        "text: with our own preprocessing the planned model's N3 gain is not distinguishable from zero once all controls are combined"
    assert all(CC[est][k]["n_bold"] == 84 for est in CC for k in ("deriv84", "rebuilt_aff", "rebuilt_syngm", "minimal84") if k in CC[est]), "Table 4 labels: 84 parcels"
    # Table 4: all controls at once, one column per estimator (draft 31)
    t4 = ["\\begin{table}[t]", "\\centering", "\\fontsize{7.5}{9}\\selectfont", "\\setlength{\\tabcolsep}{2.2pt}",
          "\\caption{All controls: N3 gains need the derivative or, with trees or BiGRU, our nuisance regression; REM gains stay small. Gain over the stronger $+$ eyes baseline "
          f"({CC['lr']['meta']['n_base']} features) on the {CC['lr']['meta']['n_epochs']:,} epochs all BOLD sets cover (N3: "
          f"{CC['lr']['deriv221']['gain']['N3']['n_subj']} subjects). 221: main "
          "features (row 1: Table~\\ref{tab:robust}'s stronger $+$ eyes rows, other epochs); 84: shared parcels' "
          f"amplitude. Baseline N3 AUC: {CC['lr']['auc_base']['N3']:.2f} (LR), {CC['hgb']['auc_base']['N3']:.2f} (trees), "
          f"{CC['gru']['auc_base']['N3']:.2f} (BiGRU). Bold: bootstrap CI above zero.}}",
          "\\label{tab:allcontrols}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
          " & \\multicolumn{3}{c}{$\\Delta_\\mathrm{N3}$} & \\multicolumn{3}{c}{$\\Delta_\\mathrm{REM}$} \\\\",
          "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
          "BOLD features & LR & trees & BiGRU & LR & trees & BiGRU \\\\", "\\midrule"]
    for k, lab in (("deriv221", "Derivative (221)"), ("deriv84", "Derivative (84)"), ("rebuilt_aff", "Rebuilt, affine (84)"),
                   ("rebuilt_syngm", "Rebuilt, nonlinear, GM (84)"), ("minimal84", "Minimal (84)")):
        cells = [gcell(CC[est][k]["gain"][st]) for st in ("N3", "REM") for est in ("lr", "hgb", "gru")]
        t4.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    t4 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tab_allcontrols.tex").write_text("\n".join(t4) + "\n")
    rng_ = lambda ks: [CC[est][k]["gain"]["N3"]["ws_mean"] for est in ("lr", "hgb", "gru") for k in ks]
    M["ccDerivEfMin"], M["ccDerivEfMax"] = sg(min(rng_(["deriv84"]))), sg(max(rng_(["deriv84"])))
    M["ccOwnMin"], M["ccOwnMax"] = sg(min(rng_(["rebuilt_aff", "rebuilt_syngm", "minimal84"]))), sg(max(rng_(["rebuilt_aff", "rebuilt_syngm", "minimal84"])))
    M["ccLrOwnMinAll"], M["ccLrOwnMaxAll"] = (sg(f([CC["lr"][k]["gain"]["N3"]["ws_mean"] for k in ("rebuilt_aff", "rebuilt_syngm", "minimal84")])) for f in (min, max))
    assert all(CC["lr"][k]["gain"]["N3"]["ws_ci"][0] < 0 for k in ("rebuilt_aff", "rebuilt_syngm", "minimal84")), "text: own pipelines' planned-model CIs include zero"
    assert all(CC[est]["deriv84"]["gain"]["N3"]["ws_ci"][0] > 0 for est in ("lr", "hgb", "gru")), "text: the derivative's 84 parcels keep a gain under all controls"
    assert all(CC[est][k]["gain"]["N3"]["ws_ci"][0] > 0 for est in ("hgb", "gru") for k in ("rebuilt_aff", "rebuilt_syngm")), \
        "text: the rebuilt pipeline's N3 gain is above zero with the flexible estimators"
    assert all(CC[est]["minimal84"]["gain"]["N3"]["ws_ci"][0] <= 0 for est in ("lr", "hgb", "gru")), "text: the minimal pipeline's N3 gain is never above zero"
    assert float(M["ccOwnMax"]) > float(M["ccLrOwnMaxAll"]), "abstract: the own-pipeline maximum comes from a nonlinear estimator"
    assert min(rng_(["deriv84", "deriv221"])) == min(rng_(["deriv84"])) and max(rng_(["deriv84", "deriv221"])) == max(rng_(["deriv221"])), \
        "text: the derivative's all-controls N3 gains run from the 84-parcel minimum to the 221-feature maximum"
    own_ = lambda est: max(CC[est][k]["gain"]["N3"]["ws_mean"] for k in ("rebuilt_aff", "rebuilt_syngm", "minimal84"))
    for est in ("lr", "hgb"):
        assert own_(est) <= 0.55 * CC[est]["deriv84"]["gain"]["N3"]["ws_mean"], f"abstract: {est} shows the preprocessing gap"
    assert abs(CC["gru"]["rebuilt_aff"]["gain"]["N3"]["ws_mean"] - CC["gru"]["deriv84"]["gain"]["N3"]["ws_mean"]) < 0.005 \
        and own_("gru") == CC["gru"]["rebuilt_aff"]["gain"]["N3"]["ws_mean"], "abstract: the BiGRU gains as much from the rebuilt pipeline"
    # the eye-over-non-BOLD comparison (41) keeps Table 3a's epochs and sets missing eye features to the subject mean
    # (wz -> nan_to_num); the stronger + eyes analyses (47, 56) drop epochs without eye features
    _eyN, _cvN = int(M["eyN"].replace(",", "")), int(M["cvN"].replace(",", ""))
    assert _eyN == MT["meta"]["n_epochs"], "the eye comparison keeps the epochs of Table 3a"
    assert 0 < _eyN - _cvN < 0.05 * _eyN, (_eyN, _cvN)
    M["eyMiss"] = f"{_eyN - _cvN:,}"
    _rm = max(((est, k) for est in ("lr", "hgb", "gru") for k in CC[est] if k not in ("meta", "auc_base")),
              key=lambda t: CC[t[0]][t[1]]["gain"]["REM"]["ws_mean"])
    _P2 = json.load(open(BASE / "results_panel_checks2.json"))["conserv"]["hgb"]["gain"]["REM"]
    assert _rm == ("hgb", "deriv221") and CC["hgb"]["deriv221"]["gain"]["REM"]["ws_ci"][0] > 0 and _P2["ws_ci"][0] <= 0, \
        "text: Table 4's largest REM gain is trees on the derivative, CI above zero only on Table 4's epochs"
    # paired pipeline differences (57_paired_pipeline_diff.py, post hoc, asked by the author on 2026-10-22)
    PD = json.load(open(BASE / "results_paired_pipeline.json"))
    for est, et in (("lr", "Lr"), ("hgb", "Hgb"), ("gru", "Gru")):
        for k, kt in (("deriv84-rebuilt_aff", "RebAff"), ("deriv84-rebuilt_syngm", "RebSyn"), ("deriv84-minimal84", "Min"),
                      ("rebuilt_aff-minimal84", "Reg")):
            r = PD["allcontrols"][est][k]["N3"]
            M[f"pd{et}{kt}"], M[f"pd{et}{kt}ci"] = sg(r["mean"]), ci(r["ci"])
    _lt = [PD["allcontrols"][e][k]["N3"] for e in ("lr", "hgb") for k in ("deriv84-rebuilt_aff", "deriv84-rebuilt_syngm", "deriv84-minimal84")]
    assert all(r["ci"][0] > 0 and r["t_lower95"] > 0 for r in _lt), "text: with LR and trees the derivative's parcels beat every one of our pipelines"
    M["pdLtMin"], M["pdLtMax"] = sg(min(r["mean"] for r in _lt)), sg(max(r["mean"] for r in _lt))
    _g = PD["allcontrols"]["gru"]
    assert _g["deriv84-rebuilt_aff"]["N3"]["ci"][0] < 0 < _g["deriv84-rebuilt_aff"]["N3"]["ci"][1] \
        and _g["deriv84-rebuilt_syngm"]["N3"]["ci"][0] < 0 < _g["deriv84-rebuilt_syngm"]["N3"]["ci"][1] \
        and _g["deriv84-minimal84"]["N3"]["ci"][0] > 0, "text: with the BiGRU only the minimal pipeline falls short of the derivative"
    L_ = PD["ladder"]
    for k, kt in (("afni84-amp_aff", "Pre"), ("afni84-amp_aff_accmotphys", "Reb"), ("amp_aff_accmotphys-amp_aff", "Reg"),
                  ("amp_syn_gm_accmotphys-amp_aff_accmotphys", "Syn")):
        M[f"pdLd{kt}"], M[f"pdLd{kt}ci"] = sg(L_[k]["N3"]["mean"]), ci(L_[k]["N3"]["ci"])
    assert L_["amp_aff_accmotphys-amp_aff"]["N3"]["ci"][0] > 0 and L_["amp_syn_gm_accmotphys-amp_aff_accmotphys"]["N3"]["ci"][0] < 0 < \
        L_["amp_syn_gm_accmotphys-amp_aff_accmotphys"]["N3"]["ci"][1], "text: the regression step adds, registration and GM do not"
    fr_ = L_["recovered_fraction_N3"]
    M["pdFrac"], M["pdFracLo"], M["pdFracHi"] = (f"{100 * v:.0f}" for v in (fr_["value"], *fr_["ci"]))
    assert 0.4 < fr_["value"] < 0.65 and fr_["ci"][0] > 0 and fr_["ci"][1] < 1, "text: rebuilding recovers about half, CI inside (0, 1)"
    M["nStrongFeat"] = str(int(M["nNbFeat"]) + int(M["nSpecFeat"]))
    M["nStrongEyeFeat"] = str(CC["lr"]["meta"]["n_base"])
    assert int(M["nStrongEyeFeat"]) == int(M["nStrongFeat"]) + 12, "text: the eye block has 12 features"
    M["ccN"] = f"{CC['lr']['meta']['n_epochs']:,}"
    M["ccRemMax"] = sg(max(CC[est][k]["gain"]["REM"]["ws_mean"] for est in ("lr", "hgb", "gru") for k in CC[est] if k not in ("meta", "auc_base")))
    # draft 36: the within-bin rows (old group c) moved to the text, with their CIs as macros
    for k, tag in (("nb+enc_nn - nb_nn", "leStrat"), ("late(nb_lr,enc) - nb_lr", "leLateStrat")):
        lo_, hi_ = LX["stratified_1h"][k]["REM"]["ci"]
        M[tag + "REMci"] = f"{sg(lo_)}, {sg(hi_)}"
    assert abs(LD["increment"]["amp_aff_gm"]["N3"]["ws_mean"] - float(M["ownNthreeMin"].replace("$-$", "-"))) < 6e-4, "text: the lowest own-pipeline N3 gain is the grey-matter-only rung"
    a0, a1, ad = (LD["auc_ws"][k]["N3"] for k in ("amp_aff", "amp_aff_accmotphys", "afni84"))
    assert 0.4 < (a1 - a0) / (ad - a0) < 0.65, "text: the regressors explain about half of the derivative's advantage"
    g0, g1, gd = (LD["increment"][k]["N3"]["ws_mean"] for k in ("amp_aff", "amp_aff_accmotphys", "afni84"))
    assert 0.4 < (g1 - g0) / (gd - g0) < 0.65, "abstract: rebuilding recovers about half of the N3-gain difference too"
    assert LD["increment"]["amp_syn_gm_accmotphys"]["N3"]["ws_mean"] <= LD["increment"]["amp_aff_accmotphys"]["N3"]["ws_mean"] + 0.005, \
        "text: nonlinear registration and grey-matter restriction add nothing"
    assert abs(dv["eye_over_nonbold"]["REM"]["ws_mean"]) < 0.01, "text: the eye block adds nothing to REM"
    _neg = sum(1 for r in rows if "&" in r and "[$-$" in r.split("&")[3] and ",\\,$-$" in r.split("&")[3])
    M["tThreeRemNeg"] = str(_neg)
    assert _neg >= 3, _neg
    _own = [MT["increment"][k]["N3"]["ws_mean"] / MT["increment"]["afni84"]["N3"]["ws_mean"] for k in ("raw84amp", "rv84amp")] + \
           [LD["increment"][k]["N3"]["ws_mean"] / LD["increment"]["afni84"]["N3"]["ws_mean"] for k in ("amp_aff", "amp_aff_accmotphys", "amp_syn_gm_accmotphys")]
    assert max(_own) <= 0.55, "Table 3 caption: our pipelines keep at most about half of the derivative's N3 gain"
    out = ["\\begin{table}[t]", "\\centering", "\\fontsize{7.5}{9}\\selectfont", "\\setlength{\\tabcolsep}{2pt}",
           "\\caption{Our pipelines keep at most about half of the derivative's N3 gain. Planned model; gain from adding the stated BOLD features to "
           "non-BOLD (REM: 95\\% CI); AUC$_\\mathrm{N3}$: that BOLD set alone; boldface: bootstrap CI above zero.}",
           "\\label{tab:secondary}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
           "Comparison & AUC$_\\mathrm{N3}$ & $\\Delta_\\mathrm{N3}$ & $\\Delta_\\mathrm{REM}$ [95\\% CI] \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if r.startswith("\\multicolumn") and i > 0:
            out.append("\\midrule")
        out.append(r)
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tab_secondary.tex").write_text("\n".join(out) + "\n")


def main():
    A = json.load(open(BASE / "results_v2_afni.json"))
    W = json.load(open(BASE / "results_v2_raw.json"))
    facts = json.load(open(BASE / "results_dataset_facts.json"))
    M = {}
    ma, mw = A["meta"], W["meta"]
    M["nSubj"], M["nRuns"] = str(ma["n_subjects"]), str(ma["n_runs"])
    M["nEpochs"] = f"{ma['n_epochs']:,}"
    M["hoursBold"] = f"{ma['n_epochs'] * 30 / 3600:.0f}"
    for st in STAGES:
        M["n" + SN[st]] = f"{ma['stage_counts'][st]:,}"
        M["elig" + SN[st]] = str(ma["subjects_eligible_ws"][st])
    M["nSubjRaw"], M["nRunsRaw"], M["nEpochsRaw"] = str(mw["n_subjects"]), str(mw["n_runs"]), f"{mw['n_epochs']:,}"
    M["nBoldFeat"], M["nBoldFeatRaw"] = str(ma["n_bold_features"]), str(mw["n_bold_features"])
    M["nMotionFeat"] = str(ma["n_motion"])
    M["deltaVal"] = f"{ma['delta']:.2f}"
    M["ageRange"] = f"{facts['age_min']:.0f}--{facts['age_max']:.0f}"
    M["nFemale"] = str(facts["n_female"])
    M["alignOK"], M["alignRuns"] = str(facts["align_ok"]), str(facts["align_runs"])
    # demographics of the derivative cohort (participants.tsv), alongside the full-cohort facts
    import csv
    Z0 = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    subs = set(map(str, np.unique(Z0["subj"])))
    rows_p = [r for r in csv.DictReader(open(BASE / "data" / "ds005127" / "participants.tsv"), delimiter="\t")
              if r["participant_id"] in subs]
    ages = [float(r["age"]) for r in rows_p]
    M["ageRangeDeriv"] = f"{min(ages):.0f}--{max(ages):.0f}"
    M["nFemaleDeriv"] = str(sum(r["sex"].strip().upper().startswith("F") for r in rows_p))

    for R, suf in [(A, ""), (W, "Raw")]:
        aw = R["auc_ws"]
        for key, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All"), ("clock", "Time"),
                         ("physio", "Phys"), ("motion", "Motion"), ("gs", "Gs"), ("physio+clock", "PhysTime")]:
            for st in STAGES:
                M[f"auc{tag}{SN[st]}{suf}"] = f3(aw[key][st]["mean"])
        for comp, tag in [("nonbold+bold - nonbold", "Nb"), ("latefusion(nonbold,bold) - nonbold", "Lf"),
                          ("motion+physio+bold - motion+physio", "Mp"), ("physio+bold - physio", "Ph"),
                          ("physio+clock+bold - physio+clock", "Pc"), ("nonbold+gs - nonbold", "Gs"),
                          ("nonbold+gs+bold - nonbold+gs", "GsBold"), ("nonbold_img+bold - nonbold_img", "Img"),
                          ("nonbold_nightfrac+bold - nonbold_nightfrac", "Nf"), ("nonbold+bold_hrf - nonbold", "Hrf"),
                          ("nonbold+bold_z - nonbold", "Z"), ("nonbold+bold_phys - nonbold", "Clean"),
                          ("nonbold+bold_physnull - nonbold", "Null"), ("nonbold+bold_gsr - nonbold", "Gsr"),
                          ("nonbold+bold_acc - nonbold", "Acc"), ("bold - gs", "BoldGs")]:
            if comp not in R["increment"]:
                continue
            for st in STAGES:
                r = R["increment"][comp][st]
                b = f"inc{tag}{SN[st]}{suf}"
                M[b] = sg(r["ws_mean"])
                M[b + "ci"] = ci(r["ws_ci"])
                M[b + "up"] = sg(r["ws_upper95"])
                M[b + "Subj"], M[b + "SubjN"] = str(r["n_pos"]), str(r["n_subj"])
                M[b + "P"] = pfmt(r["wilcoxon_p_holm"])
                M[b + "pooled"] = sg(r["pooled"])
        tm = R["time_matched_rem"]
        M["tmBold" + suf], M["tmPhys" + suf] = f3(tm["auc_ws"]["bold"]), f3(tm["auc_ws"]["physio"])
        M["tmInc" + suf], M["tmIncci" + suf] = sg(tm["inc_physio"]["ws_mean"]), ci(tm["inc_physio"]["ws_ci"])
        M["tmIncNb" + suf], M["tmIncNbci" + suf] = sg(tm["inc_nonbold"]["ws_mean"]), ci(tm["inc_nonbold"]["ws_ci"])
        pw = R["pairwise_auc_bold"]
        M["pwWNtwo" + suf], M["pwNoneNtwo" + suf], M["pwNtwoNthree" + suf] = (f2(pw["W/N2"]["ws"]), f2(pw["N1/N2"]["ws"]),
                                                                           f2(pw["N2/N3"]["ws"]))
        M["pwWNone" + suf] = f2(pw["W/N1"]["ws"])
        M["tmN" + suf], M["tmNrem" + suf] = f"{tm['n_epochs']:,}", f"{tm['n_rem']:,}"
        M["tmNb" + suf], M["tmAll" + suf] = f3(tm["auc_ws"]["nonbold"]), f3(tm["auc_ws"]["nonbold+bold"])
        M["tmIncNbup" + suf] = sg(tm["inc_nonbold"]["ws_upper95"])
        M["tmIncNbSubj" + suf], M["tmIncNbSubjN" + suf] = str(tm["inc_nonbold"]["n_pos"]), str(tm["inc_nonbold"]["n_subj"])
        for st in STAGES:
            M["elig" + SN[st] + suf] = str(R["meta"]["subjects_eligible_ws"][st])
        if "hgb" in R:
            for st in STAGES:
                r = R["hgb"]["increment"][st]
                b = f"hgb{SN[st]}{suf}"
                M[b], M[b + "ci"], M[b + "up"] = sg(r["ws_mean"]), ci(r["ws_ci"]), sg(r["ws_upper95"], 4)
                M[b + "Subj"], M[b + "SubjN"] = str(r["n_pos"]), str(r["n_subj"])
                for key, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All")]:
                    M[f"hgbAuc{tag}{SN[st]}{suf}"] = f3(R["hgb"]["auc_ws"][key][st])
        if "ladder" in R:
            L = R["ladder"]
            for name, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All"), ("motion", "Motion"),
                              ("physio", "Phys"), ("clock", "Time"), ("latefusion(nonbold,bold)", "Lf")]:
                M[f"kappa{tag}{suf}"] = f3(L[name]["kappa"])
                M[f"kappa{tag}ci{suf}"] = f"{L[name]['kappa_ci'][0]:.2f}, {L[name]['kappa_ci'][1]:.2f}"
            pp = R["ladder_paired"]
            for name, tag in [("nonbold+gs", "NbGs"), ("netFC-LR (after Tagliazucchi 2012)", "Tag"),
                              ("edgeFC-LR (after Altmann 2016)", "Alt")]:
                if name in L:
                    M[f"kappa{tag}{suf}"] = f3(L[name]["kappa"])
                    M[f"kappa{tag}ci{suf}"] = f"{L[name]['kappa_ci'][0]:.2f}, {L[name]['kappa_ci'][1]:.2f}"
            for name, tag in [("bold", "Bold"), ("nonbold+bold", "All"), ("nonbold", "Nb")]:
                M[f"mF{tag}{suf}"] = f2(L[name]["macroF1"])
                M[f"fNthree{tag}{suf}"], M[f"fREM{tag}{suf}"] = f2(L[name]["f1"]["N3"]), f2(L[name]["f1"]["REM"])
            for k, tag in [("nonbold+bold vs nonbold", "AllNb"), ("bold vs nonbold", "BoldNb"),
                           ("bold vs netFC-LR (after Tagliazucchi 2012)", "BoldTag"),
                           ("bold vs edgeFC-LR (after Altmann 2016)", "BoldAlt")]:
                if k in pp:
                    M[f"kd{tag}{suf}"], M[f"kd{tag}ci{suf}"] = sg(pp[k]["diff"]), ci(pp[k]["ci"])
                    M[f"kd{tag}Subj{suf}"], M[f"kd{tag}N{suf}"] = str(pp[k]["n_better"]), str(pp[k]["n"])
                    M[f"kd{tag}P{suf}"] = pfmt(pp[k]["p"])
            M["kappaOracle" + suf] = f3(R["oracle_previous_label_kappa"])

    # ---------------- paired N3-REM contrast and t-based bounds (33_contrast.py)
    CT = json.load(open(BASE / "results_contrast.json"))
    for src, suf in [("afni", ""), ("raw", "Raw")]:
        for k, tag in [("lr", ""), ("latefusion", "Lf"), ("hgb", "Hgb"), ("stack", "St")]:
            if k not in CT[src]:
                continue
            r = CT[src][k]
            M[f"inter{tag}{suf}"], M[f"inter{tag}ci{suf}"] = sg(r["contrast"]), ci(r["ci"])
            M[f"ctrAbove{tag}{suf}"], M[f"ctrN{tag}{suf}"] = str(r["n_n3_above_rem"]), str(r["n"])
            M[f"remUpT{tag}{suf}"] = sg(r["rem_upper95_t"])
    # text: "(10/10; \interLf to \interSt for the other three estimators)"
    o3 = {k: CT["afni"][k]["contrast"] for k in ("latefusion", "hgb", "stack")}
    assert min(o3, key=o3.get) == "latefusion" and max(o3, key=o3.get) == "stack", o3
    # ---------------- staging checks (32_staging_checks.py)
    SC = json.load(open(BASE / "results_staging_checks.json"))
    r = SC["kappa_all_minus_bold"]
    M["kdAllBold"], M["kdAllBoldci"] = sg(r["diff"]), ci(r["ci"])
    M["kdAllBoldWorse"], M["kdAllBoldN"] = str(r["n_worse"]), str(r["n"])
    for k, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All")]:
        q = SC["rem"][k]
        M[f"recEarly{tag}"], M[f"recLate{tag}"] = f2(q["recall_early"]), f2(q["recall_late"])
        M[f"precREM{tag}"], M[f"falseLate{tag}"] = f2(q["precision"]), f2(q["false_rem_late"])
        M[f"recREM{tag}"] = f2(q["recall"])
        M[f"kArg{tag}"] = f3(SC["argmax_kappa"][k])
    M["nRemEarly"] = str(SC["rem"]["bold"]["n_early"])
    # "lowest in limbic, default, salience and control networks (a-b), highest in visual cortex (c)",
    # with "after decoding BOLD raises REM F1 by precision, not recall"
    q_nb, q_all = SC["rem"]["nonbold"], SC["rem"]["nonbold+bold"]
    assert q_all["precision"] > q_nb["precision"] and q_all["recall"] < q_nb["recall"] and q_all["f1"] > q_nb["f1"]
    NM = json.load(open(BASE / "results_n3_mechanism.json"))["parcel_auc"]["N3"]["per_network"]
    low4 = sorted(NM, key=NM.get)[:4]
    assert set(low4) == {"Limbic", "Default", "SalVentAttn", "Cont"} and max(NM, key=NM.get) == "Vis", NM
    M["nthreeNetLo"], M["nthreeNetHi"], M["nthreeNetVis"] = f2(NM[low4[0]]), f2(NM[low4[-1]]), f2(NM["Vis"])
    # ---------------- extra baselines (34_extra_baselines.py)
    EB = json.load(open(BASE / "results_extra_baselines.json"))
    for k, tag in [("clock+bold - clock", "Cl"), ("motion+clock+bold - motion+clock", "Mc"),
                   ("nonbold+bold - nonbold (C=0.05)", "Reg")]:
        for st in STAGES:
            r = EB[k][st]
            b = f"inc{tag}{SN[st]}"
            M[b], M[b + "ci"], M[b + "up"] = sg(r["ws_mean"]), ci(r["ws_ci"]), sg(r["ws_upper95"])
            M[b + "Subj"], M[b + "SubjN"] = str(r["n_pos"]), str(r["n_subj"])

    # ---------------- round-2 additions
    nt = SC["no_time"]["motion+physio+bold"]
    M["kappaNoTime"], M["kdNoTime"], M["kdNoTimeci"] = f3(nt["kappa"]), sg(nt["diff_vs_bold"]), ci(nt["ci"])
    M["recEarlyNoTime"] = f2(nt["recall_early_rem"])
    for k, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All"), ("motion+physio+bold", "NoTime")]:
        M["ac" + tag], M["conf" + tag] = f2(SC["emission"][k]["lag1_r"]), f2(SC["emission"][k]["mean_max_p"])
    for st in ("W", "N1", "N2", "N3"):
        M[f"f{SN[st]}Bold"] = f2(A["ladder"]["bold"]["f1"][st])
    hr = CT["headroom"]
    for st in ("N3", "REM"):
        pc = lambda x: f"{round(100 * x) + 0:.0f}".replace("-", "$-$")       # whole percent, no "-0"
        M[f"hr{SN[st]}"] = pc(hr[st]["mean"])
        M[f"hr{SN[st]}ci"] = ", ".join(pc(x) for x in hr[st]["ci"])
    for other in ("W", "N1", "N2", "N3"):
        r = CT["rem_vs"]["lr"][other]
        M[f"rv{SN[other]}"], M[f"rv{SN[other]}ci"] = sg(r["mean"]), ci(r["ci"])
    rv_other = [CT["rem_vs"][k][o]["mean"] for k in CT["rem_vs"] for o in ("W", "N1", "N2")]
    rv_n3 = [CT["rem_vs"][k]["N3"]["mean"] for k in CT["rem_vs"]]
    M["rvOtherMin"], M["rvOtherMax"] = sg(min(rv_other)), sg(max(rv_other))
    M["rvNthreeMin"], M["rvNthreeMax"] = sg(min(rv_n3)), sg(max(rv_n3))
    M["hmmGainBold"] = sg(SC["hmm_gain"]["bold"])
    M["hmmGainAll"] = sg(SC["hmm_gain"]["nonbold+bold"])
    al = sorted(float(a_) for a_ in SC["alpha_sweep"]["bold"])
    assert SC["bold_above_combined_all_alpha"]
    M["alphaLo"], M["alphaHi"] = f"{al[0]:g}", f"{al[-1]:g}"
    M["clkBold"], M["clkNb"] = f2(CT["clock_check_n2"]["bold"]["mean"]), f2(CT["clock_check_n2"]["nonbold"]["mean"])
    M["clkSc"] = f2(CT["clock_check_n2"]["subcortex"]["mean"])
    AL = json.load(open(BASE / "results_data_checks.json"))["alignment"]
    M["alignOneVol"], M["alignRuns"] = str(AL["within_one_volume"]), str(AL["n_runs"])
    M["nDriftRuns"], M["nDriftEpochs"], M["nDriftRem"] = (str(AL["analysed_drift_runs"]), f"{AL['analysed_drift_epochs']:,}",
                                                          str(AL["analysed_drift_rem"]))
    M["nDriftAll"] = str(AL["n_runs"] - AL["within_one_epoch"])          # runs drifting by more than an epoch
    RB_ = json.load(open(BASE / "results_rem_bouts.json"))                 # 30b_rem_bouts.py
    M["remBoutMed"] = f"{RB_['median_bout_min']:.1f}"
    M["remNightTwo"], M["epNightTwo"] = f"{100 * RB_['frac_rem_night2']:.0f}", f"{100 * RB_['frac_epochs_night2']:.0f}"
    LN = A["time_matched_rem"]["inc_nonbold"]                              # late night, binary REM vs NREM
    M["lnREM"], M["lnREMci"], M["lnREMupT"] = sg(LN["ws_mean"]), ci(LN["ws_ci"]), sg(tbound(LN))
    M["pNremMax"] = pfmt(max(A["increment"]["nonbold+bold - nonbold"][st]["wilcoxon_p_holm"] for st in ("W", "N1", "N2", "N3")))
    NMr = json.load(open(BASE / "results_n3_mechanism.json"))["parcel_auc"]["REM"]["per_network"]
    M["remNetLo"], M["remNetHi"] = f2(min(NMr.values())), f2(max(NMr.values()))
    STK = json.load(open(BASE / "results_stacked.json"))["increment"]
    for st in STAGES:
        r = STK[st]
        M[f"incSt{SN[st]}"], M[f"incSt{SN[st]}ci"], M[f"incSt{SN[st]}up"] = sg(r["ws_mean"]), ci(r["ws_ci"]), sg(r["ws_upper95"])
        M[f"incSt{SN[st]}Subj"], M[f"incSt{SN[st]}SubjN"] = str(r["n_pos"]), str(r["n_subj"])
    # envelope over the four fusion estimators: largest REM upper bound, smallest N3 lower bound
    est = {"lr": A["increment"]["nonbold+bold - nonbold"], "lf": A["increment"]["latefusion(nonbold,bold) - nonbold"],
           "stack": STK, "hgb": A["hgb"]["increment"]}
    rem_up = {k: max(v["REM"]["ws_upper95"], tbound(v["REM"])) for k, v in est.items()}
    n3_lo = {k: min(v["N3"]["ws_lower95"], tbound(v["N3"], upper=False)) for k, v in est.items()}
    M["remUpMax"], M["nthreeLoMin"] = sg(max(rem_up.values())), sg(min(n3_lo.values()))
    # the text names which estimator attains each extreme; keep it true
    assert max(rem_up, key=rem_up.get) == "lf" and min(n3_lo, key=n3_lo.get) == "hgb", (rem_up, n3_lo)
    rem_means = [v["REM"]["ws_mean"] for v in est.values()]
    M["remGainMin"], M["remGainMax"] = sg(min(rem_means)), sg(max(rem_means))
    M["incLfREMupT"] = sg(tbound(A["increment"]["latefusion(nonbold,bold) - nonbold"]["REM"]))
    # sentences the text states in words; fail loudly if a rerun ever changes them
    assert all(CT["afni"][k]["n_n3_above_rem"] == CT["afni"][k]["n"] for k in ("lr", "latefusion", "hgb", "stack"))
    for k in ("lr", "stack"):
        assert est[k]["REM"]["ws_upper95"] < ma["delta"] and tbound(est[k]["REM"]) < ma["delta"], k
    _mt = json.load(open(BASE / "results_matched.json"))["increment"]
    assert all(_mt[k]["REM"]["ws_mean"] < 0 for k in ("raw84amp", "raw", "afni84", "afni"))   # "never turns positive"
    M["incNbREMupT"], M["hgbREMupT"] = sg(tbound(A["increment"]["nonbold+bold - nonbold"]["REM"])), sg(tbound(A["hgb"]["increment"]["REM"]))
    M["incStREMupT"] = sg(tbound(STK["REM"]))

    # ---------------- extra BOLD blocks (36b_extra_block.py): multi-window context, subcortex
    BLK = {}
    for blk, pre in (("mw", "mw"), ("subc", "sc"), ("lvl", "lv"), ("max", "mx")):
        f = BASE / f"results_block_{blk}.json"
        if not f.exists():
            continue
        RB = BLK[blk] = json.load(open(f))
        M[f"n{pre.capitalize()}Feat"] = str(RB["n_features"])
        for est_ in ("early", "late", "stack"):
            for q, qtag in (("all BOLD over non-BOLD", "All"), ("block beyond cortical BOLD", "Beyond")):
                G = RB["gain"][f"{est_}: {q}"]
                for st in ("N3", "REM"):
                    b_ = f"{pre}{qtag}{est_.capitalize()}{SN[st]}"
                    r = G[st]
                    M[b_], M[b_ + "ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
                    M[b_ + "up"], M[b_ + "upT"] = sg(r["ws_upper95"]), sg(r["t_upper95"])
                    M[b_ + "lo"], M[b_ + "loT"] = sg(r["ws_lower95"]), sg(r["t_lower95"])
                    M[b_ + "Subj"], M[b_ + "SubjN"] = str(r["n_pos"]), str(r["n_subj"])
                c_ = RB["contrast"][f"{est_}: {q}"]
                M[f"{pre}{qtag}{est_.capitalize()}Ctr"], M[f"{pre}{qtag}{est_.capitalize()}Ctrci"] = sg(c_["value"]), ci(c_["ci"])
                M[f"{pre}{qtag}{est_.capitalize()}CtrSubj"], M[f"{pre}{qtag}{est_.capitalize()}CtrN"] = str(c_["n_pos"]), str(c_["n"])
        for st in STAGES:
            M[f"auc{pre.capitalize()}{SN[st]}"] = f3(RB["auc_ws"]["subc"][st])
        M[f"kappaBold{pre.capitalize()}"] = f3(RB["kappa"]["bold_subc"])
    if "subc" in BLK:   # pre-registered subcortical test (next_steps.md) and its exploratory follow-up (36c)
        g_ = lambda e_, q: BLK["subc"]["gain"][f"{e_}: {q}"]["REM"]
        ea, sa = g_("early", "all BOLD over non-BOLD"), g_("stack", "all BOLD over non-BOLD")
        # the paper reports case C ("in between"): keep it true
        assert ea["ws_upper95"] < ma["delta"] and ea["t_upper95"] < ma["delta"], "early inside margin"
        assert sa["ws_upper95"] > ma["delta"] or sa["t_upper95"] > ma["delta"], "stack not inside margin (else case A)"
        assert not (ea["t_lower95"] > 0 and ea["ws_lower95"] > 0), "early not clearly positive (else maybe case B)"
        assert sa["ws_mean"] < ma["delta"], "subtitle rule: stacked mean below delta"
        assert all(BLK["subc"]["contrast"][f"{k}: all BOLD over non-BOLD"]["n_pos"] == BLK["subc"]["contrast"][f"{k}: all BOLD over non-BOLD"]["n"]
                   for k in ("early", "late", "stack"))
        FU = json.load(open(BASE / "results_subcortex_followup.json"))
        for tag, r in (("scHgb", FU["hgb"]["gain"]), ("scHgbBeyond", FU["hgb_beyond_cortex"]["gain"]),
                       ("scBeyondGs", FU["stack_beyond_cortex_and_gs"]["gain"]),
                       ("scAmpOnly", FU["stack"]["amplitudes only"]["gain"]), ("scFcOnly", FU["stack"]["correlations only"]["gain"]),
                       ("scWlStack", FU["best_aligned"]["stack: all BOLD over non-BOLD"]["gain"]),
                       ("scWlCort", FU["best_aligned"]["stack: cortical BOLD over non-BOLD"]["gain"]),
                       ("scWlBeyond", FU["best_aligned"]["stack: subcortex beyond cortex"]["gain"]),
                       ("scWlEarly", FU["best_aligned"]["early: all BOLD over non-BOLD"]["gain"])):
            for st in ("N3", "REM"):
                x = r[st]
                M[f"{tag}{SN[st]}"], M[f"{tag}{SN[st]}ci"] = sg(x["ws_mean"]), ci(x["ws_ci"])
                M[f"{tag}{SN[st]}upT"], M[f"{tag}{SN[st]}loT"] = sg(x["t_upper95"]), sg(x["t_lower95"])
                M[f"{tag}{SN[st]}Subj"], M[f"{tag}{SN[st]}SubjN"] = str(x["n_pos"]), str(x["n_subj"])
        M["scHgbCtr"], M["scHgbCtrSubj"], M["scHgbCtrN"] = sg(FU["hgb"]["contrast"]["value"]), str(FU["hgb"]["contrast"]["n_pos"]), str(FU["hgb"]["contrast"]["n"])
        drop = {k[len("without "):]: v["gain"]["REM"]["ws_mean"] for k, v in FU["stack"].items() if k.startswith("without ")}
        only = {k[len("only "):]: v["gain"]["REM"]["ws_mean"] for k, v in FU["stack"].items() if k.startswith("only ")}
        M["scDropMin"], M["scDropMax"] = sg(min(drop.values())), sg(max(drop.values()))
        M["scOnlyMin"], M["scOnlyMax"] = sg(min(only.values())), sg(max(only.values()))
        assert min(drop, key=drop.get) == "brainstem" and max(only, key=only.get) == "brainstem", (drop, only)
        sc_rem = {"early": ea["ws_mean"], "late": g_("late", "all BOLD over non-BOLD")["ws_mean"], "stack": sa["ws_mean"],
                  "hgb": FU["hgb"]["gain"]["REM"]["ws_mean"]}
        assert min(sc_rem, key=sc_rem.get) == "early" and max(sc_rem, key=sc_rem.get) == "hgb", sc_rem
        M["remGainMinSc"], M["remGainMaxSc"] = sg(min(sc_rem.values())), sg(max(sc_rem.values()))
        # "in the best-aligned runs the stacked total stays X, with Y from the cortex alone"
        assert FU["best_aligned"]["stack: cortical BOLD over non-BOLD"]["gain"]["REM"]["ws_mean"] > FU["best_aligned"]["stack: subcortex beyond cortex"]["gain"]["REM"]["ws_mean"]
        M["scBeyondStackREMp"] = pfmt(BLK["subc"]["gain"]["stack: block beyond cortical BOLD"]["REM"]["wilcoxon_p"])
        M["scShort"] = f"{ma['delta'] - sa['ws_mean']:.3f}"      # stacked mean short of delta (the rule's 'reaches delta')
        M["nWlEligREM"] = str(FU["best_aligned"]["stack: all BOLD over non-BOLD"]["gain"]["REM"]["n_subj"])
    if "mw" in BLK:   # the text gives the early..late range for "beyond the main block"
        v = {k: BLK["mw"]["gain"][f"{k}: block beyond cortical BOLD"]["REM"]["ws_mean"] for k in ("early", "late", "stack")}
        assert min(v, key=v.get) == "early" and max(v, key=v.get) == "late", v

    # ---------------- preprocessing ladder (49_preproc_ladder.py): our pipeline rebuilt step by step towards the derivative
    LD_TAGS = {"rv84amp": "RvAmp", "rv84phys": "RvPhys", "afni84": "Afni"}
    for v, vt in (("aff", "Aff"), ("aff_gm", "Gm"), ("syn", "Syn"), ("syn_gm", "SynGm")):
        for rg, rt in (("", ""), ("_acc", "Acc"), ("_accmot", "AccMot"), ("_accmotphys", "All")):
            LD_TAGS[f"amp_{v}{rg}"] = vt + rt
    if (BASE / "results_preproc_ladder.json").exists():
        LD = json.load(open(BASE / "results_preproc_ladder.json"))
        M["ldN"] = f"{LD['meta']['n_epochs']:,}"
        for key, tag in LD_TAGS.items():
            for st in ("N3", "REM"):
                r = LD["increment"][key][st]
                M[f"ldInc{tag}{SN[st]}"], M[f"ldInc{tag}{SN[st]}ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
                M[f"ldInc{tag}{SN[st]}Subj"], M[f"ldInc{tag}{SN[st]}SubjN"] = str(r["n_pos"]), str(r["n_subj"])
                M[f"ldAuc{tag}{SN[st]}"] = f3(LD["auc_ws"][key][st])
        own = {k: LD["increment"][k]["N3"]["ws_mean"] for k in LD_TAGS if k.startswith("amp_")}
        M["ldOwnNthreeMin"], M["ldOwnNthreeMax"] = sg(min(own.values())), sg(max(own.values()))
        M["ldOwnAucNthreeMin"] = f3(min(LD["auc_ws"][k]["N3"] for k in own)); M["ldOwnAucNthreeMax"] = f3(max(LD["auc_ws"][k]["N3"] for k in own))
        M["ldOwnRemMax"] = sg(max(LD["increment"][k]["REM"]["ws_mean"] for k in own))
        M["ldOwnRemMaxUpT"] = sg(max(tbound(LD["increment"][k]["REM"]) for k in own))
        M["ldBestKey"] = max(own, key=own.get).replace("amp_", "").replace("_", "\\_")
        # mean effect of each rebuilt step on the N3 gain and on the BOLD-alone N3 AUC, over the other factors:
        # alignment = syn vs aff at equal GM restriction and regression; purity = _gm vs plain at equal alignment and regression
        def pairs_for(step_name):
            if step_name == "Syn":
                return [(a, a.replace("syn", "aff")) for a in own if "syn" in a]
            return [(a, a.replace("_gm", "")) for a in own if "_gm" in a]
        for name in ("Syn", "Gm"):
            pairs = pairs_for(name)
            assert all(b in own for _, b in pairs), pairs
            M[f"ldStep{name}Inc"] = sg(np.mean([own[a] - own[b] for a, b in pairs]))
            M[f"ldStep{name}Auc"] = sg(np.mean([LD["auc_ws"][a]["N3"] - LD["auc_ws"][b]["N3"] for a, b in pairs]))
        for rg, name in (("_acc", "Acc"), ("_accmot", "AccMot"), ("_accmotphys", "All")):
            pairs = [(k, k.replace(rg, "")) for k in own if k.endswith(rg)]
            M[f"ldStep{name}Inc"] = sg(np.mean([own[a] - own[b] for a, b in pairs]))
            M[f"ldStep{name}Auc"] = sg(np.mean([LD["auc_ws"][a]["N3"] - LD["auc_ws"][b]["N3"] for a, b in pairs]))
        M["ldGapInc"] = sg(LD["increment"]["afni84"]["N3"]["ws_mean"] - LD["increment"]["amp_aff"]["N3"]["ws_mean"])
        M["ldGapAuc"] = sg(LD["auc_ws"]["afni84"]["N3"] - LD["auc_ws"]["amp_aff"]["N3"])
        M["ldClosedPct"] = f"{100 * (max(own.values()) - own['amp_aff']) / (LD['increment']['afni84']['N3']['ws_mean'] - own['amp_aff']):.0f}"
        assert all(LD["increment"][k]["N3"]["ws_mean"] > LD["increment"][k]["REM"]["ws_mean"] for k in LD_TAGS), "ladder: N3 gain above REM gain on every rung"
    # ---------------- matched epochs (28_matched.py): only the BOLD source differs
    MT = json.load(open(BASE / "results_matched.json"))
    M["mtN"] = f"{MT['meta']['n_epochs']:,}"
    for key, tag in [("raw84amp", "RawAmp"), ("raw", "Raw"), ("afni84", "Afnisub"), ("afni", "Afni"),
                     ("rv84amp", "RvAmp"), ("rv84phys", "RvPhys")]:
        for st in STAGES:
            M[f"mtAuc{tag}{SN[st]}"] = f3(MT["auc_ws"][key][st])
            r = MT["increment"][key][st]
            M[f"mtInc{tag}{SN[st]}"], M[f"mtInc{tag}{SN[st]}ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
    for st in STAGES:
        M[f"mtAucNb{SN[st]}"] = f3(MT["auc_ws"]["nonbold"][st])
    for key, tag in [("rv84amp", "RvAmp"), ("rv84phys", "RvPhys"), ("rv84mot", "RvMot")]:
        for st in STAGES:
            r = MT["increment"][key][st]
            M[f"mtInc{tag}{SN[st]}"], M[f"mtInc{tag}{SN[st]}ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
    # text: realignment and physiological regression leave the N3 gain small; the derivative's steps raise it
    lad = [MT["increment"][k]["N3"]["ws_mean"] for k in ("raw84amp", "rv84amp", "rv84phys", "afni84", "afni")]
    assert max(lad[:3]) < 0.03 and lad[3] > 0.06 and lad[4] > lad[3], lad
    assert MT["increment"]["rv84mot"]["N3"]["ws_mean"] < 0.03, "text: motion regression does not recover the derivative's N3 gain"
    own = {k: MT["increment"][k]["N3"]["ws_mean"] for k in ("raw84amp", "rv84amp", "rv84phys", "rv84mot")}
    M["mtOwnMin"], M["mtOwnMax"] = sg(min(own.values())), sg(max(own.values()))
    _shown = [own["raw84amp"], own["rv84amp"]]
    assert sg(min(_shown)) == M["mtOwnMin"] and sg(max(_shown)) == M["mtOwnMax"], "text: Table 3a's shown rows span the own-pipeline N3 range"
    ladr = [MT[q][k]["REM"]["ws_mean"] for q in ("increment",) for k in ("raw84amp", "rv84amp", "rv84phys", "afni84", "afni")]
    M["mtRemMaxEarly"] = sg(max(ladr))
    assert max(ladr) < 0
    for key, tag in [("raw84amp", "RawAmp"), ("raw", "Raw"), ("afni84", "Afnisub"), ("afni", "Afni")]:
        for st in ("N3", "REM"):
            r = MT["increment_late"][key][st]
            M[f"mtLate{tag}{SN[st]}"], M[f"mtLate{tag}{SN[st]}ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
    # "the REM gain stays at or below X under both fusion rules"; "the N3 gain grows ... under both"
    mt_rem = [MT[k][key]["REM"]["ws_mean"] for k in ("increment", "increment_late") for key in ("raw84amp", "afni84", "afni")]
    M["mtRemMax"] = sg(max(mt_rem))
    assert max(mt_rem) == MT["increment_late"]["afni"]["REM"]["ws_mean"]
    for k in ("increment", "increment_late"):
        v = [MT[k][key]["N3"]["ws_mean"] for key in ("raw84amp", "afni84", "afni")]
        assert v[0] < v[1] < v[2], v
    M["nParcelsRaw"] = str(MT["meta"]["n_common_parcels"])
    assert M["nParcelsRaw"] == "84", "Table 3 and Table 4 row labels say 84 parcels"
    n_fc_raw = mw["n_bold_features"] - MT["meta"]["n_common_parcels"]
    M["nNetfcRaw"] = str(n_fc_raw)
    M["nNetsRaw"] = str(int(round((1 + (1 + 8 * n_fc_raw) ** 0.5) / 2)))     # n(n-1)/2 pairs
    M["pctREM"] = f"{100 * ma['stage_counts']['REM'] / ma['n_epochs']:.0f}"

    # ---------------- N3 mechanism (29_n3_mechanism.py)
    ME = json.load(open(BASE / "results_n3_mechanism.json"))
    for key, tag in [("nonbold+amp", "Amp"), ("nonbold+fc", "Fc")]:
        for st in STAGES:
            r = ME["increment"][key][st]
            M[f"mech{tag}{SN[st]}"], M[f"mech{tag}{SN[st]}ci"] = sg(r["ws_mean"]), ci(r["ws_ci"])
    pa = ME["parcel_auc"]
    M["parcelNthreeMed"], M["parcelREMMed"] = f2(pa["N3"]["median"]), f2(pa["REM"]["median"])
    M["parcelNthreeBelow"] = f"{100 * (1 - pa['N3']['frac_above']):.0f}"
    M["parcelREMBelow"] = f"{100 * (1 - pa['REM']['frac_above']):.0f}"

    # ---------------- data checks (30_data_checks.py)
    DC = json.load(open(BASE / "results_data_checks.json"))
    for st in STAGES:
        M["fd" + SN[st]] = f3(DC["fd_by_stage"][st]["median_mm"])
        M["fd" + SN[st] + "Raw"] = f3(DC["fd_by_stage_raw"][st]["median_mm"])
        for src, suf in [("afni", ""), ("raw", "Raw")]:
            M[f"ret{SN[st]}{suf}"] = f"{100 * DC['retention'][src][st]['frac']:.0f}"
    M["retAll"], M["retAllRaw"] = (f"{100 * DC['retention']['afni']['all']['frac']:.0f}",
                                   f"{100 * DC['retention']['raw']['all']['frac']:.0f}")
    M["nEpochsCovered"] = f"{DC['retention']['afni']['all']['before']:,}"
    M["fdComR"] = f2(DC["fd_vs_com_r"]["median"])
    C = np.load(BASE / "confounds_ds005127.npz", allow_pickle=True)
    M["nPhysFeat"] = str(C["physio"].shape[1])
    M["nNbFeat"] = str(ma["n_motion"] + C["physio"].shape[1] + 2)

    draft10(M, A, W, ma)

    with open(PAPER / "numbers.tex", "w") as f:
        f.write("% generated by 26_paper_numbers_v2.py -- do not edit\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")

    # ---------------- Table 1: ladder on the AFNI derivative, with per-stage AUC and the gain columns
    L, aw = A["ladder"], dict(A["auc_ws"])
    # the network-FC reference model's probabilities are saved but not in auc_ws; same statistic here
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    yz, sz, pz = Z["y"], Z["subj"], Z["proba__netfc"]
    aw["netFC-LR (after Tagliazucchi 2012)"] = {}
    for i, st in enumerate(STAGES):
        v = []
        for u in np.unique(sz):
            m = sz == u
            pos = yz[m] == st
            if pos.sum() >= ma["min_epochs"] and (~pos).sum() >= ma["min_epochs"]:
                v.append(roc_auc_score(pos, pz[m, i]))
        aw["netFC-LR (after Tagliazucchi 2012)"][st] = {"mean": float(np.mean(v))}
    rows = [("Time (2)", "clock", "clock"),
            (f"Motion ({ma['n_motion']})", "motion", "motion"), (f"Physiology ({M['nPhysFeat']})", "physio", "physio"),
            (f"Non-BOLD ({M['nNbFeat']})", "nonbold", "nonbold"),
            ("Network FC (21)", "netFC-LR (after Tagliazucchi 2012)", "netFC-LR (after Tagliazucchi 2012)"),
            (f"BOLD ({ma['n_bold_features']})", "bold", "bold"),
            ("Non-BOLD$+$BOLD", "nonbold+bold", "nonbold+bold")]
    best = max(L[k]["kappa"] for _, k, _ in rows)
    out = ["\\begin{table}[t]", "\\centering", "\\fontsize{7.5}{9}\\selectfont", "\\setlength{\\tabcolsep}{1.3pt}",
           "\\caption{Five-class staging, derivative, planned model: Cohen's $\\kappa$ [95\\% CI] and F1 after HMM decoding, "
           "within-subject AUC before it. FC: network connectivity~\\cite{tagliazucchi2012}. In parentheses: gain over "
           "non-BOLD.}",
           "\\label{tab:ladder}", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
           " & \\multicolumn{3}{c}{after HMM} & \\multicolumn{2}{c}{before HMM: AUC} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-6}",
           "Model (features) & $\\kappa$ [95\\% CI] & F1$_\\mathrm{N3}$ & F1$_\\mathrm{REM}$ & N3 & REM \\\\",
           "\\midrule"]
    for label, key, akey in rows:
        h = L[key]
        kap = f"{h['kappa']:.2f}"
        if abs(h["kappa"] - best) < 1e-12:
            kap = f"\\textbf{{{kap}}}"
        cis = f" [{h['kappa_ci'][0]:.2f}, {h['kappa_ci'][1]:.2f}]" if key != "prior" else ""
        an = f"{aw[akey]['N3']['mean']:.2f}" if akey in aw else "--"
        ar = f"{aw[akey]['REM']['mean']:.2f}" if akey in aw else "--"
        if key == "nonbold+bold":     # the headline gains, visible without subtraction
            gi = A["increment"]["nonbold+bold - nonbold"]
            an += f" ({gi['N3']['ws_mean']:+.2f})".replace("-", "$-$")
            ar += f" ({gi['REM']['ws_mean']:+.2f})".replace("-", "$-$")
        out.append(f"{label} & {kap}{cis} & {h['f1']['N3']:.2f} & {h['f1']['REM']:.2f} & {an} & {ar} \\\\")
        if key in ("nonbold", "bold"):
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tab_ladder.tex").write_text("\n".join(out) + "\n")

    # ---------------- Table 2: robustness of the two headline gains
    def cell(r, dagger=False):
        if r is None:
            return "--"
        f = lambda x: f"{x:.3f}".replace("-", "$-$")     # three decimals, so the bold (CI above zero) rule can be checked
        txt = f"{f(r['ws_mean'])} [{f(r['ws_ci'][0])},\\,{f(r['ws_ci'][1])}]"
        if r["ws_ci"][0] > 0:
            txt = f"\\textbf{{{f(r['ws_mean'])}}} [{f(r['ws_ci'][0])},\\,{f(r['ws_ci'][1])}]"
        if dagger and r["ws_upper95"] < ma["delta"] and tbound(r) < ma["delta"]:
            txt += "$^\\dagger$"
        return txt

    inc_a, inc_w = A["increment"], W["increment"]
    WL = json.load(open(BASE / "results_whitelist.json"))
    M["wlRuns"], M["wlListed"], M["wlEpochs"] = str(WL["afni"]["n_runs"]), str(WL["n_whitelist"]), f"{WL['afni']['n_epochs']:,}"
    EBt = json.load(open(BASE / "results_extra_baselines.json"))
    ST = json.load(open(BASE / "results_stacked.json"))
    MX = json.load(open(BASE / "results_max_hgb.json"))
    for st in ("N3", "REM"):
        x = MX["all BOLD over non-BOLD"]["gain"][st]
        M[f"mxHgb{SN[st]}"], M[f"mxHgb{SN[st]}ci"], M[f"mxHgb{SN[st]}upT"] = sg(x["ws_mean"]), ci(x["ws_ci"]), sg(x["t_upper95"])
    M["nAllBoldFeat"] = str(ma["n_bold_features"] + BLK["max"]["n_features"])
    g_ = lambda blk, q: BLK[blk]["gain"][q]
    AB = "all BOLD over non-BOLD"
    RC = json.load(open(BASE / "results_reviewer_checks.json"))
    R2c_ = json.load(open(BASE / "results_round2_checks.json"))

    def tuned(mdl, sset):
        f = BASE / f"results_tuned_{mdl}_{sset}.json"
        return json.load(open(f))["gain"] if f.exists() else None

    SPf = BASE / "results_strong_physio.json"
    SP = json.load(open(SPf))["models"] if SPf.exists() else {}
    RVf = BASE / "results_rv2_eyes.json"
    RV = json.load(open(RVf))["A"] if RVf.exists() else None
    # the 13 data and covariate variants, printed as one range row (each one is in results_*.json)
    variants = [("global signal in both models", inc_a["nonbold+gs+bold - nonbold+gs"]),
                ("BOLD after global-signal regression", inc_a["nonbold+bold_gsr - nonbold"]),
                ("BOLD after aCompCor", inc_a["nonbold+bold_acc - nonbold"]),
                ("BOLD windows shifted 6 s", inc_a["nonbold+bold_hrf - nonbold"]),
                ("BOLD z-scored within run", inc_a["nonbold+bold_z - nonbold"]),
                ("image-derived motion added", inc_a["nonbold_img+bold - nonbold_img"]),
                ("night fraction for elapsed time", inc_a["nonbold_nightfrac+bold - nonbold_nightfrac"]),
                ("night index added", RC["night_index"]),
                ("fully uncensored epochs", RC["uncensored"]),
                ("slow physiology regressed from BOLD", R2c_["slow"]["gain"]),
                ("cubic within-run detrending", R2c_["dt"]["gain"]),
                ("cross-night training", R2c_["cross_night"]["gain"]),
                ("without drifting runs", R2c_["no_drift_runs"]["gain"]),
                ("without the minute around awakenings (tones)", json.load(open(BASE / "results_tone_proxy.json"))["gain"]),
                ("without 5 min around awakenings", json.load(open(BASE / "results_panel_checks.json"))["tones5"]["gain"]),
                ("lagged physiology in the baseline", json.load(open(BASE / "results_panel_checks.json"))["lagged"]["gain"]),
                ("without the somatomotor-auditory network", json.load(open(BASE / "results_panel_checks2.json"))["noaud"]["gain"]),
                (f"best-aligned runs ({WL['afni']['n_runs']})", WL["afni"]["increment"])] + \
               [(f"C = {c}", RC["C_sweep"][c]) for c in ("0.01", "0.1", "10.0")]
    M["nVariants"] = str(len(variants))
    # rows: (label, gain dict, group); "spec" = BOLD over the full non-BOLD model on the derivative;
    # "strong" = over the stronger non-BOLD model (+ cardiorespiratory staging features), same estimator both sides
    rob = [("Main (LR, early fusion)", inc_a["nonbold+bold - nonbold"], "spec"),
           ("Late fusion", inc_a["latefusion(nonbold,bold) - nonbold"], "spec"),
           ("Stacked fusion (nested)", ST["increment"], "spec"),
           ("Boosted trees", A["hgb"]["increment"], "spec")]
    if tuned("lr", "main"):
        rob.append(("LR, nested-tuned $C$", tuned("lr", "main"), "spec"))
    if tuned("hgb", "main"):
        rob.append(("Boosted trees, nested-tuned", tuned("hgb", "main"), "spec"))
    rob.append((f"{len(variants)} variants$^\\ddagger$", "RANGE", "spec"))
    rob += [(f"$+$ 30--300\\,s context ({BLK['mw']['n_features']})", g_("mw", f"early: {AB}"), "spec"),
            (f"$+$ subcortex ({BLK['subc']['n_features']}), early", g_("subc", f"early: {AB}"), "spec"),
            ("\\quad late fusion", g_("subc", f"late: {AB}"), "spec"),
            ("\\quad stacked", g_("subc", f"stack: {AB}"), "spec"),
            ("\\quad boosted trees", FU["hgb"]["gain"], "spec"),
            (f"$+$ regional level ({BLK['lvl']['n_features']}), stacked", g_("lvl", f"stack: {AB}"), "spec"),
            (f"All BOLD blocks ({M['nAllBoldFeat']}), early", g_("max", f"early: {AB}"), "spec"),
            ("\\quad late fusion", g_("max", f"late: {AB}"), "spec"),
            ("\\quad stacked", g_("max", f"stack: {AB}"), "spec"),
            ("\\quad boosted trees", MX[AB]["gain"], "spec")]
    if tuned("lr", "max"):
        rob.append(("\\quad LR, nested-tuned", tuned("lr", "max"), "spec"))
    if tuned("hgb", "max"):
        rob.append(("\\quad trees, nested-tuned", tuned("hgb", "max"), "spec"))
    for fam, lab in (("lr", "Stronger baseline, LR"), ("hgb", "\\quad boosted trees"), ("gru", "\\quad BiGRU")):
        if fam in SP:
            rob.append((lab, SP[fam]["gain"], "strong"))
    CV = json.load(open(BASE / "results_panel_checks2.json"))["conserv"]
    for fam, lab in (("lr", "Stronger baseline $+$ eyes, LR"), ("hgb", "\\quad boosted trees"), ("gru", "\\quad BiGRU")):
        rob.append((lab, CV[fam]["gain"], "strong"))
    # learned representations of each epoch's 90 s parcel series (53_learned_encoder.py, 54_ssl_pretrain.py), the
    # paper's readout on both; the encoder row compares two networks of the same design (as the BiGRU rows do).
    # The embedding row is the strict run (one encoder per fold, pretrained without the held-out subject); the
    # transductive run (one encoder on all runs, held-out subject's unlabeled windows included) is kept as sslTrans*
    LE = json.load(open(BASE / "results_learned_encoder.json"))
    LX = json.load(open(BASE / "results_learned_encoder_extra.json"))
    SSL = json.load(open(BASE / "results_ssl_probe_strict.json"))
    SSLT = json.load(open(BASE / "results_ssl_probe.json"))
    assert SSL["meta"].get("strict") and SSLT["meta"]["exclude"] is None
    g_le, g_ssl = LE["gain_nb+enc_over_nb"], SSL["gain"]["nb+emb - nb"]["per_stage"]
    g_sslT = SSLT["gain"]["nb+emb - nb"]["per_stage"]
    rob += [("CNN encoder", g_le, "learned"),
            ("Self-supervised embedding", g_ssl, "learned")]
    for tag, r in (("le", g_le), ("ssl", g_ssl), ("sslTrans", g_sslT)):
        for st in ("N3", "REM"):
            M[f"{tag}{SN[st]}"], M[f"{tag}{SN[st]}ci"] = sg(r[st]["ws_mean"]), ci(r[st]["ws_ci"])
            M[f"{tag}{SN[st]}up"], M[f"{tag}{SN[st]}upT"] = sg(r[st]["ws_upper95"]), sg(tbound(r[st]))
            M[f"{tag}{SN[st]}Subj"], M[f"{tag}{SN[st]}SubjN"] = str(r[st]["n_pos"]), str(r[st]["n_subj"])
    _r = g_le["REM"]
    assert _r["ws_lower95"] > 0 and _r["t_lower95"] > 0 and _r["ws_mean"] >= ma["delta"], "text: the encoder's REM gain meets the logged gain bar"
    for st in STAGES:
        M[f"leAuc{SN[st]}"], M[f"sslAuc{SN[st]}"] = f3(LE["auc_ws"]["enc"][st]), f3(SSL["auc_ws"]["emb"][st])
        M[f"sslTransAuc{SN[st]}"] = f3(SSLT["auc_ws"]["emb"][st])
    lx_lr, lx_late = LX["gain"]["nb+enc_nn - nb_lr"]["per_stage"], LX["gain"]["late(nb_lr,enc) - nb_lr"]["per_stage"]
    for tag, r in (("leLr", lx_lr), ("leLate", lx_late)):
        for st in ("N3", "REM"):
            M[f"{tag}{SN[st]}"], M[f"{tag}{SN[st]}ci"] = sg(r[st]["ws_mean"]), ci(r[st]["ws_ci"])
    for st in ("N3", "REM"):
        M[f"leStrat{SN[st]}"] = sg(LX["stratified_1h"]["nb+enc_nn - nb_nn"][st]["mean"])
        M[f"leLateStrat{SN[st]}"] = sg(LX["stratified_1h"]["late(nb_lr,enc) - nb_lr"][st]["mean"])
    # abstract: "up to leLateStratREM for a CNN encoder within 1 h bins"; Discussion: "at most about five hundredths across
    # the night and seven within 1 h bins" (the largest REM gains the paper reports, across the night and within bins)
    rem_within = [LX["stratified_1h"]["nb+enc_nn - nb_nn"]["REM"]["mean"], LX["stratified_1h"]["late(nb_lr,enc) - nb_lr"]["REM"]["mean"]]
    assert all(LX["stratified_1h"][k]["REM"]["mean"] < 0.5 * LX["stratified_1h"][k]["N3"]["mean"]
               for k in ("nb+enc_nn - nb_nn", "late(nb_lr,enc) - nb_lr")), "text: within bins the encoder's REM gain is under half its N3 gain"
    assert max(rem_within) == LX["stratified_1h"]["late(nb_lr,enc) - nb_lr"]["REM"]["mean"] and round(max(rem_within), 2) == 0.07
    assert round(max(LX["gain"]["late(nb_lr,enc) - nb_lr"]["per_stage"]["REM"]["ws_mean"], g_le["REM"]["ws_mean"]), 2) == 0.05
    # text: the encoder's late-fusion gain within 1 h bins with its CI; the encoder closes hrEncREM% of REM's remaining 1-AUC
    M["leLateStratREMci"] = ci(LX["stratified_1h"]["late(nb_lr,enc) - nb_lr"]["REM"]["ci"])
    M["hrEncREM"] = f"{100 * g_le['REM']['ws_mean'] / (1 - A['auc_ws']['nonbold']['REM']['mean']):.0f}"
    assert 20 < float(M["hrEncREM"]) < 50
    # text: the stronger baselines' REM gains stay at most spHgbREM, all six inside the margin
    _strong_rem = [r["REM"]["ws_mean"] for _, r, g in rob if g == "strong"]
    assert len(_strong_rem) == 6 and abs(max(_strong_rem) - float(M["spHgbREM"].replace("$-$", "-"))) < 6e-4
    # text: "they add lrnRemMin to lrnRemMax to REM and lrnNthreeMin to lrnNthreeMax to N3"
    lrn = [g_le, g_ssl]                       # the two specification rows; leLr*/leLate* are reported separately
    M["lrnRemMin"], M["lrnRemMax"] = sg(min(r["REM"]["ws_mean"] for r in lrn)), sg(max(r["REM"]["ws_mean"] for r in lrn))
    M["lrnNthreeMin"], M["lrnNthreeMax"] = sg(min(r["N3"]["ws_mean"] for r in lrn)), sg(max(r["N3"]["ws_mean"] for r in lrn))
    assert all(r["N3"]["ws_mean"] > 3 * max(r["REM"]["ws_mean"], 0.0) for r in lrn), "text: N3 gain at least three times the REM gain for every learned representation"
    # text: the CNN encoder detects REM from BOLD alone far better than the epoch features, the strict embedding
    # slightly; pretraining on the held-out subject's own runs (transductive) lifts the embedding's REM AUC
    b_rem = A["auc_ws"]["bold"]["REM"]["mean"]
    assert LE["auc_ws"]["enc"]["REM"] > b_rem + 0.08, "text: the encoder detects REM from BOLD alone far better than epoch features"
    assert b_rem < SSL["auc_ws"]["emb"]["REM"] < b_rem + 0.05, "text: the strict embedding detects REM slightly better than epoch features"
    assert SSLT["auc_ws"]["emb"]["REM"] > SSL["auc_ws"]["emb"]["REM"] + 0.05, "text: the transductive embedding detects REM far better than the strict one"
    assert g_ssl["REM"]["ws_upper95"] < ma["delta"] and tbound(g_ssl["REM"]) < ma["delta"], "table: the strict embedding's REM gain lies inside the margin"
    assert g_ssl["N3"]["n_pos"] == g_ssl["N3"]["n_subj"], "table: the strict embedding's N3 gain is positive in every subject"
    k2_ = lambda x: f"{x:.2f}"
    # leakage checks on the encoder: the censoring pattern alone (mask_only) and without the 14 orbitofrontal parcels
    LM = json.load(open(BASE / "results_learned_encoder_mask_only.json")); LO = json.load(open(BASE / "results_learned_encoder_no_orbitofrontal.json"))
    M["leMaskAucREM"], M["leNoOfcAucREM"] = f3(LM["auc_ws"]["enc"]["REM"]), f3(LO["auc_ws"]["enc"]["REM"])
    M["leNoOfcREM"], M["nOfcDropped"] = sg(LO["gain_nb+enc_over_nb"]["REM"]["ws_mean"]), str(len(LO["meta"]["dropped_parcels"]))
    assert LM["auc_ws"]["enc"]["REM"] < 0.6 and LM["gain_nb+enc_over_nb"]["REM"]["ws_mean"] < 0.015, "text: the censoring pattern alone carries no REM information"
    assert abs(LO["gain_nb+enc_over_nb"]["REM"]["ws_mean"] - g_le["REM"]["ws_mean"]) < 0.01 and abs(LO["auc_ws"]["enc"]["REM"] - LE["auc_ws"]["enc"]["REM"]) < 0.01, \
        "text: the encoder's REM information does not depend on the orbitofrontal parcels"
    assert float(M["hrREM"].replace("$-$", "-")) <= 0, "text: BOLD closes none of REM's remaining 1-AUC"
    M["kkEnc"], M["kkEmb"], M["kkEmbTrans"] = k2_(LE["hmm"]["enc"]["kappa"]), k2_(SSL["hmm"]["emb"]["kappa"]), k2_(SSLT["hmm"]["emb"]["kappa"])
    M["kkNbEnc"], M["kkNbEmb"] = k2_(LE["hmm"]["nb+enc"]["kappa"]), k2_(SSL["hmm"]["nb+emb"]["kappa"])
    sw = np.load(BASE / "ssl_windows.npz", allow_pickle=True)
    M["nSslWindows"], M["nSslRuns"] = f"{len(sw['subj']):,}", str(len(np.unique(sw["run"])))
    M["nSslHours"] = f"{len(sw['subj']) * 5 * 3 / 3600:.0f}"
    # text: richer estimators (incl. the strict self-supervised embedding) reach at most kkRichMax; tempering does not
    # remove the drop. Only the transductive embedding (kkEmbTrans, not in the paper) exceeds kkRichMax.
    SU = json.load(open(BASE / "results_stager_upgrades.json")); TH = json.load(open(BASE / "results_tempered_hmm.json"))
    rich = {k: v["hmm"]["1.0"]["kappa"] for k, v in SU.items() if k != "meta" and "hmm" in v and k not in ("strong:lr_nb", "strong:gru_nb")}
    rich["temporal CNN encoder"] = LE["hmm"]["enc"]["kappa"]
    rich["self-supervised embedding"] = SSL["hmm"]["emb"]["kappa"]
    M["kkRichMax"], M["nKappaVariants"] = k2_(max(rich.values())), str(len(rich))
    assert max(rich.values()) < 0.5 and SSLT["hmm"]["emb"]["kappa"] > max(rich.values())
    td = TH["paired"]["nonbold+bold vs bold"]
    M["kdTemperedAllBold"], M["kdTemperedAllBoldci"] = sg(td["diff"]), ci(td["ci"])
    assert td["ci"][1] < 0, "text: tempering the emissions does not remove the kappa drop"
    # cross-site follow-up (55_cross_site.py): the Penn State naps (ds003768) through the same chain as the rv3 rung;
    # strict per-fold encoders pretrained on both sites without the held-out NIH subject; the W/N1/N2 readout is fitted
    # on the naps' labels alone and applied to that subject (csPtoN*), against the within-site LOSO readout (csNihWithin*)
    CS = json.load(open(BASE / "results_cross_site_ssl_strict_joint.json"))
    CH = json.load(open(BASE / "results_cross_site_hand.json"))
    CE = {t: json.load(open(BASE / f"results_cross_site_ssl_{t}.json")) for t in ("nih_rv3", "joint_rv3")}
    assert CS["meta"]["strict"] and CS["meta"]["n_nih_subjects"] == int(M["nSubj"])
    M["csPsuSubj"] = str(CS["meta"]["psu"]["n_subjects"])
    M["csPsuRuns"], M["csPsuEpochs"] = str(CH["meta"]["psu"]["n_runs"]), f"{CH['meta']['psu']['n_epochs']:,}"
    for k, tag in (("psu_to_nih", "csPtoN"), ("nih_within", "csNihWithin")):
        for st in ("W", "N1", "N2"):
            M[f"{tag}Auc{SN[st]}"] = f3(CS[k]["auc_ws"][st])
        M[f"{tag}Kappa"] = k2_(CS[k]["kappa"])
        M[f"{tag}Kappaci"] = f"[{CS[k]['kappa_ci'][0]:.2f}, {CS[k]['kappa_ci'][1]:.2f}]"
    pw = [v["psu_within"][m]["kappa"] for v in CH["hand"].values() for m in ("argmax", "hmm")] + \
         [E["embedding"]["psu_within"][m]["kappa"] for E in CE.values() for m in ("argmax", "hmm")]
    n2p = [v["nih_to_psu"][m]["kappa"] for v in CH["hand"].values() for m in ("argmax", "hmm_nih_transitions")] + \
          [E["embedding"]["nih_to_psu"][m]["kappa"] for E in CE.values() for m in ("argmax", "hmm_nih_transitions")]
    M["csPsuWithinKappaMax"], M["csNtoPKappaMax"] = k2_(max(pw)), k2_(max(n2p))
    assert max(pw) < 0.2 and max(n2p) < 0.15, "text: the naps stage poorly from BOLD, locally trained or transferred"
    assert CS["psu_to_nih"]["kappa"] > 0.25 and all(CS["psu_to_nih"]["auc_ws"][st] > CS["nih_within"]["auc_ws"][st] - 0.05 for st in ("W", "N2")), \
        "text: the nap-label readout comes close to the within-site W and N2 AUC"
    # the four baseline comparisons (physiology, time, both, second half) are given in the text, not the table
    rob += [(f"Minimal pipeline, {mw['n_subjects']} subjects", inc_w["nonbold+bold - nonbold"], "own"),
            ("\\quad boosted trees", W["hgb"]["increment"], "own")]
    # abstract: "OWNMIN to OWNMAX in our own pipelines": every own-pipeline N3 gain (Fig. 1c levels incl. motion
    # regression and FC, the 16-subject minimal pipeline with LR and trees), each above its REM gain
    own_all = [MT["increment"][k] for k in ("raw84amp", "rv84amp", "rv84phys", "rv84mot", "raw")] + \
              [inc_w["nonbold+bold - nonbold"], W["hgb"]["increment"]] + \
              [LD["increment"][k] for k in LD_TAGS if k.startswith("amp_")]          # the ladder's own rungs (49)
    assert all(r["N3"]["ws_mean"] > r["REM"]["ws_mean"] for r in own_all)
    M["ownNthreeMin"] = sg(min(r["N3"]["ws_mean"] for r in own_all))
    M["ownNthreeMax"] = sg(max(r["N3"]["ws_mean"] for r in own_all))
    # (c) the N3 gain over the subjects eligible for both N3 and REM, on which the N3-REM contrast is formed
    _h = inc_a["nonbold+bold - nonbold"]
    _both = [u for u in _h["N3"]["ws_values"] if u in _h["REM"]["ws_values"]]
    M["ctrNthreeBoth"] = sg(np.mean([_h["N3"]["ws_values"][u] for u in _both]))
    M["ctrREMBoth"] = sg(np.mean([_h["REM"]["ws_values"][u] for u in _both]))
    assert abs(np.mean([_h["N3"]["ws_values"][u] - _h["REM"]["ws_values"][u] for u in _both]) - float(M["inter"].replace("$-$", "-").replace("\\xspace", ""))) < 6e-4
    # text: "REM staying inside the margin in all six" stronger-baseline rows
    assert all(r["REM"]["ws_upper95"] < ma["delta"] and tbound(r["REM"]) < ma["delta"] for _, r, g in rob if g == "strong")

    def ctr(inc):
        if inc.get("N3") is None:
            return None
        a, b = inc["N3"]["ws_values"], inc["REM"]["ws_values"]
        both = [u for u in a if u in b]
        return sum(a[u] > b[u] for u in both), len(both)

    cort = [r for lab, r, g in rob[:4]] + [r for _, r in variants]
    rv = [r["REM"]["ws_mean"] for r in cort]
    M["remVarMin"], M["remVarMax"] = sg(min(rv)), sg(max(rv))
    # the specification envelope: every derivative comparison of BOLD over a non-BOLD model with physiology and
    # time (the range row expanded into its variants)
    spec = []
    for lab, r, g in rob:
        if g in ("spec", "strong", "learned"):
            spec += [(v_lab, v_r) for v_lab, v_r in variants] if r == "RANGE" else [(lab, r)]
    M["nSpec"] = str(len(spec))
    rem_v = [r["REM"]["ws_mean"] for _, r in spec]; n3_v = [r["N3"]["ws_mean"] for _, r in spec]
    M["envRemMin"], M["envRemMax"] = sg(min(rem_v)), sg(max(rem_v))
    M["envNthreeMin"], M["envNthreeMax"] = sg(min(n3_v)), sg(max(n3_v))
    M["envRemMaxUpT"] = sg(max(tbound(r["REM"]) for _, r in spec))
    M["envNthreeMinLoT"] = sg(min(tbound(r["N3"], upper=False) for _, r in spec))
    cts = [ctr(r) for _, r in spec]
    n_full = sum(c[0] == c[1] for c in cts)
    M["nSpecCtrAll"] = str(n_full)
    M["specCtrMin"], M["specCtrMinN"] = str(min(c[0] for c in cts)), str(min(cts, key=lambda c: c[0])[1])
    M["nSpecRemCiPos"] = str(sum(r["REM"]["ws_ci"][0] > 0 for _, r in spec))
    # text: "In N of them, all with stacking, trees or the sequence model, the REM gain's CI lies above zero"
    flex = [lab for lab, r in spec if r["REM"]["ws_ci"][0] > 0]
    assert all(any(k in lab for k in ("stacked", "trees", "encoder", "embedding")) for lab in flex), flex
    # abstract: "below the N3 gain in all N specifications"
    assert all(r["N3"]["ws_mean"] > r["REM"]["ws_mean"] for _, r in spec)
    M["nLvAll"] = str(BLK["max"]["n_features"] - BLK["subc"]["n_features"] - BLK["mw"]["n_features"] - 10)
    assert ma["n_bold_features"] + 10 + BLK["subc"]["n_features"] + BLK["mw"]["n_features"] + int(M["nLvAll"]) == int(M["nAllBoldFeat"])
    M["nSpecNthreeCiPos"] = str(sum(r["N3"]["ws_ci"][0] > 0 for _, r in spec))
    M["nSpecRemDagger"] = str(sum(r["REM"]["ws_upper95"] < ma["delta"] and tbound(r["REM"]) < ma["delta"] for _, r in spec))
    # text: "the N exceptions use late fusion, boosted trees, nested tuning or stacking with added blocks"
    out_ = [lab for lab, r in spec if not (r["REM"]["ws_upper95"] < ma["delta"] and tbound(r["REM"]) < ma["delta"])]
    M["nSpecRemOut"] = str(len(out_))
    assert len(out_) + int(M["nSpecRemDagger"]) == len(spec)
    assert all(any(k in lab for k in ("late fusion", "Late fusion", "trees", "nested-tuned", "encoder", "embedding")) or lab == "\\quad stacked"
               for lab in out_), out_
    M["nSpecRemOutLearned"] = str(sum(any(k in lab for k in ("encoder", "embedding")) for lab in out_))
    _inside = [r["REM"]["ws_upper95"] < ma["delta"] and tbound(r["REM"]) < ma["delta"] for _, r in spec]   # by row, labels repeat
    M["nSpecRemCiPosOut"] = str(sum(r["REM"]["ws_ci"][0] > 0 for (lab, r), ins in zip(spec, _inside) if not ins))
    M["nSpecRemCiPosIn"] = str(sum(r["REM"]["ws_ci"][0] > 0 for (lab, r), ins in zip(spec, _inside) if ins))
    assert all(r["REM"]["ws_mean"] < 0.5 * r["N3"]["ws_mean"] for _, r in spec), "text: every REM gain is under half the N3 gain of the same specification"
    _pin = [(lab, r) for (lab, r), ins in zip(spec, _inside) if ins and r["REM"]["ws_ci"][0] > 0]
    assert len(_pin) == 1 and "trees" in _pin[0][0] and abs(_pin[0][1]["REM"]["ws_mean"] - float(M["spHgbREM"])) < 6e-4, \
        f"text: the one REM CI above zero inside the margin is the stronger baseline with boosted trees: {[l for l, _ in _pin]}"
    assert int(M["nSpecRemCiPosOut"]) + int(M["nSpecRemCiPosIn"]) == int(M["nSpecRemCiPos"])
    print("  specification rows below every-subject N3>REM:", [(lab, c) for (lab, r), c in zip(spec, cts) if c[0] < c[1]])
    lows_ = [inc["N3"]["ws_ci"][0] for _, inc, _g in rob if not isinstance(inc, str) and inc.get("N3") is not None
             and inc["N3"]["ws_ci"][0] > 0] + [r["N3"]["ws_ci"][0] for _, r in variants]
    M["tabNthreeLoMin"], M["tabNthreeLoMax"] = f"{min(lows_):.2f}", f"{max(lows_):.2f}"
    n3neg_ = [lab for lab, inc, _g in rob if not isinstance(inc, str) and inc.get("N3") is not None and inc["N3"]["ws_ci"][0] <= 0]
    assert len(n3neg_) == 1 and "inimal" in n3neg_[0] and all(r["N3"]["ws_ci"][0] > 0 for _, r in variants), \
        f"Table 2 caption: only the minimal pipeline's LR row has an N3 CI reaching zero: {n3neg_}"
    # ---- display plan (draft 25): every row whose REM CI lies above zero is printed individually; the other rows of a
    # group collapse into one range row (min to max over its n rows; k/n inside the margin). The specification
    # envelope above is computed from rob itself, so nSpec, nSpecRemDagger etc. do not depend on this plan.
    inside_ = lambda r: r["REM"]["ws_upper95"] < ma["delta"] and tbound(r["REM"]) < ma["delta"]
    pos_ = lambda r: r["REM"]["ws_ci"][0] > 0
    f3s = lambda v: f"{v:.3f}".replace("-", "$-$")
    f2s = lambda v: f"{v:.2f}".replace("-", "$-$")
    i_range = next(i for i, (lab, inc, g) in enumerate(rob) if isinstance(inc, str) and inc == "RANGE")
    i_strong0 = next(i for i, (lab, inc, g) in enumerate(rob) if g == "strong")
    i_learned0 = next(i for i, (lab, inc, g) in enumerate(rob) if g == "learned")
    i_own0 = next(i for i, (lab, inc, g) in enumerate(rob) if g == "own")
    idx_est, idx_blocks = list(range(1, i_range)), list(range(i_range + 1, i_strong0))
    idx_strong = list(range(i_strong0, i_learned0))
    assert rob[0][0].startswith("Main") and len(idx_est) == 5 and len(idx_blocks) == 12 and len(idx_strong) == 6, (len(idx_est), len(idx_blocks), len(idx_strong))

    def disp_label(i):
        lab = rob[i][0]
        if not lab.startswith("\\quad"):
            return lab
        j = i
        while rob[j][0].startswith("\\quad"):
            j -= 1
        parent = rob[j][0]
        short = ("subcortex" if "subcortex" in parent else "all blocks" if "All BOLD" in parent else
                 "stronger" if parent.startswith("Stronger non") else "stronger $+$ eyes" if "eye" in parent else parent)
        tail = lab.replace(chr(92) + "quad ", "").replace("boosted trees", "trees").replace("trees, nested-tuned", "tuned trees").replace("LR, nested-tuned", "tuned LR")
        return f"\\quad {short}, {tail}"

    def range_row(label, incs):
        n3 = [r["N3"]["ws_mean"] for r in incs]; rm = [r["REM"]["ws_mean"] for r in incs]
        n3txt = f"{f2s(min(n3))}--{f2s(max(n3))}"
        remtxt = f"{f3s(min(rm))} to {f3s(max(rm))} ({sum(inside_(r) for r in incs)}/{len(incs)}$^\\dagger$)"
        cs = [ctr(r) for r in incs]
        ctxt = "all" if all(c[0] == c[1] for c in cs) else f"{min(c[0] for c in cs)}/{cs[0][1]}"
        return f"{label} & {n3txt} & {remtxt} & {ctxt} \\\\"

    def one_row(label, inc):
        c = ctr(inc)
        x = f"{inc['N3']['ws_mean']:.2f}".replace("-", "$-$")
        n3txt = x
        return f"{label} & {n3txt} & {cell(inc.get('REM'), dagger=True)} & {'--' if c is None else f'{c[0]}/{c[1]}'} \\\\"

    def group_rows(label, idx):
        others = [rob[i][1] for i in idx if not pos_(rob[i][1])]
        rows_ = [range_row(f"{label} ({len(others)})", others)] if others else []
        rows_ += [one_row(disp_label(i), rob[i][1]) for i in idx if pos_(rob[i][1])]
        return rows_

    def short_name(i):
        return disp_label(i).replace("\\quad ", "").replace("subcortex, ", "subcortex ").replace("all blocks, ", "all-blocks ").replace("stronger, ", "stronger ").replace("$C$", "C").replace("LR, nested-tuned C", "nested-tuned LR").replace("Boosted trees, nested-tuned", "nested-tuned trees").replace("Boosted", "boosted").replace("Late", "late").replace("Stacked fusion (nested)", "stacking")
    notes_out, notes_ctr = [], []
    for gname, idx in (("other estimators", idx_est), ("added blocks", idx_blocks), ("stronger", idx_strong[:3]), ("stronger $+$ eyes", idx_strong[3:])):
        others = [i for i in idx if not pos_(rob[i][1])]
        o = [short_name(i) for i in others if not inside_(rob[i][1])]
        notes_ctr += [f"{short_name(i)} ({ctr(rob[i][1])[0]}/{ctr(rob[i][1])[1]})" for i in others if ctr(rob[i][1])[0] < ctr(rob[i][1])[1]]
        if o and len(o) == len(others) - 1 and gname == "other estimators":
            o = ["all but " + [short_name(i) for i in others if inside_(rob[i][1])][0]]
        if o:
            notes_out.append(f"{gname}, {' and '.join([', '.join(o[:-1]), o[-1]] if len(o) > 1 else o)}")
    M["tabRangeOut"], M["tabRangeCtr"] = "; ".join(notes_out), ", ".join(notes_ctr)
    out = ["\\begin{table}[t]", "\\centering", "\\fontsize{7.5}{9}\\selectfont", "\\setlength{\\tabcolsep}{1.5pt}",
           f"\\caption{{On the derivative, BOLD's N3 gain is at least {min(r['N3']['ws_mean'] for _, r in spec):.2f} with CI above zero in every specification, its REM gain negligible in {M['nSpecRemDagger']}. "
           "Gain in within-subject AUC from adding BOLD to non-BOLD, mean [95\\% CI]; the "
           f"{len(spec)} specifications are its rows, the variant row holding {len(variants)}, the nested-tuning row 2 and the stronger $+$ eyes row 3; numbers in parentheses count features (after $^\\ddagger$, variants). Every $\\Delta_\\mathrm{{N3}}$ CI lies above zero (lower bounds $\\ge${{TABLO}}). Boldface: REM bootstrap CI above zero. $^\\dagger$: negligible, both one-sided upper bounds below "
           f"$\\delta={ma['delta']:.2f}$; with boldface, a positive gain below $\\delta$. Indented rows: the block or baseline above with another "
           "estimator. N3$>$REM: subjects whose N3 gain exceeds their REM gain. $^\\ddagger$Global signal (2), WM/CSF or slow "
           "physiology regressed (2), detrending, 6\\,s label shift, within-run z, night time (2), cross-night training, "
           "epoch or run selection (5), lagged physiology, image motion, no somatomotor network, $C$ (3).}",
           "\\label{tab:robust}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
           "Specification & $\\Delta_\\mathrm{N3}$ & $\\Delta_\\mathrm{REM}$ [95\\% CI] & \\begin{tabular}{@{}c@{}}N3$>$\\\\REM\\end{tabular} \\\\", "\\midrule",
           one_row("Planned (LR, early fusion)", rob[0][1]), "\\midrule"]
    est_labels = {"Late fusion": "Late fusion", "Stacked fusion (nested)": "Stacking", "Boosted trees": "Boosted trees",
                  "LR, nested-tuned $C$": "LR, tuned $C$", "Boosted trees, nested-tuned": "Boosted trees, tuned"}
    # draft 35: the two nested-tuning rows share one range row (neither has a REM CI above zero; both N3>REM in all)
    tune_ = [i for i in idx_est if "nested-tuned" in rob[i][0]]
    assert len(tune_) == 2 and not any(pos_(rob[i][1]) for i in tune_) and all(ctr(rob[i][1])[0] == ctr(rob[i][1])[1] for i in tune_)
    import os
    COMPACT = os.environ.get("T2COMPACT") == "1"      # draft 38 A/B: compact Table 2 (rows with a REM CI above zero stay)
    if COMPACT:
        assert not any(pos_(rob[i][1]) for i in idx_est)
        out += [range_row(f"{len(idx_est)} other estimators", [rob[i][1] for i in idx_est])]
    else:
        out += [one_row(est_labels.get(rob[i][0], rob[i][0]), rob[i][1]) for i in idx_est if i not in tune_]
        out += [range_row("Nested tuning, LR \\& trees", [rob[i][1] for i in tune_])]
    assert all(inside_(r) for _, r in variants) and all(ctr(r)[0] == ctr(r)[1] for _, r in variants) and all(r["N3"]["ws_ci"][0] > 0 for _, r in variants), \
        "variant row: every variant daggered, N3 CI > 0, N3>REM in all subjects"
    n3 = [r["N3"]["ws_mean"] for _, r in variants]; rm = [r["REM"]["ws_mean"] for _, r in variants]
    out.append(f"{len(variants)} variants$^\\ddagger$ & {f2s(min(n3))}--{f2s(max(n3))} & "
               f"{f3s(min(rm))} to {f3s(max(rm))}, all$^\\dagger$ & all \\\\")
    M["varNthreeMin"], M["varNthreeMax"], M["varREMmin"], M["varREMmax"] = sg(min(n3)), sg(max(n3)), sg(min(rm)), sg(max(rm))
    blk_labels = {"\\quad LR, nested-tuned": "\\quad LR, tuned $C$", "\\quad trees, nested-tuned": "\\quad boosted trees, tuned"}
    def flat_label(i):
        lab = rob[i][0]
        if not lab.startswith("\\quad"):
            return lab.replace(", early", ", early fusion").replace("$+$ ", "")
        j = i
        while rob[j][0].startswith("\\quad"):
            j -= 1
        parent = rob[j][0].split(",")[0].replace("$+$ ", "")
        parent = (parent[0].upper() + parent[1:]).replace("All BOLD blocks", "All blocks")
        return parent + ", " + blk_labels.get(lab, lab).replace("\\quad ", "").replace("boosted trees, tuned", "tuned trees")
    if COMPACT:
        bpos = [i for i in idx_blocks if pos_(rob[i][1])]
        out += ["\\midrule"] + [one_row(flat_label(i), rob[i][1]) for i in bpos]
        out += [range_row(f"{len(idx_blocks) - len(bpos)} other added-block rows", [rob[i][1] for i in idx_blocks if i not in bpos])]
    else:
        out += ["\\midrule"] + [one_row(blk_labels.get(rob[i][0], rob[i][0]), rob[i][1]) for i in idx_blocks]
    # draft 33: the stronger-baseline groups collapse like the others (rows with a REM CI above zero stay individual);
    # the minimal pipeline's rows leave the table (outside the specifications; the text gives its trees REM gain)
    assert not [i for i in idx_strong[3:] if pos_(rob[i][1])]
    assert all(ctr(rob[i][1])[0] == ctr(rob[i][1])[1] for i in idx_strong[3:]), "the collapsed stronger + eyes row: N3>REM in all subjects"
    if COMPACT:
        spos = [i for i in idx_strong[:3] if pos_(rob[i][1])]
        out += ["\\midrule"] + [one_row("Stronger baseline, boosted trees", rob[i][1]) for i in spos]
        out += [range_row("Stronger baseline, LR \\& BiGRU", [rob[i][1] for i in idx_strong[:3] if i not in spos])]
    else:
        out += ["\\midrule"] + [one_row(rob[i][0], rob[i][1]) for i in idx_strong[:3]]
    out += [range_row("Stronger $+$ eyes, all three", [rob[i][1] for i in idx_strong[3:]])]
    out += ["\\midrule"] + [one_row(rob[i][0], rob[i][1]) for i in range(i_learned0, i_own0)]
    M["tabNthreeLoMin"] = f"{min([rob[i][1]['N3']['ws_ci'][0] for i in range(i_own0) if not isinstance(rob[i][1], str)] + [r['N3']['ws_ci'][0] for _, r in variants]):.2f}"
    assert float(M["tabNthreeLoMin"]) > 0, "Table 2 caption: every derivative N3 CI lies above zero"
    out = [l.replace("{TABLO}", M["tabNthreeLoMin"]) for l in out]
    if COMPACT:
        out = [l.replace(f"the variant row holding {len(variants)}, the nested-tuning row 2 and the stronger $+$ eyes row 3",
                         "range rows hold their leading number of specifications (stronger $+$ eyes: 3), with $k/n^\\dagger$ of them negligible and the lowest N3$>$REM count")
                .replace("Indented rows: the block or baseline above with another estimator. ", "") for l in out]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    M["nTabRows"] = str(sum(1 for l in out if l.endswith("\\\\") and "&" in l) - 1)
    (PAPER / "tab_robust.tex").write_text("\n".join(out) + "\n")
    # abstract: "envNthreeMin to spHgbNthree against stronger cardiorespiratory and eye baselines"
    num = lambda k: float(M[k].replace("$-$", "-"))
    strong = [num(k) for k in ("spLrNthree", "spHgbNthree", "spGruNthree", "cvLrNthree", "cvHgbNthree", "cvGruNthree")]
    assert min(strong) == num("envNthreeMin") and max(strong) == num("spHgbNthree"), strong
    table3(M, ma)
    with open(PAPER / "numbers.tex", "w") as f:
        f.write("% generated by 26_paper_numbers_v2.py -- do not edit\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print(f"wrote {len(M)} macros, Table 1 and Table 2")


if __name__ == "__main__":
    main()
