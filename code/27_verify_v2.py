"""Independent check of the v2 paper's headline numbers.

Re-derives, from the saved per-epoch probabilities and HMM paths (predictions_v2_<source>.npz) and
with sklearn's own metrics rather than 24_experiments_v2's helpers, the numbers the abstract and the
results rest on, and compares them with the macros in paper_isbi/numbers.tex and the cells of
tab_ladder.tex. Also lists every decimal number typed directly into main.tex, which should be none.

    source <venv>/bin/activate
    python 27_verify_v2.py
"""

import re
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

BASE = Path(__file__).parent
PAPER = BASE / "paper_isbi"
STAGES = ["W", "N1", "N2", "N3", "REM"]
SN = {"W": "W", "N1": "None", "N2": "Ntwo", "N3": "Nthree", "REM": "REM"}
MIN_EP = 20
FAIL = []


def macros():
    M = {}
    for line in (PAPER / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$", line)
        if m:
            M[m.group(1)] = m.group(2)
    return M


def num(s):
    return float(s.replace("$-$", "-").replace(",", ""))


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        FAIL.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:28s} paper {want:+.4f}  recomputed {got:+.4f}")


def ws(y, p, s, st, i):
    out = {}
    for u in np.unique(s):
        m = s == u
        pos = y[m] == st
        if pos.sum() >= MIN_EP and (~pos).sum() >= MIN_EP:
            out[u] = roc_auc_score(pos, p[m, i])
    return out


def main():
    M = macros()
    for src, suf in [("afni", ""), ("raw", "Raw")]:
        print(f"== {src}")
        Z = np.load(BASE / f"predictions_v2_{src}.npz", allow_pickle=True)
        y, s = Z["y"], Z["subj"]
        if src == "afni":
            check("nEpochs", len(y), num(M["nEpochs"]), 0)
            check("nSubj", len(np.unique(s)), num(M["nSubj"]), 0)
            for st in STAGES:
                check(f"n{SN[st]}", (y == st).sum(), num(M["n" + SN[st]]), 0)
        for st_i, st in enumerate(STAGES):
            a = ws(y, Z["proba__nonbold+bold"], s, st, st_i)
            b = ws(y, Z["proba__nonbold"], s, st, st_i)
            d = np.array([a[k] - b[k] for k in a])
            check(f"incNb{SN[st]}{suf}", d.mean(), num(M[f"incNb{SN[st]}{suf}"]), 6e-4)
            check(f"incNb{SN[st]}{suf}Subj", (d > 0).sum(), num(M[f"incNb{SN[st]}{suf}Subj"]), 0)
            check(f"incNb{SN[st]}{suf}SubjN", len(d), num(M[f"incNb{SN[st]}{suf}SubjN"]), 0)
            for key, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("clock", "Time")]:
                v = np.mean(list(ws(y, Z[f"proba__{key}"], s, st, st_i).values()))
                check(f"auc{tag}{SN[st]}{suf}", v, num(M[f"auc{tag}{SN[st]}{suf}"]), 6e-4)
        for key, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All")]:
            check(f"kappa{tag}{suf}", cohen_kappa_score(y, Z[f"hmm__{key}"]), num(M[f"kappa{tag}{suf}"]), 6e-4)
        if src == "afni":
            k = cohen_kappa_score(y, Z["hmm__bold"]) - cohen_kappa_score(y, Z["hmm__netFC-LR (after Tagliazucchi 2012)"])
            check("kdBoldTag", k, num(M["kdBoldTag"]), 6e-4)
            k = cohen_kappa_score(y, Z["hmm__bold"]) - cohen_kappa_score(y, Z["hmm__nonbold"])
            check("kdBoldNb", k, num(M["kdBoldNb"]), 6e-4)
            # Table 1 kappa cells
            tab = (PAPER / "tab_ladder.tex").read_text()
            for label, key in [("BOLD (", "bold"), ("Non-BOLD (", "nonbold"), ("Head motion", "motion"),
                               ("Physiology (", "physio"), ("Elapsed time", "clock"),
                               ("7-network FC", "netFC-LR (after Tagliazucchi 2012)"), ("Non-BOLD + BOLD", "nonbold+bold")]:
                row = next(l for l in tab.splitlines() if l.startswith(label))
                cellk = float(re.search(r"(\d\.\d{2})", row).group(1))      # two decimals in draft 10
                check(f"Table1 {key[:20]}", cohen_kappa_score(y, Z[f"hmm__{key}"]), cellk, 5.1e-3)
            row = next(l for l in tab.splitlines() if l.startswith("Non-BOLD + BOLD"))
            gains = [num(g_) for g_ in re.findall(r"\(([^)]*)\)", row)]       # the headline gains in parentheses
            for st, g_ in zip(("N3", "REM"), gains):
                i_ = STAGES.index(st)
                a_, b_ = ws(y, Z["proba__nonbold+bold"], s, st, i_), ws(y, Z["proba__nonbold"], s, st, i_)
                check(f"Table1 gain {st}", np.mean([a_[k] - b_[k] for k in a_]), g_, 5.1e-3)

    print("== paired N3-REM contrast, staging checks (new in draft 2)")
    from scipy.stats import t as tdist
    from sklearn.metrics import f1_score
    for src, suf in [("afni", ""), ("raw", "Raw")]:
        Z = np.load(BASE / f"predictions_v2_{src}.npz", allow_pickle=True)
        y, s = Z["y"], Z["subj"]
        lf = Z["proba__nonbold"] * Z["proba__bold"]
        lf = lf / lf.sum(1, keepdims=True)
        for tag, pa, pb in [("", Z["proba__nonbold+bold"], Z["proba__nonbold"]), ("Lf", lf, Z["proba__nonbold"]),
                            ("Hgb", Z["proba__hgb:nonbold+bold"], Z["proba__hgb:nonbold"])]:
            g = {}
            for st in ("N3", "REM"):
                i = STAGES.index(st)
                a_, b_ = ws(y, pa, s, st, i), ws(y, pb, s, st, i)
                g[st] = {u: a_[u] - b_[u] for u in a_}
            both = sorted(set(g["N3"]) & set(g["REM"]))
            d = np.array([g["N3"][u] - g["REM"][u] for u in both])
            check(f"inter{tag}{suf}", d.mean(), num(M[f"inter{tag}{suf}"]), 6e-4)
            check(f"ctrAbove{tag}{suf}", (d > 0).sum(), num(M[f"ctrAbove{tag}{suf}"]), 0)
            check(f"ctrN{tag}{suf}", len(d), num(M[f"ctrN{tag}{suf}"]), 0)
            r = np.array(list(g["REM"].values()))
            check(f"remUpT{tag}{suf}", r.mean() + tdist.ppf(0.95, len(r) - 1) * r.std(ddof=1) / np.sqrt(len(r)),
                  num(M[f"remUpT{tag}{suf}"]), 6e-4)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, nf = Z["y"].astype(str), Z["subj"].astype(str), Z["night_frac"]
    hb, ha = Z["hmm__bold"].astype(str), Z["hmm__nonbold+bold"].astype(str)
    check("kdAllBold", cohen_kappa_score(y, ha) - cohen_kappa_score(y, hb), num(M["kdAllBold"]), 6e-4)
    worse = sum(cohen_kappa_score(y[s == u], ha[s == u]) < cohen_kappa_score(y[s == u], hb[s == u]) for u in np.unique(s))
    check("kdAllBoldWorse", worse, num(M["kdAllBoldWorse"]), 0)
    early = (nf < 0.5) & (y == "REM")
    for tag, h in [("Bold", hb), ("All", ha)]:
        check(f"recEarly{tag}", np.mean(h[early] == "REM"), num(M[f"recEarly{tag}"]), 6e-3)
        check(f"fREM{tag}", f1_score(y == "REM", h == "REM"), num(M[f"fREM{tag}"]), 6e-3)
        check(f"kArg{tag}", cohen_kappa_score(y, np.array(STAGES)[Z[f"proba__{'bold' if tag == 'Bold' else 'nonbold+bold'}"].argmax(1)]),
              num(M[f"kArg{tag}"]), 6e-4)
    tab = (PAPER / "tab_ladder.tex").read_text()
    for label, key in [("BOLD (", "bold"), ("Non-BOLD + BOLD", "nonbold+bold"), ("Non-BOLD (", "nonbold")]:
        row = next(l for l in tab.splitlines() if l.startswith(label))
        cell = float(row.split("&")[2])
        check(f"Table1 F1REM {key}", f1_score(y == "REM", Z[f"hmm__{key}"].astype(str) == "REM"), cell, 6e-3)

    print("== round-2 numbers: kappa without time, headroom, REM vs stage, clock check, stacking, envelope")
    import importlib.util
    spec = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
    e24 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e24)
    run, epoch = Z["run"].astype(str), Z["epoch"]
    h = e24.hmm_decode(Z["proba__motion+physio+bold"], y, s, run, epoch).astype(str)
    check("kappaNoTime", cohen_kappa_score(y, h), num(M["kappaNoTime"]), 6e-4)
    check("kdNoTime", cohen_kappa_score(y, h) - cohen_kappa_score(y, hb), num(M["kdNoTime"]), 6e-4)
    pa_, pb_ = Z["proba__nonbold+bold"], Z["proba__nonbold"]
    for st, key in [("N3", "hrNthree"), ("REM", "hrREM")]:
        i = STAGES.index(st)
        a_, b_ = ws(y, pa_, s, st, i), ws(y, pb_, s, st, i)
        check(key, 100 * np.mean([a_[u] - b_[u] for u in a_]) / np.mean([1 - b_[u] for u in a_]), num(M[key]), 0.5)  # ratio of means
    i = STAGES.index("REM")
    for other, key in [("W", "rvW"), ("N1", "rvNone"), ("N2", "rvNtwo"), ("N3", "rvNthree")]:
        g = []
        for u in np.unique(s):
            m = (s == u) & np.isin(y, ["REM", other])
            pos = y[m] == "REM"
            if pos.sum() >= MIN_EP and (~pos).sum() >= MIN_EP:
                g.append(roc_auc_score(pos, pa_[m, i]) - roc_auc_score(pos, pb_[m, i]))
        check(key, np.mean(g), num(M[key]), 6e-4)
    for k, key in [("bold", "clkBold"), ("nonbold", "clkNb")]:
        v = []
        for u in np.unique(s):
            m = (s == u) & (y == "N2")
            late = nf[m] >= 0.5
            if late.sum() >= MIN_EP and (~late).sum() >= MIN_EP:
                v.append(roc_auc_score(late, Z[f"proba__{k}"][m, i]))
        check(key, np.mean(v), num(M[key]), 6e-3)
    S2 = np.load(BASE / "predictions_stacked.npz", allow_pickle=True)
    assert (S2["y"] == Z["y"]).all()
    per = {}
    for st, key in [("N3", "incStNthree"), ("REM", "incStREM")]:
        j = STAGES.index(st)
        a_, b_ = ws(y, S2["stack_both"], s, st, j), ws(y, S2["stack_nonbold"], s, st, j)
        per[st] = np.array([a_[u] - b_[u] for u in a_])
        check(key, per[st].mean(), num(M[key]), 6e-4)
    # envelope: the printed REM ceiling must cover every estimator's t-bound, the N3 floor lie under every t-bound
    lf_ = Z["proba__nonbold"] * Z["proba__bold"]
    lf_ = lf_ / lf_.sum(1, keepdims=True)
    ests = {"lr": (pa_, pb_), "lf": (lf_, pb_), "hgb": (Z["proba__hgb:nonbold+bold"], Z["proba__hgb:nonbold"]),
            "stack": (S2["stack_both"], S2["stack_nonbold"])}
    up_max, lo_min = -1, 1
    for k, (qa, qb) in ests.items():
        for st in ("N3", "REM"):
            j = STAGES.index(st)
            a_, b_ = ws(y, qa, s, st, j), ws(y, qb, s, st, j)
            d = np.array([a_[u] - b_[u] for u in a_])
            hw = tdist.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            if st == "REM":
                up_max = max(up_max, d.mean() + hw)
            else:
                lo_min = min(lo_min, d.mean() - hw)
    check("remUpMax (max REM t-bound)", up_max, num(M["remUpMax"]), 6e-4)
    check("nthreeLoMin (min N3 t-bound)", lo_min, num(M["nthreeLoMin"]), 6e-4)

    print("== round-3 claims: emission-weight sweep, four-estimator ranges")
    for alpha in (0.5, 1.0, 3.0):
        kk = {}
        for k in ("bold", "nonbold+bold"):
            q = np.clip(Z[f"proba__{k}"], 1e-12, 1) ** alpha
            q = q / q.sum(1, keepdims=True)
            kk[k] = cohen_kappa_score(y, e24.hmm_decode(q, y, s, run, epoch).astype(str))
        okk = kk["bold"] > kk["nonbold+bold"]
        if not okk:
            FAIL.append(f"alpha {alpha}")
        print(f"  {'ok  ' if okk else 'FAIL'} alpha {alpha}: BOLD {kk['bold']:.3f} > combined {kk['nonbold+bold']:.3f}")
    check("hmmGainBold", cohen_kappa_score(y, hb) - cohen_kappa_score(y, np.array(STAGES)[Z["proba__bold"].argmax(1)]),
          num(M["hmmGainBold"]), 6e-4)
    rem_means, rv_o, rv_3 = [], [], []
    for k, (qa, qb) in ests.items():
        j = STAGES.index("REM")
        a_, b_ = ws(y, qa, s, "REM", j), ws(y, qb, s, "REM", j)
        rem_means.append(np.mean([a_[u] - b_[u] for u in a_]))
        for other in ("W", "N1", "N2", "N3"):
            g = []
            for u in np.unique(s):
                m = (s == u) & np.isin(y, ["REM", other])
                pos = y[m] == "REM"
                if pos.sum() >= MIN_EP and (~pos).sum() >= MIN_EP:
                    g.append(roc_auc_score(pos, qa[m, j]) - roc_auc_score(pos, qb[m, j]))
            (rv_3 if other == "N3" else rv_o).append(np.mean(g))
    check("remGainMin", min(rem_means), num(M["remGainMin"]), 6e-4)
    check("remGainMax", max(rem_means), num(M["remGainMax"]), 6e-4)
    check("rvOtherMin", min(rv_o), num(M["rvOtherMin"]), 6e-4)
    check("rvOtherMax", max(rv_o), num(M["rvOtherMax"]), 6e-4)
    check("rvNthreeMin", min(rv_3), num(M["rvNthreeMin"]), 6e-4)
    check("rvNthreeMax", max(rv_3), num(M["rvNthreeMax"]), 6e-4)
    a_, b_ = {}, {}
    for st in ("N3", "REM"):
        j = STAGES.index(st)
        A3, B3 = ws(y, S2["stack_both"], s, st, j), ws(y, S2["stack_nonbold"], s, st, j)
        a_[st] = {u: A3[u] - B3[u] for u in A3}
    both = sorted(set(a_["N3"]) & set(a_["REM"]))
    check("interSt", np.mean([a_["N3"][u] - a_["REM"][u] for u in both]), num(M["interSt"]), 6e-4)

    print("== subcortex (predictions_block_subc.npz, predictions_subcortex_followup.npz)")
    from scipy.stats import t as tdist
    PB = np.load(BASE / "predictions_block_subc.npz", allow_pickle=True)
    PF = np.load(BASE / "predictions_subcortex_followup.npz", allow_pickle=True)
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s = PB["y"].astype(str), PB["subj"].astype(str)
    assert (PF["y"].astype(str) == y).all() and (Z["y"].astype(str) == y).all()

    def gain(pa, pb, st):
        j = STAGES.index(st)
        A_, B_ = ws(y, pa, s, st, j), ws(y, pb, s, st, j)
        return {u: A_[u] - B_[u] for u in A_}

    def tup(d):
        v = np.array(list(d.values()))
        return v.mean() + tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

    lf3 = PB["late3"]
    pairs = {"scAllEarly": (PB["early3"], Z["proba__nonbold"]), "scAllLate": (lf3, Z["proba__nonbold"]),
             "scAllStack": (PB["stack3"], PB["stack1"]), "scBeyondStack": (PB["stack3"], PB["stack2"]),
             "scHgb": (PF["hgb3"], Z["proba__hgb:nonbold"]), "scBeyondGs": (PF["stack_gs4"], PF["stack_gs3"])}
    for tag, (pa, pb) in pairs.items():
        for st in ("N3", "REM"):
            d = gain(pa, pb, st)
            check(f"{tag}{SN[st]}", np.mean(list(d.values())), num(M[f"{tag}{SN[st]}"]), 6e-4)
            if st == "REM":
                check(f"{tag}REMSubj", sum(v > 0 for v in d.values()), num(M[f"{tag}REMSubj"]), 0)
                if tag in ("scAllEarly", "scAllStack"):
                    check(f"{tag}REMupT", tup(d), num(M[f"{tag}REMupT"]), 6e-4)
        g3, gr = gain(pa, pb, "N3"), gain(pa, pb, "REM")
        both = sorted(set(g3) & set(gr))
        if tag in ("scAllStack", "scHgb"):
            key = "scAllStackCtrN" if tag == "scAllStack" else "scHgbCtrSubj"
            want = len(both) if tag == "scAllStack" else sum(g3[u] > gr[u] for u in both)
            check(key, want, num(M[key]), 0)
    for pa_key in ("early3", "late3", "stack3"):   # "N3 still exceeds REM in all ... under early, late and stacked"
        pa, pb = PB[pa_key], (PB["stack1"] if pa_key == "stack3" else Z["proba__nonbold"])
        g3, gr = gain(pa, pb, "N3"), gain(pa, pb, "REM")
        both = sorted(set(g3) & set(gr))
        check(f"ctr10of10 {pa_key}", sum(g3[u] > gr[u] for u in both), len(both), 0)
    v = np.mean(list(ws(y, PB["block"], s, "REM", STAGES.index("REM")).values()))
    check("aucScREM", v, num(M["aucScREM"]), 6e-4)
    ea = gain(PB["early3"], Z["proba__nonbold"], "REM")
    sa = gain(PB["stack3"], PB["stack1"], "REM")
    hg = gain(PF["hgb3"], Z["proba__hgb:nonbold"], "REM")
    la = gain(lf3, Z["proba__nonbold"], "REM")
    m4 = [np.mean(list(d.values())) for d in (ea, la, sa, hg)]
    check("remGainMinSc", min(m4), num(M["remGainMinSc"]), 6e-4)
    check("remGainMaxSc", max(m4), num(M["remGainMaxSc"]), 6e-4)
    d = gain(lf3, Z["proba__nonbold"], "REM")
    v = np.array(list(d.values()))
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    lo_, hi_ = M["scAllLateREMci"].split(", ")
    # an independent bootstrap (24's increment() resamples all 12 subjects from a shared RNG stream), so the
    # endpoints agree only to Monte Carlo error (~0.003 at 2,000 resamples)
    check("scAllLateREMci lo (MC)", np.percentile(bs, 2.5), num(lo_), 5e-3)
    check("scAllLateREMci hi (MC)", np.percentile(bs, 97.5), num(hi_), 5e-3)
    # best-aligned runs: stacks refitted on the subset (36c)
    wm = PF["wl_mask"].astype(bool)
    yw, sw = y[wm], s[wm]

    def gain_w(pa, pb, st):
        j = STAGES.index(st)
        A_, B_ = ws(yw, pa, sw, st, j), ws(yw, pb, sw, st, j)
        return np.mean([A_[u] - B_[u] for u in A_])

    check("scWlStackREM", gain_w(PF["wl_stack3"], PF["wl_stack1"], "REM"), num(M["scWlStackREM"]), 6e-4)
    check("scWlCortREM", gain_w(PF["wl_stack2"], PF["wl_stack1"], "REM"), num(M["scWlCortREM"]), 6e-4)
    check("scWlBeyondREM", gain_w(PF["wl_stack3"], PF["wl_stack2"], "REM"), num(M["scWlBeyondREM"]), 6e-4)
    import json
    FU = json.load(open(BASE / "results_subcortex_followup.json"))   # transcription only (no saved probabilities)
    drop = [v_["gain"]["REM"]["ws_mean"] for k_, v_ in FU["stack"].items() if k_.startswith("without ")]
    check("scDropMin", min(drop), num(M["scDropMin"]), 6e-4)
    check("scDropMax", max(drop), num(M["scDropMax"]), 6e-4)
    p_sc = PB["block"][:, STAGES.index("REM")]
    nf = Z["night_frac"]
    vv = []
    for u in np.unique(s):
        m_ = (s == u) & (y == "N2")
        late = nf[m_] >= 0.5
        if late.sum() >= MIN_EP and (~late).sum() >= MIN_EP:
            vv.append(roc_auc_score(late, p_sc[m_]))
    check("clkSc", np.mean(vv), num(M["clkSc"]), 6e-3)

    print("== regional level and all BOLD blocks (predictions_block_lvl/max.npz, predictions_max_hgb.npz)")
    for blk, pre in (("lvl", "lv"), ("max", "mx")):
        Q = np.load(BASE / f"predictions_block_{blk}.npz", allow_pickle=True)
        assert (Q["y"].astype(str) == y).all()
        for tag, (pa, pb) in {f"{pre}AllEarly": (Q["early3"], Z["proba__nonbold"]), f"{pre}AllLate": (Q["late3"], Z["proba__nonbold"]),
                              f"{pre}AllStack": (Q["stack3"], Q["stack1"]), f"{pre}BeyondEarly": (Q["early3"], Z["proba__nonbold+bold"]),
                              f"{pre}BeyondLate": (Q["late3"], Q["late2"]), f"{pre}BeyondStack": (Q["stack3"], Q["stack2"])}.items():
            for st in ("N3", "REM"):
                check(f"{tag}{SN[st]}", np.mean(list(gain(pa, pb, st).values())), num(M[f"{tag}{SN[st]}"]), 6e-4)
    QH = np.load(BASE / "predictions_max_hgb.npz", allow_pickle=True)
    for st in ("N3", "REM"):
        check(f"mxHgb{SN[st]}", np.mean(list(gain(QH["hgb_max"], Z["proba__hgb:nonbold"], st).values())), num(M[f"mxHgb{SN[st]}"]), 6e-4)
    # the specification envelope: every derivative row of Table 2, from its sources (independent selection)
    import json as _j
    L = lambda f: _j.load(open(BASE / f))
    A_, ST_, WL_ = L("results_v2_afni.json"), L("results_stacked.json"), L("results_whitelist.json")
    BL = {b: L(f"results_block_{b}.json") for b in ("mw", "subc", "lvl", "max")}
    keys_a = ["nonbold+bold - nonbold", "latefusion(nonbold,bold) - nonbold", "nonbold+gs+bold - nonbold+gs",
              "nonbold+bold_gsr - nonbold", "nonbold+bold_acc - nonbold", "nonbold+bold_hrf - nonbold", "nonbold+bold_z - nonbold",
              "nonbold_img+bold - nonbold_img", "nonbold_nightfrac+bold - nonbold_nightfrac"]
    rows = [A_["increment"][k] for k in keys_a] + [ST_["increment"], A_["hgb"]["increment"], WL_["afni"]["increment"],
            BL["mw"]["gain"]["early: all BOLD over non-BOLD"], FU["hgb"]["gain"], L("results_max_hgb.json")["all BOLD over non-BOLD"]["gain"]]
    rows += [BL["subc"]["gain"][f"{k}: all BOLD over non-BOLD"] for k in ("early", "late", "stack")]
    rows += [BL["lvl"]["gain"]["stack: all BOLD over non-BOLD"]]
    rows += [BL["max"]["gain"][f"{k}: all BOLD over non-BOLD"] for k in ("early", "late", "stack")]
    # draft 10: night index, uncensored epochs, the C sweep, nested tuning and the stronger non-BOLD baseline
    RC_ = L("results_reviewer_checks.json")
    rows += [RC_["night_index"], RC_["uncensored"]] + [RC_["C_sweep"][c] for c in ("0.01", "0.1", "10.0")]
    R2_ = L("results_round2_checks.json")
    rows += [R2_[k]["gain"] for k in ("slow", "dt", "cross_night", "no_drift_runs")]
    rows.append(L("results_tone_proxy.json")["gain"])
    for mdl, sset in (("lr", "main"), ("hgb", "main"), ("lr", "max"), ("hgb", "max")):
        if (BASE / f"results_tuned_{mdl}_{sset}.json").exists():
            rows.append(L(f"results_tuned_{mdl}_{sset}.json")["gain"])
    if (BASE / "results_strong_physio.json").exists():
        rows += [L("results_strong_physio.json")["models"]["gru"]["gain"]]
    P1_, P2_ = L("results_panel_checks.json"), L("results_panel_checks2.json")
    rows += [P1_["tones5"]["gain"], P1_["lagged"]["gain"], P2_["noaud"]["gain"]]
    rows += [P2_["conserv"][f]["gain"] for f in ("lr", "hgb", "gru")]
    check("nSpec", len(rows), num(M["nSpec"]), 0)
    check("envRemMin", min(r["REM"]["ws_mean"] for r in rows), num(M["envRemMin"]), 6e-4)
    check("envRemMax", max(r["REM"]["ws_mean"] for r in rows), num(M["envRemMax"]), 6e-4)
    check("envNthreeMin", min(r["N3"]["ws_mean"] for r in rows), num(M["envNthreeMin"]), 6e-4)
    check("envNthreeMax", max(r["N3"]["ws_mean"] for r in rows), num(M["envNthreeMax"]), 6e-4)

    def tb_(r, up):
        v = np.array(list(r["ws_values"].values()))
        h = tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean() + h if up else v.mean() - h
    check("envRemMaxUpT", max(tb_(r["REM"], True) for r in rows), num(M["envRemMaxUpT"]), 6e-4)
    check("envNthreeMinLoT", min(tb_(r["N3"], False) for r in rows), num(M["envNthreeMinLoT"]), 6e-4)
    full = sum(all(r["N3"]["ws_values"][u] > r["REM"]["ws_values"][u] for u in r["REM"]["ws_values"] if u in r["N3"]["ws_values"])
               for r in rows)
    check("nSpecCtrAll", full, num(M["nSpecCtrAll"]), 0)
    MT_ = L("results_matched.json")
    check("mtRemMax", max(MT_[k][q]["REM"]["ws_mean"] for k in ("increment", "increment_late") for q in ("raw84amp", "afni84", "afni")),
          num(M["mtRemMax"]), 6e-4)

    print("== REM precision/recall after decoding; N3 network gradient")
    from sklearn.metrics import precision_score, recall_score
    yz = Z["y"].astype(str)
    for key, tag in (("nonbold", "Nb"), ("nonbold+bold", "All")):
        hp = Z[f"hmm__{key}"].astype(str)
        check(f"recREM{tag}", recall_score(yz == "REM", hp == "REM"), num(M[f"recREM{tag}"]), 6e-3)
        check(f"precREM{tag}", precision_score(yz == "REM", hp == "REM"), num(M[f"precREM{tag}"]), 6e-3)
    NP = np.load(BASE / "n3_parcel_map.npz", allow_pickle=True)
    nets_ = np.array([str(n) for n in NP["nets"]])
    per = {n: NP["N3"][nets_ == n].mean() for n in np.unique(nets_)}
    low = sorted(per, key=per.get)[:4]
    check("nthreeNetLo", per[low[0]], num(M["nthreeNetLo"]), 6e-3)
    check("nthreeNetHi", per[low[-1]], num(M["nthreeNetHi"]), 6e-3)
    check("nthreeNetVis", max(per.values()), num(M["nthreeNetVis"]), 6e-3)

    print("== draft-9 additions (REM bouts, night split, late night, REM network range, drift)")
    yb, runb = Z["y"].astype(str), None
    D9 = _j.load(open(BASE / "results_rem_bouts.json"))
    check("remBoutMed", D9["median_bout_min"], num(M["remBoutMed"]), 0.05)
    check("remNightTwo", 100 * D9["frac_rem_night2"], num(M["remNightTwo"]), 0.5)
    ln = A_["time_matched_rem"]["inc_nonbold"]
    check("lnREM", ln["ws_mean"], num(M["lnREM"]), 6e-4)
    rp = {n: NP["REM"][nets_ == n].mean() for n in np.unique(nets_)}
    check("remNetLo", min(rp.values()), num(M["remNetLo"]), 6e-3)
    check("remNetHi", max(rp.values()), num(M["remNetHi"]), 6e-3)
    AL_ = L("results_data_checks.json")["alignment"]
    check("nDriftAll", AL_["n_runs"] - AL_["within_one_epoch"], num(M["nDriftAll"]), 0)

    print("== draft-10 additions: review checks, tuning, stronger baseline, nuisance audit, own v2, eyes")
    import json as _jj
    from scipy.stats import t as _tdist

    def gain_from(pa, pb, yy, ss, st):
        i = STAGES.index(st)
        a, b = ws(yy, pa, ss, st, i), ws(yy, pb, ss, st, i)
        return np.array([a[k] - b[k] for k in a]), {k: a[k] - b[k] for k in a}

    def tup(v):
        return v.mean() + _tdist.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

    def check_pair(tag, pa, pb, yy, ss, sts=("N3", "REM"), ctr=True):
        out = {}
        for st in sts:
            d, dd = gain_from(pa, pb, yy, ss, st)
            out[st] = dd
            check(f"{tag}{SN[st]}", d.mean(), num(M[f"{tag}{SN[st]}"]), 6e-4)
            check(f"{tag}{SN[st]}Subj", (d > 0).sum(), num(M[f"{tag}{SN[st]}Subj"]), 0)
            if st == "REM":
                check(f"{tag}REMupT", tup(d), num(M[f"{tag}REMupT"]), 6e-4)
        if ctr and f"{tag}CtrSubj" in M:
            both = [u for u in out["N3"] if u in out["REM"]]
            check(f"{tag}CtrSubj", sum(out["N3"][u] > out["REM"][u] for u in both), num(M[f"{tag}CtrSubj"]), 0)
            check(f"{tag}CtrN", len(both), num(M[f"{tag}CtrN"]), 0)

    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, run, ep = Z["y"].astype(str), Z["subj"].astype(str), Z["run"].astype(str), Z["epoch"].astype(int)
    # two-decimal kappas used in the text and Table 1
    for key, tag in [("bold", "Bold"), ("nonbold", "Nb"), ("nonbold+bold", "All")]:
        check(f"kk{tag}", cohen_kappa_score(y, Z[f"hmm__{key}"].astype(str)), num(M[f"kk{tag}"]), 5.1e-3)
    from sklearn.metrics import f1_score as _f1
    for st in STAGES:
        check(f"fone{SN[st]}Bold", _f1(y == st, Z["hmm__bold"].astype(str) == st), num(M[f"fone{SN[st]}Bold"]), 6e-3)
    # bouts: same-stage runs of consecutive analysed epochs, recomputed here
    L_ = np.zeros(len(y), int)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r); idx = idx[np.argsort(ep[idx])]
        a0 = 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or y[idx[k]] != y[idx[a0]] or ep[idx[k]] != ep[idx[k - 1]] + 1:
                L_[idx[a0:k]] = k - a0; a0 = k
    check("boutRemEp", np.median(L_[y == "REM"]) / 2, num(M["boutRemEp"]), 0.05)
    check("boutNthreeEp", np.median(L_[y == "N3"]) / 2, num(M["boutNthreeEp"]), 0.05)
    m5 = L_ >= 10
    check_pair("bl", Z["proba__nonbold+bold"][m5], Z["proba__nonbold"][m5], y[m5], s[m5])
    n2 = np.array([r_.split("_")[1] == "ses-2" for r_ in run])
    check_pair("nt", Z["proba__nonbold+bold"][n2], Z["proba__nonbold"][n2], y[n2], s[n2])
    # nested tuning, stronger baseline, own v2: from their saved predictions
    for mdl, sset, tag in (("lr", "main", "tuLr"), ("hgb", "main", "tuHgb"), ("lr", "max", "tuLrMx"), ("hgb", "max", "tuHgbMx")):
        f_ = BASE / f"predictions_tuned_{mdl}_{sset}.npz"
        if f_.exists() and f"{tag}REM" in M:
            T_ = np.load(f_, allow_pickle=True)
            assert (T_["y"].astype(str) == y).all()
            check_pair(tag, T_["with_bold"], T_["nonbold"], y, s)
    f_ = BASE / "predictions_strong_physio.npz"
    if f_.exists():
        SPp = np.load(f_, allow_pickle=True)
        assert (SPp["y"].astype(str) == y).all()
        for fam, tag in (("lr", "spLr"), ("hgb", "spHgb"), ("gru", "spGru")):
            if f"{fam}_all" in SPp.files and f"{tag}REM" in M:
                check_pair(tag, SPp[f"{fam}_all"], SPp[f"{fam}_nb"], y, s, sts=STAGES)
                v = np.mean(list(ws(y, SPp[f"{fam}_nb"], s, "N3", 3).values()))
                check(f"{tag}AucNbNthree", v, num(M[f"{tag}AucNbNthree"]), 6e-4)
    f_ = BASE / "predictions_rv2_eyes.npz"
    if f_.exists():
        RVp = np.load(f_, allow_pickle=True)
        yv, sv = RVp["y"].astype(str), RVp["subj"].astype(str)
        for k, tag in (("minimal", "rvMin"), ("v2", "rvTwo"), ("v2_ric", "rvRic"), ("v2_phys", "rvPhys")):
            check_pair(tag, RVp[f"all_{k}"], RVp["nonbold"], yv, sv)
            v = np.mean(list(ws(yv, RVp[f"bold_{k}"], sv, "REM", 4).values()))
            check(f"{tag}AucREM", v, num(M[f"{tag}AucREM"]), 6e-4)
    PRj = _jj.load(open(BASE / "results_derivative_physio_r2.json"))
    subj_ = sorted({r_["subject"] for r_ in PRj["runs"]})
    for k, tag in (("deriv_slow", "prDerivSlow"), ("deriv_retroicor", "prDerivRic")):
        v = np.mean([np.mean([r_[k] for r_ in PRj["runs"] if r_["subject"] == u]) for u in subj_])
        check(tag, v, num(M[tag]), 6e-4)

    print("== draft-13 additions: time-stratified AUC and average precision (recomputed from predictions)")
    from sklearn.metrics import average_precision_score as _ap
    Zs = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    ys, ss = Zs["y"].astype(str), Zs["subj"].astype(str)
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location("e24v", BASE / "24_experiments_v2.py"); _e = _iu.module_from_spec(_sp); _sp.loader.exec_module(_e)
    hrs = _e.load("afni")["clock"][:, 0]
    for st, tag in (("REM", "strREM"), ("N3", "strNthree")):
        i_ = STAGES.index(st); g_ = []
        for u in np.unique(ss):
            m_ = ss == u; y01 = ys[m_] == st
            if y01.sum() >= 20 and (~y01).sum() >= 20:
                def sa(p_):
                    num = den = 0.0
                    for h in np.unique(np.floor(hrs[m_])):
                        b_ = np.floor(hrs[m_]) == h
                        a1, a0 = y01[b_].sum(), (~y01[b_]).sum()
                        if a1 >= 5 and a0 >= 5:
                            num += roc_auc_score(y01[b_], p_[b_]) * a1 * a0; den += a1 * a0
                    return num / den if den else np.nan
                a_, b_ = sa(Zs["proba__nonbold+bold"][m_, i_]), sa(Zs["proba__nonbold"][m_, i_])
                if np.isfinite(a_) and np.isfinite(b_):
                    g_.append(a_ - b_)
        check(tag, np.mean(g_), num(M[tag]), 6e-4)
        v_ = [_ap(ys[ss == u] == st, Zs["proba__nonbold+bold"][ss == u, i_]) - _ap(ys[ss == u] == st, Zs["proba__nonbold"][ss == u, i_])
              for u in np.unique(ss) if (ys[ss == u] == st).sum() >= 20]
        check("ap" + tag[3:], np.mean(v_), num(M["ap" + tag[3:]]), 6e-4)

    print("== hand-typed decimals in main.tex (should be none outside comments)")
    tex = [l.split("%")[0] for l in (PAPER / "main.tex").read_text().splitlines()]
    hits = [(i + 1, m.group(0)) for i, l in enumerate(tex) for m in re.finditer(r"(?<![\w\\{])[-+]?\d+\.\d+", l)]
    for ln, h in hits:
        print(f"  line {ln}: {h}")
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")


if __name__ == "__main__":
    main()
