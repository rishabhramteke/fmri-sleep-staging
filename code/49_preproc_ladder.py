"""Which preprocessing step raises the N3 amplitude signal? A ladder from our realigned pipeline towards the
derivative, rebuilt in our own native-space stream and scored on the matched epochs and shared parcels of 28_matched.py.

The clue (results_matched.json): on identical parcels and epochs, BOLD alone separates N3 at AUC 0.654 from our
realigned parcel amplitudes and at 0.836 from the derivative's, so the derivative's amplitude is a cleaner signal, not
merely one more independent of the non-BOLD baseline. The derivative's steps that ours lacks, each rebuilt here
without AFNI:
  alignment  nonlinear T1->MNI registration (dipy SyN, cross-correlation metric, after the affine) instead of the
             affine alone; the atlas and the tissue priors are carried through it into native EPI
  purity     parcel means restricted to grey matter, by the MNI GM probability carried through the same chain
             (>= GM_MIN), or by an intensity tissue class of the mean EPI (k-means, 3 classes; middle class = GM)
  nuisance   regression, at feature time, of white-matter PCs + CSF mean (aCompCor-like; the derivative's 3dTproject
             matrix is undocumented), the 24 motion parameters, and RETROICOR + slow physiological regressors
Parcel-series variants per run: aff (= 06b_raw_v2.py), aff_gm, aff_tis, syn, syn_gm, syn_tis, syn_gm_tis.

    source <venv>/bin/activate
    python 49_preproc_ladder.py --sample [--workers 3]      # derivatives/rv3_ds005127/<sub>/<run>_rv3.npz (~3 h)
    python 49_preproc_ladder.py --features                 # features_v2_rv3.npz (84-parcel amplitude per variant)
    python 49_preproc_ladder.py --gains                    # results_preproc_ladder.json
"""
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2

BASE = Path(__file__).parent
RAW = BASE / "data" / "ds005127"
MOT = BASE / "derivatives" / "motion_ds005127"
OUT = BASE / "derivatives" / "rv3_ds005127"
FD_CENSOR = 0.5
GM_MIN = 0.5
VARIANTS = ("aff", "aff_gm", "aff_tis", "syn", "syn_gm", "syn_tis", "syn_gm_tis")
REGRESSIONS = ("", "_acc", "_accmot", "_accmotphys")
SYN_ITERS = [10, 10, 5]
N_WM_PCS = 5


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ sampling stage
def brain_mask(mf):
    """Largest bright connected component of the mean EPI, closed and filled."""
    b = mf > 0.3 * np.percentile(mf, 99)
    lab, n = ndimage.label(b)
    if n == 0:
        return b
    sizes = ndimage.sum(b, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def tissue_classes(mf, brain):
    """k-means (3) on mean-EPI intensity inside the brain, seeded at the 30th/60th/90th percentiles: 0 dark (white
    matter, partial volume with bone), 1 middle (grey matter), 2 bright (CSF, vessels). Registration-independent."""
    x = mf[brain].astype(np.float64)
    seeds = np.percentile(x, [30, 60, 90])[:, None]
    cent, lab = kmeans2(x[:, None], seeds, minit="matrix")
    order = np.argsort(cent.ravel())
    cls = np.full(mf.shape, -1, np.int8)
    cls[brain] = np.argsort(order)[lab]
    return cls, np.sort(cent.ravel())


def skull_strip(t1, t1_aff, bm_img, mni_to_t1):
    """Rough skull strip of the T1 by the MNI brain mask dilated 3 voxels (6 mm) carried through the affine, so the
    skull does not drive the nonlinear step (the template is skull-stripped)."""
    from dipy.align.imaffine import AffineMap
    bm = ndimage.binary_dilation(np.asanyarray(bm_img.dataobj) > 0, iterations=3).astype(np.float32)
    amap = AffineMap(np.linalg.inv(mni_to_t1), t1.shape, t1_aff, bm.shape, bm_img.affine)
    return t1 * (amap.transform(bm, interpolation="nearest") > 0.5)


def syn_t1_to_mni(t1_img, template, tmpl_aff, mni_to_t1, bm_img):
    """Nonlinear T1->MNI (dipy SyN, CC metric) starting from the affine of the 06_preprocess chain."""
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    t1 = skull_strip(np.asanyarray(t1_img.dataobj, dtype=np.float32), t1_img.affine, bm_img, mni_to_t1)
    sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=SYN_ITERS)
    return sdr.optimize(template, t1, tmpl_aff, t1_img.affine, prealign=mni_to_t1)


def to_native_affine(img_data, img_aff, native_from_mni, shape, faff, interp):
    from dipy.align.imaffine import AffineMap
    amap = AffineMap(native_from_mni, shape, faff, img_data.shape, img_aff)
    return amap.transform(np.asarray(img_data, np.float32), interpolation=interp)


def to_native_syn(img_data, mapping, t1_img, epi2t1, shape, faff, interp):
    """MNI image -> T1 space through the inverse nonlinear map -> native EPI through the rigid EPI->T1."""
    from dipy.align.imaffine import AffineMap
    in_t1 = mapping.transform_inverse(np.asarray(img_data, np.float32), interpolation=interp)
    amap = AffineMap(np.linalg.inv(epi2t1), shape, faff, in_t1.shape, t1_img.affine)
    return amap.transform(in_t1, interpolation=interp)


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def parcel_means(V, lab):
    """Parcel means over the sampled voxels: lab (N,) in -1..199, -1 = not in this variant."""
    m = lab >= 0
    acc = np.zeros((200, V.shape[1]))
    np.add.at(acc, lab[m], V[m])
    sizes = np.bincount(lab[m], minlength=200)
    ts = np.full((200, V.shape[1]), np.nan, np.float32)
    ok = sizes > 0
    ts[ok] = (acc[ok] / sizes[ok, None]).astype(np.float32)
    return ts, sizes.astype(np.int32)


def process_subject(sub, max_runs=0, shard=(0, 1)):
    pp = _load("pp", "06_preprocess.py")
    rv = _load("rv", "06b_raw_v2.py")
    from nilearn.datasets import (fetch_atlas_schaefer_2018, load_mni152_brain_mask, load_mni152_gm_template,
                                  load_mni152_template, load_mni152_wm_template)
    funcs = sorted(RAW.glob(f"{sub}/**/func/*task-sleep*.nii.gz"))
    t1s = sorted(RAW.glob(f"{sub}/**/anat/*T1w.nii.gz"))
    tmpl_img = load_mni152_template(resolution=2)
    template = np.asanyarray(tmpl_img.dataobj, dtype=np.float32)
    atlas_img = nib.load(fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)["maps"])
    atlas = np.asanyarray(atlas_img.dataobj, dtype=np.float32)
    gm_img, wm_img, bm_img = load_mni152_gm_template(resolution=2), load_mni152_wm_template(resolution=2), load_mni152_brain_mask(resolution=2)
    gm_p, wm_p = np.asanyarray(gm_img.dataobj, dtype=np.float32), np.asanyarray(wm_img.dataobj, dtype=np.float32)
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    todo = [f for f in funcs if not (OUT / sub / f"{f.name.replace('_bold.nii.gz', '')}_rv3.npz").exists()
            and (MOT / sub / f"{f.name.replace('_bold.nii.gz', '')}_motion.npz").exists()]
    if max_runs:
        todo = todo[:max_runs]
    todo = todo[shard[0]::shard[1]]                 # this worker's share of the subject's remaining runs
    if not todo:
        return sub, "nothing to do"
    t1_path = pp.select_t1(t1s, todo[0], template, tmpl_img, atlas_img)
    t1_img = nib.load(str(t1_path))
    cache, anchors, log, mapping = {}, {}, [], None
    for f in todo:
        t0 = time.time()
        stem = f.name.replace("_bold.nii.gz", "")
        M = np.load(MOT / sub / f"{stem}_motion.npz")
        mf, faff, fimg = pp.mean_epi(f)
        ses = next((p for p in f.name.split("_") if p.startswith("ses-")), "ses-?")
        nfm, epi2t1 = pp.build_chain(mf, faff, t1_path, template, tmpl_img.affine, cache=cache, warm=anchors.get(ses))
        lab_aff = pp.atlas_to_native(atlas_img, nfm, mf.shape, faff)
        n_aff = len(np.unique(lab_aff[lab_aff > 0]))
        if n_aff < pp.MIN_PARCELS and anchors.get(ses) is not None:        # as 06b: retry cold
            nfm2, aff2 = pp.build_chain(mf, faff, t1_path, template, tmpl_img.affine, cache=cache, warm=None)
            lab2 = pp.atlas_to_native(atlas_img, nfm2, mf.shape, faff)
            if len(np.unique(lab2[lab2 > 0])) > n_aff:
                lab_aff, nfm, epi2t1 = lab2, nfm2, aff2
                n_aff = len(np.unique(lab_aff[lab_aff > 0]))
        if n_aff >= pp.MIN_PARCELS and ses not in anchors:
            anchors[ses] = epi2t1
        if mapping is None:
            ts0 = time.time()
            mapping = syn_t1_to_mni(t1_img, template, tmpl_img.affine, cache["t12mni"], bm_img)
            log.append(f"SyN T1->MNI {time.time() - ts0:.0f} s")
            print(f"  {sub} {log[-1]}", flush=True)
        # tissue priors and labels in native EPI, through the affine chain and through SyN
        gm_aff = to_native_affine(gm_p, gm_img.affine, nfm, mf.shape, faff, "linear")
        lab_syn = np.rint(to_native_syn(atlas, mapping, t1_img, epi2t1, mf.shape, faff, "nearest")).astype(np.int16)
        gm_syn = to_native_syn(gm_p, mapping, t1_img, epi2t1, mf.shape, faff, "linear")
        wm_syn = to_native_syn(wm_p, mapping, t1_img, epi2t1, mf.shape, faff, "linear")
        n_syn = len(np.unique(lab_syn[lab_syn > 0]))
        brain = brain_mask(mf)
        cls, cent = tissue_classes(mf, brain)
        gm_cls = cls == 1
        q = dict(dice_aff=dice(gm_aff >= GM_MIN, gm_cls), dice_syn=dice(gm_syn >= GM_MIN, gm_cls),
                 gmfrac_aff=float(gm_cls[lab_aff > 0].mean()), gmfrac_syn=float(gm_cls[lab_syn > 0].mean()),
                 prior_gm_aff=float((gm_aff[lab_aff > 0] >= GM_MIN).mean()), prior_gm_syn=float((gm_syn[lab_syn > 0] >= GM_MIN).mean()),
                 n_aff=n_aff, n_syn=n_syn, centroids=cent.tolist())
        inpar = (lab_aff > 0) | (lab_syn > 0)
        wm_mask = (wm_syn >= 0.9) & brain & ~inpar
        wm_mask = ndimage.binary_erosion(wm_mask, iterations=1)
        csf_mask = (gm_syn < 0.1) & (wm_syn < 0.1) & brain & (cls == 2) & ~inpar
        if csf_mask.sum() < 20:
            csf_mask = (cls == 2) & brain & ~inpar
        if wm_mask.sum() < 50:
            wm_mask = (wm_syn >= 0.9) & brain & ~inpar
        q.update(n_wm=int(wm_mask.sum()), n_csf=int(csf_mask.sum()))
        # one pass over the volumes: every voxel any variant or nuisance signal needs
        data = np.asanyarray(fimg.dataobj, dtype=np.float32)
        T = data.shape[3]
        params = M["params"].astype(np.float64)
        if len(params) != T:
            log.append(f"{stem}: motion length {len(params)} != {T}, skipped")
            continue
        union = inpar | wm_mask | csf_mask
        ijk = np.argwhere(union)
        V = rv.sample_realigned(data, ijk, params, fimg.header.get_zooms()[:3])
        del data
        la, ls = lab_aff[tuple(ijk.T)].astype(int) - 1, lab_syn[tuple(ijk.T)].astype(int) - 1
        ga, gs_ = gm_aff[tuple(ijk.T)], gm_syn[tuple(ijk.T)]
        gc = gm_cls[tuple(ijk.T)]
        sel = {"aff": la, "aff_gm": np.where(ga >= GM_MIN, la, -1), "aff_tis": np.where(gc, la, -1),
               "syn": ls, "syn_gm": np.where(gs_ >= GM_MIN, ls, -1), "syn_tis": np.where(gc, ls, -1),
               "syn_gm_tis": np.where(gc & (gs_ >= GM_MIN), ls, -1)}
        fd = M["fd"].astype(np.float64)
        cens = fd > FD_CENSOR
        cens = cens | np.r_[cens[1:], False]
        out = {}
        for k, lab in sel.items():
            ts, sizes = parcel_means(V, lab)
            out["ts_" + k] = rv.despike(ts)
            out["n_" + k] = sizes
        wmv, csfv = wm_mask[tuple(ijk.T)], csf_mask[tuple(ijk.T)]
        out["wm"] = V[wmv].mean(0).astype(np.float32) if wmv.any() else np.full(T, np.nan, np.float32)
        out["csf"] = V[csfv].mean(0).astype(np.float32) if csfv.any() else np.full(T, np.nan, np.float32)
        out["gs"] = V[la >= 0].mean(0).astype(np.float32)
        valid = ~cens
        pcs = np.full((N_WM_PCS, T), np.nan, np.float32)
        if wmv.sum() >= N_WM_PCS + 5 and valid.sum() > N_WM_PCS + 5:
            X = V[wmv]
            X = (X - X[:, valid].mean(1, keepdims=True)) / (X[:, valid].std(1, keepdims=True) + 1e-6)
            U, S, Vt = np.linalg.svd(X[:, valid], full_matrices=False)
            pcs = (U[:, :N_WM_PCS].T @ X / np.sqrt(len(X))).astype(np.float32)
        out["wm_pcs"] = pcs
        np.savez_compressed(OUT / sub / f"{stem}_rv3.npz", censored=cens, qc=json.dumps(q), tr=float(fimg.header.get_zooms()[3]),
                            n_vol=T, **out)
        log.append(f"{stem}: parcels aff {n_aff} syn {n_syn}; GM-class fraction of parcel voxels aff {q['gmfrac_aff']:.2f} "
                   f"syn {q['gmfrac_syn']:.2f}; Dice(prior GM, EPI GM class) aff {q['dice_aff']:.2f} syn {q['dice_syn']:.2f}; "
                   f"wm {q['n_wm']} csf {q['n_csf']} vox; {time.time() - t0:.0f} s")
        print(f"  {sub} {log[-1]}", flush=True)
    return sub, "; ".join(log[-2:])


def remaining(sub):
    return [f for f in sorted(RAW.glob(f"{sub}/**/func/*task-sleep*.nii.gz"))
            if not (OUT / sub / f"{f.name.replace('_bold.nii.gz', '')}_rv3.npz").exists()
            and (MOT / sub / f"{f.name.replace('_bold.nii.gz', '')}_motion.npz").exists()]


def sample(a):
    subs = ([a.subject] if a.subject else sorted(p.name for p in RAW.glob("sub-*") if p.is_dir()))
    # tasks: (subject, shard) with at most --runs-per-task runs each, longest subjects first, so the tail is short
    tasks = []
    for s in subs:
        n = len(remaining(s))
        k = max(1, -(-n // a.runs_per_task)) if a.runs_per_task else 1
        tasks += [(s, (i, k), n) for i in range(k)]
    tasks.sort(key=lambda x: -x[2])
    print(f"{len(tasks)} tasks over {len(subs)} subjects, {sum(x[2] for x in tasks if x[1][0] == 0)} runs remaining, {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_subject, s, a.max_runs, sh): (s, sh) for s, sh, _ in tasks}
        for fu in as_completed(futs):
            try:
                s, msg = fu.result()
                print(f"[done {s} shard {futs[fu][1][0] + 1}/{futs[fu][1][1]}] {msg} ({time.time() - t0:.0f} s)", flush=True)
            except Exception as err:
                print(f"[FAILED {futs[fu]}] {err!r}", flush=True)


# ------------------------------------------------------------------ feature stage
def features():
    f22 = _load("f22", "22_features_v2.py")
    feat, conf = f22.feat, f22.conf
    net_ids, nets = feat.network_ids()
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    raw_runs = {r.name.replace("_parcels.npz", ""): r for r in (BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")}
    src = {stem: OUT / r.parent.name / f"{stem}_rv3.npz" for stem, r in raw_runs.items()
           if (OUT / r.parent.name / f"{stem}_rv3.npz").exists()}
    print(f"rv3: {len(src)} runs; {common.sum()} shared parcels", flush=True)
    keys, blocks, qcs = [], {}, []
    for i, (stem, p) in enumerate(sorted(src.items()), 1):
        eps = feat.run_epochs(raw_runs[stem])
        if not eps:
            continue
        z = np.load(p)
        T = int(z["n_vol"])
        if T != np.load(raw_runs[stem])["ts"].shape[1]:
            print(f"  skip {stem}: volume count differs from raw", flush=True)
            continue
        qcs.append(json.loads(str(z["qc"])))
        valid = ~z["censored"]
        gs = z["gs"].astype(np.float64)
        gs = np.where(valid, (gs - gs[valid].mean()) / (gs[valid].std() + 1e-9), np.nan)
        # nuisance sets: aCompCor-like (WM PCs + CSF), + 24 motion parameters, + RETROICOR and slow physiology
        Racc = np.column_stack([z["wm_pcs"].T, z["csf"]]).astype(np.float64)
        Racc = (Racc - Racc[valid].mean(0)) / (Racc[valid].std(0) + 1e-9)
        mp = np.load(MOT / raw_runs[stem].parent.name / f"{stem}_motion.npz")["params"].astype(np.float64)
        dm = np.vstack([np.zeros((1, 6)), np.diff(mp, axis=0)])
        Rm = np.hstack([mp, dm, mp ** 2, dm ** 2])
        Rm = (Rm - Rm[valid].mean(0)) / (Rm[valid].std(0) + 1e-9)
        Pphys = None
        pf = conf.PHYS / raw_runs[stem].parent.name / f"{stem}_physio.npz"
        if pf.exists():
            P = dict(np.load(pf, allow_pickle=True))
            qc, qr = conf.run_quality(P)
            grid, S = conf.physio_signals(P, T, 3.0)
            R1, R2 = conf.retroicor(P, T, 3.0), conf.slow_regressors(grid, S, T, 3.0)
            if qc >= 0.7 and qr >= 0.7 and R1 is not None and R2 is not None and conf.trigger_consistent(P, T, 3.0):
                Pphys = np.hstack([R1, R2])
        regs = {"": None, "_acc": Racc, "_accmot": np.hstack([Racc, Rm]),
                "_accmotphys": np.hstack([Racc, Rm, Pphys]) if Pphys is not None else None}
        if not np.isfinite(Racc).all():
            regs = {"": None}
        for v in VARIANTS:
            ts = f22.psc(z["ts_" + v].astype(np.float64), valid)
            for tag in REGRESSIONS:
                R = regs.get(tag, None)
                x = ts if tag == "" else (f22.regress(ts, R, valid) if R is not None else None)
                for ep in eps:
                    if x is None:
                        amp = np.full(common.sum(), np.nan)
                    else:
                        amp, _, _ = f22.epoch_block(x, gs, ep, common, net_ids, len(nets), float(z["tr"]), T)
                    blocks.setdefault(f"amp_{v}{tag}", []).append(amp)
        for ep in eps:
            keys.append((stem, ep["epoch"]))
        if i % 25 == 0 or i == len(src):
            print(f"  [{i}/{len(src)}] epochs {len(keys)}", flush=True)
    out = {k: np.array(v, np.float32) for k, v in blocks.items()}
    np.savez_compressed(BASE / "features_v2_rv3.npz", run=np.array([k[0] for k in keys]), epoch=np.array([k[1] for k in keys]),
                        common=common, qc=json.dumps(qcs), **out)
    print({k: round(float(np.isnan(v).any(1).mean()), 3) for k, v in out.items()})


# ------------------------------------------------------------------ gains stage
def gains(chunk=None):
    e = _load("e24", "24_experiments_v2.py")
    ST = e.STAGES
    Dr, Da, Dv = e.load("raw"), e.load("afni"), e.load("rv2")
    V3 = np.load(BASE / "features_v2_rv3.npz", allow_pickle=True)
    k3 = {(r, int(p)): i for i, (r, p) in enumerate(zip(V3["run"].astype(str), V3["epoch"]))}
    kr = {(r, int(p)): i for i, (r, p) in enumerate(zip(Dr["run"], Dr["epoch"]))}
    ka = {(r, int(p)): i for i, (r, p) in enumerate(zip(Da["run"], Da["epoch"]))}
    kv = {(r, int(p)): i for i, (r, p) in enumerate(zip(Dv["run"], Dv["epoch"]))}
    # the intensity-class (*_tis) variants empty some shared parcels in most runs (the EPI k-means class is a weak tissue
    # reference, Dice 0.3-0.5 with the prior), so they are left out of the ladder; GM-prior restriction covers purity
    names = [f"amp_{v}{t}" for v in VARIANTS if "tis" not in v for t in REGRESSIONS]
    fin = np.ones(len(V3["run"]), bool)
    for n in names:
        fin &= np.isfinite(V3[n]).all(1)
    keys = sorted(k for k in set(kr) & set(ka) & set(kv) & set(k3) if fin[k3[k]])
    ir, ia, iv, i3 = (np.array([d[k] for k in keys]) for d in (kr, ka, kv, k3))
    y, s = Dr["stage"][ir], Dr["subj"][ir]
    assert (y == Da["stage"][ia]).all() and (s == Da["subj"][ia]).all() and (y == Dv["stage"][iv]).all()
    common = np.load(BASE / "features_v2_raw.npz", allow_pickle=True)["common"].astype(bool)
    wz = e.wz
    nb = np.hstack([wz(Dr["motion"][ir], s), wz(Dr["physio"][ir], s), Dr["clock"][ir]])      # as 28_matched.py
    bold = {"rv84amp": wz(Dv["amp"][iv], s), "rv84phys": wz(Dv["amp_phys"][iv], s), "afni84": wz(Da["amp"][ia][:, common], s)}
    for n in names:
        bold[n] = wz(V3[n][i3].astype(np.float64), s)
    if chunk:
        i_, n_ = (int(v) for v in chunk.split("/"))
        bold = {k: bold[k] for k in list(bold)[i_::n_]}
    qcs = json.loads(str(V3["qc"]))
    R = {"meta": dict(n_epochs=len(keys), n_subjects=int(len(np.unique(s))), n_matched28=13448,
                      stage_counts={st: int((y == st).sum()) for st in ST}, variants=list(VARIANTS), regressions=list(REGRESSIONS)),
         "qc": {k: dict(median=float(np.median([q[k] for q in qcs])), min=float(np.min([q[k] for q in qcs])), max=float(np.max([q[k] for q in qcs])))
                for k in ("dice_aff", "dice_syn", "gmfrac_aff", "gmfrac_syn", "prior_gm_aff", "prior_gm_syn", "n_aff", "n_syn", "n_wm", "n_csf")}}
    print(json.dumps(R["meta"]), flush=True)
    print(json.dumps(R["qc"], indent=0), flush=True)
    P = {"nonbold": e.loso(nb, y, s, ST)}
    R["auc_ws"], R["increment"] = {}, {}
    for k, X in bold.items():
        P[k] = e.loso(X, y, s, ST)
        P["nonbold+" + k] = e.loso(np.hstack([nb, X]), y, s, ST)
        R["auc_ws"][k] = {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P[k][:, i], s).values()))) for i, st in enumerate(ST)}
        inc = {st: e.increment((y == st).astype(int), P["nonbold+" + k][:, i], P["nonbold"][:, i], s) for i, st in enumerate(ST)}
        for st in ST:
            inc[st].pop("boot_ws")
        R["increment"][k] = inc
        print(f"  {k:24s} alone N3 {R['auc_ws'][k]['N3']:.3f} REM {R['auc_ws'][k]['REM']:.3f} | gain N3 {inc['N3']['ws_mean']:+.3f} "
              f"[{inc['N3']['ws_ci'][0]:+.3f},{inc['N3']['ws_ci'][1]:+.3f}] {inc['N3']['n_pos']}/{inc['N3']['n_subj']}  "
              f"REM {inc['REM']['ws_mean']:+.3f} up {inc['REM']['ws_upper95']:+.3f}", flush=True)
    R["auc_ws"]["nonbold"] = {st: float(np.mean(list(e.ws_auc((y == st).astype(int), P["nonbold"][:, i], s).values()))) for i, st in enumerate(ST)}
    tag = f"_part{chunk.replace('/', 'of')}" if chunk else ""
    (BASE / f"results_preproc_ladder{tag}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / f"predictions_preproc_ladder{tag}.npz", y=y, subj=s, run=Dr["run"][ir], epoch=Dr["epoch"][ir],
                        **{"proba__" + k: v.astype(np.float32) for k, v in P.items()})
    print(f"saved results_preproc_ladder{tag}.json, predictions_preproc_ladder{tag}.npz")


def merge():
    parts = sorted(BASE.glob("results_preproc_ladder_part*.json"))
    R = json.load(open(parts[0]))
    Z = dict(np.load(BASE / parts[0].name.replace("results", "predictions").replace(".json", ".npz"), allow_pickle=True))
    for p in parts[1:]:
        r = json.load(open(p))
        assert r["meta"]["n_epochs"] == R["meta"]["n_epochs"]
        R["auc_ws"].update(r["auc_ws"]); R["increment"].update(r["increment"])
        z = np.load(BASE / p.name.replace("results", "predictions").replace(".json", ".npz"), allow_pickle=True)
        assert (z["y"] == Z["y"]).all() and np.allclose(z["proba__nonbold"], Z["proba__nonbold"], atol=1e-6)
        Z.update({k: z[k] for k in z.files if k.startswith("proba__")})
    (BASE / "results_preproc_ladder.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(BASE / "predictions_preproc_ladder.npz", **Z)
    print(f"merged {len(parts)} parts: {len(R['increment'])} models -> results_preproc_ladder.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--gains", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--runs-per-task", type=int, default=0, help="split a subject's remaining runs into tasks of this size (0 = one task per subject)")
    ap.add_argument("--chunk", default=None, help="gains: 'i/n' computes the i-th of n chunks of the model list (writes a part file)")
    ap.add_argument("--merge", action="store_true", help="gains: merge the part files into results_preproc_ladder.json")
    a = ap.parse_args()
    if a.sample:
        sample(a)
    if a.features:
        features()
    if a.gains:
        gains(a.chunk)
    if a.merge:
        merge()


if __name__ == "__main__":
    main()
