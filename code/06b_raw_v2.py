"""Own pipeline, version 2 (post hoc): realigned, despiked and motion-censored, with no physiological correction, plus
the eye signal. Answers two review points:
  * the minimal pipeline's REM control was uninformative, because that stream cannot detect REM at all (BOLD-alone
    REM AUC 0.50); a stream that sees REM WITHOUT any physiological regression is the informative control;
  * the eyes: rapid eye movements are REM's defining sign, and EPI images the eyeballs when the field of view
    reaches them (MR eye tracking, Frey et al. 2021).

Per run (06_preprocess registration chain, recomputed because its affines were not saved):
  1. Schaefer-200 labels in native EPI (mean EPI -> T1 rigid -> MNI affine, dipy MI), as 06_preprocess;
  2. every labelled voxel and every eye voxel is sampled in each volume through that volume's rigid realignment
     (23_realign.py parameters, the same convention: volume sampled at R x + t about the volume centre), trilinear;
  3. parcel means; despiking per parcel (running median of 9 frames, |residual| > 5 robust s.d. replaced);
  4. censoring by framewise displacement > 0.5 mm (Power FD; about AFNI's 0.3 mm motion norm) (the frame and the one before), a motion-only rule;
  5. eyes: bright (vitreous) components of the mean EPI outside the dilated brain, anterior and inferior to the
     brain; eye signals = mean, first two principal components of the voxel series, left-right difference.
No slice-timing correction, no nonlinear warp, no nuisance regression (22_features_v2.py --source rv2 adds the
physiological variants: RETROICOR only, as the derivative did, and RETROICOR + slow HR/RV/pulse regressors).

    source <venv>/bin/activate
    python 06b_raw_v2.py --workers 3           # writes derivatives/rv2_ds005127/<sub>/<run>_rv2.npz (~2-3 h)
    python 06b_raw_v2.py --qc                  # eye-mask QC montage -> derivatives/rv2_ds005127/eye_qc.png
"""
import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BASE = Path(__file__).parent
RAW = BASE / "data" / "ds005127"
MOT = BASE / "derivatives" / "motion_ds005127"
OUT = BASE / "derivatives" / "rv2_ds005127"
FD_CENSOR = 0.5


def _pp():
    s = importlib.util.spec_from_file_location("pp", BASE / "06_preprocess.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def rotm(p):
    """23_realign.rot_matrix in numpy: R = Rz Ry Rx, translation t (mm)."""
    tx, ty, tz, rx, ry, rz = p
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.array([tx, ty, tz])


def sample_realigned(data, ijk, params, zooms):
    """Voxel series at reference-grid voxels ijk (N, 3), each volume sampled through its own rigid transform."""
    shape = np.array(data.shape[:3])
    centre = (shape - 1) / 2
    z = np.array(zooms, float)
    x = (ijk - centre) * z
    out = np.empty((len(ijk), data.shape[3]), np.float32)
    for t in range(data.shape[3]):
        R, tr = rotm(params[t])
        q = (x @ R.T + tr) / z + centre
        out[:, t] = ndimage.map_coordinates(data[..., t], q.T, order=1, mode="nearest")
    return out


def despike(ts, size=9, k=5.0):
    out = ts.copy()
    for p in range(ts.shape[0]):
        x = ts[p]
        if not np.isfinite(x).all():
            continue
        med = ndimage.median_filter(x, size=size, mode="nearest")
        r = x - med
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        bad = np.abs(r) > k * mad
        out[p, bad] = med[bad]
    return out


def find_eyes(mf, labels, aff):
    """Eyeballs: bright (> half the 99th intensity percentile) connected components of the mean EPI other than the
    brain (the largest one), whose centroid lies > 50 mm anterior to, below, and 15-50 mm lateral of the brain's;
    up to the two largest of >= 8 voxels. Independent of the atlas labels."""
    bright = mf > 0.5 * np.percentile(mf, 99)
    lab, n = ndimage.label(bright, structure=np.ones((3, 3, 3)))
    ijk = np.indices(mf.shape).reshape(3, -1).T
    W = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    if n == 0:
        return np.zeros(mf.shape, bool), W.mean(0)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    brain_id = int(np.argmax(sizes)) + 1
    cb = W[lab.ravel() == brain_id].mean(0)
    flat = lab.ravel()
    ok = []
    for i in np.argsort(sizes)[::-1]:
        cid = i + 1
        if cid == brain_id or sizes[i] < 8:
            continue
        rel = W[flat == cid].mean(0) - cb
        if rel[1] > 50 and rel[2] < 0 and 15 <= abs(rel[0]) <= 50:
            ok.append(cid)
        if len(ok) == 2:
            break
    return np.isin(lab, ok), cb


def eye_signals(E, valid, eye_ijk, aff, cb):
    """Mean, first two PCs (voxel series z-scored over valid frames) and left-right difference, as 4 x T."""
    T = E.shape[1]
    out = np.full((4, T), np.nan, np.float32)
    if len(E) < 15:
        return out
    mu = E[:, valid].mean(1, keepdims=True)
    psc = 100 * (E - mu) / (np.abs(mu) + 1e-6)
    out[0] = psc.mean(0)
    Zs = (psc - psc[:, valid].mean(1, keepdims=True)) / (psc[:, valid].std(1, keepdims=True) + 1e-6)
    U, S, Vt = np.linalg.svd(Zs[:, valid], full_matrices=False)
    comps = U[:, :2].T @ Zs                                   # project all frames on the valid-frame PCs
    out[1:3] = comps / np.sqrt(len(E))
    xw = (aff[:3, :3] @ eye_ijk.T).T[:, 0] + aff[0, 3]
    L, Rr = xw < cb[0], xw >= cb[0]
    if L.sum() >= 5 and Rr.sum() >= 5:
        out[3] = psc[L].mean(0) - psc[Rr].mean(0)
    return out


def process_subject(sub, max_runs=0):
    pp = _pp()
    from nilearn.datasets import fetch_atlas_schaefer_2018, load_mni152_template
    funcs = sorted(RAW.glob(f"{sub}/**/func/*task-sleep*.nii.gz"))
    t1s = sorted(RAW.glob(f"{sub}/**/anat/*T1w.nii.gz"))
    tmpl_img = load_mni152_template(resolution=2)
    template = np.asanyarray(tmpl_img.dataobj, dtype=np.float32)
    atlas_img = nib.load(fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)["maps"])
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    todo = []
    for f in funcs:
        stem = f.name.replace("_bold.nii.gz", "")
        if not (OUT / sub / f"{stem}_rv2.npz").exists() and (MOT / sub / f"{stem}_motion.npz").exists():
            todo.append(f)
    if max_runs:
        todo = todo[:max_runs]
    if not todo:
        return sub, "nothing to do"
    t1 = pp.select_t1(t1s, todo[0], template, tmpl_img, atlas_img)
    cache, anchors, log = {}, {}, []
    for f in todo:
        t0 = time.time()
        stem = f.name.replace("_bold.nii.gz", "")
        M = np.load(MOT / sub / f"{stem}_motion.npz")
        mf, faff, fimg = pp.mean_epi(f)
        ses = next((p for p in f.name.split("_") if p.startswith("ses-")), "ses-?")
        warm = anchors.get(ses)
        nfm, new_aff = pp.build_chain(mf, faff, t1, template, tmpl_img.affine, cache=cache, warm=warm)
        labels = pp.atlas_to_native(atlas_img, nfm, mf.shape, faff)
        n_present = len(np.unique(labels[labels > 0]))
        if n_present < pp.MIN_PARCELS and warm is not None:
            nfm2, aff2 = pp.build_chain(mf, faff, t1, template, tmpl_img.affine, cache=cache, warm=None)
            lab2 = pp.atlas_to_native(atlas_img, nfm2, mf.shape, faff)
            if len(np.unique(lab2[lab2 > 0])) > n_present:
                labels, new_aff = lab2, aff2
                n_present = len(np.unique(labels[labels > 0]))
        if n_present >= pp.MIN_PARCELS and ses not in anchors:
            anchors[ses] = new_aff
        eye, cb = find_eyes(mf, labels, faff)
        data = np.asanyarray(fimg.dataobj, dtype=np.float32)
        T = data.shape[3]
        params = M["params"].astype(np.float64)
        if len(params) != T:
            log.append(f"{stem}: motion length {len(params)} != {T}, skipped")
            continue
        zooms = fimg.header.get_zooms()[:3]
        lab_ijk = np.argwhere(labels > 0)
        eye_ijk = np.argwhere(eye)
        V = sample_realigned(data, np.vstack([lab_ijk, eye_ijk]), params, zooms)
        del data
        lv = labels[tuple(lab_ijk.T)].astype(int) - 1
        sizes = np.bincount(lv, minlength=200)
        ts = np.full((200, T), np.nan, np.float32)
        acc = np.zeros((200, T))
        np.add.at(acc, lv, V[:len(lab_ijk)])
        ts[sizes > 0] = (acc[sizes > 0] / sizes[sizes > 0, None]).astype(np.float32)
        ts = despike(ts)
        fd = M["fd"].astype(np.float64)
        cens = fd > FD_CENSOR
        cens = cens | np.r_[cens[1:], False]
        E = V[len(lab_ijk):]
        eye_sig = eye_signals(E, ~cens, eye_ijk, faff, cb)
        np.savez_compressed(OUT / sub / f"{stem}_rv2.npz", ts=ts, censored=cens, parcel_sizes=sizes,
                            n_parcels_present=n_present, registration_ok=bool(n_present >= pp.MIN_PARCELS),
                            eye=eye_sig, eye_nvox=int(len(eye_ijk)), eye_ijk=eye_ijk.astype(np.int16),
                            tr=float(fimg.header.get_zooms()[3]), n_vol=T)
        log.append(f"{stem}: {n_present}/200 parcels, eye voxels {len(eye_ijk)}, censored {cens.mean():.3f}, "
                   f"{time.time() - t0:.0f} s")
        print(f"  {sub} {log[-1]}", flush=True)
    return sub, "; ".join(log[-2:])


def qc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = sorted(p.name for p in OUT.glob("sub-*"))
    fig, ax = plt.subplots(2, (len(subs) + 1) // 2, figsize=(2.2 * ((len(subs) + 1) // 2), 4.6))
    ax = ax.ravel()
    for a in ax:
        a.axis("off")
    for k, sub in enumerate(subs):
        files = sorted((OUT / sub).glob("*_rv2.npz"))
        if not files:
            continue
        nv = [int(np.load(f)["eye_nvox"]) for f in files]
        f = files[int(np.argmax(nv))]
        stem = f.name.replace("_rv2.npz", "")
        raw = next(RAW.glob(f"{sub}/**/func/{stem}_bold.nii.gz"))
        img = nib.load(raw)
        mf = np.asanyarray(img.dataobj[..., 0], dtype=np.float32)
        ijk = np.load(f)["eye_ijk"]
        if len(ijk):
            j = int(np.median(ijk[:, 1]))
            sl = mf[:, j, :]
            m = np.zeros_like(sl, bool)
            sel = ijk[ijk[:, 1] == j]
            m[sel[:, 0], sel[:, 2]] = True
        else:
            sl, m = mf[:, mf.shape[1] // 2 + 13, :], np.zeros(mf[:, 0, :].shape, bool)
        ax[k].imshow(sl.T, origin="lower", cmap="gray")
        ax[k].imshow(np.ma.masked_where(~m.T, m.T), origin="lower", cmap="autumn", alpha=0.8)
        ax[k].set_title(f"{sub}\neye vox: median {int(np.median(nv))}, max {max(nv)}", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "eye_qc.png", dpi=110)
    print("saved", OUT / "eye_qc.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--subjects", default=None, help="comma-separated list")
    ap.add_argument("--qc", action="store_true")
    a = ap.parse_args()
    if a.qc:
        return qc()
    subs = ([a.subject] if a.subject else a.subjects.split(",") if a.subjects
            else sorted(p.name for p in RAW.glob("sub-*") if p.is_dir()))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_subject, s): s for s in subs}
        for fu in as_completed(futs):
            try:
                s, msg = fu.result()
                print(f"[done {s}] {msg} ({time.time() - t0:.0f} s)", flush=True)
            except Exception as err:
                print(f"[FAILED {futs[fu]}] {err!r}", flush=True)


if __name__ == "__main__":
    main()
