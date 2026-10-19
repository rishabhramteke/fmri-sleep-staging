"""Lightweight preprocessing: bring the Schaefer-200 atlas into each run's native EPI space.

No FSL/ANTs/AFNI/Docker on this machine, so registration is done with dipy's mutual-information
affine registration. The design choice that makes this tractable: instead of resampling every 4D
series into MNI (943 volumes x 96x72x50 per ds005127 run), we register ONCE per run and carry the
ATLAS backwards into native space, then extract 200 parcel timeseries. Output per run is ~200 x T
floats, a few hundred KB instead of gigabytes.

Chain:  mean EPI  --rigid-->  subject T1  --affine-->  MNI152
        atlas (MNI) --inverse--> native EPI, nearest-neighbour

    source <venv>/bin/activate
    python 06_preprocess.py --subject sub-00003 --dataset ds005127 --runs 1
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric
from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D
from nilearn.datasets import fetch_atlas_schaefer_2018, load_mni152_template
from scipy import sparse

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "derivatives"
LEVELS = [1000, 500, 200]            # iterations per resolution level
SIGMAS = [3.0, 1.0, 0.0]
FACTORS = [4, 2, 1]
MIN_PARCELS = 150                    # below this, registration is treated as failed


def mean_epi(path, max_vols=200):
    """Mean of up to max_vols evenly spaced volumes. Loading 943 volumes to average is wasteful."""
    img = nib.load(str(path))
    n = img.shape[3]
    idx = np.linspace(0, n - 1, min(max_vols, n)).astype(int)
    acc = None
    for i in idx:
        v = np.asanyarray(img.dataobj[..., i], dtype=np.float32)
        acc = v if acc is None else acc + v
    return acc / len(idx), img.affine, img


def register(static, static_aff, moving, moving_aff, transform, starting=None):
    # sampling_proportion=None evaluates mutual information over EVERY voxel. Combined with a
    # 10,000-iteration coarse level that produced 2-4 hour registrations when the starting affine
    # was poor. Subsampling to 25% is standard practice and costs nothing in accuracy here.
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.25)
    areg = AffineRegistration(metric=metric, level_iters=LEVELS, sigmas=SIGMAS, factors=FACTORS,
                              verbosity=0)
    return areg.optimize(static, moving, transform, None, static_aff, moving_aff,
                         starting_affine=starting)


def build_chain(mean_func, func_aff, t1_path, template, template_aff, cache=None, warm=None):
    """mean EPI -> T1 (rigid) -> MNI (affine). Returns the composed native<-MNI mapping.

    T1->MNI is identical for every run of a subject, so it is computed once and cached. EPI->T1 is
    warm-started from the previous run's solution, which converges far faster than from scratch.
    """
    t1_img = nib.load(str(t1_path))
    t1 = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    t1_aff = t1_img.affine

    t0 = time.time()
    # EPI -> T1. Cross-modality, so mutual information and a coarse-to-fine warm start.
    if warm is None:
        start = register(t1, t1_aff, mean_func, func_aff, TranslationTransform3D()).affine
    else:
        start = warm
    epi2t1 = register(t1, t1_aff, mean_func, func_aff, RigidTransform3D(), start)
    print(f"    EPI->T1 rigid done in {time.time() - t0:.0f}s"
          f"{' (warm-started)' if warm is not None else ''}", flush=True)

    if cache is not None and "t12mni" in cache:
        t12mni_aff = cache["t12mni"]
        print("    T1->MNI reused from cache", flush=True)
    else:
        t0 = time.time()
        tr2 = register(template, template_aff, t1, t1_aff, TranslationTransform3D())
        t12mni_aff = register(template, template_aff, t1, t1_aff, AffineTransform3D(), tr2.affine).affine
        print(f"    T1->MNI affine done in {time.time() - t0:.0f}s", flush=True)
        if cache is not None:
            cache["t12mni"] = t12mni_aff

    native_from_mni = np.linalg.inv(epi2t1.affine) @ np.linalg.inv(t12mni_aff)
    return native_from_mni, epi2t1.affine


def atlas_to_native(atlas_img, native_from_mni, func_shape, func_aff):
    amap = AffineMap(native_from_mni,
                     domain_grid_shape=func_shape, domain_grid2world=func_aff,
                     codomain_grid_shape=atlas_img.shape, codomain_grid2world=atlas_img.affine)
    atlas = np.asanyarray(atlas_img.dataobj, dtype=np.float32)
    return np.rint(amap.transform(atlas, interpolation="nearest")).astype(np.int16)


def extract_all(func_path, labels, n_parcels=200):
    """Parcel timeseries + motion proxies in ONE pass over the data.

    The first version streamed the 4D file twice and looped 200 parcels per volume, costing ~21 min
    per run, which is 94 hours across ds005127. This loads the series once (about 1.3 GB for a
    943-volume ds005127 run) and reduces parcels with a single sparse matrix product, which is one
    BLAS call instead of 188,600 fancy-index operations.

    Motion is a translation-only centre-of-mass proxy plus DVARS. Adequate as a nuisance regressor
    and for flagging high-motion epochs; NOT a substitute for realignment if voxelwise results are
    ever claimed (there is no mcflirt/3dvolreg on this machine).
    """
    img = nib.load(str(func_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)          # x,y,z,t
    nx, ny, nz, nt = data.shape
    flat = data.reshape(-1, nt)                                   # voxels x time

    lab = labels.ravel()
    keep = lab > 0
    rows = lab[keep].astype(np.int64) - 1
    cols = np.flatnonzero(keep)
    sizes = np.bincount(rows, minlength=n_parcels).astype(np.float32)
    weights = np.where(sizes[rows] > 0, 1.0 / sizes[rows], 0.0).astype(np.float32)
    M = sparse.csr_matrix((weights, (rows, cols)), shape=(n_parcels, flat.shape[0]))
    ts = np.asarray(M @ flat, dtype=np.float32)                   # parcels x time
    ts[sizes == 0] = np.nan

    # DVARS: RMS frame-to-frame change over in-brain voxels
    brain = flat[keep]
    dv = np.concatenate([[0.0], np.sqrt((np.diff(brain, axis=1) ** 2).mean(axis=0))])

    # centre of mass per volume, vectorised over the three axes
    tot = flat.sum(axis=0) + 1e-9
    gx = np.arange(nx, dtype=np.float32)
    gy = np.arange(ny, dtype=np.float32)
    gz = np.arange(nz, dtype=np.float32)
    cx = (data.sum(axis=(1, 2)) * gx[:, None]).sum(axis=0) / tot
    cy = (data.sum(axis=(0, 2)) * gy[:, None]).sum(axis=0) / tot
    cz = (data.sum(axis=(0, 1)) * gz[:, None]).sum(axis=0) / tot
    com = np.stack([cx, cy, cz], axis=1)
    fd = np.concatenate([[0.0], np.abs(np.diff(com, axis=0)).sum(axis=1)])

    return ts, sizes.astype(int).tolist(), fd.astype(np.float32), dv.astype(np.float32)


def select_t1(t1s, probe_func, template, tmpl_img, atlas_img):
    """Pick the T1 that gives the best atlas coverage, not simply the first on disk.

    Subjects here can have several T1s of very different quality: sub-00065's ses-0 mprage has a
    median intensity of 120 against a p99 of 3586 (heavy background), and every one of its 21 runs
    failed registration because that scan was chosen alphabetically. Registration now costs ~30 s, so
    trying each candidate once per subject is cheap insurance.
    """
    if len(t1s) == 1:
        print(f"  T1: {t1s[0].name} (only candidate)", flush=True)
        return t1s[0]
    mf, faff, _ = mean_epi(probe_func)
    best, best_n = None, -1
    for cand in t1s:
        try:
            nfm, _ = build_chain(mf, faff, cand, template, tmpl_img.affine, cache=None, warm=None)
            lab = atlas_to_native(atlas_img, nfm, mf.shape, faff)
            n = len(np.unique(lab[lab > 0]))
        except Exception as e:
            print(f"  T1 candidate {cand.name}: registration errored ({e})", flush=True)
            continue
        print(f"  T1 candidate {cand.name}: {n}/200 parcels", flush=True)
        if n > best_n:
            best, best_n = cand, n
    print(f"  T1 chosen: {best.name} ({best_n}/200)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ds005127")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--runs", type=int, default=1, help="how many runs to process (0 = all)")
    ap.add_argument("--task", default="sleep")
    args = ap.parse_args()

    root = DATA / args.dataset
    funcs = sorted(root.glob(f"{args.subject}/**/func/*task-{args.task}*.nii.gz"))
    t1s = sorted(root.glob(f"{args.subject}/**/anat/*T1w.nii.gz"))
    if not funcs or not t1s:
        raise SystemExit(f"missing data: {len(funcs)} func, {len(t1s)} T1 for {args.subject}")
    if args.runs:
        funcs = funcs[:args.runs]
    print(f"{args.subject}: {len(funcs)} run(s) to process, {len(t1s)} T1 candidate(s)", flush=True)

    tmpl_img = load_mni152_template(resolution=2)
    template = np.asanyarray(tmpl_img.dataobj, dtype=np.float32)
    atlas_img = nib.load(fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)["maps"])

    outdir = OUT / args.dataset / args.subject
    outdir.mkdir(parents=True, exist_ok=True)

    cache, anchors = {}, {}
    t1 = select_t1(t1s, funcs[0], template, tmpl_img, atlas_img)
    for f in funcs:
        stem = f.name.replace("_bold.nii.gz", "")
        dest = outdir / f"{stem}_parcels.npz"
        if dest.exists():
            print(f"  {stem}: already done, skipping", flush=True)
            continue
        print(f"  {stem}", flush=True)
        t_start = time.time()
        mf, faff, fimg = mean_epi(f)
        # Anchor per session: chaining each run off the previous one let the affine drift, and each
        # degraded seed cost the optimizer more iterations to recover (47s -> 8033s within one
        # subject). Anchoring to the first GOOD affine of the session stops the feedback loop.
        ses = next((p for p in f.name.split("_") if p.startswith("ses-")), "ses-?")
        warm = anchors.get(ses)
        native_from_mni, new_aff = build_chain(mf, faff, t1, template, tmpl_img.affine,
                                               cache=cache, warm=warm)
        labels = atlas_to_native(atlas_img, native_from_mni, mf.shape, faff)
        covered = int((labels > 0).sum())
        n_present = len(np.unique(labels[labels > 0]))
        print(f"    atlas in native space: {covered} voxels, {n_present}/200 parcels present", flush=True)

        # A bad warm start propagates: one failed registration becomes the starting affine for the
        # next run, which then fails identically. Retry cold, and never pass a bad affine forward.
        if n_present < MIN_PARCELS and warm is not None:
            print(f"    only {n_present}/200 parcels: warm start suspected bad, retrying cold",
                  flush=True)
            native_from_mni, warm_retry = build_chain(mf, faff, t1, template, tmpl_img.affine,
                                                      cache=cache, warm=None)
            labels_retry = atlas_to_native(atlas_img, native_from_mni, mf.shape, faff)
            n_retry = len(np.unique(labels_retry[labels_retry > 0]))
            print(f"    cold retry: {n_retry}/200 parcels present", flush=True)
            if n_retry > n_present:
                labels, n_present, new_aff = labels_retry, n_retry, warm_retry
                covered = int((labels > 0).sum())
        if n_present < MIN_PARCELS:
            print(f"    WARNING: only {n_present}/200 parcels after retry. Registration failed; "
                  f"this run is flagged and will not seed the anchor.", flush=True)
        elif ses not in anchors:
            anchors[ses] = new_aff        # first good affine of this session becomes the anchor
        ts, sizes, fd, dv = extract_all(f, labels)
        np.savez_compressed(dest, ts=ts, parcel_sizes=np.array(sizes), fd=fd, dvars=dv,
                            tr=float(fimg.header.get_zooms()[3]), n_parcels_present=n_present,
                            registration_ok=bool(n_present >= MIN_PARCELS))
        print(f"    saved {dest.name}: {ts.shape} in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
