"""Stream the dataset authors' AFNI derivative (fmri_ric2alreg) and reduce it to parcel signals.

The derivative is the authors' own preprocessing: 3dDespike, slice-wise RETROICOR (3dREMLfit with
slice-based regressors), 3dTshift (alt+z2), motion correction combined with a nonlinear warp to TT_N27
at 2 mm (3dNwarpApply), then 3dTproject with motion censoring and nuisance regression; the published
image is residuals + temporal mean. It fixes what our own minimal pipeline lacks (realignment,
slice timing, slice-wise physiological correction) with choices made by people who know the data.

276.5 GB will not fit, and does not need to: each run is streamed from S3 straight through a gzip
decompressor, one volume at a time (NIfTI is x-fastest, so a volume is a contiguous block), and
reduced on the fly to Schaefer-200 parcel means (atlas from 19_atlas_to_tt.py), the global signal,
white-matter and CSF means, and a 4,000-voxel white-matter sample for aCompCor. Nothing large is
written to disk. Censored frames are recovered exactly: censored residuals are zero, so those frames
equal the temporal mean image.

    source <venv>/bin/activate
    python 20_fetch_afni_deriv.py [--workers 6]
"""

import argparse
import gzip
import importlib.util
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE = Path(__file__).parent
OUT = BASE / "derivatives" / "afni_ds005127"
RNG = np.random.default_rng(0)
N_WM_SAMPLE = 4000

_spec = importlib.util.spec_from_file_location("fetch", BASE / "03_fetch_sleep_scoring.py")
fetch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_mod)


def build_operators():
    A = np.load(OUT / "atlas_tt.npz")
    atlas, wm, csf, mask = A["atlas"], A["wm"], A["csf"], A["epi_mask"]
    flat = atlas.ravel(order="F")                         # NIfTI voxel order: x fastest
    nvox = flat.size
    rows, cols, vals = [], [], []
    sizes = np.zeros(200, int)
    for k in range(1, 201):
        idx = np.flatnonzero(flat == k)
        sizes[k - 1] = len(idx)
        if len(idx):
            rows += [k - 1] * len(idx); cols += list(idx); vals += [1.0 / len(idx)] * len(idx)
    extra = [mask.ravel(order="F"), wm.ravel(order="F"), csf.ravel(order="F")]
    for j, m in enumerate(extra):
        idx = np.flatnonzero(m)
        rows += [200 + j] * len(idx); cols += list(idx); vals += [1.0 / len(idx)] * len(idx)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(203, nvox), dtype=np.float64)
    wm_idx = np.flatnonzero(wm.ravel(order="F"))
    wm_sample = np.sort(RNG.choice(wm_idx, min(N_WM_SAMPLE, len(wm_idx)), replace=False))
    return W, sizes, wm_sample, tuple(A["grid_shape"])


def stream_run(key, W, wm_sample, shape, attempts=4):
    url = f"{fetch_mod.BUCKET}/{urllib.parse.quote(key)}"
    for a in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp, gzip.GzipFile(fileobj=resp) as gz:
                hdr = gz.read(352)
                dim = np.frombuffer(hdr[40:56], np.int16)
                vox_offset = int(np.frombuffer(hdr[108:112], np.float32)[0])
                datatype = int(np.frombuffer(hdr[70:72], np.int16)[0])
                assert datatype == 16, f"expected float32, got datatype {datatype}"
                nx, ny, nz, nt = (int(x) for x in dim[1:5])
                assert (nx, ny, nz) == shape, f"grid {nx, ny, nz} != atlas grid {shape}"
                gz.read(vox_offset - 352)
                nbytes = nx * ny * nz * 4
                sig = np.empty((W.shape[0], nt), np.float32)      # 203 rows for the cortical run
                wms = np.empty((len(wm_sample), nt), np.float32)
                for t in range(nt):
                    buf = gz.read(nbytes)
                    if len(buf) != nbytes:
                        raise IOError(f"short read at volume {t}")
                    v = np.frombuffer(buf, np.float32).astype(np.float64)
                    sig[:, t] = W @ v
                    wms[:, t] = v[wm_sample]
                return sig, wms, nt
        except Exception:
            if a == attempts - 1:
                raise
            time.sleep(5 * (a + 1))


def one(key, W, sizes, wm_sample, shape):
    stem = key.rsplit("/", 1)[-1].replace("_bold.nii.gz", "")
    sub = stem.split("_")[0]
    dest = OUT / sub / f"{stem}_afni.npz"
    if dest.exists():
        return stem, "skip", 0.0
    t0 = time.time()
    sig, wms, nt = stream_run(key, W, wm_sample, shape)
    ts = sig[:200]
    # censored frames equal the temporal mean image exactly (residuals zeroed, then + mean)
    mu = ts.mean(1, keepdims=True)
    scale = ts.std(1, keepdims=True) + 1e-6
    censored = (np.abs(ts - mu) / scale).max(0) < 1e-4
    # aCompCor on the white-matter sample: top 5 PCs of uncensored, demeaned, variance-normalised series
    X = wms[:, ~censored]
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    U, S, Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    pcs = np.zeros((5, nt), np.float32)
    pcs[:, ~censored] = Vt[:5]
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, ts=ts, gs=sig[200], wm=sig[201], csf=sig[202], wm_pcs=pcs,
                        censored=censored, parcel_sizes=sizes, n_vol=nt, tr=3.0)
    return stem, f"ok T={nt} censored={censored.mean():.3f}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated run stems, for testing")
    args = ap.parse_args()
    W, sizes, wm_sample, shape = build_operators()
    keys, _ = fetch_mod.s3_list("ds005127/derivatives/fmri_ric2alreg/")
    used = set(np.load(BASE / "features_ds005127.npz", allow_pickle=True)["run"].astype(str))
    keys = [k for k, _ in keys if k.rsplit("/", 1)[-1].replace("_bold.nii.gz", "") in used]
    if args.only:
        keep = set(args.only.split(","))
        keys = [k for k in keys if k.rsplit("/", 1)[-1].replace("_bold.nii.gz", "") in keep]
    # longest first, so the tail of the job is not one 4 GB file
    sizes_s3 = dict(fetch_mod.s3_list("ds005127/derivatives/fmri_ric2alreg/")[0])
    keys.sort(key=lambda k: -sizes_s3.get(k, 0))
    print(f"{len(keys)} runs to process ({sum(sizes_s3[k] for k in keys) / 1e9:.0f} GB streamed)", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, k, W, sizes, wm_sample, shape): k for k in keys}
        for f in as_completed(futs):
            done += 1
            try:
                stem, status, dt = f.result()
                print(f"[{done}/{len(keys)}] {status} {stem} {dt:.0f}s", flush=True)
            except Exception as e:
                print(f"[{done}/{len(keys)}] FAIL {futs[f]}: {e}", flush=True)
    print(f"afni runs on disk: {len(list(OUT.rglob('*_afni.npz')))}")


if __name__ == "__main__":
    main()
