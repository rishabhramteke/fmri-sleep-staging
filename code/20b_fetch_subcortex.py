"""Stream the AFNI derivative again, reducing each volume to the 15 subcortical structures (19b).

Same streaming as 20_fetch_afni_deriv.py (its stream_run, one volume at a time from S3 through gzip);
only the reduction operator differs. The censor mask is taken from the cortical run file written by
20, so both blocks share it exactly.

    source <venv>/bin/activate
    python 20b_fetch_subcortex.py [--workers 6]   # writes derivatives/afni_ds005127_subc/<sub>/<run>_subc.npz
"""

import argparse
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE = Path(__file__).parent
CORT = BASE / "derivatives" / "afni_ds005127"
OUT = BASE / "derivatives" / "afni_ds005127_subc"
_s = importlib.util.spec_from_file_location("f20", BASE / "20_fetch_afni_deriv.py")
f20 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f20)


def operator():
    S = np.load(CORT / "subcortex_tt.npz")
    flat = S["atlas"].ravel(order="F")
    n = len(S["names"])
    rows, cols, vals = [], [], []
    for k in range(1, n + 1):
        idx = np.flatnonzero(flat == k)
        rows += [k - 1] * len(idx); cols += list(idx); vals += [1.0 / len(idx)] * len(idx)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, flat.size)), S["names"]


def one(key, W, names, shape):
    stem = key.rsplit("/", 1)[-1].replace("_bold.nii.gz", "")
    sub = stem.split("_")[0]
    dest = OUT / sub / f"{stem}_subc.npz"
    claim = dest.with_suffix(".claim")
    if dest.exists() or claim.exists():
        return stem, "skip", 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    claim.touch()                                        # lets a second process skip a run in progress
    t0 = time.time()
    sig, _, nt = f20.stream_run(key, W, np.array([], int), shape)
    cort = np.load(CORT / sub / f"{stem}_afni.npz")
    assert int(cort["n_vol"]) == nt
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.npz")
    np.savez_compressed(tmp, ts=sig, names=names, censored=cort["censored"], n_vol=nt, tr=3.0)
    tmp.rename(dest)
    claim.unlink(missing_ok=True)
    return stem, f"ok T={nt}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smallest-first", action="store_true",
                    help="second process working from the other end of the size-sorted list (both skip finished runs)")
    args = ap.parse_args()
    W, names = operator()
    shape = tuple(np.load(CORT / "atlas_tt.npz")["grid_shape"])
    have = {p.name.replace("_afni.npz", "") for p in CORT.rglob("*_afni.npz")}
    listing = dict(f20.fetch_mod.s3_list("ds005127/derivatives/fmri_ric2alreg/")[0])
    keys = sorted((k for k in listing if k.rsplit("/", 1)[-1].replace("_bold.nii.gz", "") in have),
                  key=lambda k: listing[k] if args.smallest_first else -listing[k])
    print(f"{len(keys)} runs, {sum(listing[k] for k in keys) / 1e9:.0f} GB, {len(names)} structures", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, k, W, names, shape): k for k in keys}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                stem, status, dt = f.result()
                print(f"[{i}/{len(keys)}] {status} {stem} {dt:.0f}s", flush=True)
            except Exception as e:
                print(f"[{i}/{len(keys)}] FAIL {futs[f]}: {e}", flush=True)


if __name__ == "__main__":
    main()
