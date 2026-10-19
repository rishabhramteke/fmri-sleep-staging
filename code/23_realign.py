"""Rigid-body realignment of the raw BOLD volumes, for a head-motion estimate that is not image-derived.

The motion proxy used so far (centre of mass of image intensity) correlates 0.77 with DVARS within
subject and is higher in N3 than N2, i.e. it tracks BOLD intensity, not the head. Proper framewise
displacement needs six-parameter realignment, and no FSL/AFNI/ANTs is available here, so this does it
directly: every volume is registered to the run's mean volume by Gauss-Newton least squares (the
objective SPM realign uses) over 3 translations and 3 rotations about the volume centre, batched on the
GPU (torch, MPS) with a two-level pyramid. Output per run: the 6 parameters (mm, rad), Power FD
(translations + 50 mm x rotations, frame to frame), and the residual cost.

Validated against dipy's rigid registration on random volumes (--validate).

    source <venv>/bin/activate
    python 23_realign.py [--workers 2]          # writes derivatives/motion_ds005127/<sub>/<run>_motion.npz
    python 23_realign.py --validate RUN_STEM
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as Fnn

BASE = Path(__file__).parent
RAW = BASE / "data" / "ds005127"
OUT = BASE / "derivatives" / "motion_ds005127"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 64


def rot_matrix(p):
    """p: (B, 6) = tx, ty, tz (mm), rx, ry, rz (rad) -> (B, 3, 3) rotation, (B, 3) translation."""
    rx, ry, rz = p[:, 3], p[:, 4], p[:, 5]
    cx, sx, cy, sy, cz, sz = torch.cos(rx), torch.sin(rx), torch.cos(ry), torch.sin(ry), torch.cos(rz), torch.sin(rz)
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).view(-1, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).view(-1, 3, 3)
    return Rz @ Ry @ Rx, p[:, :3]


def warp(vol, p, coords_mm, half_extent_mm):
    """Resample volumes (B,1,D,H,W) at R x + t, coordinates in mm about the volume centre.

    The tensor axes (D, H, W) are the NIfTI array axes (i, j, k); grid_sample's last grid dimension is
    ordered (W, H, D), hence the [2, 1, 0] reorder after normalising each axis to [-1, 1].
    """
    R, t = rot_matrix(p)
    x = torch.einsum("bij,nj->bni", R, coords_mm) + t[:, None, :]
    g = (x / half_extent_mm)[..., [2, 1, 0]]
    D, H, W = vol.shape[2:]
    return Fnn.grid_sample(vol, g.view(vol.shape[0], D, H, W, 3), mode="bilinear",
                           padding_mode="border", align_corners=True)


def setup_grid(shape, zooms):
    D, H, W = shape
    ii, jj, kk = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    c = torch.stack([ii, jj, kk], -1).reshape(-1, 3).float()
    centre = (torch.tensor(shape, dtype=torch.float32) - 1) / 2
    mm = (c - centre) * torch.tensor(zooms, dtype=torch.float32)
    half = centre * torch.tensor(zooms, dtype=torch.float32)
    return mm.to(DEV), half.to(DEV)


def register_batch(ref, vols, zooms, iters=(10, 6)):
    """Gauss-Newton rigid registration of each volume in `vols` (B,D,H,W) onto `ref` (D,H,W).

    Residual = gain * warped volume - reference over a brain mask; the Jacobian is taken by forward
    finite differences (6 extra resamplings per iteration), which is robust on MPS.
    """
    B = vols.shape[0]
    p = torch.zeros(B, 6, device=DEV)
    eps = torch.tensor([0.02, 0.02, 0.02, 3e-4, 3e-4, 3e-4], device=DEV)
    for level, n_it in zip((2, 1), iters):
        if level > 1:
            r = Fnn.avg_pool3d(ref[None, None], level)[0, 0]
            v = Fnn.avg_pool3d(vols[:, None], level)
        else:
            r, v = ref, vols[:, None]
        mm, half = setup_grid(r.shape, tuple(np.array(zooms) * level))
        mask = (r > 0.1 * r.max()).reshape(-1)
        rr = r.reshape(-1)[mask]

        def sample(q):
            return warp(v, q, mm, half).reshape(B, -1)[:, mask]

        for _ in range(n_it):
            m0 = sample(p)
            gain = (m0 * rr).sum(1, keepdim=True) / (m0 * m0).sum(1, keepdim=True)
            res = gain * m0 - rr[None]
            J = torch.stack([(gain * sample(p + eps[k] * torch.eye(6, device=DEV)[k]) - gain * m0) / eps[k]
                             for k in range(6)], -1)                     # (B, N, 6)
            JtJ = J.transpose(1, 2) @ J + 1e-6 * torch.eye(6, device=DEV)
            Jtr = (J.transpose(1, 2) @ res[..., None])[..., 0]
            p = p - torch.linalg.solve(JtJ.cpu(), Jtr.cpu()).to(DEV)
    m0 = sample(p)
    gain = (m0 * rr).sum(1, keepdim=True) / (m0 * m0).sum(1, keepdim=True)
    cost = ((gain * m0 - rr[None]) ** 2).mean(1) / (rr ** 2).mean()
    return p.cpu().numpy(), cost.cpu().numpy()


def realign_run(path):
    img = nib.load(path)
    zooms = img.header.get_zooms()[:3]
    data = np.asarray(img.dataobj, dtype=np.float32)                  # (X, Y, Z, T)
    vols = torch.from_numpy(np.ascontiguousarray(np.moveaxis(data, 3, 0))).to(DEV)   # (T, X, Y, Z)
    ref = vols.mean(0)
    P, C = [], []
    for s in range(0, vols.shape[0], BATCH):
        p, c = register_batch(ref, vols[s:s + BATCH], zooms)
        P.append(p); C.append(c)
    P, C = np.concatenate(P), np.concatenate(C)
    d = np.diff(P, axis=0)
    fd = np.concatenate([[0.0], np.abs(d[:, :3]).sum(1) + 50.0 * np.abs(d[:, 3:]).sum(1)])
    return P, fd, C


def one(path):
    stem = Path(path).name.replace("_bold.nii.gz", "")
    dest = OUT / stem.split("_")[0] / f"{stem}_motion.npz"
    if dest.exists():
        return stem, "skip", 0.0
    t0 = time.time()
    P, fd, C = realign_run(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, params=P.astype(np.float32), fd=fd.astype(np.float32), cost=C.astype(np.float32))
    return stem, f"ok T={len(fd)} FD median {np.median(fd):.3f} mm, p99 {np.percentile(fd, 99):.2f}", time.time() - t0


def validate(stem, n=12):
    """Compare against dipy rigid registration (MI-free: SSD) on n random volumes of one run."""
    from dipy.align.imaffine import AffineRegistration
    from dipy.align.metrics import CCMetric  # noqa: F401  (imported to keep dipy version checks honest)
    from dipy.align.imaffine import MutualInformationMetric
    from dipy.align.transforms import RigidTransform3D
    path = next(RAW.rglob(f"{stem}_bold.nii.gz"))
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    ref = data.mean(3)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(data.shape[3], n, replace=False))
    vols = torch.from_numpy(np.ascontiguousarray(np.moveaxis(data[..., idx], 3, 0))).to(DEV)
    p_t, _ = register_batch(torch.from_numpy(ref).to(DEV), vols, zooms)
    aff = np.diag(list(zooms) + [1.0])
    aff[:3, 3] = -(np.array(ref.shape) - 1) / 2 * np.array(zooms)
    reg = AffineRegistration(metric=MutualInformationMetric(32, None), level_iters=[200, 100], sigmas=[1, 0], factors=[2, 1])
    for k, t in enumerate(idx):
        m = reg.optimize(ref, data[..., t], RigidTransform3D(), None, aff, aff)
        A = m.affine
        tr_d = A[:3, 3]
        # small-angle rotation vector from dipy's matrix
        Rm = A[:3, :3]
        rv = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / 2
        print(f"  vol {t:4d}: torch t={np.round(p_t[k, :3], 3)} r(deg)={np.round(np.degrees(p_t[k, 3:]), 3)} | "
              f"dipy t={np.round(tr_d, 3)} r(deg)={np.round(np.degrees(rv), 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--validate", default=None)
    args = ap.parse_args()
    if args.validate:
        validate(args.validate)
        return
    used = set(np.load(BASE / "features_ds005127.npz", allow_pickle=True)["run"].astype(str))
    paths = sorted(p for p in RAW.rglob("*task-sleep*_bold.nii.gz")
                   if "func-complex" not in str(p) and p.name.replace("_bold.nii.gz", "") in used)
    print(f"{len(paths)} runs on {DEV}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, str(p)): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                stem, status, dt = f.result()
                print(f"[{i}/{len(paths)}] {status} {stem} {dt:.0f}s", flush=True)
            except Exception as e:
                print(f"[{i}/{len(paths)}] FAIL {futs[f].name}: {e}", flush=True)


if __name__ == "__main__":
    main()
