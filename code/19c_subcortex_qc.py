"""QC of the subcortical masks (19b) on the derivative's own EPI: an overlay, and voxel retention per run.

19b keeps Harvard-Oxford voxels inside the EPI mask of the one run 19 registered to. This streams the first
N_VOL volumes of three derivative runs (three subjects, the smallest files) from OpenNeuro's S3 bucket, and
for each structure reports the share of its affine-mapped Harvard-Oxford voxels (outside the cortical
atlas) that lie inside that run's own intensity mask (same threshold as 19). A structure losing voxels in
every run is cut by the field of view or by susceptibility dropout; one losing them in only the masking run
was cut by that run. Writes derivatives/afni_ds005127/subcortex_tt_qc.png (three planes, first run) and
subcortex_tt_qc.json.

    source <venv>/bin/activate
    python 19c_subcortex_qc.py
"""

import gzip
import importlib.util
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
from nilearn import datasets

BASE = Path(__file__).parent
OUT = BASE / "derivatives" / "afni_ds005127"
N_VOL = 20
_s = importlib.util.spec_from_file_location("f20", BASE / "20_fetch_afni_deriv.py")
f20 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f20)


def first_volumes(key, shape):
    url = f"{f20.fetch_mod.BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=300) as resp, gzip.GzipFile(fileobj=resp) as gz:
        hdr = gz.read(352)
        dim = np.frombuffer(hdr[40:56], np.int16)
        vox_offset = int(np.frombuffer(hdr[108:112], np.float32)[0])
        assert int(np.frombuffer(hdr[70:72], np.int16)[0]) == 16
        assert tuple(int(x) for x in dim[1:4]) == shape
        gz.read(vox_offset - 352)
        n = int(np.prod(shape))
        vols = [np.frombuffer(gz.read(n * 4), np.float32).reshape(shape, order="F") for _ in range(N_VOL)]
    return np.mean(vols, 0)


def main():
    A = np.load(OUT / "atlas_tt.npz")
    S = np.load(OUT / "subcortex_tt.npz", allow_pickle=True)
    shape, grid_aff = tuple(int(x) for x in A["grid_shape"]), A["grid_affine"]
    names = [str(n) for n in S["names"]]
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    img = ho["maps"] if isinstance(ho["maps"], nib.Nifti1Image) else nib.load(ho["maps"])
    labels = [str(l) for l in ho["labels"]]
    amap = AffineMap(A["affine_mni_to_tt"], shape, grid_aff, img.shape, img.affine)
    tt = amap.transform(np.asarray(img.dataobj).astype(np.float32), interpolation="nearest").astype(np.int16)
    full = {n: (tt == labels.index(n)) & (A["atlas"] == 0) for n in names}
    # same row order as 20b: the runs analysed, one per subject, smallest files first
    have = {p.name.replace("_afni.npz", "") for p in OUT.rglob("*_afni.npz")}
    listing = dict(f20.fetch_mod.s3_list("ds005127/derivatives/fmri_ric2alreg/")[0])
    keys, seen = [], set()
    for k in sorted((k for k in listing if k.rsplit("/", 1)[-1].replace("_bold.nii.gz", "") in have), key=listing.get):
        sub = k.rsplit("/", 1)[-1].split("_")[0]
        if sub not in seen:
            keys.append(k); seen.add(sub)
        if len(keys) == 3:
            break
    R = {"n_vol": N_VOL, "runs": [], "retention_masking_run": {n: float(((S["atlas"] == j + 1).sum()) / full[n].sum())
                                                              for j, n in enumerate(names)}, "retention": {}}
    means = []
    for k in keys:
        m = first_volumes(k, shape)
        means.append(m)
        R["runs"].append(k.rsplit("/", 1)[-1])
        mask = m > 0.2 * np.percentile(m[m > 0], 98)
        for n in names:
            R["retention"].setdefault(n, []).append(float((full[n] & mask).sum() / full[n].sum()))
        print(f"  {R['runs'][-1]}: done", flush=True)
    for n in names:
        print(f"  {n:18s} masking run {R['retention_masking_run'][n]:.2f} | these runs "
              + " ".join(f"{v:.2f}" for v in R["retention"][n]))
    (OUT / "subcortex_tt_qc.json").write_text(json.dumps(R, indent=1))

    m, lab = means[0], S["atlas"]
    ctr = [int(np.median(np.nonzero(lab == names.index(n) + 1)[i])) for i, n in
           zip((0, 1, 2), ("Left Thalamus", "Left Amygdala", "Brain-Stem"))]
    cmap = plt.get_cmap("tab20", 16)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, (sl, title) in zip(ax, [((ctr[0], slice(None), slice(None)), "sagittal (left thalamus)"),
                                   ((slice(None), ctr[1], slice(None)), "coronal (left amygdala)"),
                                   ((slice(None), slice(None), ctr[2]), "axial (brainstem)")]):
        a.imshow(np.rot90(m[sl]), cmap="gray")
        L = np.rot90(lab[sl]).astype(float)
        a.imshow(np.ma.masked_where(L == 0, L), cmap=cmap, vmin=0, vmax=15, alpha=0.55, interpolation="nearest")
        a.set_title(title, fontsize=8); a.axis("off")
    fig.suptitle(f"Harvard-Oxford subcortex on TT_N27 (19b), mean of first {N_VOL} volumes of {R['runs'][0]}", fontsize=8)
    fig.savefig(OUT / "subcortex_tt_qc.png", dpi=130, bbox_inches="tight")
    print("saved subcortex_tt_qc.png and subcortex_tt_qc.json")


if __name__ == "__main__":
    main()
