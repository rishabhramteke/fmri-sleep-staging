"""Carry Harvard-Oxford subcortical structures into TT_N27 with the affine already estimated by 19.

The Schaefer-200 atlas is cortical only, so thalamus, striatum, hippocampus, amygdala and brainstem
(central to REM physiology) are not represented in the main BOLD block. This takes the 15 grey-matter
structures of the Harvard-Oxford subcortical atlas (maximum-probability, 25% threshold, MNI152 2 mm),
maps them with the MNI -> TT_N27 affine stored in atlas_tt.npz (nearest neighbour), and keeps voxels
inside the EPI mask and outside the cortical parcels.

    source <venv>/bin/activate
    python 19b_subcortex_to_tt.py             # writes derivatives/afni_ds005127/subcortex_tt.npz
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
from nilearn import datasets

BASE = Path(__file__).parent
OUT = BASE / "derivatives" / "afni_ds005127"
KEEP = ["Left Thalamus", "Left Caudate", "Left Putamen", "Left Pallidum", "Brain-Stem", "Left Hippocampus",
        "Left Amygdala", "Left Accumbens", "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
        "Right Hippocampus", "Right Amygdala", "Right Accumbens"]


def main():
    A = np.load(OUT / "atlas_tt.npz")
    shape, grid_aff = tuple(A["grid_shape"]), A["grid_affine"]
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    img = ho["maps"] if isinstance(ho["maps"], nib.Nifti1Image) else nib.load(ho["maps"])
    labels = [str(l) for l in ho["labels"]]
    data = np.asarray(img.dataobj).astype(np.int16)
    amap = AffineMap(A["affine_mni_to_tt"], shape, grid_aff, data.shape, img.affine)
    tt = amap.transform(data.astype(np.float32), interpolation="nearest").astype(np.int16)
    out = np.zeros(shape, np.int16)
    sizes = []
    for j, name in enumerate(KEEP, 1):
        m = (tt == labels.index(name)) & A["epi_mask"].astype(bool) & (A["atlas"] == 0)
        out[m] = j
        sizes.append(int(m.sum()))
        print(f"  {j:2d} {name:18s} {m.sum():5d} voxels", flush=True)
    np.savez_compressed(OUT / "subcortex_tt.npz", atlas=out, names=np.array(KEEP), sizes=np.array(sizes))
    print("saved subcortex_tt.npz")


if __name__ == "__main__":
    main()
