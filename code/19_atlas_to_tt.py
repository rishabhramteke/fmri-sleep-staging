"""Carry the Schaefer-200 atlas (MNI152) into the TT_N27 space of the dataset authors' AFNI derivative.

derivatives/fmri_ric2alreg is already motion-corrected and nonlinearly warped to TT_N27 at 2 mm, so
one affine from MNI152 to TT_N27 serves every run. It is estimated once by mutual information
(dipy: translation -> rigid -> affine) between the MNI152 template and the temporal mean of a
derivative run, and applied with nearest-neighbour interpolation to the atlas and to MNI tissue masks.

    source <venv>/bin/activate
    python 19_atlas_to_tt.py <derivative_run.nii.gz>     # writes derivatives/afni_ds005127/atlas_tt.npz
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric, transform_centers_of_mass
from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D
from nilearn import datasets, image

BASE = Path(__file__).parent
OUT = BASE / "derivatives" / "afni_ds005127"


def main(run_path):
    img = nib.load(run_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    mean = data.mean(axis=3)
    mask = mean > 0.2 * np.percentile(mean[mean > 0], 98)
    tmpl = datasets.load_mni152_template(resolution=2)
    static, static_aff = mean, img.affine
    moving, moving_aff = np.asarray(tmpl.dataobj, np.float32), tmpl.affine
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    reg = AffineRegistration(metric=metric, level_iters=[1000, 500, 200], sigmas=[3, 1, 0], factors=[4, 2, 1])
    c = transform_centers_of_mass(static, static_aff, moving, moving_aff)
    t = reg.optimize(static, moving, TranslationTransform3D(), None, static_aff, moving_aff, starting_affine=c.affine)
    r = reg.optimize(static, moving, RigidTransform3D(), None, static_aff, moving_aff, starting_affine=t.affine)
    a = reg.optimize(static, moving, AffineTransform3D(), None, static_aff, moving_aff, starting_affine=r.affine)
    print("MNI->TT affine:\n", np.round(a.affine, 3))

    def carry(src_img, interp):
        amap = AffineMap(a.affine, static.shape, static_aff, src_img.shape[:3], src_img.affine)
        return amap.transform(np.asarray(src_img.dataobj, np.float32), interpolation=interp)

    sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas = carry(nib.load(sch["maps"]), "nearest").astype(np.int16)
    atlas[~mask] = 0
    tis = {}
    for name, fn in [("gm", datasets.load_mni152_gm_mask), ("wm", datasets.load_mni152_wm_mask)]:
        tis[name] = carry(image.resample_to_img(fn(resolution=2), tmpl, interpolation="nearest"), "nearest") > 0.5
    # CSF: brain mask minus GM and WM, eroded by being kept only where the template is dark
    brain = carry(image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), tmpl, interpolation="nearest"),
                  "nearest") > 0.5
    warped_t1 = carry(tmpl, "linear")
    csf = brain & ~tis["gm"] & ~tis["wm"] & (warped_t1 < np.percentile(warped_t1[brain], 15)) & mask
    wm = tis["wm"] & mask
    present = np.array([(atlas == k).sum() for k in range(1, 201)])
    print(f"parcels with >=10 voxels inside the EPI mask: {(present >= 10).sum()}/200; "
          f"median voxels {np.median(present):.0f}; WM {wm.sum()} CSF {csf.sum()} voxels")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "atlas_tt.npz", atlas=atlas, wm=wm, csf=csf, epi_mask=mask,
                        affine_mni_to_tt=a.affine, grid_affine=static_aff, grid_shape=np.array(static.shape))
    # QC overlay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, sl in zip(axs, [(slice(None), slice(None), static.shape[2] // 2),
                            (slice(None), static.shape[1] // 2, slice(None)),
                            (static.shape[0] // 2, slice(None), slice(None))]):
        ax.imshow(np.rot90(mean[sl]), cmap="gray")
        ax.contour(np.rot90(warped_t1[sl]), levels=[np.percentile(warped_t1[brain], 50)], colors="r", linewidths=0.6)
        ax.imshow(np.rot90(np.ma.masked_equal(atlas[sl], 0)), cmap="tab20", alpha=0.35, interpolation="nearest")
        ax.axis("off")
    fig.savefig(OUT / "atlas_tt_qc.png", dpi=110, bbox_inches="tight")
    print("saved atlas_tt.npz and atlas_tt_qc.png")


if __name__ == "__main__":
    main(sys.argv[1])
