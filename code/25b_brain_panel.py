"""Brain views for Fig. 1d: per-parcel within-subject AUC of 30 s BOLD amplitude, N3 and REM vs the rest.

n3_parcel_map.npz (29_n3_mechanism.py) holds, for each Schaefer-200 parcel, the within-subject AUC of its
epoch amplitude for N3 and for REM (0.5 = no difference; below 0.5 = lower amplitude in that stage). This
paints them on the left fsaverage5 pial surface (lateral and medial) and saves four small PNGs that
25_figures_v2.py places as panel d.

    source <venv>/bin/activate
    python 25b_brain_panel.py                 # writes paper_isbi/figures/brain_{N3,REM}_{lateral,medial}.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting, surface
from scipy.spatial import cKDTree

BASE = Path(__file__).parent
OUT = BASE / "paper_isbi" / "figures"
VMIN, VMAX = 0.30, 0.70     # symmetric about 0.5, the no-difference midpoint
CMAP = "RdBu_r"             # diverging: blue = lower amplitude in the stage, red = higher, white = 0.5


def main():
    Z = np.load(BASE / "n3_parcel_map.npz", allow_pickle=True)
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch["labels"]]
    names = [n for n in names if n != "Background"]
    assert [str(x) for x in Z["labels"]] == names[:200], "parcel order differs from the Schaefer atlas"
    img = sch["maps"] if isinstance(sch["maps"], nib.Nifti1Image) else nib.load(sch["maps"])
    lab = np.asarray(img.dataobj).astype(int)
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    # parcel label per vertex; vertices the projection misses (between the sampled voxels) take the label of
    # the nearest labelled vertex within 6 mm, so the medial wall stays empty but no speckles remain
    vlab = surface.vol_to_surf(img, fs["pial_left"], interpolation="nearest_most_frequent", radius=3.0).astype(int)
    coords = surface.load_surf_mesh(fs["pial_left"]).coordinates
    have = vlab > 0
    d, j = cKDTree(coords[have]).query(coords[~have])
    fill = np.where(d < 6.0, vlab[have][j], 0)
    vlab[~have] = fill
    for st in ("N3", "REM"):
        tex = np.where(vlab > 0, np.r_[np.nan, Z[st]][vlab], np.nan)
        for view in ("lateral", "medial"):
            fig = plt.figure(figsize=(2.0, 1.5))
            ax = fig.add_subplot(111, projection="3d")
            # flat light grey where there is no parcel (medial wall), so it does not read as a 3-D artefact
            plotting.plot_surf(fs["infl_left"], tex, hemi="left", view=view, bg_map=np.where(np.isfinite(tex), 1.0, 0.0), bg_on_data=False,
                               cmap=CMAP, vmin=VMIN, vmax=VMAX, colorbar=False, axes=ax, figure=fig)
            fig.savefig(OUT / f"brain_{st}_{view}.png", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)
        print(f"  {st}: median parcel AUC {np.median(Z[st]):.3f}", flush=True)
    print("saved", sorted(p.name for p in OUT.glob("brain_*.png")))


if __name__ == "__main__":
    main()
