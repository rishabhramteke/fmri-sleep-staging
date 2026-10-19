"""Facts the paper reports about the data rather than the models.

  * realignment FD by stage (per-subject medians, then median over subjects), raw stream, 16 subjects;
    within-subject correlation with the old centre-of-mass proxy
  * per-stage retention of scored, BOLD-covered epochs after the physiology-QC and censoring filters,
    for both streams (the committee asked for per-stage exclusion rates)
  * epoch-to-volume alignment: the volume count implied by the scoring file's MRIon column
    (sum x 30 s / TR) against the acquired count, for every run, and the analysed derivative runs that
    drift by more than one epoch (10 volumes), with their REM content

    source <venv>/bin/activate
    python 30_data_checks.py                  # writes results_data_checks.json
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)
STAGES = e.STAGES


def main():
    R = {}
    # FD by stage on the main (derivative) cohort and epochs; column 0 of the motion block is the 30 s
    # FD mean in both streams. The raw-stream version is kept for the 16-subject statement.
    for src, key in (("afni", "fd_by_stage"), ("raw", "fd_by_stage_raw")):
        D = e.load(src)
        fd, y, sj = D["motion"][:, 0], D["stage"], D["subj"]
        R[key] = {}
        for st in STAGES:
            v = [np.median(fd[(sj == u) & (y == st)]) for u in np.unique(sj) if ((sj == u) & (y == st)).sum() >= e.MIN_EP]
            R[key][st] = dict(median_mm=float(np.median(v)), n_subj=len(v))
    com = D["imgmotion"][:, 0]
    r = [np.corrcoef(fd[sj == u], com[sj == u])[0, 1] for u in np.unique(sj)]
    R["fd_vs_com_r"] = dict(median=float(np.median(r)), min=float(min(r)), max=float(max(r)))

    d = e.bl.load()
    C = np.load(BASE / "confounds_ds005127.npz", allow_pickle=True)
    R["retention"] = {}
    for src in ("afni", "raw"):
        V = np.load(BASE / f"features_v2_{src}.npz", allow_pickle=True)
        key = {(r_, ep): i for i, (r_, ep) in enumerate(zip(d["run"], d["epoch"]))}
        rows = np.array([key[(r_, ep)] for r_, ep in zip(V["run"].astype(str), V["epoch"])])
        yy = d["stage"][rows]
        kept = e.load(src)
        R["retention"][src] = {st: dict(before=int((yy == st).sum()), after=int((kept["stage"] == st).sum()),
                                        frac=float((kept["stage"] == st).sum() / max(1, (yy == st).sum())))
                               for st in STAGES}
        R["retention"][src]["all"] = dict(before=int(len(yy)), after=int(len(kept["stage"])),
                                          frac=float(len(kept["stage"]) / len(yy)))
    import pandas as pd
    sc = BASE / "data" / "ds005127" / "derivatives" / "sleep_scoring_mr"
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    used, zr, zy = set(Z["run"].astype(str)), Z["run"].astype(str), Z["y"].astype(str)
    rows = []
    for f in sorted((BASE / "derivatives" / "ds005127").rglob("*_parcels.npz")):
        stem = f.name.replace("_parcels.npz", "")
        c = list(sc.rglob(stem + ".txt"))
        if not c:
            continue
        z = np.load(f)
        T, tr = z["ts"].shape[1], float(z["tr"])
        on = pd.read_csv(c[0], sep="\t")["MRIon"].to_numpy(float)
        rows.append((stem, T, on.sum() * 30 / tr - T))
    d = np.array([r[2] for r in rows])
    drift = [r[0] for r in rows if abs(r[2]) > 10 and r[0] in used]
    R["alignment"] = dict(n_runs=len(rows), within_one_volume=int((np.abs(d) <= 1.0001).sum()),
                          within_one_epoch=int((np.abs(d) <= 10).sum()), median_abs_vol=float(np.median(np.abs(d))),
                          analysed_drift_runs=len(drift),
                          analysed_drift_epochs=int(np.isin(zr, drift).sum()),
                          analysed_drift_rem=int((np.isin(zr, drift) & (zy == "REM")).sum()))
    print(json.dumps(R, indent=1))
    (BASE / "results_data_checks.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
