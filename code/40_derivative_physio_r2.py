"""What did the derivative's preprocessing remove? Physiological variance left in its parcel signals (post hoc).

The derivative's AFNI history (in each file's NIfTI header extension) records: 3dDespike; slice-wise RETROICOR fitted by
3dREMLfit (-slibase_sm, the afni_proc ricor block); 3dTshift; 3dNwarpApply (volreg + nonlinear warp); 3dTproject
-polort 0 -ort X.nocensor.xmat.1D with a combined censor file. The -ort matrix is not distributed. This script asks,
per run and parcel, how much variance two regressor sets still explain, against the same regressors circularly
shifted by half a run (same spectra and dimensionality, timing destroyed):
  retroicor  2nd-order cardiac and respiratory phase regressors (8), from the dataset's own beat/breath detections
  slow       heart rate, RV, finger pulse amplitude and respiratory rate at lags 0-15 s (24; 12_confounds)
on (a) the derivative and (b) our minimal pipeline (no physiological correction), same runs, uncensored frames.
If (a) is at the null for both sets, the derivative had that variance removed; if (a) keeps slow variance above the
null, the REM test on the derivative cannot be a product of removed slow physiology.

    source <venv>/bin/activate
    python 40_derivative_physio_r2.py          # writes results_derivative_physio_r2.json (~5 min)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("conf", BASE / "12_confounds.py")
conf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(conf)
AFNI = BASE / "derivatives" / "afni_ds005127"


def r2(Y, R, valid):
    """Mean over parcels of in-sample R^2 of R (plus intercept and linear trend) on valid frames, adjusted for
    the number of regressors."""
    T = valid.sum()
    X = np.column_stack([np.ones(T), np.linspace(-1, 1, T), R[valid]])
    Yv = Y[:, valid].T
    good = np.isfinite(Yv).all(0) & (Yv.std(0) > 0)
    Yv = Yv[:, good]
    X0 = X[:, :2]
    res0 = Yv - X0 @ np.linalg.lstsq(X0, Yv, rcond=None)[0]
    res1 = Yv - X @ np.linalg.lstsq(X, Yv, rcond=None)[0]
    raw = 1 - (res1 ** 2).sum(0) / (res0 ** 2).sum(0)
    k = R.shape[1]
    adj = 1 - (1 - raw) * (T - 3) / (T - 3 - k)
    return float(np.mean(adj))


def main():
    out = {"runs": []}
    for f in sorted(AFNI.glob("sub-*/*_afni.npz")):
        stem = f.name.replace("_afni.npz", "")
        sub = f.parent.name
        z = np.load(f)
        ts, cens, T, tr = z["ts"].astype(np.float64), z["censored"].astype(bool), int(z["n_vol"]), float(z["tr"])
        raw_f = BASE / "derivatives" / "ds005127" / sub / f"{stem}_parcels.npz"
        pf = conf.PHYS / sub / f"{stem}_physio.npz"
        if not raw_f.exists() or not pf.exists():
            continue
        zr = np.load(raw_f)
        if zr["ts"].shape[1] != T:
            continue
        P = dict(np.load(pf, allow_pickle=True))
        qc, qr = conf.run_quality(P)
        grid, S = conf.physio_signals(P, T, tr)
        R1, R2 = conf.retroicor(P, T, tr), conf.slow_regressors(grid, S, T, tr)
        if not (qc >= 0.7 and qr >= 0.7 and R1 is not None and R2 is not None and conf.trigger_consistent(P, T, tr)):
            continue
        valid = ~cens
        if valid.sum() < 200:
            continue
        rawts = zr["ts"].astype(np.float64)
        rawts = rawts[np.isfinite(rawts).all(1)]
        row = dict(run=stem, subject=sub, n_valid=int(valid.sum()))
        for name, Y in (("deriv", ts), ("raw", rawts)):
            for rn, R in (("retroicor", R1), ("slow", R2)):
                row[f"{name}_{rn}"] = r2(Y, R, valid)
                row[f"{name}_{rn}_null"] = r2(Y, np.roll(R, T // 2, axis=0), valid)
        out["runs"].append(row)
    keys = [k for k in out["runs"][0] if k not in ("run", "subject", "n_valid")]
    subj = sorted({r["subject"] for r in out["runs"]})
    # subject means first, then the mean and range over subjects
    per_subj = {k: [np.mean([r[k] for r in out["runs"] if r["subject"] == u]) for u in subj] for k in keys}
    out["summary"] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in per_subj.items()}
    for name in ("deriv", "raw"):
        for rn in ("retroicor", "slow"):
            d = np.array(per_subj[f"{name}_{rn}"]) - np.array(per_subj[f"{name}_{rn}_null"])
            out["summary"][f"{name}_{rn}_excess"] = dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()),
                                                         n_above=int((d > 0).sum()), n=len(d))
    out["n_runs"], out["n_subjects"] = len(out["runs"]), len(subj)
    for k, v in out["summary"].items():
        print(f"  {k:28s} {v}")
    (BASE / "results_derivative_physio_r2.json").write_text(json.dumps(out, indent=1))
    print(f"saved results_derivative_physio_r2.json ({out['n_runs']} runs, {out['n_subjects']} subjects)")


if __name__ == "__main__":
    main()
