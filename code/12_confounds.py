"""Epoch-level confound features (motion, cardiorespiratory) and physiology-cleaned brain features.

This answers the reviewer objection the project had no answer to: is the stage signal in BOLD actually
cardiorespiratory? Heart rate falls and becomes regular in N3, becomes irregular in REM, breathing
slows and deepens with depth, and peripheral vasoconstriction (finger pulse amplitude) tracks arousal
and couples strongly to BOLD during sleep (Ozbay et al. 2019). All of these could be read out of BOLD
without anything neural. Three tests, from weakest to strongest:

  1. PHYSIOLOGY-ONLY DECODER. Multi-window summaries of heart rate, HRV, pulse amplitude, respiratory
     rate and depth, RV and RVT, given the same multi-window budget as motion (see "Do NOT redo":
     under-specified confound baselines manufacture results).
  2. FEATURE-LEVEL ORTHOGONALISATION. Brain features residualised on motion + physiology summaries
     (fitted on training subjects only; done in 13_experiments.py).
  3. SIGNAL-LEVEL CLEANING. Physiological regressors (RETROICOR 2nd order, plus heart rate, RV and
     pulse amplitude at lags 0-15 s, which spans the cardiac and respiratory response functions of
     Chang 2009 / Birn 2008) are regressed out of every parcel timeseries within run, and the brain
     features are recomputed from the cleaned signal. If the stage information were cardiorespiratory
     variance in BOLD, it would leave with the regressors.

Motion uses the same 5 windows x 4 statistics on FD and DVARS = 40 features (the documented budget).

Epochs are aligned exactly as in 07_features.py (cumulative MRIon) and keyed by (run, epoch), so the
output joins row-for-row onto features_ds005127.npz.

    source <venv>/bin/activate
    python 12_confounds.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DERIV = BASE / "derivatives" / "ds005127"
PHYS = BASE / "derivatives" / "physio_ds005127"
OUT = BASE / "confounds_ds005127.npz"

_spec = importlib.util.spec_from_file_location("feat", BASE / "07_features.py")
feat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feat)

WINDOWS_S = (30, 60, 90, 150, 300)      # centred on the epoch; 30 s is the epoch itself
LAGS_S = (0, 3, 6, 9, 12, 15)           # FIR lags for slow physiological regressors
IBI_RANGE = (0.33, 2.0)                 # 30-180 bpm
BREATH_RANGE = (1.5, 15.0)              # 4-40 breaths/min
MIN_BEAT_FRAC = 0.6                     # an epoch window needs this much plausible coverage


# ------------------------------------------------------------------ physiology -> continuous signals
def clean_intervals(t, lo, hi):
    """Event times -> (midpoint, interval) pairs, dropping implausible and outlying intervals."""
    if t is None or len(t) < 3:
        return np.array([]), np.array([])
    iv = np.diff(t)
    mid = t[:-1] + iv / 2
    ok = (iv >= lo) & (iv <= hi)
    if ok.sum() > 10:
        med = pd.Series(iv).where(ok).rolling(15, center=True, min_periods=3).median().to_numpy()
        ok &= np.abs(iv - med) < 0.4 * med          # reject missed / double detections
    return mid[ok], iv[ok]


def on_grid(t_ev, v_ev, grid, maxgap):
    """Interpolate event-valued samples onto a grid; NaN where the nearest event is too far away."""
    out = np.full(len(grid), np.nan)
    if len(t_ev) < 3:
        return out
    out = np.interp(grid, t_ev, v_ev)
    idx = np.clip(np.searchsorted(t_ev, grid), 1, len(t_ev) - 1)
    gap = np.minimum(np.abs(grid - t_ev[idx - 1]), np.abs(t_ev[idx] - grid))
    out[gap > maxgap] = np.nan
    out[(grid < t_ev[0]) | (grid > t_ev[-1])] = np.nan
    return out


def physio_signals(P, n_vol, tr):
    """1 Hz continuous physiological signals over the run, time zero = first volume."""
    grid = np.arange(0, n_vol * tr, 1.0)
    cm, ibi = clean_intervals(P["card_t"], *IBI_RANGE)
    rm, bi = clean_intervals(P["resp_t"], *BREATH_RANGE)
    S = {}
    S["hr"] = on_grid(cm, 60.0 / ibi if len(ibi) else ibi, grid, 3.0)
    # successive IBI differences for RMSSD (vagal tone), placed at the later interval
    if len(ibi) > 2:
        S["dibi2"] = on_grid(cm[1:], np.diff(ibi) ** 2, grid, 3.0)
    else:
        S["dibi2"] = np.full(len(grid), np.nan)
    ct, ca = np.asarray(P["card_t"]), np.asarray(P["card_amp"])
    S["pulse_amp"] = on_grid(ct, ca, grid, 3.0) if len(ct) > 3 else np.full(len(grid), np.nan)
    S["resp_rate"] = on_grid(rm, 60.0 / bi if len(bi) else bi, grid, 15.0)
    rt, ra = np.asarray(P["resp_t"]), np.asarray(P["resp_amp"])
    S["resp_amp"] = on_grid(rt, ra, grid, 15.0) if len(rt) > 3 else np.full(len(grid), np.nan)
    if len(rm) > 3:
        amp_at = np.interp(rm, rt, ra)
        S["rvt"] = on_grid(rm, amp_at / bi, grid, 15.0)
    else:
        S["rvt"] = np.full(len(grid), np.nan)
    # RV (Chang 2009): SD of the respiratory waveform in a 6 s window
    wt, rw = P["wave_t"], P["resp_wave"]
    rv = np.full(len(grid), np.nan)
    if len(wt) > 10:
        s = pd.Series(rw, index=wt)
        hz = 1.0 / np.median(np.diff(wt))
        sd = s.rolling(int(6 * hz), center=True, min_periods=int(3 * hz)).std().to_numpy()
        rv = np.interp(grid, wt, sd, left=np.nan, right=np.nan)
    S["rv"] = rv
    return grid, S


def trigger_consistent(P, n_vol, tr):
    """Physio time zero is the first MR trigger; trust it only if the trigger train spans the run.

    219 of 229 runs record T+1 trigger onsets spanning 0..T*TR (a closing trigger) and a few span
    (T-1)*TR; 4 runs have dropped or extra triggers (e.g. a 115 s gap with 42 TRs too many in
    sub-00029 ses-2 T0304), where volume k cannot be placed at k*TR. Those runs get no physiology.
    """
    mt = np.asarray(P["mr_time"])
    if len(mt) < 3:
        return False
    on = mt[np.concatenate([[True], np.diff(mt) > 0.5])]
    span = on[-1] - on[0]
    return min(abs(span - (n_vol - 1) * tr), abs(span - n_vol * tr)) <= 1.5 * tr


def run_quality(P):
    """Fraction of plausible cardiac and respiratory intervals; a channel below 0.7 is unusable."""
    def frac(t, lo, hi):
        if t is None or len(t) < 10:
            return 0.0
        iv = np.diff(t)
        return float(((iv >= lo) & (iv <= hi)).mean())
    return frac(P["card_t"], *IBI_RANGE), frac(P["resp_t"], *BREATH_RANGE)


# ------------------------------------------------------------------ BOLD-level physiological model
def retroicor(P, n_vol, tr, order=2):
    """Cardiac and respiratory phase Fourier regressors at each volume's mid-acquisition time.

    Volume-level, not slice-level: no slice timing is distributed and the parcels pool slices, so this
    is a partial correction. At TR 3 s cardiac aliasing is the main thing it catches.
    """
    t = np.arange(n_vol) * tr + tr / 2
    regs = []
    for key in ("card_t", "resp_t"):
        ev = np.asarray(P[key])
        if len(ev) < 5:
            return None
        idx = np.clip(np.searchsorted(ev, t) - 1, 0, len(ev) - 2)
        ph = 2 * np.pi * (t - ev[idx]) / np.maximum(ev[idx + 1] - ev[idx], 1e-3)
        ph = np.where((t < ev[0]) | (t > ev[-1]), 0.0, ph)
        for m in range(1, order + 1):
            regs += [np.cos(m * ph), np.sin(m * ph)]
    return np.column_stack(regs)


def slow_regressors(grid, S, n_vol, tr):
    """HR, RV, pulse amplitude and respiratory rate, sampled at volumes, at lags 0-15 s (FIR)."""
    t = np.arange(n_vol) * tr + tr / 2
    regs = []
    for key in ("hr", "rv", "pulse_amp", "resp_rate"):
        x = pd.Series(S[key]).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(x).any():
            return None
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
        for lag in LAGS_S:
            regs.append(np.interp(t - lag, grid, x, left=x[0], right=x[-1]))
    return np.column_stack(regs)


def clean_ts(ts, R):
    """Regress physiological regressors (plus intercept and linear trend) out of each parcel."""
    T = ts.shape[1]
    X = np.column_stack([np.ones(T), np.linspace(-1, 1, T), R])
    out = np.full_like(ts, np.nan)
    good = np.isfinite(ts).all(axis=1)
    Y = ts[good].T
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    out[good] = (Y - X @ beta).T
    return out


# ------------------------------------------------------------------ epoch summaries
def window_stats(x, a, b, stats):
    seg = x[a:b]
    fin = np.isfinite(seg)
    if seg.size == 0 or fin.mean() < MIN_BEAT_FRAC:
        return [np.nan] * len(stats)
    seg = seg[fin]
    f = {"mean": np.mean, "sd": np.std, "max": np.max, "p90": lambda v: np.percentile(v, 90),
         "rmssd": lambda v: np.sqrt(np.mean(v))}
    return [float(f[s](seg)) for s in stats]


def epoch_windows(center_s, total_s):
    for w in WINDOWS_S:
        yield max(0.0, center_s - w / 2), min(total_s, center_s + w / 2)


def main():
    net_ids, nets = feat.network_ids()
    n_nets = len(nets)
    runs = sorted(DERIV.rglob("*_parcels.npz"))
    rows, motion, physio, amp_c, fc_c, gs = [], [], [], [], [], []
    runlog = []
    for i, r in enumerate(runs, 1):
        eps = feat.run_epochs(r)
        if not eps:
            continue
        stem = r.name.replace("_parcels.npz", "")
        z = np.load(r)
        ts, tr, fd, dv = z["ts"], float(z["tr"]), z["fd"], z["dvars"]
        T = ts.shape[1]
        pf = PHYS / r.parent.name / f"{stem}_physio.npz"
        P = dict(np.load(pf, allow_pickle=True)) if pf.exists() else None
        if P is not None:
            qc, qr = run_quality(P)
            grid, S = physio_signals(P, T, tr)
            R1, R2 = retroicor(P, T, tr), slow_regressors(grid, S, T, tr)
            ok_phys = (qc >= 0.7 and qr >= 0.7 and R1 is not None and R2 is not None
                       and trigger_consistent(P, T, tr))
        else:
            qc = qr = 0.0
            ok_phys = False
        runlog.append(dict(run=stem, subject=r.parent.name, n_vol=T, physio_file=P is not None,
                           card_frac=qc, resp_frac=qr, physio_ok=ok_phys))

        # physiology-cleaned parcel timeseries, z-scored within run exactly like 07_features
        if ok_phys:
            good = np.isfinite(ts).all(axis=1)
            tsz = np.full_like(ts, np.nan)
            tsz[good] = (ts[good] - ts[good].mean(1, keepdims=True)) / (ts[good].std(1, keepdims=True) + 1e-9)
            tsc = clean_ts(tsz, np.hstack([R1, R2]))
            tsc[good] = (tsc[good] - tsc[good].mean(1, keepdims=True)) / (tsc[good].std(1, keepdims=True) + 1e-9)
        # global signal (mean over present parcels of the run-z-scored series)
        good = np.isfinite(ts).all(axis=1)
        g = ts[good]
        g = ((g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + 1e-9)).mean(0)

        for ep in eps:
            a, n = ep["start_vol"], ep["n_vol"]
            b = a + n
            c_vol = (a + b) / 2
            rows.append((r.parent.name, stem, ep["epoch"], ep["stage"]))
            # motion: 5 windows x {mean, sd, max, p90} x {FD, DVARS}
            mv = []
            for w in WINDOWS_S:
                h = w / tr / 2
                wa, wb = int(max(0, round(c_vol - h))), int(min(T, round(c_vol + h)))
                for x in (fd, dv):
                    mv += window_stats(x, wa, wb, ("mean", "sd", "max", "p90"))
            motion.append(mv)
            # global-signal amplitude, same windows
            gv = []
            for w in WINDOWS_S:
                h = w / tr / 2
                wa, wb = int(max(0, round(c_vol - h))), int(min(T, round(c_vol + h)))
                gv += window_stats(g, wa, wb, ("mean", "sd"))
            gs.append(gv)
            # physiology: 5 windows x {mean, sd} x 6 signals + RMSSD
            pv = []
            if ok_phys:
                cs = c_vol * tr
                for wa_s, wb_s in epoch_windows(cs, T * tr):
                    ia, ib = int(wa_s), int(np.ceil(wb_s))
                    for key in ("hr", "pulse_amp", "resp_rate", "resp_amp", "rv", "rvt"):
                        pv += window_stats(S[key], ia, ib, ("mean", "sd"))
                    pv += window_stats(S["dibi2"], ia, ib, ("rmssd",))
            else:
                pv = [np.nan] * (len(WINDOWS_S) * 13)
            physio.append(pv)
            # physiology-cleaned brain features, recomputed with the 07_features recipe
            if ok_phys:
                half = int(round(feat.FC_WINDOW_S / 2 / tr))
                c0, c1 = max(0, (a + b) // 2 - half), min(T, (a + b) // 2 + half)
                amp, _, fc = feat.featurise(dict(seg=tsc[:, a:b], win=tsc[:, c0:c1], good=ep["good"]),
                                            net_ids, n_nets)
            else:
                amp = np.full(200, np.nan); fc = np.full(n_nets * (n_nets + 1) // 2, np.nan)
            amp_c.append(amp); fc_c.append(fc)
        if i % 20 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] epochs {len(rows)}, runs with usable physio "
                  f"{sum(x['physio_ok'] for x in runlog)}", flush=True)

    md = pd.DataFrame(rows, columns=["subject", "run", "epoch", "stage"])
    pnames = [f"{k}_{s}_w{w}" for w in WINDOWS_S
              for k in ("hr", "pulse_amp", "resp_rate", "resp_amp", "rv", "rvt") for s in ("mean", "sd")]
    pnames = []
    for w in WINDOWS_S:
        pnames += [f"{k}_{s}_w{w}" for k in ("hr", "pulse_amp", "resp_rate", "resp_amp", "rv", "rvt")
                   for s in ("mean", "sd")] + [f"rmssd_w{w}"]
    mnames = [f"{x}_{s}_w{w}" for w in WINDOWS_S for x in ("fd", "dvars") for s in ("mean", "sd", "max", "p90")]
    np.savez_compressed(OUT, subject=md.subject.values, run=md.run.values, epoch=md.epoch.values,
                        stage=md.stage.values, motion=np.array(motion, np.float32),
                        physio=np.array(physio, np.float32), gs=np.array(gs, np.float32),
                        amp_clean=np.array(amp_c, np.float32), fc_clean=np.array(fc_c, np.float32),
                        physio_names=np.array(pnames), motion_names=np.array(mnames))
    pd.DataFrame(runlog).to_csv(BASE / "results_physio_runs.csv", index=False)
    ph = np.array(physio)
    valid = np.isfinite(ph).all(axis=1)
    print(f"\n{len(md)} epochs; physio complete for {valid.sum()} ({100*valid.mean():.1f}%)")
    print(md.assign(v=valid).groupby("stage").v.mean().round(3).to_string())
    print(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
