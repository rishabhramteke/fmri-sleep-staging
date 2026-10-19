"""Stronger cardiorespiratory features for the non-BOLD baseline (post hoc; the review asked for the features of the
heart-rate-variability sleep-staging literature, not only window means and s.d.s).

Per epoch, in windows of 150 s and 300 s centred on it, from the dataset's own beat and breath detections
(physio_prepro, cleaned exactly as in 12_confounds.clean_intervals):
  HRV spectrum   log LF (0.04-0.15 Hz) and log HF (0.15-0.4 Hz) power of the beat-interval series (4 Hz
                 interpolation, linear detrend, Welch 64 s segments), log LF/HF, HF in normalised units
  HRV fractal    DFA alpha-1 of the beat intervals (scales 4-16 beats), a standard deep-sleep marker
  breathing      coefficient of variation and normalised RMSSD of breath intervals (regular in N3, irregular in
                 REM), coefficient of variation of breath amplitude
  pulse          coefficient of variation of finger pulse amplitude
= 9 features x 2 windows = 18. Runs failing 12_confounds' physiology QC get NaN, as there.

    source <venv>/bin/activate
    python 12b_physio_spectral.py             # writes features_physio_spectral.npz (~5 min)
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.signal import detrend, welch

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("conf", BASE / "12_confounds.py")
conf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(conf)
feat = conf.feat
WINDOWS = (150, 300)
FS = 4.0
NAMES = ["log_lf", "log_hf", "log_lfhf", "hf_nu", "dfa_a1", "breath_cv", "breath_rmssd_n", "resp_amp_cv", "pulse_amp_cv"]


def dfa_alpha1(ibi):
    if len(ibi) < 50:
        return np.nan
    yv = np.cumsum(ibi - ibi.mean())
    sc, F = np.arange(4, 17), []
    for n in sc:
        k = len(yv) // n
        seg = yv[:k * n].reshape(k, n)
        t = np.arange(n)
        res = seg - np.array([np.polyval(np.polyfit(t, r, 1), t) for r in seg])
        F.append(np.sqrt((res ** 2).mean()))
    return float(np.polyfit(np.log(sc), np.log(F), 1)[0])


def hrv_spectrum(mid, ibi, a, b):
    m = (mid >= a) & (mid < b)
    if m.sum() < 20 or ibi[m].sum() < 0.6 * (b - a):
        return [np.nan] * 4
    g = np.arange(max(a, mid[m][0]), min(b, mid[m][-1]), 1 / FS)
    if len(g) < 256:
        return [np.nan] * 4
    x = detrend(np.interp(g, mid[m], ibi[m] * 1000.0))
    f, p = welch(x, fs=FS, nperseg=256, noverlap=128)
    df = f[1] - f[0]
    lf = p[(f >= 0.04) & (f < 0.15)].sum() * df
    hf = p[(f >= 0.15) & (f < 0.40)].sum() * df
    return [np.log(lf + 1e-6), np.log(hf + 1e-6), np.log((lf + 1e-6) / (hf + 1e-6)), hf / (lf + hf + 1e-9)]


def cv(v):
    return float(np.std(v) / (abs(np.mean(v)) + 1e-9)) if len(v) >= 5 else np.nan


def main():
    runs = sorted(conf.DERIV.rglob("*_parcels.npz"))
    keys, X = [], []
    n_ok = 0
    for i, r in enumerate(runs, 1):
        eps = feat.run_epochs(r)
        if not eps:
            continue
        stem = r.name.replace("_parcels.npz", "")
        z = np.load(r)
        tr, T = float(z["tr"]), z["ts"].shape[1]
        pf = conf.PHYS / r.parent.name / f"{stem}_physio.npz"
        ok = False
        if pf.exists():
            P = dict(np.load(pf, allow_pickle=True))
            qc, qr = conf.run_quality(P)
            grid, S = conf.physio_signals(P, T, tr)
            ok = (qc >= 0.7 and qr >= 0.7 and conf.retroicor(P, T, tr) is not None
                  and conf.slow_regressors(grid, S, T, tr) is not None and conf.trigger_consistent(P, T, tr))
        if ok:
            n_ok += 1
            cm, ibi = conf.clean_intervals(P["card_t"], *conf.IBI_RANGE)
            rm, bi = conf.clean_intervals(P["resp_t"], *conf.BREATH_RANGE)
            rt, ra = np.asarray(P["resp_t"]), np.asarray(P["resp_amp"])
            ct, ca = np.asarray(P["card_t"]), np.asarray(P["card_amp"])
        for ep in eps:
            keys.append((stem, ep["epoch"]))
            if not ok:
                X.append([np.nan] * (len(NAMES) * len(WINDOWS)))
                continue
            c = (ep["start_vol"] + ep["n_vol"] / 2) * tr
            row = []
            for w in WINDOWS:
                a, b = c - w / 2, c + w / 2
                row += hrv_spectrum(cm, ibi, a, b)
                mb = (cm >= a) & (cm < b)
                row.append(dfa_alpha1(ibi[mb]) if ibi[mb].sum() >= 0.6 * w else np.nan)
                mr = (rm >= a) & (rm < b)
                bb = bi[mr]
                row.append(cv(bb) if bb.sum() >= 0.6 * w else np.nan)
                row.append(float(np.sqrt(np.mean(np.diff(bb) ** 2)) / bb.mean()) if len(bb) >= 5 and bb.sum() >= 0.6 * w
                           else np.nan)
                row.append(cv(ra[(rt >= a) & (rt < b)]))
                row.append(cv(ca[(ct >= a) & (ct < b)]))
            X.append(row)
        if i % 25 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] epochs {len(keys)}, physio-ok runs {n_ok}", flush=True)
    X = np.array(X, np.float32)
    names = [f"{n}_w{w}" for w in WINDOWS for n in NAMES]
    np.savez_compressed(BASE / "features_physio_spectral.npz", run=np.array([k[0] for k in keys]),
                        epoch=np.array([k[1] for k in keys]), X=X, names=np.array(names))
    print("NaN fraction per feature:", dict(zip(names, np.round(np.isnan(X).mean(0), 3))))


if __name__ == "__main__":
    main()
