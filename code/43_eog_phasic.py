"""Phasic vs tonic REM from the EOG, and a search for tone markers (post hoc; review of draft 11).

The authors' gradient- and pulse-artefact-corrected EEG (derivatives/eeg_gac_cbc, BrainVision, VECTORIZED float32,
250 Hz) stores channels contiguously, so only the two EOG channels (E1, E2) are range-requested from the public
bucket (~2/65 of the 39.5 GB). The scoring files' rows are the EEG's 30 s epochs (row i = EEG samples
[7500 i, 7500 (i + 1))), the same index our epochs carry.

  density   per epoch, the fraction of 1 s windows whose E1-E2 peak-to-peak amplitude (0.3-5 Hz) exceeds that subject's
            95th percentile in N2: a rapid-eye-movement density. Its mean by stage checks the alignment (W, REM high)
  phasic    REM epochs at or above the subject's median REM density; tonic = below. BOLD's gain (the main models'
            saved per-epoch probabilities, no refit) for each subset against all non-REM epochs
  markers   every marker type in every run's .vmrk, to find tone/stimulus events for the arousal-tone confound

    source <venv>/bin/activate
    python 43_eog_phasic.py                      # writes eog_density.npz, results_eog_phasic.json (~20 min)
"""
import collections
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

BASE = Path(__file__).parent


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, BASE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


fetch = _load("f03", "03_fetch_sleep_scoring.py")
x = _load("x36b", "36b_extra_block.py")
e = x.e
ST = e.STAGES
PFX = "ds005127/derivatives/eeg_gac_cbc"
FS, EPOCH = 250, 30


def get(key, rng=None, tries=4):
    url = f"{fetch.BUCKET}/{urllib.parse.quote(key)}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"fetch failed: {key}")


def one_run(stem):
    sub, ses = stem.split("_")[0], stem.split("_")[1]
    base = f"{PFX}/{sub}/{ses}/{stem}_eeg"
    hdr = get(base + ".vhdr").decode("latin1")
    n = int(re.search(r"DataPoints=(\d+)", hdr).group(1))
    assert "DataOrientation=VECTORIZED" in hdr and "IEEE_FLOAT_32" in hdr
    chans = re.findall(r"^Ch(\d+)=([^,]*),", hdr, re.M)
    idx = {name: int(k) - 1 for k, name in chans}
    eo = []
    for name in ("E1", "E2"):
        c = idx[name]
        eo.append(np.frombuffer(get(base + ".dat", (c * n * 4, (c + 1) * n * 4 - 1)), "<f4"))
    mk = get(base + ".vmrk").decode("latin1").splitlines()
    types = collections.Counter(",".join(l.split("=", 1)[1].split(",")[:2]) for l in mk if l.startswith("Mk"))
    if len(eo[0]) < 10 * FS:                     # a run with (almost) no EEG: no EOG epochs
        return stem, np.array([], np.float32), dict(types)
    # per second: fraction of anti-phase (E1-E2 correlation < -0.5) large deflections, the signature of horizontal
    # saccades (EEG slow waves move both electrodes in phase); 0.5-8 Hz
    sos = butter(4, [0.5, 8.0], btype="band", fs=FS, output="sos")
    x1, x2 = sosfiltfilt(sos, eo[0].astype(np.float64)), sosfiltfilt(sos, eo[1].astype(np.float64))
    n_s = len(x1) // FS
    w1, w2 = x1[:n_s * FS].reshape(n_s, FS), x2[:n_s * FS].reshape(n_s, FS)
    w1c, w2c = w1 - w1.mean(1, keepdims=True), w2 - w2.mean(1, keepdims=True)
    r = (w1c * w2c).sum(1) / (np.sqrt((w1c ** 2).sum(1) * (w2c ** 2).sum(1)) + 1e-9)
    pd = np.ptp((w1 - w2) / 2, axis=1)
    return stem, np.column_stack([r, pd]).astype(np.float32), dict(types)


def main():
    t0 = time.time()
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, run, ep = Z["y"].astype(str), Z["subj"].astype(str), Z["run"].astype(str), Z["epoch"].astype(int)
    runs = sorted(set(run))
    P2P, MK = {}, collections.Counter()
    other = collections.Counter()
    with ThreadPoolExecutor(8) as ex:
        for k, (stem, p2p, types) in enumerate(ex.map(one_run, runs), 1):
            P2P[stem] = p2p
            MK.update(types)
            for t_, c in types.items():
                if not re.match(r"(Pulse Artifact|SyncStatus|Response,R128|New Segment|TPULSE)", t_):
                    other[t_] += c
            if k % 20 == 0:
                print(f"  [{k}/{len(runs)}] {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(BASE / "eog_p2p.npz", **{k: v for k, v in P2P.items()})
    # per-epoch seconds above the subject's N2 95th percentile
    sec = {}
    for i in range(len(y)):
        p = P2P[run[i]]
        a, b = EPOCH * ep[i], EPOCH * (ep[i] + 1)
        sec[i] = p[a:b] if len(p) and b <= len(p) else np.zeros((0, 2), np.float32)
    dens = np.full(len(y), np.nan)
    for u in np.unique(s):
        m = np.flatnonzero(s == u)
        med = np.median(np.concatenate([sec[i][:, 1] for i in m if len(sec[i])]))
        for i in m:
            if len(sec[i]) >= 25:
                dens[i] = float(((sec[i][:, 0] < -0.5) & (sec[i][:, 1] > med)).mean())
    R = {"markers_all": dict(MK.most_common(30)), "markers_other": dict(other), "n_runs": len(runs),
         "density_by_stage": {st: float(np.nanmean([np.nanmean(dens[(s == u) & (y == st)]) for u in np.unique(s)
                                                    if ((s == u) & (y == st)).sum() >= 20])) for st in ST}}
    # EOG density alone as a REM detector (alignment check)
    ok = np.isfinite(dens)
    R["eog_auc_rem"] = float(np.mean(list(e.ws_auc((y[ok] == "REM").astype(int), dens[ok], s[ok]).values())))
    # phasic / tonic split and BOLD's gain for each
    phasic = np.zeros(len(y), bool)
    for u in np.unique(s):
        m = (s == u) & (y == "REM") & ok
        if m.sum():
            phasic[m] = dens[m] >= np.median(dens[m])
    tonic = (y == "REM") & ok & ~phasic
    i_rem = ST.index("REM")
    Pnb, Pall = Z["proba__nonbold"], Z["proba__nonbold+bold"]
    for tag, sub_ in (("phasic", phasic), ("tonic", tonic)):
        keep = ok & ((y != "REM") | sub_)
        y01 = sub_[keep].astype(int)
        inc = e.increment(y01, Pall[keep, i_rem], Pnb[keep, i_rem], s[keep])
        inc.pop("boot_ws")
        v = list(inc["ws_values"].values())
        inc["t_upper95"], inc["t_lower95"] = x.tb(v), x.tb(v, False)
        R[tag] = inc
        R[tag]["auc_nonbold"] = float(np.mean(list(e.ws_auc(y01, Pnb[keep, i_rem], s[keep]).values())))
        R[tag]["mean_density"] = float(np.nanmean(dens[sub_]))
        print(f"  {tag}: gain {inc['ws_mean']:+.3f} [{inc['ws_ci'][0]:+.3f},{inc['ws_ci'][1]:+.3f}] t up {inc['t_upper95']:+.3f} "
              f"({inc['n_pos']}/{inc['n_subj']}); non-BOLD AUC {R[tag]['auc_nonbold']:.3f}; density {R[tag]['mean_density']:.2f}", flush=True)
    print("  density by stage:", {k: round(v, 3) for k, v in R["density_by_stage"].items()}, " EOG REM AUC", round(R["eog_auc_rem"], 3))
    print("  non-standard markers:", R["markers_other"])
    np.savez_compressed(BASE / "eog_density.npz", run=run, epoch=ep, density=dens, phasic=phasic)
    (BASE / "results_eog_phasic.json").write_text(json.dumps(R, indent=1))
    print(f"saved results_eog_phasic.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
