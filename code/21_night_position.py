"""Night position from wall-clock time, per night.

09_baselines.py ranked the RETAINED epochs and divided by the subject's maximum over both nights, so
scanner-off gaps and EEG-unscorable epochs (a label-derived exclusion) compressed time, and one of
each subject's nights topped out near 0.4. Here every epoch gets its wall-clock time from the scoring
file (first row's HH:MM plus 30 s per row, midnight-wrapped), and night position is the fraction of
that night elapsed, from the first to the last scored row of the session. Labels are never used.

    source <venv>/bin/activate
    python 21_night_position.py          # writes night_position_ds005127.npz (row-aligned to features)
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SCORING = BASE / "data" / "ds005127" / "derivatives" / "sleep_scoring_mr"


def minutes(hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    return (h + 24 if h < 12 else h) * 60 + m          # noon-to-noon clock, so midnight does not wrap


def main():
    F = np.load(BASE / "features_ds005127.npz", allow_pickle=True)
    run, epoch = F["run"].astype(str), F["epoch"].astype(int)
    t_of, span = {}, {}
    drift = []
    for f in sorted(SCORING.rglob("*.txt")):
        d = pd.read_csv(f, sep="\t", dtype=str)
        # one file (sub-00078 ses-2 T2336) has a corrupted header "fa" for the Time column
        tcol = d.columns[0]
        t0 = minutes(d[tcol].iloc[0])
        t = t0 + 0.5 * np.arange(len(d))
        drift.append(abs(minutes(d[tcol].iloc[-1]) - t[-1]))
        stem = f.stem
        t_of[stem] = t
        night = "_".join(stem.split("_")[:2])            # sub-XXXXX_ses-N
        lo, hi = span.get(night, (np.inf, -np.inf))
        span[night] = (min(lo, t[0]), max(hi, t[-1]))
    frac = np.empty(len(run))
    clock = np.empty(len(run))
    for i, (r, e) in enumerate(zip(run, epoch)):
        lo, hi = span["_".join(r.split("_")[:2])]
        clock[i] = t_of[r][e]
        frac[i] = (t_of[r][e] - lo) / (hi - lo)
    print(f"scoring files {len(t_of)}, nights {len(span)}; max |last-row clock - reconstructed| "
          f"{max(drift):.1f} min; night length median {np.median([h - l for l, h in span.values()]) / 60:.2f} h")
    print(f"night fraction range {frac.min():.3f}-{frac.max():.3f}")
    np.savez_compressed(BASE / "night_position_ds005127.npz", run=run, epoch=epoch, night_frac=frac,
                        clock_min=clock)


if __name__ == "__main__":
    main()
