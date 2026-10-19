"""Arousal-tone proxy (post hoc; review of draft 11). The dataset distributes no tone markers (no stimulus events in any
EEG marker file; 43_eog_phasic.py). Tones were delivered mostly in slow-wave sleep and each one ended in an awakening
(Moehlman et al. 2019), so the epochs around every awakening from N2 or N3 hold the tones and their evoked responses.
This re-scores the main models' saved per-epoch probabilities without the 2 epochs (60 s) before and after every
N2/N3 -> W transition.

    source <venv>/bin/activate
    python 44_tone_proxy.py                       # writes results_tone_proxy.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("x36b", BASE / "36b_extra_block.py")
x = importlib.util.module_from_spec(_s)
_s.loader.exec_module(x)


def main():
    Z = np.load(BASE / "predictions_v2_afni.npz", allow_pickle=True)
    y, s, run, ep = Z["y"].astype(str), Z["subj"].astype(str), Z["run"].astype(str), Z["epoch"].astype(int)
    near = np.zeros(len(y), bool)
    n_aw = 0
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        idx = idx[np.argsort(ep[idx])]
        for k in range(1, len(idx)):
            if y[idx[k]] == "W" and y[idx[k - 1]] in ("N2", "N3") and ep[idx[k]] - ep[idx[k - 1]] <= 2:
                n_aw += 1
                t = ep[idx[k]]
                near |= (run == r) & (ep >= t - 2) & (ep <= t + 2)
    keep = ~near
    G = x.gains(Z["proba__nonbold+bold"][keep], Z["proba__nonbold"][keep], y[keep], s[keep])
    R = dict(n_awakenings=n_aw, n_epochs_removed=int(near.sum()),
             removed_by_stage={st: int((near & (y == st)).sum()) for st in ("W", "N1", "N2", "N3", "REM")},
             gain={st: {k: v for k, v in G[st].items() if k != "boot_ws"} for st in G}, contrast=x.contrast(G))
    print(f"awakenings from N2/N3: {n_aw}; epochs removed {near.sum()} {R['removed_by_stage']}")
    print(f"N3 {G['N3']['ws_mean']:+.3f} ({G['N3']['n_pos']}/{G['N3']['n_subj']})  REM {G['REM']['ws_mean']:+.3f} "
          f"t up {G['REM']['t_upper95']:+.3f}  N3>REM {R['contrast']['n_pos']}/{R['contrast']['n']}")
    (BASE / "results_tone_proxy.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
