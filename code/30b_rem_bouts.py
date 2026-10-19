"""REM representativeness in the main analysis (derivative epochs): bout lengths and night split.

A REM bout is a run of consecutive analysed REM epochs within one scan run (an epoch gap, a censored or
excluded epoch, or another stage ends it). Night = session (ses-1, ses-2).

    source <venv>/bin/activate
    python 30b_rem_bouts.py                   # writes results_rem_bouts.json
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("e24", BASE / "24_experiments_v2.py")
e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e)


def main():
    D = e.load("afni")
    y, run, ep = D["stage"], D["run"], D["epoch"].astype(int)
    bouts = []
    for r in np.unique(run):
        m = run == r
        o = np.argsort(ep[m])
        yy, ee = y[m][o], ep[m][o]
        n = 0
        for i in range(len(yy)):
            if yy[i] == "REM" and (n == 0 or ee[i] == ee[i - 1] + 1):
                n += 1
            else:
                if n:
                    bouts.append(n)
                n = 1 if yy[i] == "REM" else 0
        if n:
            bouts.append(n)
    bouts = np.array(bouts)
    ses = np.array([re.search(r"ses-(\d)", r).group(1) for r in run])
    rem = y == "REM"
    R = dict(n_bouts=int(len(bouts)), median_bout_min=float(np.median(bouts) * 0.5), mean_bout_min=float(bouts.mean() * 0.5),
             max_bout_min=float(bouts.max() * 0.5), frac_rem_night2=float((rem & (ses == "2")).sum() / rem.sum()),
             frac_epochs_night2=float((ses == "2").sum() / len(y)))
    print(R)
    (BASE / "results_rem_bouts.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
