"""Assemble the public release folder for the ISBI 2027 paper (published at github.com/rishabhramteke/fmri-sleep-staging).

release_isbi2027/
  code/         every script the paper's numbers, tables and figure come from, in run order (see README)
  results/      every results_*.json read by 26_paper_numbers_v2.py
  predictions/  per-epoch out-of-fold probabilities (predictions_*.npz), so every AUC can be recomputed
  paper/        the generated numbers.tex, tables and figure (the manuscript itself is not included)
  log/          ANALYSIS_LOG.md: next_steps.md, the dated analysis log (includes the subcortical rule recorded at
                12:14 CEST 2026-10-15, before any subcortical feature file existed), lightly redacted (local paths,
                an unrelated closed-out analysis, scores of internal pre-submission reviews) -- only with --with-log
  MANIFEST.tsv  path, bytes, SHA-256 and file modification time of every file (so the order rule -> features
                -> results can be checked against the timestamps of the original files)

    source <venv>/bin/activate
    python make_release.py [--with-log]
"""
import argparse
import hashlib
import re
import shutil
import time
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "release_isbi2027"
CODE = ["03_fetch_sleep_scoring.py", "06_preprocess.py", "06b_raw_v2.py", "07_features.py", "09_baselines.py",
        "11_fetch_physio.py", "12_confounds.py", "12b_physio_spectral.py", "16_paper_numbers.py", "19_atlas_to_tt.py",
        "19b_subcortex_to_tt.py", "19c_subcortex_qc.py", "20_fetch_afni_deriv.py", "20b_fetch_subcortex.py",
        "21_night_position.py", "22_features_v2.py", "22b_subcortex_features.py", "22c_multiwindow_bold.py",
        "22d_level_features.py", "23_realign.py", "24_experiments_v2.py", "25_figures_v2.py", "25b_brain_panel.py",
        "26_paper_numbers_v2.py", "27_verify_v2.py", "28_matched.py", "29_n3_mechanism.py", "30_data_checks.py",
        "30b_rem_bouts.py", "31_whitelist.py", "32_staging_checks.py", "33_contrast.py", "34_extra_baselines.py",
        "35_stacked.py", "36_subcortex.py", "36b_extra_block.py", "36c_subcortex_followup.py", "36d_max_hgb.py",
        "37_reviewer_checks.py", "38_tuning.py", "39_strong_physio.py", "40_derivative_physio_r2.py", "41_rv2_eyes.py",
        "42_round2_checks.py", "43_eog_phasic.py", "44_tone_proxy.py", "45_round4_checks.py", "46_panel_checks.py",
        "47_panel_checks2.py", "48_panel_checks3.py", "make_release.py"]
PAPER = ["numbers.tex", "tab_ladder.tex", "tab_robust.tex", "figures/fig_gain_v2.pdf"]

README = """# Whole-night fMRI sleep staging: BOLD adds to deep sleep, little to REM

Code, results, per-epoch predictions and the dated analysis log for the ISBI 2027 submission of the same title
(Rishabh Ramteke, independent researcher). Licence: MIT (code); the data are not redistributed here. Data: OpenNeuro ds005127 (v2.0.2,
doi:10.18112/openneuro.ds005127.v2.0.2; raw images from snapshot 2.0.1, identical for every file used here) and the
dataset authors' AFNI derivative (derivatives/fmri_ric2alreg), streamed from the public OpenNeuro S3 bucket.

Environment: Python 3.14, numpy 2.4, scipy 1.17, scikit-learn 1.8, nibabel 5.4, nilearn 0.13, dipy 1.12, torch 2.13.

Run order (each script documents its inputs and outputs in its docstring):
  data        03 (scoring, metadata, raw images) -> 11 (physiology peaks) -> 06 (minimal pipeline) -> 07 -> 12 -> 12b
              -> 23 (realignment, FD) -> 21 (elapsed time) -> 19, 19b (atlases to TT_N27) -> 20, 20b (derivative
              stream) -> 22 --source raw|afni, 22b, 22c, 22d -> 06b, 22 --source rv2
  analysis    24 --source afni|raw -> 28 29 30 30b 31 32 33 34 35 36 36b(--block subc|mw|lvl|max) 36c 36d
              -> 37 38 39 40 41 -> 42 43 44 45 46 47 48 (review checks)
  paper       25b -> 25 (figure) -> 26 (numbers.tex, both tables) -> 27 (independent re-derivation from the
              saved predictions, must print ALL PASS; its last check reads the manuscript, which is not included)

results/ holds every JSON the paper's numbers come from; predictions/ the out-of-fold probabilities per epoch.
MANIFEST.tsv lists SHA-256 and modification time of every file; TIMELINE.tsv the modification times of the files
that bear on the pre-specified subcortical rule. These times come from the author's file system: they document
the order of work but are not third-party timestamps.

log/ANALYSIS_LOG.md is the working log kept during the analysis (next_steps.md), with local paths, notes on an
unrelated closed-out analysis and the scores of internal pre-submission reviews removed. Analysis decisions and
their dates are unchanged. Files it mentions under dreams/ are internal notes and are not included.
"""

LICENSE = """MIT License

Copyright (c) 2026 Rishabh Ramteke

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

VENV = "<venv>"


def redact_log(text):
    """The public copy of next_steps.md: no local paths, no unrelated analysis, no internal review scores."""
    out = []
    for line in text.splitlines():
        if re.search(r"personality|Fallback after|^User decisions:|^dated log public; will read|Reviews so far|^Loop stopped|^licence; \$200|^Open user decisions|^Open \(user's"
                     r"|^dropped\), EDAS|^verifiable\), EDAS|^309", line):
            continue
        line = line.replace(VENV, "<venv>").replace("/Users/rishabh.ramteke/Documents/sleep_mri", ".")
        line = re.sub(r" ?\((\d)/5\)", "", line)
        line = re.sub(r",? ?score \d/5", "", line)
        line = re.sub(r": \d/5( weak accept)?\.", ".", line)
        line = re.sub(r" \(4/3/4 -> meta 4 conditional\)", "", line)
        line = re.sub(r": 4/4/3, meta 4 conditional\.", ".", line)
        line = re.sub(r" = \d \([^)]*(accept|reject|borderline)[^)]*\)", "", line)
        line = re.sub(r" = 4 / 4 / 4( \(stable\))?", "", line)
        line = re.sub(r" \(all weak accept\)", "", line)
        line = line.replace("; Fable", "")
        line = re.sub(r"\buser's (call|decision)", r"author's \1", line).replace("the user must", "the author must")
        line = line.replace("(user decision)", "(author's decision)")
        line = re.sub(r" \(\d/\d/\d\)", "", line)
        line = re.sub(r" \(\d(, weak reject|; clarity \d)\)", "", line)
        out.append(line)
    return "\n".join(out) + "\n"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-log", action="store_true")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    items = [(BASE / c, OUT / "code" / c) for c in CODE]
    items += [(p, OUT / "results" / p.name) for p in sorted(BASE.glob("results_*.json"))]
    items += [(p, OUT / "predictions" / p.name) for p in sorted(BASE.glob("predictions_*.npz"))]
    items += [(BASE / "paper_isbi" / q, OUT / "paper" / q) for q in PAPER]
    if a.with_log:
        items.append((BASE / "next_steps.md", OUT / "log" / "ANALYSIS_LOG.md"))
    rows = ["path\tbytes\tsha256\tmtime_original"]
    for src, dst in items:
        if not src.exists():
            print("missing", src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.suffix == ".py":
            dst.write_text(dst.read_text().replace(VENV, "<venv>"))
        elif dst.name == "ANALYSIS_LOG.md":
            dst.write_text(redact_log(src.read_text()))
        rows.append(f"{dst.relative_to(OUT)}\t{src.stat().st_size}\t{sha(src)}\t"
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(src.stat().st_mtime))}")
    # timeline of the files that bear on the pre-specified subcortical rule (recorded in next_steps.md at
    # 2026-10-15 12:14 CEST): modification times of the originals, including intermediates not shipped here
    tl = ["file\tmtime_original"]
    for f in ("22b_subcortex_features.py", "36_subcortex.py", "derivatives/afni_ds005127_subc/atlas_tt.npz",
              "features_v2_afni_subc.npz", "results_subcortex.json", "results_block_subc.json",
              "results_subcortex_followup.json", "dreams/review_committee_2026-10-14.md", "results_v2_raw.json",
              "results_v2_afni_quick.json",
              "results_v2_afni.json"):
        q = BASE / f
        if q.exists():
            tl.append(f"{f}\t{time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(q.stat().st_mtime))}")
    (OUT / "TIMELINE.tsv").write_text("\n".join(tl) + "\n")
    (OUT / "README.md").write_text(README)
    (OUT / "LICENSE").write_text(LICENSE)
    (OUT / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"release_isbi2027: {len(rows) - 1} files, {tot / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
