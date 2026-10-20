# Whole-night fMRI sleep staging: BOLD adds far more to deep sleep than to REM

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
