# sleep_mri — next steps

Resume file. Read this first. Last updated 2026-10-13.

## STATUS 2026-10-13: Project B paper drafted for ISBI 2027 (deadline 2026-10-26 23:59 EDT)

ML4H 2026 (2026-09-10) was missed. The paper is `paper_isbi/main.pdf` (4 pages, official ISBI
template, single-blind, EDAS submission). Venue facts verified 2026-10-13 in
`dreams/venue_check_2026-10.md`: no concurrent review anywhere until 2027-01-12; preprint allowed;
(2027-01-24).

**The framing changed, because the confound control changed the answer.** With properly specified
non-neural rivals (40 multi-window motion features, 65 cardiorespiratory features from the dataset's
own `physio_prepro` peak detection, night position) through the SAME HMM decoder:

- BOLD alone: REM AUC 0.739, N3 0.777, 5-class HMM kappa 0.257 [0.196, 0.311].
- Non-neural alone: REM AUC 0.820 (!), N3 0.744, kappa 0.236.
- Adding BOLD to non-neural: REM -0.005 [-0.038, +0.027]; N3 +0.057 [+0.026, +0.084], 12/13 subjects.
- Best EEG-free stager (non-neural + BOLD): kappa 0.315 [0.263, 0.370], +0.079 over non-neural, 14/16.
- Physiology-cleaned BOLD loses most REM information; N3 keeps more.

Title: "REM sleep is decodable from whole-night human fMRI, but not beyond physiology". The old
"+0.150 REM brain-over-motion margin" does NOT survive: it came from an under-specified motion
baseline and no physiology. Do not resurrect it.

Pipeline for the paper (all reproducible, ~1.5 h total):
`11_fetch_physio.py` (range-request fetch of physio peaks, 229 runs, ~10 min) ->
`12_confounds.py` (motion/physio features + physiology-cleaned BOLD) -> `13_experiments.py` (~50 min)
-> `14_brain_multiwindow.py` + `15_followups.py` + `15b_robustness.py` -> `16_paper_numbers.py`
(writes paper_isbi/numbers.tex + tab_ladder.tex; no hand-typed numbers) -> `17_figures.py` ->
`latexmk -pdf main.tex` in paper_isbi -> `18_verify_paper.py` (independent recomputation, must pass).

Still to do before submission: the author must register on EDAS and submit; decide on a bioRxiv/arXiv
preprint and a public code release (both outward-facing, author's call); the novelty re-search report is
`dreams/novelty_rerun_2026-10.md`.

---

## UPDATE 2026-10-14: committee review and the v2 rebuild in progress

Two internal reviews (`dreams/review_committee_2026-10-14.md` is the full one with a plan
to 4) found: no realignment in our pipeline; the centre-of-mass "FD" and DVARS are image-derived, so the
"non-neural" model contained BOLD; the night-position bug (09_baselines divided by the max over both
nights); the physiological-cleaning test is non-specific (time-shifted regressors remove as much);
the N3 gain is largely global signal; no non-superiority test for the REM null. Novelty re-search
(`dreams/novelty_rerun_2026-10.md`): Preti et al. OHBM 2026 poster 414 trained a supervised SVM with
W/N1/N2/N3/REM on human BOLD, but REM only from narcoleptic daytime naps. Any priority claim must say
"healthy adults, whole-night" and cite Preti 2026.

v2 pipeline (all new, run in this order):
- `19_atlas_to_tt.py`: Schaefer-200 + tissue masks into TT_N27 (the AFNI derivative space); QC png.
- `20_fetch_afni_deriv.py`: streams the authors' AFNI derivative (fmri_ric2alreg; despike, slice-wise
  RETROICOR, tshift, volreg+nonlinear warp, censoring, nuisance regression) from S3 straight through
  gzip, volume by volume, to parcel/GS/WM/CSF signals + censor mask. 185 runs, ~268 GB, ~4 h.
- `21_night_position.py`: wall-clock night position (fixes the bug); hours since lights-out is the
  causal covariate used in v2.
- `23_realign.py`: our own 6-DOF rigid realignment (torch/MPS Gauss-Newton), validated: beats dipy
  on 10/10 volumes by independent NCC. Gives Power FD for a non-BOLD motion confound.
- `12_confounds.py` now has a trigger-consistency QC (4 runs fail).
- `22_features_v2.py --source raw|afni`: PSC amplitude, fixed-parcel network FC, GS block, HRF-shift,
  physio-cleaned + time-shifted-null (raw), aCompCor + GSR (afni).
- `24_experiments_v2.py --source raw|afni`: within-subject AUC primary, increments with bootstrap CI,
  one-sided upper bound vs delta=0.03, Holm, late fusion, HGB, time-matched REM, pairwise AUCs, ladder.
- `25_figures_v2.py`: new Fig. 1 (AUC per stage + increment forest, both pipelines).
Decision rule (from the review): if the N3 increment CI excludes zero on the AFNI derivative, keep the
headline "BOLD adds deep-sleep information beyond non-BOLD signals; REM information is redundant with
physiology, motion and time of night"; otherwise switch to the cautionary headline.

## (Superseded 2026-08-03 plan, kept for the record)

## Do these next, in order

1. **Reframe the paper around the per-stage gradient, not REM alone.** N3 is 0.771 against REM's
   0.770, with more data (15.45 h vs 12.88 h) and all 16 subjects rather than 13. "REM and N3 are
   decodable, N1 and wake are not" is probably a stronger and more defensible claim than REM alone,
   and it uses the same numbers. Decide this before writing, because it changes the framing throughout.
2. **Re-run the recorded novelty search shortly before submission.** The whole contribution rests on a
   negative ("no supervised REM decoder on human fMRI"), and negatives rot. The search protocol and
   verbatim queries are in `dreams/README.md`; the full reproducible methods paragraph is in the
   workflow output referenced there. Scoop risk is real: the NIH group owns this dataset and already
   trains supervised classifiers on it for a different label.
3. **Extract the cardiac and respiratory traces.** ds005127 ships 49.3 GB of physio that was never
   downloaded. Reviewers will ask whether the REM signal is cardiorespiratory rather than neural, and
   right now there is no answer. This is the single most likely reviewer objection left standing.
4. **Write the baseline ladder properly** (spec in `dreams/README.md`): persistence, clock-only,
   prior-only, motion-only, and reimplementations of Tagliazucchi 2012 and Altmann 2016 under LOSO
   extended to five classes. The published bar has never been evaluated with REM as a class.
5. **Then start Project A** on the same machinery.

## Do NOT redo these

Five things were tested and failed. All are documented with numbers in `dreams/README.md`.

- Edge-level FC (3,486 features): overfits at n=16, drags 0.785 to 0.659.
- Imputing missing parcels with subject medians: injects subject identity, drops 0.734 to 0.655.
- Segment-conditioned FC: killed by an oracle test. Bout duration alone (0.684) beats
  bout-conditioned connectivity. Do not rebuild it.
- Semi-Markov explicit-duration decoding: loses to a plain first-order HMM with data-estimated
  transitions (kappa 0.240 vs 0.267). Duration histograms are too noisy at 16 subjects.
- Four-feature motion baselines: understate the confound. Motion needs the same multi-window budget
  as the brain features, or the comparison is invalid.

**The current best model is the simple one** (`10_model.py`, HMM row): within-subject z amplitude +
network FC, logistic emissions, first-order HMM with learned transitions. kappa 0.267.

## Gotchas that will bite whoever picks this up

- **Epoch index does not map to volume index.** Use the cumulative sum of the `MRIon` column. Assuming
  contiguity misaligns labels in about two thirds of runs while still producing plausible output.
- **The ds005127 `Score` coding is undocumented.** It is 6=W, 4=N1 (REM when the REM flag is set),
  3=N2, 2=N3, 0=unscorable. Validated against sleep architecture (REM sits +2.26 h later in the night
  than N3; 93.8% of REM is in the late half). Confirm against Picchioni 2022 before publishing.
- **Within-subject z-scoring matters** and is not optional: it moves brain up and motion down.
- **Only 84 of 200 parcels survive** intersection across subjects. Do not "fix" this with imputation
  (see above).
- Registration is dipy mutual-information; there is no FSL/ANTs/AFNI/Docker on this machine.
  `06_preprocess.py` warns below 150/200 parcels and records a `registration_ok` flag.

## Environment, so nothing has to be rediscovered

- Always `source <venv>/bin/activate`. Python 3.14, numpy 2.4, scipy 1.17,
  sklearn 1.8, nibabel 5.4, nilearn 0.13, dipy 1.12, pandas 2.3, torch 2.11 with MPS available.
- Machine: Apple Silicon, 14 cores, 48 GB RAM. A 19,440^2 float64 kernel (~3 GB) fits.
- **No FSL, ANTs, AFNI, FreeSurfer, Docker or Singularity.** Registration is dipy mutual-information.
  If you ever install ANTs, `06_preprocess.py` would get much faster and could gain a nonlinear step.
- OpenNeuro downloads go over plain HTTPS against the public S3 bucket, no credentials.
  `03_fetch_sleep_scoring.py` has a hard 5 MB per-file cap for text and named targets for imaging.

## Re-running from scratch

```bash
source <venv>/bin/activate
cd .
python 03_fetch_sleep_scoring.py                      # hypnograms + metadata, ~1.2 MB
python 03_fetch_sleep_scoring.py --imaging ds005127-sleep   # 67.8 GB, refuses if disk is tight
python 03_fetch_sleep_scoring.py --imaging ds005127-anat    # 1.8 GB
python 04_sleep_inventory.py                          # coding key validation + REM inventory
for s in $(ls data/ds005127 | grep '^sub-'); do
  python -u 06_preprocess.py --dataset ds005127 --subject $s --runs 0
done                                                  # ~3-4 h total, resumable, skips done runs
python 07_features.py                                 # epoch alignment + features -> 257 MB npz
python 09_baselines.py                                # full ladder
python 10_model.py                                    # final model + ablations
```

## File map

| file | what it does |
|------|--------------|
| `dreams/README.md` | Project B: results, venue, novelty adjudication, negative results |
| `dreams/systematic_search.md` | **the recorded priority search**: 137 verbatim queries, 84 candidates adjudicated, and the methods paragraph to paste into the paper |
| `dreams/venue_and_design.md` | verified venue deadlines and the full algorithm design panel with critiques |
| `hidden_sleep/README.md` | Project A: thesis, threats, ground-truth datasets |
| `01_psqi_phenotype_qc.py` | HCP PSQI phenotype QC |
| `03_fetch_sleep_scoring.py` | OpenNeuro downloads; `--imaging <target>` for the 5 imaging targets |
| `04_sleep_inventory.py` | hypnogram parsing, coding-key validation, per-subject inventory |
| `05_arousal_readout.py` | ds003768 rest-run arousal test (negative; global-signal features are chance) |
| `06_preprocess.py` | atlas-to-native registration, parcel timeseries, motion proxies |
| `07_features.py` | epoch alignment via cumulative MRIon, amplitude/FC/edge/nuisance features |
| `08_decode.py` | LOSO decoding, baseline ladder, within-subject z, orthogonalisation |

Data: `data/ds005127` (65 GB), `data/ds003768` (21 GB), `derivatives/` (121 MB parcel timeseries),
`features_ds005127.npz` (257 MB).

## Open questions

- Is the REM signal cardiorespiratory? Unanswerable until the physio traces are extracted (item 3).
- Does sub-00065 have a genuinely smaller EPI field of view? It reaches 200/200 parcels but with
  47,688 in-atlas voxels against ~88,000 elsewhere. Check before pooling its features.
- Is 5-class kappa 0.179 publishable as a first, or too weak to lead with? It is the first such number
  for fMRI, but it is "slight agreement" on the Landis-Koch scale. Frame carefully.

## UPDATE 2026-10-15: v2 analysis complete, paper rewritten (draft 1)

Data: AFNI derivative covers only 12 subjects (191 runs on S3, 185 in our run set; 179 runs / 13,836
epochs after physio QC + <=50% censoring). Raw pipeline = 16 subjects / 16,671 epochs (replication).
Realignment FD (23_realign.py) done for 226/226 runs; FD W 0.118 > N1/N2/N3 ~0.098 > REM 0.085 mm.

Headline (derivative, within-subject AUC gain of non-BOLD+BOLD over non-BOLD):
W +0.060, N1 +0.071, N2 +0.080, N3 +0.139 [+0.107,+0.173] 12/12, REM -0.011 [-0.033,+0.012],
upper95 +0.008 < delta 0.03. Interaction N3-REM +0.151 [+0.111,+0.192]. kappa BOLD 0.449 (best),
non-BOLD 0.232, both 0.410, netFC 0.354. Time alone REM AUC 0.813 > BOLD 0.760.
REM caveats: physio-only baseline REM +0.039 [+0.005,+0.081] (vanishes with time); late fusion REM
upper95 +0.037 (> delta); HGB REM +0.016, upper95 +0.030; raw HGB REM +0.024 [+0.007,+0.044].
Raw pipeline: N3 +0.039 [-0.003,+0.086] (n.s.), REM -0.030. 28_matched.py: same epochs, N3 gain
+0.025 (raw 84 amp) -> +0.085 (AFNI 84 amp) -> +0.142 (AFNI 200+FC); REM <= 0 throughout.
29_n3_mechanism.py: amp alone +0.112, FC alone +0.116, GS +0.027; amp lower in N3 in 90% of parcels.

New scripts: 27_verify_v2.py (83 checks ALL PASS), 28_matched.py, 29_n3_mechanism.py, 30_data_checks.py.
24_experiments_v2.py now parallel over LOSO folds (joblib, N_JOBS=6) with an exact rank AUC.
Paper: paper_isbi/main.tex rewritten (v1 kept as main_v1.tex); tables tab_ladder.tex + tab_robust.tex
and figures/fig_gain_v2.pdf generated by 26_paper_numbers_v2.py / 25_figures_v2.py.
Snapshot: dreams/paper_v2_draft1_2026-10-15.pdf. Layout assumes paid page 5 (references only).


## UPDATE 2026-10-15 (later): review round 1 -> draft 2
Review (dreams/review_committee_2026-10-15.md). All numbers matched; problems were claims vs
tables + one bug. Fixed in draft 2 (dreams/paper_v2_draft2_2026-10-15.pdf):
- REM claim recalibrated ("barely"; LF +0.014 / HGB +0.016 / raw HGB +0.024 CI>0 stated); headline
  is now the paired N3-REM contrast (+0.160 [+0.117,+0.203], 10/10; LF +0.133, HGB +0.136, raw +0.100).
- BUG: 24's N3-REM interaction was unpaired (two separate bootstraps, 12 vs 10 subjects). Fixed in 24
  (paired, own RNG seed 0) and 33_contrast.py; 24 rerun in background (logs_24_*_r2.log); backups of
  the previous results in results_backup_2026-10-15/.
- kappa drop with non-BOLD reported (-0.039 [-0.069,-0.008], 9/12): elapsed time + HMM kill
  early-night REM (recall 0.70 -> 0.00) -- 32_staging_checks.py. Table 1 now has F1_REM (mF1 dropped).
- Matched-epoch wording corrected (REM rises -0.034 -> -0.011, stays < 0).
- 34_extra_baselines.py: BOLD over time alone REM +0.018 (n.s.), motion+time +0.048, C=0.05 REM
  -0.009 (up +0.012). Table 2 now 19 rows incl. baselines and best-aligned runs (31_whitelist.py).
- Title: "What does BOLD add to whole-night fMRI sleep staging? Much for deep sleep, little for REM".
- Priority-search paragraph folded into contribution (i); pedregosa/rosenberg refs dropped.
27_verify_v2.py: 118 checks ALL PASS. Round-2 review running -> dreams/review_committee_2026-10-15_round2.md

## UPDATE 2026-10-15 (afternoon): review round 2 -> draft 3
Round 2 (dreams/review_committee_2026-10-15_round2.md). Fixed in draft 3 (dreams/paper_v2_draft3_2026-10-15.pdf):
- kappa-drop cause corrected: not elapsed time -- motion+physio+BOLD w/o time loses the same
  (-0.040 [-0.067,-0.011]); slow non-BOLD features make emissions autocorrelated (lag-1 r 0.92 vs BOLD
  0.60, combined 0.74) and the first-order HMM over-trusts them; argmax kappa combined 0.332 > BOLD 0.307.
- W is not NREM (wording); scope claims limited to derivative / full model; "CI above zero" rule;
  one bound rule (bootstrap AND t below delta) for daggers; per-stage F1 replaces macro-F1/pairwise.
- New: 35_stacked.py nested stacking (N3 +0.160, REM +0.010 [-0.007,+0.029], up 0.026/t 0.028);
  envelope over 4 estimators REM bound <= +0.041, N3 bound >= +0.102; headroom N3 62% vs REM -18%;
  REM vs W/N1/N2 gains <= 0, vs N3 +0.012; clock check within N2 BOLD 0.58 vs non-BOLD 0.90;
  alignment: 212/229 runs within one volume; 6 drifting analysed runs hold no REM.
- 27_verify_v2.py: 131 checks ALL PASS.
- IN PROGRESS: subcortical block (19b atlas -> 20b streaming 268 GB, ETA ~15:20 -> 22b features ->
  36_subcortex.py). Round-3 review running (dreams/review_committee_2026-10-15_round3.md).

## PRE-REGISTERED DECISION RULE FOR THE SUBCORTICAL TEST (recorded 2026-10-15 12:14 CEST, before any subcortical
## feature file or result exists; 20b streaming still in progress)
Primary statistic: REM within-subject dAUC of non-BOLD + cortical + subcortical BOLD over non-BOLD,
under (1) early fusion (LR, the paper's primary estimator) and (2) nested stacking (three-block stack
vs non-BOLD stack). Secondary: the same with non-BOLD + cortical BOLD as reference ("beyond cortex").
Bounds: one-sided 95% subject-bootstrap AND t-based, delta = 0.03, unchanged.
  Case A (inside margin): both upper bounds < delta under both estimators.
  Case B (clearly positive): both one-sided lower bounds > 0 under both estimators AND stacked mean >= delta.
  Case C (in between): anything else.
Case B additionally requires, before any change of the paper's REM claim: all four estimators,
leave-one-group-out over the 5 subcortical groups, the paired N3-REM contrast with subcortex, and the
best-aligned-runs subset. Features fixed in advance: 15 Harvard-Oxford structure amplitudes + 5 groups
x 7 cortical networks FC (22b_subcortex_features.py as written at this time).

## UPDATE 2026-10-15 (~12:45): review round 3 -> draft 4
Round 3 (dreams/review_committee_2026-10-15_round3.md). Fixed in draft 4
(dreams/paper_v2_draft4_2026-10-15.pdf):
- kappa mechanism: "HMM over-trusts" also refuted (alpha sweep 0.5..3: BOLD > combined at every weight);
  text now states only: decoding adds +0.141 to BOLD, +0.077 to combined.
- bounds labelled (bootstrap vs t), envelope with standard rounding (+0.040 LF t / +0.103 HGB t), guards
  in 26 assert which estimator attains them; headroom sentence deleted (ratio-of-means kept in JSON:
  N3 61%, REM -8%); REM-vs-competitor stated as range over 4 estimators (-0.021..+0.020; vs N3 +0.012..+0.028).
- abstract rewritten (REM -0.011..+0.016 across four estimators); stacking in the headline REM sentence;
  both nights / LOSO; delta provenance ("after a pilot on our own pipeline"); DVARS wording; "et al." refs.
- NEW 22c_multiwindow_bold.py + 36b_extra_block.py --block mw: 61 multi-window BOLD features beyond the
  main block change REM by -0.010 (early) .. +0.005 (late), stacked +0.002 (t up +0.017) -> context
  length is not the reason; Table 2 row "BOLD + 300 s context".
- 27_verify_v2.py: 143 checks ALL PASS.
- Waiting: subcortical stream (two processes, ~18 MB/s, ETA ~16:00) -> background job runs 22b, 36, 36b --block subc.

## RESULT OF THE PRE-REGISTERED SUBCORTICAL TEST (2026-10-15 ~16:05 CEST; stream done 15:54, 185/185 runs, 0 FAIL)
36b_extra_block.py --block subc (results_block_subc.json), REM, all BOLD (cortex + subcortex) over non-BOLD:
  early fusion : -0.0037 [-0.034,+0.028]; upper b +0.023 / t +0.028 (< delta); lower t -0.035; 5/10
  stacking     : +0.0292 [+0.005,+0.056]; upper b +0.052 / t +0.055 (> delta); lower b +0.008 / t +0.003; 7/10
-> NOT A (stacked upper bounds > delta); NOT B (early lower bounds < 0, and stacked mean 0.0292 < delta 0.03,
   i.e. 0.0008 short); => CASE C (in between), as pre-registered. Subtitle unchanged (rule: change only if
   stacked mean >= delta) -- but the margin is 0.0008, so the text must not lean on the subtitle.
Secondary (beyond cortex): early +0.008 [-0.008,+0.021] 8/10; late +0.018 [+0.006,+0.031]; stack +0.020
  [+0.010 lo b] 9/10. N3 gain with subcortex +0.143 (early), +0.169 (stack); N3-REM contrast 10/10 under
  early, late, stack. Subcortex alone REM AUC 0.811 (cortical BOLD 0.760). kappa BOLD 0.449 -> 0.451.
Follow-up (exploratory, labelled so): 36c_subcortex_followup.py (HGB, leave-one-group-out, amp/FC only,
  GS as stacked block, best-aligned runs).
Follow-up (36c, EXPLORATORY), REM, all BOLD over non-BOLD unless stated:
  HGB with subcortex +0.041 [+0.019,+0.066] 9/10 (beyond cortex +0.025, 10/10); N3-REM 9/10 under HGB.
  stacking, leave-one-group-out +0.023 (w/o brainstem) .. +0.031 (w/o striatum); single group +0.017..+0.022
  (brainstem highest); amplitudes only +0.017, correlations only +0.028 -> diffuse, FC-led, no single group.
  beyond cortex + GS (GS as 4th stacked block): +0.018 [+0.005,+0.030] 9/10 -> not a global-signal effect.
  control: 61-feature multi-window block beyond cortex, stacked: +0.002 -> not a generic stacking artefact.
  best-aligned runs (9 REM subjects): stack all +0.030 [+0.001,+0.066]; beyond cortex only +0.006.
Paper (draft 5): case C wording + new "Subcortex." paragraph; abstract states the subcortical gain range
  (early -0.004 .. HGB +0.041) and "remains open"; subtitle kept per rule (stacked 0.0292 < 0.03).
Draft 5 snapshot: dreams/paper_v2_draft5_2026-10-15.pdf (= paper_isbi/main_v2_draft5.tex), 5 pages, 0 overfull,
  27_verify_v2.py 171 checks ALL PASS (new: subcortical gains/contrasts/AUC re-derived from
  predictions_block_subc.npz + predictions_subcortex_followup.npz). Space: dropped "Stronger L2" row, merged
  the robustness sentence, bib "et al." for >3 authors, shorter ethics/acks. New guards in 26: case-C
  conditions, subtitle rule, 10/10 contrasts, brainstem extremes, +-0.02 cortical-variant claim.
Review round 4 launched on draft 5 (-> dreams/review_committee_2026-10-15_round4.md).

## UPDATE 2026-10-15 (evening): review round 4 -> draft 6
Round 4 (dreams/review_committee_2026-10-15_round4.md) confirmed the rule was recorded first and applied as
written (case C). Fixed in draft 6 (dreams/paper_v2_draft6_2026-10-15.pdf = main_v2_draft6.tex):
- P0-1/2 Subcortex paragraph rewritten: rule verdict only for early + stacking ("nor clearly positive");
  beyond-cortex labelled pre-specified (+0.020); exploratory part now prints late fusion (+0.032
  [-0.004,+0.069]) and HGB; best-aligned stacked total +0.030 with +0.024 from cortex alone (not just +0.006).
- P0-3 "+-0.02 in every variant" was false (300 s context -0.0216): now \remVarMin..\remVarMax
  (-0.022..+0.016) from 26, guard fixed; no hand-typed 0.02 left.
- P1: subtitle "Much for deep sleep, far less for REM" (contrast holds under every estimator); scope words
  ("early-fusion REM gain never turns positive" x2, "can hardly hurt", "cortical BOLD" in Discussion);
  Limitations name the brainstem and the single affine; "basal ganglia"; subcortex clock check 0.48
  (\clkSc, 33_contrast.py); ISBI ethics wording restored.
- Cuts: per-stage F1 sentence, first-half REM recall sentence, own-pipeline HGB sentence, Table 2 caption.
- Fig. 1c: "Own 84+FC" moved behind a separator (off the path); REM line no longer appears to pass through it.
- NEW 19c_subcortex_qc.py -> derivatives/afni_ds005127/subcortex_tt_qc.{png,json}: overlay on mean EPI;
  retention in 3 other runs: L amygdala 0.28-0.81, brainstem 0.72-0.89, others >= 0.86 (rebuttal material).
- 36c now saves best-aligned stacks; 27: 179 checks ALL PASS (bootstrap CI endpoints to MC error 5e-3).

## UPDATE 2026-10-16: full committee review (dreams/review_committee_2026-10-16_full.md) -> draft 7
Round 5 (dreams/review_committee_2026-10-15_round5.md). Full review (PDF-only part,
then everything): science careful; barriers to a strong accept = number-dense text, hedged REM message,
unstated rule, narrow BOLD representation ("absence of evidence"). Strategy: rebuild around the robust
headline (N3 >> REM in every subject, every estimator) and bound REM with the richest BOLD model.
NEW experiments:
- 22d_level_features.py -> features_v2_afni_lvl.npz (epoch-mean PSC: 200 parcels + 15 structures).
  36b --block lvl: REM beyond cortex early -0.010 / late -0.001 / stacked +0.000 -> regional level adds nothing.
- 36b --block max (subc + mw + lvl + GS = 336 extra; 557 BOLD features): REM early -0.031, late +0.025,
  stacked +0.026 [+0.003,+0.050]; N3 +0.147 / +0.172 / +0.186; N3>REM 10/10 under all three.
- 36d_max_hgb.py: boosted trees on all blocks: REM +0.040 [+0.017,+0.066], N3 +0.176, N3>REM 10/10.
- 28_matched.py now also late fusion: N3 +0.010 -> +0.090 -> +0.144; REM -0.055 -> +0.001 -> +0.014
  (REM <= +0.014 under both fusion rules).
- Envelope over 22 derivative specifications (Table 2): REM -0.031..+0.041, N3 +0.109..+0.186; N3>REM in
  every subject in 21/22 (9/10 for HGB+subcortex); min N3 t-lower +0.072 > max REM t-upper +0.068.
Draft 7 (dreams/paper_v2_draft7_2026-10-16.pdf = main_v2_draft7.tex): abstract/contributions rebuilt on the
envelope; Methods state the subcortical rule, pre-specified vs exploratory, one significance convention,
defines the late-night row, describes the extra blocks; Results = 6 short findings; Table 2 gains a
N3>REM column + rows (subcortex x4, regional level, all blocks x4), N3 CI dropped; Discussion leads with
the stage split, REM's autonomic signature, and a recommendation for fMRI state decoders.
5 pages, 0 overfull; 27_verify_v2.py 214 checks ALL PASS.

## UPDATE 2026-10-16 (later): review round 7 -> draft 8
Round 7 (dreams/review_committee_2026-10-16_round7.md): no selective reporting; P0 wording fixes. Applied in
draft 8 (dreams/paper_v2_draft8_2026-10-16.pdf = main_v2_draft8.tex):
- abstract rewritten (13 numbers; "preprocessing and coverage"; ends with the recommendation for fMRI state decoders);
- REM F1 reconciled: BOLD raises REM F1 0.39->0.45 via precision 0.28->0.38, recall 0.62->0.57 (\recREM*);
- headroom restored (N3 61% vs REM -8%) in place of the fragile envelope-bound clause;
- matched-epoch paragraph: "Better BOLD helps N3; REM only stops losing" + BOLD-alone REM AUC 0.503->0.695->0.760;
- scope words ("moves REM by hundredths", "in every specification", "no better than", "with up to all 557");
- Table 2 bold = "bootstrap CI above zero (descriptive)"; subcortex framed as "the small exception";
- N3 network gradient (limbic/default/salience/control 0.34-0.36 vs visual 0.48) replaces vasomotor speculation;
- Discussion trims; delta clause; nuisance wording (3dTproject); rule sentence restructured; Fig. 1c caption.
- NEW Fig. 1d brain panel (25b_brain_panel.py -> paper_isbi/figures/brain_*.png; per-parcel N3/REM amplitude AUC
  on fsaverage5, left lateral/medial); Fig. 1 relaid out (4 panels, "Both" legend, wider panel b counts).
5 pages, 0 overfull; 27_verify_v2.py 221 checks ALL PASS.

## UPDATE 2026-10-16 (evening): fresh PDF-only read of draft 8 -> draft 9
Fresh read (dreams/review_committee_2026-10-16_round8_fresh.md). Applied in draft 9
(dreams/paper_v2_draft9_2026-10-16.pdf = main_v2_draft9.tex):
- abstract rewritten (first sentences, "22 model variants", "preprocessing, all 200 parcels and network FC");
- intro "validity problem unavoidable"; contribution (iv) de-confounded;
- Methods: two nights held out together; alignment (212/229 within a volume; 11 drift > an epoch, 6 analysed,
  no REM); nuisance wording (regressor matrix not distributed); minimal pipeline described (no realignment,
  physiological correction or nuisance regression); RV [Chang] / RVT [Birn] fixed; early fusion named;
- Results: BOLD alone has the highest REM F1 (0.54) and "claims concern per-epoch information before
  decoding"; REM bound notation; late-night test reported (-0.005 [-0.043,+0.039]); Holm p <= 0.01;
  subcortex beyond-cortex Wilcoxon p = 0.02; NEW: REM result not made by nuisance regression (own pipeline,
  none: -0.030; own RETROICOR+RVT+HR: -0.024); "least reduced in visual cortex";
- Discussion reordered (asymmetry is what holds under every estimator); Limitations: median REM bout 2.5 min,
  66% of REM epochs (48% of all) on night 2 (30b_rem_bouts.py), phasic/tonic pooled; delta pilot provenance.
- Table 2: REM to 3 decimals; caption names the 22 specification rows and defines best-aligned.
- Table 1: "Prior only" row dropped. Refs: birn2008, chow2013, picchioni2022 no longer cited.
- Fig. 1: "Both" violet (green = CI above zero only; palette validated), all text >= 7 pt, "n > 0", panel c
  ticks "Own 84 / Deriv. 84 / Deriv. 200+FC / Own 84+FC", panel d = N3 only (lateral + medial, larger, labelled;
  REM network range 0.47-0.52 in caption), figure 2.2 in tall.
5 pages (technical content ends on p. 4; p. 5 = ethics, acks, refs), 0 overfull; 27: 227 checks ALL PASS.

## UPDATE 2026-10-16 (night): PDF-only cold review of draft 9 -> draft 10 (in progress)
Cold review (scratchpad phase1_cold_review.md): W1 weak physiology baseline (no spectral HRV, no sequence
model); W2 REM null estimator-dependent; W3 nothing tuned; W4 delta/pre-spec unverifiable; W5 derivative's nuisance
regression unknown; W6 bout geometry; W7 clock limits rs-fMRI relevance; W8 number overload; W9 stager detail;
W10 HRV/CSF/eye literature; W11 kappa estimand, censored fraction, night index, online use.
All post hoc (exploratory) checks, new scripts:
- 37_reviewer_checks.py: bouts >=5 min N3 +0.130 / REM -0.004 (t +0.027); uncensored epochs N3 +0.136 / REM -0.013;
  night index N3 +0.116 / REM -0.014; night 2 only REM +0.006 (t +0.029); C 0.01..10 REM -0.012..+0.001;
  Holm p W .004 N1 .010 N2 .002 N3 .002; contrast p .002; second half t bound +0.035 (> delta); REM claim holds for
  any margin > +0.011; REM epochs sit in longer bouts than N3 (epoch-weighted median 11.5 vs 7.0 min).
- 38_tuning.py: nested LOSO tuning (inner subject-grouped 4-fold, stage-averaged AUC). LR main (old grid): REM +0.000
  (t +0.023), N3 +0.151; LR max: REM +0.004, N3 +0.174. Rerun with C down to 0.001 + trees in progress.
- 12b_physio_spectral.py (18 HRV/breathing features) + 39_strong_physio.py (LR, HGB, BiGRU, same family both sides): running.
- 40_derivative_physio_r2.py: derivative AFNI history (NIfTI header): 3dDespike, slice-wise RETROICOR via 3dREMLfit,
  3dTshift, 3dNwarpApply, 3dTproject -ort X.nocensor.xmat.1D, censor combined_3. RETROICOR variance removed
  (adj R2 -0.007 vs null -0.001, 0/12 above); slow HR/RV/pulse/resp variance retained (0.132 vs 0.005, 12/12 above).
- 06b_raw_v2.py (own pipeline v2: realigned through 23_realign params, despiked, FD>0.5 censored, no phys correction;
  eye ROI where the FOV reaches the eyes) + 22 --source rv2 + 41_rv2_eyes.py: running (~3 h stream).
Paper: abstract/intro/methods/results rewritten (neutral headings, evaluation protocol as contribution, AFNI history,
censored fraction conservative, kappa pooled, transductive z-scoring, HGB settings, tuning, BiGRU, delta "a fifth of N3"
and margin-free statement, correlated specifications); Table 1 two-level header + gains; Table 2 one range row for 13
variants + tuned + stronger baseline rows; Fig. 1 (b stage labels, muted own pipeline, +-delta moved, d at 719 ppi,
panel-d title); refs: Fonseca 2015, Maquet 1996, Fukunaga 2006, Frazier 2005 (replaces Makris), Frey 2021; journal
abbreviations; ds005127 v2.0.2 DOI (v2.0.1 has none; only change is a file we do not use).
Release package script: make_release.py (local only; publishing is the author's decision).
Draft 10 finished 2026-10-17 (dreams/paper_v2_draft10_2026-10-17.pdf = paper_isbi/main_v2_draft10.tex): 5 pages
(technical content ends on p. 4; p. 5 = ethics, acks, 20 refs), 0 overfull, 27_verify_v2.py 356 checks ALL PASS.
Final results of the post hoc checks:
- stronger non-BOLD (+18 HRV/breathing features), same estimator both sides: N3 +0.118 (LR), +0.126 (trees),
  +0.061 [+0.023,+0.108] 11/12 (BiGRU; non-BOLD N3 AUC 0.773 -> 0.909); REM -0.007 / +0.015 / -0.010, all inside delta.
  BiGRU W/N1/N2 gains stay CI > 0. Cardiorespiratory-only BiGRU 4-class kappa 0.32. In the abstract.
- nested tuning (C 0.0001-10, interior picks 0.001-0.003): LR REM +0.008 (t +0.038) N3 +0.155; LR all blocks REM +0.022
  N3 +0.180 (9/10); trees REM +0.010 N3 +0.135; trees all blocks REM +0.035 [+0.006,+0.073] N3 +0.170.
- 34 specifications: REM -0.031..+0.041, N3 +0.061..+0.186; N3>REM in every subject in 31; REM CI > 0 in 6 (all flexible).
- own pipeline v2 (realigned, despiked, FD>0.5 censored, same 84 parcels, 16 subj): BOLD-alone N3 AUC 0.726 -> 0.778, REM
  0.596 -> 0.603 (no gain in REM sensitivity); gains N3 +0.049 / REM -0.024; with RETROICOR + slow regressors +0.034 / -0.019.
- eyes (found in all 16 subjects; 97% of derivative epochs): alone REM AUC 0.69, but +0.001 [-0.016,+0.016] over non-BOLD,
  +0.004 over non-BOLD + BOLD; N3 +0.063 over non-BOLD; BOLD over non-BOLD + eyes: N3 +0.083 [+0.063,+0.104] 12/12.
Release folder rebuilt (release_isbi2027/, 97+ files, TIMELINE.tsv); NOT published (author's decision).

## UPDATE 2026-10-17: fresh PDF-only review of draft 10 -> draft 11
Review: dreams/review_committee_2026-10-17_draft10_cold.md. Fixed in draft 11 (dreams/paper_v2_draft11_2026-10-17.pdf):
- title "More for deep sleep than for REM"; abstract 7 numbers ("epoch-level BOLD features", BiGRU +0.061, REM point
  estimates up to +0.041 small but positive with flexible learners/subcortex, "shared mainly with time of night");
- removed "(for scale, a fifth of the N3 gain)" (implied delta scaled on the derivative); delta scale now "a fifth of the
  non-BOLD model's remaining REM 1-AUC" (guarded); "fixed with delta in a dated analysis log";
- fixed "stacking, trees or the sequence model" (no BiGRU row has REM CI > 0; guard tightened); 557 sum (215 level);
  "all 200 parcels covered"; visual parcels rising in N3 (13 of 19); Fig. 1c baseline; bout medians; tone awakenings
  biased to SWS; scorer agreement 74.8%; REM retention 96%; acronyms; feature counts as products; Results tagged
  primary / pre-specified secondary / exploratory; dilution sentence; Snoek 2019, Chang 2016, Hong 2009, Song 2022 cited;
  bibliography \small; Fig. 1 one-sided delta lines, flat medial wall, legend "Derivative".
- NEW 42_round2_checks.py: slow physiology regressed from the derivative N3 +0.132 [+0.101,+0.165] 12/12, REM -0.018;
  cubic detrending N3 +0.139 REM -0.011; BOLD predicts elapsed time r 0.28 (same after detrending); cross-night N3 +0.161
  REM -0.028; balanced log-loss N3 +0.26, REM -0.12; without drifting runs unchanged. All four folded into the
  variant row (17 variants; 38 specifications; REM inside margin in 26; N3>REM every subject in 35).
5 pages, 0 overfull, 27_verify_v2.py 356 checks ALL PASS.

## UPDATE 2026-10-17 (later): fresh PDF-only review of draft 11 -> draft 12
Review: dreams/review_committee_2026-10-17_draft11_cold.md. Fixed in draft 12 (dreams/paper_v2_draft12_2026-10-17.pdf):
- N3-REM superiority test restored as a number (+0.160 [+0.117,+0.203], p .002, 10/10), also in the abstract; REM bounds
  under nested-tuned LR stated (+0.034 / +0.038: outside delta); "below the mean N3 gain"; abstract first sentence.
- Fig. 1c = step-wise ablation on 13,448 shared epochs (28_matched.py + rv2 levels): N3 gain 1 minimal +0.021,
  2 realigned +0.009, 3 + physiological regression +0.014, 4 derivative same parcels +0.085, 5 derivative 200+FC +0.142;
  REM <= -0.007 throughout. Derivative not band-passed (10% of power > 0.1 Hz; results_derivative_spectrum.json).
- tones: no tone/stimulus markers in any EEG .vmrk (43_eog_phasic.py); proxy 44_tone_proxy.py excludes +-60 s around 262
  awakenings from N2/N3: N3 +0.138, REM -0.010 (18th variant).
- EOG (range-requested E1/E2 from eeg_gac_cbc, ICA-cleaned): anti-phase eye-movement density detects REM only at AUC 0.57;
  REM split by it: BOLD gains -0.004 / -0.025 (no phasic advantage; marker weak) -> reported with the caveat.
- decoded F1 gains: N3 +0.31 ... REM +0.07; kappa drop narrows with regularisation (0.43 at C 0.01).
- pipeline names Derivative / Minimal / Realigned throughout; Table 1 bold defined; texttt TT_N27; 50 = 15 + 35;
  float parameters so Table 2 stays on p. 4.
5 pages, 0 overfull, 27_verify_v2.py 356 checks ALL PASS. 39 specifications.

## UPDATE 2026-10-17 (evening): fresh review of draft 12 -> draft 13
Review: dreams/review_committee_2026-10-17_draft12_cold.md. Draft 13 (dreams/paper_v2_draft13_2026-10-17.pdf):
- abstract ~150 words, leads with the comparative claim (N3-REM +0.160, all subjects, all 39 specifications), REM as an
  upper bound (-0.011 pre-specified; <= +0.041 flexible), NREM gains preprocessing-dependent, "woken by tones".
- NEW 45_round4_checks.py: time controlled by design (AUC within 1 h bins): REM -0.013 [-0.053,+0.036], N3 +0.156;
  BOLD over physiology alone within-hour +0.015 [-0.027,+0.064] (whole night +0.039 [+0.005,+0.081]);
  average precision N3 +0.261 / REM -0.026; logit-AUC N3-REM +1.41 [+1.12,+1.67] 10/10;
  WM+CSF negative control adds +0.016 to N3, BOLD +0.123 [+0.094,+0.158] beyond it; N3 on REM-eligible +0.148.
- decoding reconciled: BOLD alone best REM F1 (0.54) because time ranks REM but cannot place bouts (F1 0.27);
  class-balanced posteriors as scaled likelihoods; BiGRU details; derivative defined; "7-network FC"; Chow 2013 cited;
  stacked subcortical mean misses delta by 0.001 (stated).
5 pages, 0 overfull, 27_verify_v2.py 360 checks ALL PASS.

## UPDATE 2026-10-17 (night): three-specialist panel on draft 13 -> draft 14
Panel reviews: dreams/review_panel_2026-10-17_draft13_{fmri,sleep,stats}.md. Draft 14
(dreams/paper_v2_draft14_2026-10-17.pdf) adds, from 46_panel_checks.py / 47_panel_checks2.py / 28_matched.py:
- combined-conservative baseline (non-BOLD + 18 cardiorespiratory + eye block): N3 +0.084 (LR), +0.081 (trees),
  +0.067 (BiGRU, 11/12); REM -0.011 / +0.004 / -0.029, inside delta for all three; N3>REM 10/10. In the abstract.
- 24-parameter motion regression in the realigned pipeline: N3 +0.012 (does not recover the derivative's +0.085).
- pairwise: REM vs W -0.009, vs N1 -0.021, vs N2 -0.014; N3 vs N2 +0.179 [+0.135,+0.228] 12/12.
- lagged physiology baseline N3 +0.149 REM -0.008; tones +-5 min (26% removed) N3 +0.135 REM -0.008; without the
  somatomotor-auditory network N3 +0.134 REM -0.004; decoded F1 CIs (N3 +0.31 [0.24,0.38], REM +0.07 [0.02,0.11]);
  30 s-only non-BOLD emissions do not remove the kappa drop (0.41).
- REM stated as small and estimator-dependent; time-controlled upper bound given; "+0.039 over the night, +0.015 within
  hours"; "nuisance-signal baseline"; CO2/BP unrecorded; Results compacted. 43 specifications.
5 pages, 0 overfull, 27_verify_v2.py 360 checks ALL PASS.

## UPDATE 2026-10-17 (late night): panel on draft 14 -> draft 15, review loop stopped
Panel reviews: dreams/review_panel_2026-10-17_draft14_{fmri,sleep,stats}.md. Draft 15
(dreams/paper_v2_draft15_2026-10-17.pdf, paper_isbi/main_v2_draft15.tex), from 48_panel_checks3.py and the reviews:
- abstract: "to our knowledge"; N3 +0.139 (derivative), +0.061 to +0.126 against the stronger cardiorespiratory/eye
  baselines (guard in 26: range = min/max of the six stronger-baseline rows); decoded REM F1 0.54 (BOLD) vs 0.39.
- Methods: the 9x2 stronger features listed; "pulse rate"; percentile bootstrap, dependent folds; the pilot's result
  disclosed (own pipeline N3 +0.057, REM -0.005, before the derivative); ethics as secondary analysis of de-identified data.
- Results: kappa drop diagnosed (fused posteriors more confident 0.60 vs 0.55, smoother lag-1 0.74 vs 0.60);
  "consistent with time-of-night information", within-hour bound +0.031 just above delta; pairwise CIs;
  N3 amplitude drop stronger at 0.01-0.1 Hz (median AUC 0.38, 90% of parcels) than 0.1-0.16 Hz (0.47, 76%),
  larger where tSNR is higher (rho -0.39), all 200 parcels >= 80% covered in every run.
- Discussion: "beyond the recorded ... as modelled"; repetition trimmed. Fig. 1c ticks in words.
5 pages (p. 5 ethics/acknowledgments/references only), 0 overfull, 27_verify_v2.py 360 checks ALL PASS; release rebuilt.

## UPDATE 2026-10-19: draft 16 = 4 pages, no fee; code public; submission sheet
- Title: "Whole-night fMRI sleep staging: BOLD adds to deep sleep, little to REM" (alternatives in
  submission_form.txt).
- Fit to 4 pages: references in IEEE initials style, \footnotesize, 19 refs (dropped chang2016, chow2013, song2022,
  fukunaga2006, ozbay2019, frazier2005); Table 2 \scriptsize and shorter captions; figure 2.05 in tall; text cuts of
  secondary detail only (alignment and retention counts, 30 s-only kappa check, F1 CIs, bout geometry and
  cross-night clauses [both still in Table 2's variant range], EOG phasic/tonic, visual-parcel count, detrended
  clock r, Discussion repetition). Every pre-specified result and every main control is kept.
- Abstract: abstract range fixed to +0.061 to +0.126 (guard in 26); AI disclosure now names all sections.
- Code: https://github.com/rishabhramteke/fmri-sleep-staging (personal account, public, MIT). Built by
  make_release.py --with-log (now ships 42-48, no manuscript, redacted log/ANALYSIS_LOG.md, LICENSE).
  Local git checkout: public_repo/.
- submission_form.txt: copy-paste sheet for EDAS (title, abstract, keywords, topics, declarations).
4 pages, Letter, all fonts embedded, 0 overfull, 27_verify_v2.py 360 checks ALL PASS.
Snapshot: dreams/paper_v2_draft16_2026-10-19.pdf (= paper_isbi/main_v2_draft16.tex).
Note: if accepted, ISBI requires in-person presentation in Lausanne and conference registration (fees not posted).
