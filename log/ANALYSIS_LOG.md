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
  secondary detail only (alignment and retention counts, 30 s-only kappa check, F1 CIs, the bout-geometry
  check [bouts >= 5 min; now only in results_reviewer_checks.json] and the cross-night clause [still in Table 2's
  variant range], EOG phasic/tonic, visual-parcel count, detrended clock r, Discussion repetition). Every pre-specified result and every main control is kept.
- Abstract: abstract range fixed to +0.061 to +0.126 (guard in 26); AI disclosure now names all sections.
- Code: https://github.com/rishabhramteke/fmri-sleep-staging (personal account, public, MIT). Built by
  make_release.py --with-log (now ships 42-48, no manuscript, redacted log/ANALYSIS_LOG.md, LICENSE).
  Local git checkout: public_repo/.
- submission_form.txt: copy-paste sheet for EDAS (title, abstract, keywords, topics, declarations).
4 pages, Letter, all fonts embedded, 0 overfull, 27_verify_v2.py 360 checks ALL PASS.
Snapshot: dreams/paper_v2_draft16_2026-10-19.pdf (= paper_isbi/main_v2_draft16.tex).
Note: if accepted, ISBI requires in-person presentation in Lausanne and conference registration (fees not posted).

## UPDATE 2026-10-19 (later): venue confirmed
Title 1 kept. Venue: ISBI 2027 (not ISMRM 2027, whose 28 Oct abstract deadline overlaps ISBI's no-concurrent-review
rule). Accepted ISBI papers must be presented in person in Lausanne, else withdrawn from the proceedings.

Review: dreams/review_cold_2026-10-19_draft16.md (4 MUST-FIX, 20 optional). Draft 17
(dreams/paper_v2_draft17_2026-10-19.pdf = paper_isbi/main_v2_draft17.tex):
- M1: abstract's +0.126 had no support in the paper -> Table 2 restores "Stronger non-BOLD, LR / trees / BiGRU"
  (+0.118 / +0.126 / +0.061; REM inside delta in all) and relabels "Stronger + eye block"; both are now in the
  specification envelope (43 -> 45; 27_verify_v2.py updated). Abstract gives own pipelines +0.009 to +0.050
  (new macros ownNthreeMin/Max with a guard that N3 > REM in every own-pipeline analysis).
- M2: robustness sentence now names the 12 exceptions (late fusion, trees, nested tuning, stacking with added
  blocks; guard on the labels) and the 6 CI-above-zero rows (trees and such stacking).
- M3: 1 h-bin AUC defined in Methods; Results sentence split; Discussion "consistent with being shared".
- M4: regional level defined (200 parcels; 215 with the structures).
- Optional: subcortical 50 = 15 + 35; minimal pipeline lacks the visual network; per-epoch kappa named; pilot
  described (minimal pipeline, earlier features, same data, not independent); tone times not distributed;
  transitions (not awakenings) counted; AP numbers; WM/CSF from the derivative; 24P not plotted; run-wise band-pass;
  AUC/HMM/FC/tSNR expanded; AUC-vs-F1 reconciliation sentence; Ozbay 2019 restored; URL no longer breaks at a hyphen;
  REM night-2 wording; "led by the brainstem" dropped.
- Cuts to stay at 4 pages: estimator hyperparameters (in the code), posterior-smoothness numbers, RETROICOR R2
  values, REM-AUC ladder clause, pairwise REM CIs, kappa-difference CI, coverage clause, network list, frey2021,
  "non-superiority" keyword; Table 1 \scriptsize; figure 1.95 in.
4 pages (right column of p. 4 has < 1 line spare), 0 overfull, 27_verify_v2.py 360 checks ALL PASS.
submission_form.txt updated (abstract, keywords).

## UPDATE 2026-10-19 (night): experiments against the remaining weaknesses
Launched: 49_preproc_ladder.py (which derivative step
cleans the N3 amplitude signal: SyN alignment, grey-matter restriction, tissue/motion/physio regression, on the
matched epochs; clue: BOLD-alone N3 AUC 0.654 ours vs 0.836 derivative on identical parcels and epochs),
52_cross_dataset.py (ds005127-trained BOLD stager applied to ds003768 Penn State naps, W/N1/N2). Done:
50_tempered_hmm.py (nested emission exponent does not repair the kappa drop: fused minus BOLD-alone -0.047
[-0.079, -0.014], 3/12), 51_stager_upgrades.py (kappa 0.40-0.46 in 15 of 17 BOLD-containing variants at tau 1, two stacking variants lower; best 0.459). Paper
untouched tonight (still draft 17); integration planned for draft 18.
Cross-site result (21:30): ds005127-trained BOLD stager on ds003768 naps (30 subjects, 5,502 epochs, W/N1/N2):
kappa 0.059 [0.023, 0.097], AUC W 0.60 N1 0.53 N2 0.63; within-ds003768 LOSO with the same features 0.07 (argmax) /
0.10 (HMM), bold+time 0.09 / 0.12. Weak everywhere: naps from the minimal pipeline are near floor; 40% of nap epochs
are called N3/REM (within-subject z-scoring assumes a whole night). At most one honest Limitations clause.

## 2026-10-19 (late): two learned-representation experiments, at the author's request
- 53_learned_encoder.py (running): a small temporal CNN on the raw 90 s parcel series of each epoch (201 x 30 input,
  LOSO, early stopping on two training subjects, 3 seeds), as one more estimator row: does a LEARNED BOLD
  representation change the N3 / REM gains over the non-BOLD model? Answers the reviewer's "feature design may be the
  bottleneck for REM". Readout: the paper's within-subject AUC gain, N3 - REM contrast, HMM kappa.
- 54_ssl_pretrain.py (code ready; to be launched when the GPU is free): JEPA-style self-supervised pretraining on the
  whole-night derivative corpus (185 runs, ~145 h, 90 s windows, stride 5 -> ~34k windows): frame tokens, 4-layer
  transformer, EMA target encoder, predictor, cross-time block masking alternating with cross-parcel masking, smooth-L1
  in latent space. Then a frozen linear probe with the paper's LR readout: nb+emb vs nb, nb+bold+emb vs nb+bold, emb
  alone, HMM kappa. Rationale: no sleep-fMRI foundation model exists; whole-night data is the natural pretraining
  corpus for brain-state dynamics (cf. Brain-JEPA, BrainLM on UK Biobank awake rest). Pretraining is label-free but
  transductive over subjects by default; --exclude-subject gives a strict per-fold check. Follow-up material unless it
  finishes and is clear-cut before 26 Oct; it is not needed for the ISBI paper's claims.

## 2026-10-20 (01:40): overnight results
- Learned encoder (53): enc alone AUC W 0.856 N1 0.733 N2 0.797 N3 0.953 REM 0.867; nb+enc - nb N3 +0.152 [+0.122,
  +0.179] 12/12, REM +0.048 [+0.028, +0.070] 10/10 (vs the LR non-BOLD baseline +0.017 [-0.028, +0.056]; late fusion
  +0.052 [+0.024, +0.080]); N3-REM +0.107 10/10; within 1 h bins REM +0.062. Leakage checks clean: censoring pattern
  alone REM AUC 0.526 (gain +0.010); without the 14 orbitofrontal parcels REM AUC 0.863, gain +0.052. HMM kappa enc
  0.459, nb+enc 0.448.
- Self-supervised JEPA-style pretraining (54): 80 epochs -> frozen embedding AUC W 0.863 N1 0.740 N2 0.812 N3 0.953
  REM 0.864; nb+emb - nb N3 +0.172 [+0.131, +0.217] 12/12, REM +0.018 [+0.003, +0.035] upper 0.032; beyond the
  hand-crafted features +0.027 N3 / +0.011 REM; HMM kappa emb 0.478 (highest of all stagers), nb+emb 0.451.
  Undertraining check (300-epoch run, checkpoints every 50, each probed): the probe is flat from epoch 100 on (emb REM
  AUC 0.866-0.876, N3 gain +0.176 to +0.180, REM gain +0.018 to +0.026, kappa 0.467-0.476); loss keeps falling
  (0.118 -> 0.053) and the embedding spread shrinks (0.23 -> 0.14) without collapse. 80 epochs was enough; the paper
  uses the 80-epoch run.
- Paper (draft 18 in progress): two Table 2 rows (temporal CNN encoder; self-supervised embedding) -> 47
  specifications, REM envelope -0.031 to +0.048, 14 exceptions (2 learned), CI above zero in 8; Methods describes both
  representations; Results give enc/emb REM AUC vs 0.760, the leakage numbers, the +0.017 to +0.052 REM and +0.152 to
  +0.180 N3 gains; kappa clause (18 richer variants <= 0.46, embedding 0.48, tempering does not remove the drop).
  4 pages, 0 overfull, 27_verify_v2.py 385 checks ALL PASS. Still to add: the preprocessing-ladder result (gains
  rerun 01:40 without the *_tis variants, whose k-means tissue class emptied shared parcels in most runs).

## 2026-10-20 (02:30): draft 18 complete
Preprocessing ladder (49, 12,694 matched epochs, identical non-BOLD model): our realigned pipeline (affine atlas) BOLD-
alone N3 AUC 0.660, gain -0.007; adding WM/CSF regressors 0.699 / +0.013, + 24 motion parameters 0.734 / +0.032,
+ RETROICOR and slow physiology 0.764 / +0.044 (9/11); derivative on the same parcels 0.851 / +0.085. Nonlinear (SyN)
registration changes the N3 gain by +0.001 on average (it raises BOLD-alone REM AUC from 0.47 to 0.69, without any REM
gain over non-BOLD), grey-matter restriction by -0.003. So nuisance regression accounts for about 55% of the
derivative's N3 advantage; the rest (slice-wise RETROICOR within AFNI, the undocumented 3dTproject matrix, 2 mm
resampling) was not rebuilt. REM gain <= +0.004 on every rung. The *_tis (EPI intensity class) variants were dropped
(they emptied shared parcels). Paper: Results sentence + Discussion sentence replaced ("is open" -> "about half is
nuisance regression; half unattributed").
Draft 18: dreams/paper_v2_draft18_2026-10-20.pdf, paper_isbi/main_v2_draft18.tex. 4 pages, 0 overfull, 0 underfull
(spconf's \section penalty of -300 had been breaking the last column early; set to +150 in the preamble, headings
unchanged), 406 checks ALL PASS, fonts embedded. Figure 1 height 1.95 -> 1.60 in. submission_form.txt abstract updated
(161 words, 47 specifications, REM -0.031 to +0.048 "learned representations included").

Its nine MUST-FIX items were consistency and wording defects, all fixed in draft 19:
(a) learned-representation ranges now the two Table 2 rows (+0.018 to +0.048 REM, +0.152 to +0.172 N3; the encoder's
late-fusion +0.052 and 1 h-bin +0.062 are named separately); (b) the abstract's own-pipeline range now includes the
ladder rungs (-0.012 to +0.050); (c) the N3 gain over the 10 contrast subjects is given (+0.148); (d) the 1 h-bin REM
upper bound +0.031 is labelled "above delta: not inside the margin"; (e) the derivative-variance numbers are defined
(R2 of slow physiological regressors vs time-shifted copies); (f) "hence the pre-specified test ..." reworded;
(g) abstract: "on average in all 47 specifications and in every subject in 44 of them"; (h) CI above zero in 7 of the
14 exceptions (an 8th bold row sits inside the margin); (i) Discussion: "five hundredths across the night (six within
1 h bins)". Also: the N3 signature is shown to exist without any regression (89% of parcels in the minimal pipeline:
not a regression-fit artefact), the self-supervised pretraining is disclosed as including held-out subjects, and
Figure 1 lost its brain panel (unreadable at print size; its number is in the text). Layout: spconf's \section broke
the last column early before the two short closing sections; those two headings are now set by \plainsection (same
look, numbered via the section counter). One citation (altmann2016) dropped to make room. 4 pages, 0 overfull,
406 checks ALL PASS. Snapshot dreams/paper_v2_draft19_2026-10-20.pdf. Review loop stopped here (3-4 for days).
Strict (held-out-subject-excluded) pretraining runs in the background for the follow-up; not needed for the paper.

## 2026-10-20 (04:10): strict self-supervised pretraining finished -> draft 20
Strict run (54_ssl_pretrain.py --pretrain --exclude-subject per fold: 12 encoders x 50 epochs, about 10 min each, six
in parallel, no collapse, final embedding spread 0.22-0.23; then --probe-strict): embedding alone W/N1/N2/N3/REM AUC
0.838/0.711/0.782/0.926/0.781; nb+emb minus nb: N3 +0.147 [+0.112, +0.188] 12/12, REM -0.004 [-0.021, +0.013]
(one-sided upper bounds +0.010 bootstrap, +0.012 t: inside the margin), N3-REM contrast +0.158 (10/10); HMM kappa
embedding 0.455, nb+embedding 0.418. Transductive run (one encoder on all 185 runs, held-out subject's unlabeled
windows included, 80 epochs): REM AUC 0.864, REM gain +0.018 [+0.003, +0.035], N3 +0.172, kappa 0.478. At matched
50 epochs the transductive encoder still gives REM AUC 0.847 and REM gain +0.010, so the gap is the held-out subject's
own windows in pretraining (subject-specific structure), not training length.
Decision: the Table 2 row is now the strict run (LOSO-consistent like every other row); the transductive REM AUC is
quoted in the Results sentence as the comparison ("0.864 once the held-out subject's own runs enter pretraining").
Consequences: REM inside the margin in 34 specifications, 13 exceptions (was 14), CI above zero in 6 of them (was 7);
learned-representation ranges REM -0.004 to +0.048, N3 +0.147 to +0.152; kappa clause "19 variants, from boosted trees
to the temporal CNN and the self-supervised embedding, reach at most 0.46"; Methods "pretrained label-free per fold on
the training subjects' runs". Abstract unchanged (47 specifications, 44, REM -0.031 to +0.048), so submission_form.txt
stands. 26_paper_numbers_v2.py reads results_ssl_probe_strict.json for the row and keeps the transductive run as
sslTrans* macros; 27_verify_v2.py recomputes both from predictions_ssl_probe_strict.npz / predictions_ssl_probe.npz.
Draft 20: dreams/paper_v2_draft20_2026-10-20.pdf, paper_isbi/main_v2_draft20.tex. 4 pages, 0 overfull, 419 checks
ALL PASS, fonts embedded. Release rebuilt. Nothing left running.

## 2026-10-20 (08:05): follow-up started: cross-site transfer and a joint self-supervised corpus (55_cross_site.py)
Why the earlier transfer sat at floor: within-ds003768 LOSO with the same minimal-pipeline features is itself at floor
(kappa 0.07; W-vs-N2 within-subject AUC 0.62), and a label-shift scan (-8..+8 epochs) peaks at zero lag, so the labels
are aligned; the ladder on ds005127 showed that the minimal pipeline's amplitude is the weak signal (BOLD-alone N3 AUC
0.654 vs 0.836 for the nuisance-regressed derivative). So ds003768 goes through the same chain as ds005127's rv3 rung:
23_realign.py motion, 49 chain (affine/SyN atlas, GM prior, WM PCs + CSF, 24 motion parameters; no physiology exists
for ds003768), on all 255 runs (189 sleep + 66 rest; the rest runs are sleep-scored too: W 708, N1 341, N2 217).
Hypnogram parser now reads the 4-column file (sub-01) as well: 254 runs, 33 subjects, 6,859 scorable epochs.
Other public sleep-scored fMRI: none. NIH sleep2 (ds006163) is embargoed (release planned 2028); OpenNeuro/NEMAR
searches find only ds003768 and ds005127. "More data" therefore means ds003768 (43-64 h) and, later, unlabelled
resting-state corpora (decided after the joint-corpus result below).
Running (a shell chain calling 55_cross_site.py and 54_ssl_pretrain.py): motion (3 workers, 15-22 s/run) -> sample (4 workers, runs-per-task 3) ->
features_ds003768_rv3.npz -> results_cross_site_hand.json (within-PSU LOSO, NIH->PSU, PSU->NIH for amp_{aff,aff_gm,syn,
syn_gm}{,_acc,_accmot}, identical pipeline both sides) -> ssl_windows_joint_rv3.npz (ssl_windows.npz + PSU at TR 3 s) ->
results_cross_site_ssl_nih_rv3.json (NIH-only encoder on PSU) -> joint pretraining (transductive 80 epochs; 12 strict
per-fold encoders, 50 epochs, 4 at a time) -> results_ssl_probe_joint.json, results_ssl_probe_strict_joint.json
(does the second site's unlabelled data change the ds005127 N3/REM gains?), results_cross_site_ssl_joint_rv3.json.
Preview (a second chain) on the existing minimal-pipeline PSU series: 10,394 windows (189 sleep runs,
31 subjects, 43 h), 5,592 labelled windows; results_cross_site_ssl_nih_minimal.json, ssl_checkpoint_jointmin.pt,
results_cross_site_ssl_jointmin_minimal.json, results_ssl_probe_jointmin.json. 54 gained --windows; 49 TASK_GLOB.
Machine load is 50-60 from other jobs, so timings are uncertain.

## 2026-10-20 (09:10): preview on the minimal-pipeline Penn State series (results_cross_site_ssl_*.json, results_ssl_probe_jointmin.json)
Joint corpus = ssl_windows.npz (28,933 NIH windows) + 10,394 PSU windows (189 sleep runs, 31 subjects, 43 h, minimal
pipeline, resampled 2.1 -> 3 s); one transductive encoder, 80 epochs (ssl_checkpoint_jointmin.pt).
- NIH probe, transductive (compare results_ssl_probe.json, NIH-only, 80 epochs: N3 +0.172, REM +0.018, kappa 0.478,
  embedding REM AUC 0.864): joint N3 +0.175 [+0.134, +0.217] 12/12, REM +0.025 (upper +0.046, t +0.049), contrast
  +0.157, kappa 0.477, embedding REM AUC 0.870. The second site's 43 h change nothing on ds005127.
- Cross-site, NIH-only encoder (ssl_checkpoint.pt, PSU never seen): NIH -> PSU kappa 0.03 (AUC W 0.52, N1 0.51,
  N2 0.55); within-PSU LOSO on the embedding kappa 0.09 (hand features 0.07); PSU -> NIH kappa 0.11 (W 0.60, N2 0.64).
- Cross-site, joint encoder: NIH -> PSU still at floor (kappa 0.02-0.03); within-PSU kappa 0.11 (AUC W 0.64, N2 0.66);
  PSU -> NIH kappa 0.24, within-subject AUC W 0.74, N1 0.57, N2 0.79 over the 12 NIH subjects. So a readout trained on
  the nap labels transfers to the whole-night data once the encoder has seen both sites label-free, but not the other
  way round: the Penn State side (minimal pipeline, 15-min naps, N1-heavy) is the weak link, as the within-site floor
  says. The nuisance-regressed chain (running) is the real test. 55 now also reports a three-class NIH model
  (nih3_to_psu) so both directions use the same class set.
Sampling throughput: 2.1 runs/min with 8 or 12 workers (CPU saturated by other jobs; load 120-180), 257 s per run.

## 2026-10-20 (12:35): cross-site results with the nuisance-regressed chain on both sites (chain finished 11:12)
Penn State through the 49 chain: 255 runs sampled (SyN 15 s per subject; 2-4 runs/min with 8-12 workers on a saturated
machine), features_ds003768_rv3.npz: 6,859 epochs, 33 subjects, 234 runs with finite features (62 rest runs).
Hand features, identical pipeline both sides (results_cross_site_hand.json; NIH side 14,845 epochs, 16 subjects):
- within-PSU LOSO (W/N1/N2) stays near floor on every rung: kappa 0.07-0.09 (argmax), <= 0.15 with HMM, within-subject
  AUC W 0.60-0.64, N1 ~0.5, N2 0.60-0.67; time-in-run alone gives 0.075. Nuisance regression adds little (N2 AUC 0.63
  -> 0.66). So the 15-min, N1-heavy naps are intrinsically hard for epoch features: the floor was the dataset, not only
  the pipeline.
- NIH -> PSU: kappa 0.05-0.11 (AUC W ~0.6, N2 0.65-0.69), i.e. as good as a locally trained model; both low.
- PSU -> NIH: kappa 0.08-0.15 (W 0.61-0.67, N2 0.59-0.68).
Embeddings on the regressed PSU series (results_cross_site_ssl_nih_rv3.json, _joint_rv3.json):
- NIH-only encoder (PSU never seen): NIH -> PSU kappa 0.075 (W 0.63, N2 0.68); within-PSU 0.14/0.16 HMM (W 0.69,
  N2 0.70; best within-PSU result of all); PSU -> NIH kappa 0.32, within-subject AUC W 0.82, N1 0.57, N2 0.85.
- Joint encoder (49 h of PSU added): NIH -> PSU 0.088 (W 0.66, N2 0.68); within-PSU 0.15 (W 0.70, N2 0.70);
  PSU -> NIH kappa 0.35, AUC W 0.84, N1 0.57, N2 0.86.
  Reading: a readout fitted on another site's nap labels stages our nights' W/N1/N2 well through the embedding; the
  reverse is capped by the naps' own ceiling (within-PSU kappa <= 0.16 for everything).
Does the second site's unlabelled data help ds005127? (results_ssl_probe_joint.json, results_ssl_probe_strict_joint.json)
- transductive: N3 +0.175, REM +0.021 (up +0.037, t +0.039), kappa 0.472, embedding REM AUC 0.873 (NIH-only: +0.172,
  +0.018, 0.478, 0.864): unchanged.
- strict: N3 +0.156 [+0.118, +0.196] 12/12, REM -0.000 [-0.017, +0.019] (up +0.016, t +0.018), contrast +0.162, kappa
  0.473, embedding alone REM 0.803 / N3 0.935 (NIH-only strict: +0.147, -0.004, 0.455, 0.781 / 0.926). The embedding
  itself improves a little with 1.7x data (kappa +0.02, REM AUC +0.02); what BOLD adds beyond the recorded signals does
  not move. Scaling verdict: no case for piling on unlabelled resting-state corpora for the paper's question.
Next (running): nih_within W/N1/N2 LOSO reference on both feature types, saved predictions for 27, and the strict
PSU -> NIH number (per-fold joint encoders, --ssl-eval-strict); then decide on a one-sentence Discussion/Limitations
update (single-site limitation) if it fits in 4 pages.

## 2026-10-20 (17:40): strict cross-site number and draft 21
Strict PSU -> NIH (55_cross_site.py --ssl-eval-strict; per-fold encoders pretrained on both sites without the held-out
NIH subject; W/N1/N2 readout fitted on the Penn State labels alone): kappa 0.29 [0.24, 0.34], within-subject AUC
W 0.815, N1 0.556, N2 0.840; within-NIH LOSO W/N1/N2 reference with the same encoders: kappa 0.40, W 0.843, N1 0.693,
N2 0.872. Transductive versions: 0.32-0.35 vs 0.42 (NIH-only and joint encoders). Hand features, within-NIH W/N1/N2
reference: kappa 0.13-0.23 (rv3 rungs). So wake and N2 transfer from another site's short scans nearly at the
within-site level; N1 does not (0.56 vs 0.69). Within-PSU kappa <= 0.16 for every feature set (argmax max 0.144);
NIH -> PSU <= 0.11.
Draft 21 (dreams/paper_v2_draft21_2026-10-20.pdf, paper_isbi/main_v2_draft21.tex): the Limitations sentence "12
subjects from one site" now reports the strict transfer (AUC 0.815 W, 0.840 N2 against 0.843, 0.872 within site; the
short scans stage poorly, within-site kappa <= 0.16, so N3 and REM transfer are untested); ds003768 added to the
compliance statement. Room made by trimming: the F1-by-stage clause, "0.155 with 200 parcels and FC" (still in
Fig. 1c), "subject-grouped inner CV", "pulse-oximeter and belt", "its posteriors are smoother" clause; Figure 1 height
1.50 -> 1.40 in, panel-b counts side by side (derivative | minimal; the stacked counts collided). Abstract unchanged,
submission_form.txt stands. 26 has a guarded cross-site block (cs* macros, asserts on the claims); 27 recomputes the
strict AUCs and kappas from predictions_cross_site_ssl_strict_joint.npz and bounds the within-PSU maximum. 4 pages,
0 overfull, 0 undefined, fonts embedded, 428 checks ALL PASS.
Scaling verdict (for the follow-up write-up): 1.7x pretraining data from a second site improves the embedding itself
slightly (strict kappa 0.455 -> 0.473, REM AUC 0.781 -> 0.803) but leaves what BOLD adds beyond the recorded signals
unchanged (strict N3 +0.147 -> +0.156, REM -0.004 -> -0.000). No public sleep-scored fMRI beyond ds003768/ds005127
(ds006163 embargoed to 2028); unlabelled resting-state scaling not pursued.

Review: dreams/review_cold_2026-10-20_draft21.md. Nine MUST-FIX items, all addressed in draft 22
(dreams/paper_v2_draft22_2026-10-20.pdf, paper_isbi/main_v2_draft22.tex); title kept (author's decision):
1. N3 claim scoped to the distributed preprocessing: abstract opens "With the dataset authors' preprocessing", the own
   pipelines are "but only -0.012 to +0.050"; contribution (iii) "across 47 specifications of the distributed
   preprocessing"; Discussion "our own pipelines recover at most +0.050 (minimal pipeline +0.039 [-0.003, +0.086])".
2. Abstract ranges traceable: -0.012 is the ladder's affine + GM rung (now "all rungs -0.012 to +0.044" in Results),
   +0.050 the minimal-pipeline trees (Table 2); REM range "-0.031 to +0.048 across specifications, up to +0.073 for a
   CNN encoder within 1 h bins" (late fusion within bins = the largest REM gain reported; guarded in 26).
3. REM: abstract "inside the margin for the planned across-night model, not within 1 h bins"; CI above zero in 7 of
   47; Discussion drops the time-of-night mechanism ("most of BOLD's REM information is shared with the recorded
   signals, but a small REM-specific part is not excluded"; seven hundredths within bins).
4. "Pre-specified" -> "planned after a pilot on the same subjects"; Methods: the log entry of 14 Oct 2026 fixed the
   statistic with its bootstrap bound against delta and Holm (the t bound came later); delta = a fifth of the non-BOLD
   REM 1-AUC; the planned time control was the second half (t bound +0.035, too imprecise), 1 h bins added later;
   the N3-REM contrast was in the code's first run on the derivative (made paired after a bug fix).
5. Cross-site sentence rewritten: single-site limitation stands; strict per-fold self-supervised encoders, W/N1/N2
   AUC 0.815/0.556/0.840, kappa 0.29 [0.24, 0.34] vs 0.843/0.693/0.872, kappa 0.40 within site; tests neither N3,
   REM nor BOLD's gain; ds003768 in Data (10-15 min rest and nap scans) and cited (Gu et al. 2023, Data Brief 48).
6. Eyes: the eye block adds +0.063 to N3 and lowers BOLD's N3 gain from +0.142 to +0.083 (97% of epochs; stronger
   baseline +0.118 -> +0.084); stronger baselines' own performance (N3 AUC 0.785 LR, 0.909 BiGRU; kappa 0.23, 0.35).
7. Numbers: derivative on the ladder's 12,694 epochs recomputed (+0.0851 vs +0.0852 on 13,448); the undrawn motion
   level dropped from the Fig. 1c sentence; "262 N2/N3-to-wake transitions"; context block defined (8 amplitude series
   x 5 windows + 21 FC over 300 s); amplitude over the epoch's 30 s; "best stager in Table 1".
8. Fig. 1: all text >= 7 pt, both count columns labelled (deriv., min.), panel c encodings, subjects and short labels
   in the caption; height 1.32 in.
9. Table 2 caption gives the bold N3 lower bounds (0.02-0.15); decoded F1: BOLD alone 0.54 [0.42, 0.64]; added to
   non-BOLD +0.307 [+0.24, +0.38] for N3 and +0.068 [+0.02, +0.11] for REM (three decimals; Table 1 shows rounded F1).
Also: JEPA cited (Assran et al. CVPR 2023); article numbers printed as "Art. no."; Maquet 1996 replaced by Hong 2009
for REM's regional activity (space). Space: intro paragraph 2, subcortex rule, stronger-baseline feature list,
tempering clause, physiology-alone across/within-bin clause, band/tSNR clause, Fig. 1c sentence compressed; tighter
float and bibliography spacing in the preamble (margins and fonts unchanged). 4 pages, 0 overfull, fonts embedded,
27_verify_v2.py 435 checks ALL PASS (new: CNN within-bin REM gains, decoded F1 and its bootstrap CI).

(dreams/paper_v2_draft23_2026-10-20.pdf), the objective errors and cheap clarifications:
- abstract: "+0.073 for a CNN encoder by late fusion within 1 h bins" (was mislabelled "within 1 h bins"); the N3-REM
  difference "set after a pilot" (it was in the analysis code, not the dated log; Methods says so; "planned" dropped
  from Results);
- Methods: regional level "200, 215 with the structures" (the draft-22 compression had left "215" against Table 2's
  200); the censored fraction is in the non-BOLD model "for the derivative" only (the 16-subject analyses lack it);
  pretraining windows "90 s at 15 s steps"; subcortex rule worded as logged;
- Results: "On the derivative, BOLD raises AUC ..."; "t upper bound +0.031" within bins; stronger baselines "give N3
  gains of +0.061 to +0.126" (not all six are the smallest); transfer AUCs per stage as a list (kappa dropped);
  the time-alone AUC/F1 sentence dropped (Table 1 row);
- Discussion: early fusion ranks REM below non-BOLD alone (0.836 vs 0.847), so Delta AUC bounds what these models use;
- layout: \belowcaptionskip 3 pt (Table 1's top rule overprinted its caption); citation order [5, 6].
Not done (author's constraints or no space): retitling (asked by both latest reviews; title kept at the author's
request), tables at >= 7 pt, abstract additions (subcortical secondary, decoded REM F1), Fig. 1a marker offsets, the
per-item count of the 21 variants. 4 pages, 0 overfull, fonts embedded, 435 checks ALL PASS. submission_form.txt
lists new requests rather than unresolved old ones.

## 2026-10-21: draft 24 = draft 23 with the new title (author's decision)
to scope the title; the author chose this wording). Updated in main.tex, submission_form.txt and the release README.
4 pages, 0 overfull. Snapshot dreams/paper_v2_draft24_2026-10-21.pdf. Asking the ds005127 authors for the
derivative's nuisance matrix: not pursued (no time before the deadline).

## 2026-10-21: clarity pass (dreams/clarity_plan_2026-10-21.md) -> drafts 25, 26, 27
What moved where (draft 24 -> 27):
- Abstract: 6 to 7 numbers (+0.160 [CI], +0.139, +0.050, 0.03, +0.048); "specification" defined in it; the REM sentence
  says the within-bin result is too imprecise rather than a verdict.
- Methods: "The test" (statistic, eligibility, CI, both bounds, N3-REM test, post-hoc 1 h bins with pair weighting,
  what a specification is) separated from "What was fixed when" (14 Oct log: statistic, delta, bootstrap bound, Holm;
  pilot with an earlier feature set; t bound and bins later; logged second-half control too imprecise, t bound
  +0.035; N3-REM test in code, not the log, so not pre-registered, bootstrap made paired; subcortical rule logged).
  Feature counts of the extra BOLD blocks reduced to the total; stacking, eye block, eligibility defined at first use.
- Results in the planned order: what BOLD adds (with the N3 gain +0.148 and REM -0.011 over the 10 subjects of the
  N3-REM test) -> time of night -> logged subcortex (rule first, then verdict) -> robustness (Table 2) -> what
  explains N3 (shared epochs, rebuild on its own 12,694 epochs with its starting point named, eye block on 97% of
  epochs) -> five-class staging. Cut to clauses: physiology-alone across/within bins, tones and somatomotor ranges,
  residual slow physiology numbers, lagged physiology, tSNR rho, SSL transductive AUC, leakage AUCs, per-epoch
  kappa kept only as the evidence for the HMM explanation; transfer check to one Limitations sentence.
- Discussion: planned-model non-superiority does not bound BOLD's REM information; flexible estimators suggest a
  REM-specific part of the order of the margin; margin verdicts at third-decimal differences called borderline.
- Table 1: F1 of N3 added; 7.5 pt; caption maps every column. Table 2: 34 -> 16 rows (every REM CI above zero on its
  own row; other rows as range rows with the count inside the margin; exceptions and below-10/10 rows named in the
  caption); 7.5 pt. Fig. 1: three marker rows in panel a, visible chance line, count columns clear of the CI bars,
  panel c y label and REM line dotted (delta dashed).
- 26: Table 2 display plan generated from the same rob list (nSpec and all envelope macros unchanged), new macros
  hrEncREM, leLateStratREMci, tabRangeOut, tabRangeCtr, with asserts. 27: Table 1 parser for the new labels and the F1
  columns (449 checks ALL PASS).
Reviews (fresh subagents, PDF only):
  Verified items fixed -> draft 26.
  N3-REM arithmetic, post-hoc bins, second-half result, rebuild start, ungrammatical Discussion sentence, Fig. 1c).
Declined: retitling (asked again by both ISBI reviewers; the title is the author's decision), new analyses
(combined-controls N3 gain, N3-REM CI under each control, per-variant enumeration of the 21 data variants), N3 CIs in
Table 2 (lower bound given in the caption), tables above 7.5 pt (no space). Stopping rule reached (two rounds).
Draft 27: dreams/paper_v2_draft27_2026-10-21.pdf, paper_isbi/main_v2_draft27.tex. 4 pages, 0 overfull, 0 undefined,
fonts embedded, 449 checks ALL PASS. submission_form.txt abstract updated (192 words). Public repo not updated.

## 2026-10-21 (14:00): all controls combined (56_combined_controls.py) and drafts 28-30
New analysis (asked by the author): BOLD's gain over the combined-conservative baseline (non-BOLD + 18 cardiorespiratory
staging features + eye block, as 47 conserv) for the derivative's features and for our own pipelines' 84-parcel
amplitude, on the 12,509 epochs every source covers (12 subjects; 11 assessable for N3, 9 for both stages), with LR,
boosted trees and the BiGRU on both sides. N3 gain:
  derivative 221 features: LR +0.083 [+0.056,+0.110] 11/11; trees +0.088; BiGRU +0.055 [+0.019,+0.098] 10/11
  derivative 84 parcels:   LR +0.046; trees +0.052; BiGRU +0.035
  rebuilt (affine, all regressors): LR +0.019 [-0.007,+0.042]; trees +0.024 [+0.003,+0.040]; BiGRU +0.036 [+0.006,+0.074]
  rebuilt (nonlinear + GM):          LR +0.009; trees +0.026 [+0.015,+0.037]; BiGRU +0.027 [+0.006,+0.056]
  minimal 84 parcels:      LR +0.000; trees +0.012; BiGRU +0.003
REM gain at most +0.015 (trees, derivative), N3-REM difference positive in 9/9 for the derivative with LR and BiGRU.
Reading: with the derivative the N3 gain survives every control at once (+0.055 to +0.088); with our own
preprocessing it is +0.000 to +0.036 (CI including zero with the planned model).
Paper (drafts 28-30): new title "What BOLD adds to whole-night fMRI sleep staging: preprocessing-dependent for deep
sleep, small for REM" (author to confirm; question form offered); Table 2 one row per specification; Table 3
(secondary comparisons with pipeline, baseline, epoch set, estimator per row: shared epochs, rebuilt pipeline, all
controls, within 1 h bins, subcortex), generated by 26 and checked by 27; Fig. 1 one column (panels a, b; old panel c
= Table 3a); transfer check dropped; abstract REM claim tied to endpoint and estimator (up to +0.073), N3 under all
(others 3), first-time reader 3. Draft 30: dreams/paper_v2_draft30_2026-10-21.pdf; 4 pages, 488 checks ALL PASS;
submission_form.txt title and abstract updated.

## 2026-10-21 (evening): draft 31
Reviewer fixes on draft 30 (matched 84-parcel comparison in the abstract and Discussion; verdict vocabulary
"non-superiority is shown / not shown"; dated (i)/(ii)/(iii) timeline in "What was fixed when"; rebuilt variants named
the same everywhere; variant ranges stated). New Table 4 (tab_allcontrols.tex): the all-controls gain per estimator
(LR, trees, BiGRU) for N3 and REM, five BOLD sets; it replaces the old Table 3c. Table 3 now has groups (a) shared
epochs, (b) rebuilt pipeline, (c) within 1 h bins; the subcortex group and the duplicate 200-parcel row are gone and
the logged subcortex verdict is one sentence with its two numbers. Abstract: "with richer cardiorespiratory features
and the eyeballs' EPI signal added to the baseline, 84 parcels common to all pipelines add +0.035 to +0.052 to N3 from
the derivative but +0.000 to +0.036 from our own pipelines". Discussion: from our own pipelines the parcels add little,
above zero only through a flexible estimator (new asserts in 26: rebuilt CIs above zero with trees and BiGRU; minimal
never above zero; Table 3/4 labels tied to the 84-parcel count). Typed "84" replaced by \nParcelsRaw.
27_verify_v2.py: Table 3 checks follow the new labels and groups (N3 and REM for every rebuilt row); new Table 4 checks
(every cell against the recomputed combined-controls gains, and the boldface rule against the saved CIs).
Draft 31: dreams/paper_v2_draft31_2026-10-21.pdf, paper_isbi/main_v2_draft31.tex. 4 pages, 0 overfull, 0 undefined,
552 checks ALL PASS. submission_form.txt abstract updated (208 words). Public repo not updated (still 7bfc585).

## 2026-10-21 (night): draft 32 (clarity pass on the r31 reviews)
inconsistent (title "small" vs encoder +0.048/+0.073, F1 +0.068); subcortex rule never stated and contradicted by the
boldface in Table 2; Robustness paragraph overloaded; drifting names and three unexplained epoch sets; Fig. 1 x-labels
clipped. Verified: the logged rule (next_steps.md, 2026-10-15 12:14) and its result (stacked +0.0292, 0.0008 short);
physiology 65 = (6 signals x mean, s.d. + RMSSD) x 5 windows; motion 25 = 4 statistics x 5 windows + 5 censored
fractions (derivative only; our pipelines use 20); the epoch sets differ because each comparison keeps only epochs
where all its feature sets are defined; the two +0.085 values differ unrounded (0.0852 vs 0.0851); the eighth bold
REM entry in Table 2 is the minimal pipeline with boosted trees, outside the 47 specifications.
Changes (no analysis changed): abstract as an N3 ladder (+0.139 over non-BOLD, +0.083 with all controls, +0.046 from
the 84 shared parcels, +0.000 to +0.019 from our pipelines, at most +0.036 nonlinear) and one REM verdict; "negligible"
defined once and used throughout; stage names and abbreviations expanded; Methods with one name per object (context,
self-supervised embedding, rebuilt, nonlinear, GM), feature counts decomposed, the eye block's use stated; analysis log
as plain sentences; Results split into "N3 above REM in every specification" and "REM depends on the estimator" (the
inside-margin positive entry explained, the encoder controls' purpose and the untested eye-block case stated, the
minimal-pipeline trees REM gain reported as a possible artefact); subcortex rule stated; Discussion reconciles the REM
message (BOLD carries REM information, mostly shared; REM gains under half the N3 gains of the same models);
Limitations name the two readings of the unattributed N3 half. Tables: Table 2 N3 column no longer boldfaced (all but
one CI above zero, asserted), ‡ list glossed; Table 3 row "Rebuilt, affine, before regression" and epoch-set note;
Table 4 caption gives the epoch-set reason. Fig. 1 bottom margin raised (labels no longer clipped), caption names the
model and feature sets. New asserts in 26 for every new claim (p floor, under-half ratios, rebuilt-only above zero,
inside-margin positive entry, 84-parcel labels). 27: Table 3 labels updated.
Draft 32: dreams/paper_v2_draft32_2026-10-21.pdf, paper_isbi/main_v2_draft32.tex; 4 pages, 0 overfull, 552 checks
(first-time reader). Public repo not updated; EDAS not submitted.

## 2026-10-21 (late night): draft 33 (r32 reviews)
Verified before fixing: Table 2's stronger + eyes rows use 13,100 epochs and the stronger rows all 13,836, so the old
"+0.118 -> +0.084" mixed epoch sets; replaced by the matched comparison on the 13,448 epochs of Table 3a (+0.142 ->
+0.083, eyeRefBold/eyeBoldOver). Table 4's identical-looking REM cells for the derivative's 221 and 84 features differ
unrounded (-0.01216/-0.01226, -0.03905/-0.03859; recomputed by 27). Within-bin per-model AUCs are not stored, so the
REM AUC-vs-F1 gap is stated as unresolved, not explained.
Changes (no analysis changed): abstract names the baseline of each N3 number, says rebuilding recovers about half
(new assert, gain metric too), REM verdict "not shown negligible within bins", CNN "untested against the eye signal";
Methods: map of epoch sets (13,836 / 13,448 / 12,694 / 13,100 / 12,509), baseline sizes (92 / 110 / 122 as macros),
regional level defined, delta's "for scale" framed as post hoc; Results: preprocessing stated like for like first
(+0.046 vs +0.000 to +0.019, same parcels and baseline) with its reading (moved from Limitations); eye effect on matched
epochs; REM AUC-vs-F1 gap acknowledged; Discussion credits +0.048 to the encoder and names the REM F1 gain. Tables:
Table 2 N3 column fully unbolded (variant range too), stronger-baseline groups collapsed to range rows (REM-CI-positive
row kept), minimal-pipeline rows dropped (text gives the trees REM gain with CI); Table 3 physiological-regression row
dropped (the text's range is attained by the shown rows, asserted), captions say which CIs are not shown. Fig. 1
height 1.44 in. Acknowledgments merged into the ethics section; two conference names abbreviated in references.bib.
Draft 33: dreams/paper_v2_draft33_2026-10-21.pdf; 4 pages, 552 checks ALL PASS; EDAS abstract 223 words. Fresh

## 2026-10-21 (night, later): draft 34 (r33 reviews)
contradicted Table 4 (rebuilt pipeline above zero with trees and the BiGRU); with the BiGRU the rebuilt pipeline matches
the derivative's 84 parcels (+0.036 vs +0.035), so the preprocessing gap holds for LR and trees only (asserted); Table
2's grouped "8/10" row made "44 of 47" uncheckable. Eye features: the eye-over-non-BOLD comparison (41_rv2_eyes.py) keeps
Table 3a's 13,448 epochs and sets missing eye features to the subject mean (wz -> nan_to_num) on 348 of them, while the
stronger + eyes analyses (47, 56) drop those epochs (13,100); rerun of 41 on the current features reproduced every number
exactly (backup in backups_2026-10-21/), and the paper now states the handling (macro eyMiss).
Changes (no analysis changed): abstract compares like for like per estimator (LR +0.046 vs +0.000 to +0.019; trees a
similar gap; BiGRU +0.036 rebuilt vs +0.035 derivative) and drops the +0.083 and "about half" from the abstract;
Discussion fixed and "small" defined (REM gains below half the N3 gain under every estimator); preprocessing and
all-controls paragraphs labelled post hoc, "about half" tied to the planned model in both metrics; how the regressors
could inject stage information stated; Table 2 stronger rows individual again, the stronger + eyes row named "all three"
and parentheses mean feature counts only; boldface labelled bootstrap CI in Tables 2 to 4; Table 4 caption says row 1
repeats Table 2's stronger + eyes row on fewer epochs; "negligible" defined as no gain of delta or more; "REM
concentrates late"; variant N3 range restored; ethics text shortened.
Draft 34: dreams/paper_v2_draft34_2026-10-21.pdf; 4 pages, 552 checks ALL PASS; EDAS abstract updated.

## 2026-10-21 (night, latest): draft 35 (r34 reviews)
5 of 6 right. Verified and fixed: Table 4 caption said row 1 "repeats" Table 2's stronger + eyes row (same
specification, different epochs; now said so, with each estimator's baseline N3 AUC 0.84 / 0.83 / 0.91); "BOLD alone
ranks REM worst" was false against Table 1 (now "below non-BOLD"); "below half the N3 gain under every estimator" failed
for the minimal pipeline with trees (+0.010 vs +0.012; now "every derivative specification", which is asserted); the
abstract's "stronger baseline" included the eyes, unlike Methods (now "a baseline enlarged by ..."), and the abstract's
preprocessing check is flagged post hoc; untabulated numbers (+0.024 minimal trees REM, +0.142 -> +0.083 eye effect)
state their basis, and the coincidence with Table 4's +0.083 is flagged; "censoring mask"; "stage sleep" without "well";
"margin fixed before the main analysis"; the weak-bound caveat moved into Results; Table 2's nested-tuning rows share a
range row; Methods' CNN naming unified. Draft 35: dreams/paper_v2_draft35_2026-10-21.pdf; 4 pages, 552 checks ALL PASS.

## 2026-10-22: draft 36 (r35 reviews: statistician, skim reader)
Verified and fixed: the CNN encoder's REM gain (+0.048 [+0.028,+0.070], 10/10, one-sided lower bounds +0.031 bootstrap
and +0.028 t) meets the bar the log set for the subcortical block, so the text now says it "would count as a gain"
(asserted); the like-for-like within-bin value is +0.062 (same comparison, nb+enc_nn - nb_nn), +0.073 is late fusion,
so the abstract now gives +0.048 (+0.062 within bins); the regression caveat now applies to our rebuilt pipeline too
(its recovered half also comes from regressing stage-varying signals); delta's scale framed as "for scale ... here";
the analysis log names what is confirmatory; five Table 3 REM CIs below zero are now mentioned (macro tThreeRemNeg);
abstract like for like and "47 specifications on the distributed data"; Introduction opens with the confound question;
Discussion concrete (no drift-inflated encoder AUC); caption take-home lines for Fig. 1 and Tables 2 to 4 (asserted);
Table 3 group (c) folded into the text with CIs (leStratREMci, leLateStratREMci); 27 updated (548 checks).
Open, needing the author: (1) title "small for REM" vs the encoder meeting the logged gain bar; (2) the title's
"preprocessing-dependent" has no paired test of pipeline differences (could be computed from saved predictions; a new
Draft 36: dreams/paper_v2_draft36_2026-10-22.pdf; 4 pages, 548 checks ALL PASS; EDAS abstract updated.
Draft 37 (2026-10-22, rest of r35A's statistical audit): "does not raise" instead of "lowers" for the planned model's
REM AUC (CI includes zero); "reliably positive" dropped (bootstrap-only, uncorrected), Table 2 caption too; "similar
point gain" instead of "matches"; "do not raise it" instead of "add nothing"; ceiling effect scoped to the planned model;
contribution (iii) worded as robustness; post-hoc tags on the 1 h bins and the per-subject N3-REM claim in the abstract,
which is scoped to the planned model; the 16-subject epoch count stated (16,671); Table 4's largest REM gain flagged as
above zero on its epochs only (asserted against Table 2's row); analysis log merged ("only these are confirmatory").
Draft 37: dreams/paper_v2_draft37_2026-10-22.pdf; 4 pages, 548 checks ALL PASS; EDAS abstract updated.
Still open for the author: title wording, a paired test for the pipeline differences, scope.

## 2026-10-22 (late morning): author decisions -> draft 38, paired test, table A/B
Author: general title; run the paired test behind "preprocessing-dependent"; check how trimming tables affects review
scores; remove unnecessary references.
- Title: "What BOLD adds to whole-night fMRI sleep staging" (question, no result; one line). submission_form.txt updated.
- NEW 57_paired_pipeline_diff.py (post hoc, saved predictions only; results_paired_pipeline.json): per-subject
  differences of with-BOLD AUCs (= difference of gains, shared baseline), bootstrap CI, t bounds, Wilcoxon.
  All controls (Table 4, N3): derivative 84 minus rebuilt affine / rebuilt nonlinear GM / minimal:
    LR    +0.028 [+0.013,+0.048] / +0.037 [+0.022,+0.053] / +0.046 [+0.026,+0.066]
    trees +0.028 [+0.013,+0.044] / +0.026 [+0.010,+0.042] / +0.039 [+0.015,+0.067]
    BiGRU -0.001 [-0.026,+0.017] / +0.008 [-0.005,+0.022] / +0.032 [+0.016,+0.049]
  rebuilt affine minus minimal: LR +0.018 [-0.008,+0.044], trees +0.012 [-0.016,+0.036], BiGRU +0.033 [+0.013,+0.054].
  Ladder (Table 3b, plain baseline, LR): derivative minus before-regression +0.092 [+0.058,+0.128]; minus rebuilt
  affine +0.041 [+0.017,+0.073]; regression step +0.051 [+0.019,+0.088]; nonlinear + GM -0.014 [-0.035,+0.009];
  recovered fraction 55% [25%, 81%].
  Reading: preprocessing matters under every estimator (minimal always falls short), but whether the derivative beats
  our rebuild depends on the estimator (LR, trees yes; BiGRU no).
  26 macros pd* with asserts; 27 recomputes the paired means (555 checks ALL PASS).
- References: snoek2019 removed (the only citation not backing a definition, a method or a claim); the rest stay.
- Table A/B: identical text (paper_isbi/main_v2_draft38.tex), Table 2 full (X, one row per specification, ~8 lines
  onto page 5) vs compact (Y, rows with a REM CI above zero kept, others in range rows; 4 pages; T2COMPACT=1 for 26).
  Blind copies dreams/ab_2026-10-22/review_copy_X.pdf, review_copy_Y.pdf (key in dreams/ab_2026-10-22_KEY.md); two
  identical-prompt ISBI reviewers per copy (review_X1, X2, Y1, Y2).
  want more table evidence (one-sided bounds, Table 4 CIs), not less. Decision: keep the full Table 2.
  All four flag the CNN encoder never tested over the eye block (53_learned_encoder.py run took ~16 min; an eye-block
  variant is a new analysis, awaiting the author).
Draft 38 (dreams/paper_v2_draft38_2026-10-22.pdf): new title; paired tests in abstract and Results; full Table 2;
Table 3a folded into the text (Table 3 = rebuilt pipeline only; tThreeRemNeg now 3); Fig. 1 caption "across the night,
negligibly to REM"; Discussion "may carry some REM information ... untested against the eye block"; contribution (iii)
"on the distributed data"; snoek2019 dropped. 4 pages, 555 checks ALL PASS; EDAS title and abstract updated.
