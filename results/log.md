# Experiment log

One entry per day: what ran, what we saw, what we decided.

## 2026-11-09
- Code for the full pipeline written; 27 unit/integration tests pass on CPU.
- Smoke test on synthetic DriveLM/nuScenes-format data: build -> prior/mock predict -> eval -> tables -> failure images all run.
- Linking switched to IoU with DriveLM's own 2D box when available: on synthetic data, point-only linking mislinked
  truncated objects at image edges (category agreement 97% -> 100%).
- Not run yet (needs GPU + real data): B1 SigLIP2, B2/R1/R2 Qwen3-VL.

## 2026-11-10
- Downloaded DriveLM-nuScenes v1.1 (`v1_1_train_nus.json`, `drivelm_nus_imgs_train.zip`); v1.0 is not on Hugging Face.
- `key_object_infos` fields and image paths match the reader. Perception wording did NOT match the planned P1-P4 regexes:
  30 templates, mostly region-level ("objects to the front right", "status of the trucks that are to the ...").
  Question types rewritten to P1, P2, P2o, P3, P4, P5, P6 (+ other); yes/no parser no longer reads "There are ..." as yes.
- Added: P4 answers checked against projected nuScenes boxes; stratified training sample and val (100 per type).
- `make data` on the real data: LABEL CHECK PASSED. 162,480 perception questions all classified ("other" = the 4 known free-form templates).
  Link rate 91.1% overall; by DriveLM category: Vehicle 99.3%, Pedestrian 99.8%, Cycle 99.7%, Others 86.9%, Traffic element 25.2%.
  Status agreement 89.5%, category agreement 88.1%, P4 vs nuScenes 89.9%.
- Decision: traffic elements are not linked (nuScenes has no such boxes, the 25% "links" were poles/barriers), and a predicted
  tag that matches a DriveLM GT tag is never counted as a hallucination. Re-run `make data` to refresh label_check.json.
- Label diagnosis (diag.txt): status 89.7% with stopped = stopped (79.8% if stopped = moving); category mismatches
  mostly from our keyword map (van, rider, bare "vehicle") -> fixed; P4 agreement falls when filtering to visible boxes.
- Changes: exact nuScenes class per P4 template, P4 definition variants measured (parked incl. stopped, neighbour cameras),
  rows contradicting nuScenes flagged `nusc_disagree`, Table 4 (all vs consistent subset), P3 scored vs DriveLM answer
  with nuScenes as second check. Next: re-run `make data`, read `p4_nuscenes_agreement_by_variant`.
- B2 examples checked by hand: the parser reads model answers correctly except "standing" (not mapped to stopped),
  which also mislabeled GT answers such as "One of the pedestrians is standing, and three are moving" (P6) and
  pedestrian "Standing." (P2o). Fixed; `eval_perception.py` now refreshes labels from data/processed so finished runs
  are re-scored without re-predicting. B2 zero-shot < B0 on most types is mostly genuine (no tag format, verbose lists).

## 2026-11-10 (afternoon)
- R1 (LoRA, 6 cameras, 40k balanced samples, 2,500 steps) done. vs B0: P1 F1 55.0 (halluc 5.4%, grounding 89.7%),
  P2o 80.9 vs 39.4, P3 89.9 vs 60.7, P4 80.8 vs 73.1, P5 79.7 vs 56.7; P2 31.8 = B0 (COLLAPSE), P6 55.4 vs 53.7.
  Loss 1.86 -> 0.34 by step 220, then flat around 0.26-0.31.
- P2 val: 91 going ahead, 6 turn right, 3 turn left, 0 stopped -> P2 asks turn direction of moving objects; R1 gets 0/9 turns.
- P6: mixed groups (moving+stopped) right 6/25; pure groups 50/68.
- P6 cycles ask about the rider ("One motorcycle is without a rider.", 1,993 train answers) -> were unscored (label None).
  Added rider / no_rider labels (not folded into moving/stopped). Re-run `make data` and re-evaluate all runs.
- Ablations on the R1 checkpoint: CAM_FRONT only -> P1 F1 on other cameras 0, P3 89.9 -> 69.7, P2o 80.9 -> 64.8,
  P5 79.7 -> 66.8 (close to B0): R1 uses the images of the asked camera, not only the question template.
  No camera names -> P1 0 (cannot write the CAM part of tags), P3 77.5: R1 relies on the name tokens (trained with names).
- B1 SigLIP2 crop: P3 73.0 (> B2 68.5), P2o 37.8, P2 13.3; ~0.016 s/question. B2 re-run at R1's pixel budget (588 tokens):
  p50 0.46 s vs R1 0.22 s. After adding rider labels, P6 macro-F1: B0 36.9, R1 53.6.
- Tables: types a model does not answer are shown as "–" and excluded from parse-fail; latency printed with 3 decimals.
- P2 on all 1,153 val P2 questions (83 turns, 7.2%): B0 turn recall 0%, R1 1.2% (exact) / 2.4% (any turn).
- Batch-2 speed test failed: transformers 5 returns a token-aligned extra tensor; collator now pads it (untested on GPU,
  so long runs stay at batch 1). B1 latency now measured on answered questions only.
- R2 config fixed before running: grid pixel budget 6x (same per-camera resolution as R1), camera dropout 0.15 as R1.
- Night run: R2 (grid), r1_10k, r1_10k_turnx8 trained and scored. BUG: run lookup used glob `*_r1_10k_*`, which also
  matches `r1_10k_turnx8`, so `p2full_r1_10k` scored the turnx8 adapter (rows identical). Fixed (exact match on the
  experiment name, Makefile uses the same helper); `p2full_r1_10k` must be re-run. Table 1 rows were not affected.
- r1_10k_turnx8 on all 1,153 val P2: turn recall (any) 73.5%, exact 34.9%, turn precision 7.6%, balanced acc 46.7
  (R1 40k: 2.4 / 1.2 / 11.1 / 33.9), accuracy 92.3 -> 67.4: oversampling trades many false turns for turn recall.

## 2026-11-10 (evening)
- p2full_r1_10k re-run after the lookup fix: turn recall 0/83 (same as R1 40k without oversampling).
- R2 grid vs R1 separate: no gain at equal per-camera resolution (P3 -6.7, P6 -6.6, more tokens, slower) -> keep separate.
- Data scaling 10k -> 40k: P1 +5.3, P2o +10.7, P3 +4.5 although train loss was flat. My earlier claim "flat loss ->
  little gain from more data" was wrong.
- Changing only the training sample moves other types by +-5-9 points (r1_10k vs r1_10k_turnx8) -> added 95%
  percentile-bootstrap CIs over questions (B=1000, seed 0) to every headline metric (`ci95` in _report.json,
  Table 6, `make reeval`). CIs cover val sampling noise only; run-to-run variance is measured with a second seed.
- New configs: r1_seed1 (R1, training seed 1, same split/val) and r1_80k (third data-scaling point).

## 2026-11-10 (23:30) — after the review questions
- CIs now resample whole val scenes (cluster bootstrap); question-level CIs kept as `ci95_question` for comparison.
- New: score split by the camera a question is about (`by_question_camera`, Table 7); P1 tags per answer;
  P2 any-turn precision and predicted-turn rate (Table 5). From the existing turnx8 numbers: ~383 predicted turns
  (33% of P2), any-turn precision ~15.9% vs 7.2% base rate; right direction among caught turns 29/61 = 48% (chance).
  My earlier claim that turnx8 "learned part of the turn direction" was wrong: it learned "turn-ish", not direction.
- 10k vs 40k was confounded with training length (1 epoch each: 625 vs 2,500 steps). New config r1_10k_4ep
  (10k x 4 epochs = 2,500 steps).
- Missing baseline: blind LoRA with the same recipe (r1_textonly, no images). Third seed: r1_seed2.
- scripts/export_audit.py: sample 100 scored answers for a hand check of the keyword scorer.

## 2026-11-11 (12:45) — overnight runs: blind LoRA, seed 1, 10k x 4 epochs
- r1_textonly (no images, same recipe): P1 19.7 (halluc 19%), P2o 61.8, P3 70.8, P4 71.7, P5 59.2, P6 34.6.
  Fine-tuning on text alone already beats B0 on P2o (+22) and P3 (+10): B0 underestimated the question prior.
- r1_seed1 vs r1_6cam (seed 0): every metric within 2 points -> training-seed noise is small at 40k; the +-5-9
  between r1_10k and r1_10k_turnx8 came from changing the training data, not from the seed.
- r1_10k_4ep (2,500 steps like R1): P1 55.1 = R1 -> the 10k->40k P1 gain was training length, not data.
  P2o 73.6 vs R1 80.9: status questions do benefit from more data.
- New: paired scene bootstrap (scripts/compare_runs.py, `make paired`) for A - B on the same questions.
- New: scripts/label_oracle.py (`make label-oracle`): do P2 turn labels follow the object's past or future
  heading change (nuScenes box yaw, 1-3 s)? Threshold rule tuned on train = oracle upper bound.
- New: b2_fewshot (zero-shot + 2 text examples per question type) for review question 10.
- Hand check of 100 scored R1 answers (20 each of P2o, P3, P4, P5, P6; checked by the scorer's author, not an
  independent rater): scorer right on 99/100. One miss: "Road sign." was not a traffic element (fixed; any "sign"
  now counts). Convention noted: trailer / construction vehicle fold into truck (2 P5 rows).
  Model errors seen: P6 mixed groups (7/20 GT "some moving, some parked") -> R1 always answers one status (0/7);
  P5 misses minority classes (15 missed vs 2 extra classes in 20 answers) -> recall, not hallucination.

## 2026-11-11 (18:10) — integrity check (scripts/check_runs.py)
- All 14 runs on the fixed val share one question-id hash (also after rebuilding data today); the 4 full-P2 runs
  share another. Image tokens, seeds, training sizes and adapter sources match each config; every run has its own
  answers except p2full_b0 = p2full_r1_10k, both "Going ahead." on all 1,153 questions (collapse, not a mix-up).
- Seed 2: P1 54.25, P2o 82.17, P3 88.76, P4 81.47, P5 77.07, P6 52.73 -> 3 seeds within 2.7 points.
- b2_fewshot: parse-fail 16.7 -> 4.6, P5 21.8 -> 64.0, but P1 3.3 with 54.6% hallucinated tags.
- `make verify`: one command re-checks everything (tests, check_runs, re-score, paired CIs, label oracle, tables)
  and compares every number printed in the report (results/report_numbers.json) with the fresh results.

## 2026-11-11 (19:00) — README rewritten with final results; phase-2 plan (Ke_hoach_giai_doan2.pdf)
- Removed configs/exp/r1_80k.yaml (never run: 80k x 1 epoch would again mix data size with training length).
