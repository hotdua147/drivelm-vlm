# Multi-camera driving-scene VLM on DriveLM-nuScenes, verified against nuScenes 3D boxes

Fine-tune **Qwen3-VL-2B** with LoRA to answer the **perception** questions of DriveLM-nuScenes from
**6 camera images**, and check every answer against **projected nuScenes 3D boxes**, not only against
reference text. Phase 1 (perception) of an end-to-end driving project; trained on one 16 GB GPU in ~4 h.

**Reports (Vietnamese):** technical report `Technical_report_giai_doan1.pdf` (14 pages, paper format: findings, paired CIs, reproducibility statement, glossary); detailed report `Bao_cao_ky_thuat_giai_doan1_day_du.pdf` (15 pages); short version `Bao_cao_giai_doan1_DriveLM_perception.pdf` (5 pages). **Phase-2 plan:** `Ke_hoach_giai_doan2.pdf`.
**Every number of the short report is re-checked by one command:** `make verify` (239/239 match, 2026-11-11).

## Results (fixed val: 700 questions, 100 per type, 91 held-out scenes)

| Model | P1 F1 | Halluc. | P2o | P3 | P4 | P5 | P6 | Latency p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0: most frequent answer per question template | – | – | 39.4 | 60.7 | 73.1 | 56.7 | 36.9 | – |
| B1: SigLIP2 zero-shot on a crop at the tag | – | – | 37.8 | 73.0 | – | – | – | 0.03 s* |
| B2: Qwen3-VL-2B zero-shot | 0.0 | – | 32.7 | 68.5 | 59.3 | 21.8 | 8.1 | 0.46 s |
| B2 + 2 example answers (few-shot) | 3.3 | 54.6% | 35.9 | 60.7 | 63.0 | 64.0 | 13.5 | 0.21 s |
| LoRA, same recipe, **no images** | 19.7 | 19.0% | 61.8 | 70.8 | 71.7 | 59.2 | 34.6 | 0.10 s |
| **R1: LoRA, 6 named camera images** (seed 0) | **55.0** | 5.4% | **80.9** | **91.0** | 80.8 | **79.7** | **53.6** | 0.22 s |
| R1 seeds 1 / 2 | 54.9 / 54.2 | 4.8 / 6.8% | 82.4 / 82.2 | 89.9 / 88.8 | 80.0 / 81.5 | 79.3 / 77.1 | 55.4 / 52.7 | 0.23 s |
| R2: LoRA, one 3x2 grid image | 54.9 | 5.7% | 79.3 | 84.3 | 77.3 | 76.6 | 47.0 | 0.24 s |

P1: important objects as `<c,CAM,x,y>` tags, Hungarian F1 at 100 px. P2o/P6 macro-F1, P3 accuracy, P4 balanced
accuracy, P5 category-set F1. RTX 5060 Ti 16 GB. *B1 runs an encoder on one crop and generates no text.

**Findings** (A − B differences from a paired scene bootstrap on the same questions, 95% CI):

* **Images add +19 to +35 points** over the blind model trained with the same recipe on P1, P2o, P3, P5, P6
  (P1 +35.4 [+29.9, +40.9]) and cut hallucinated tags from 19.0% to 5.4%. P4 is the exception (+9.1 [−0.6, +19.8]).
* **Text alone already explains part of the gain:** the blind model beats B0 by +22.4 on P2o, so comparing with B0
  only would overstate what the images do.
* **Fine-tuning learns localisation, not only the format:** with two example answers, the base model writes tags
  (parse failures 16.7% → 4.6%, P5 21.8 → 64.0) but reaches only 3.3 P1 F1 with 54.6% hallucinated tags.
* **R1 uses the image of the camera asked about:** shown CAM_FRONT only, on questions about the other five cameras
  P1 drops from 51.9 to 0 and P2o/P3/P5/P6 fall 18–44 points, to about the blind model.
* **Separate images vs grid:** equal except P3 (+6.7 [+1.1, +14.0] for separate images); the grid costs more tokens (630 vs 588) and time.
* **Data vs training length:** 10k → 40k samples (1 epoch) raises P1 by 5.3, but 10k samples × 4 epochs (same 2,500
  steps) reaches the same P1; at equal steps no metric shows a significant data effect.
* **Seeds:** three R1 seeds stay within 2.7 points on every metric; the val set size (±4 to ±20 points) is the main uncertainty.
* **Motion is the limit of a single frame.** On all 1,153 val "moving status" questions (83 turns) R1 almost never
  predicts a turn (balanced accuracy 33.9). Oversampling turns ×8 detects "some turn" better than chance but gets the
  direction right on 48% of caught turns. An oracle on the true nuScenes box heading reaches **75.0** from the past
  1 s and **88.0** from the next 2 s: DriveLM's turn labels are partly about the future, and past motion alone leaves
  large headroom → phase 2 adds multi-frame input.

## Label and scorer checks

* DriveLM key-object tags linked to a projected nuScenes box: **99.1%** (traffic lights/signs are not annotated in nuScenes).
* DriveLM labels agree with nuScenes: moving status **89.7%**, category **94.3%**, region yes/no **93.7%**.
  Disagreeing rows are flagged (`nusc_disagree`) and scored separately (Table 4).
* The keyword scorer was hand-checked on 100 R1 answers (20 per closed type): **99/100** correct; the one miss
  ("Road sign") is fixed. Checked by the scorer's author, not an independent rater.
* `scripts/check_runs.py`: all runs share the same question ids, image tokens / seeds / training sizes / adapters
  match each config, and every run has its own answers (the only identical pair answers "Going ahead." everywhere).

## Experiments

| ID | What | Config |
| --- | --- | --- |
| B0 | most frequent training answer per question template | `b0_prior` |
| B1 | SigLIP2 zero-shot, 320 px crop around the tag in the question | `b1_siglip` |
| B2 | Qwen3-VL-2B-Instruct zero-shot / with 2 text examples per question type | `b2_zeroshot`, `b2_fewshot` |
| T | LoRA exactly like R1 but without images | `r1_textonly` |
| R1 | LoRA r=16 on the language model, 6 named cameras, 40k samples, 1 epoch, camera dropout 0.15 | `r1_6cam`, `r1_seed1`, `r1_seed2` |
| R2 | as R1 with one 3x2 grid image | `r2_grid` |
| D | 10k samples × 1 epoch / × 4 epochs; 10k with turn questions oversampled ×8 | `r1_10k`, `r1_10k_4ep`, `r1_10k_turnx8` |
| A | R1 checkpoint evaluated with CAM_FRONT only / without camera names | `abl_front_only`, `abl_no_camname` |
| P2 | moving status on all 1,153 val questions | `p2full_*` |

Question types (DriveLM-nuScenes v1.1, 162,480 perception questions over 4,072 keyframes):

| Type | Template | Scored as |
| --- | --- | --- |
| P1 | What are the important objects in the current scene? | tags: Hungarian F1 at 50/100/200 px, grounding, hallucination |
| P2 | What is the moving status of object `<tag>`? | going ahead / turn left / turn right |
| P2o | What is the observed status of object `<tag>`? | moving / stationary, macro-F1 |
| P3 | What is the visual description of `<tag>`? | object category, accuracy |
| P4 | Are there {parked, moving, ...} {cars, ...} to the {region}? | yes/no, balanced accuracy |
| P5 | What are objects to the {region} of the ego car? | F1 over the set of categories |
| P6 | What is the status of the {objects} to the {region}? | moving / stationary / rider, macro-F1 |

A **hallucinated tag** lies inside no projected nuScenes box, more than 100 px from the nearest one, and matches no
DriveLM reference tag (so correct traffic lights are not counted). **Uncertainty**: 95% percentile bootstrap over
whole val scenes (Table 6) and paired bootstrap for A − B (`results/paired.md`).

## How to run

```bash
pip install -r requirements.txt
make test                         # 43 tests, CPU only
make smoke                        # synthetic data in the exact DriveLM/nuScenes formats, end to end
```

Data (non-commercial licences: DriveLM CC BY-NC-SA 4.0, nuScenes terms of use):

```
data/raw/drivelm/v1_1_train_nus.json            # huggingface.co/datasets/OpenDriveLab/DriveLM (gated)
data/raw/nuscenes/samples/CAM_*/...jpg          # drivelm_nus_imgs_train.zip from the same repo
data/raw/nuscenes/v1.0-trainval/*.json          # nuscenes.org, Trainval metadata (0.43 GB), no sensor blobs
```

```bash
make data                         # build data/processed + label checks (exits 1 if a check fails)
make run EXP=r1_6cam              # train (GPU, ~4 h on 16 GB) -> predict -> score; same for every config above
make paired                       # paired comparisons -> results/paired.md
make label-oracle                 # do turn labels follow past or future heading? -> results/label_oracle.md
make tables                       # results/tables.md (Tables 1-7)
make verify                       # re-check everything and every number of the report (~10 min, CPU)
```

Each run writes `runs/<MMDD>_<exp>_<data_version>_s<seed>/`: merged `config.yaml`, `git_sha.txt`,
`pip_freeze.txt`, `command.txt`, `_preds.jsonl`, `_report.json`, and for training runs the adapter,
`label_check_train.json` (what the loss sees) and generation samples.

## Repository

```
configs/        data/, model/, exp/ (one file per experiment, `_base_` inheritance)
src/dlp/data/   question templates and parsers, nuScenes projection (numpy, no devkit), tag-to-box linking,
                label checks, scene split, dataset / collator (answer-only loss)
src/dlp/models/ prior baseline, SigLIP2 baseline, Qwen3-VL (load, LoRA, generate, few-shot), mock (tests)
src/dlp/eval/   Hungarian matching, grounding / hallucination, metrics, scene and paired bootstrap
scripts/        build, train, predict, eval, tables, paired comparisons, label oracle, run check,
                report check, scorer audit export, failure images, synthetic data
tests/          43 tests incl. an end-to-end run on synthetic data
results/        log.md (dated decisions and corrections), report_numbers.json, audit_r1_checked.csv
```

## Licences

Code: see `LICENSE` (choose one before publishing). Data is **not** in this repository: DriveLM is CC BY-NC-SA 4.0,
nuScenes is under its own non-commercial terms of use; download both yourself (see "How to run").
`results/audit_r1_checked.csv` quotes 100 DriveLM questions and answers for the scorer audit (CC BY-NC-SA 4.0, OpenDriveLab).
`runs/*/` keeps only metrics, merged configs and commands; predictions and adapters stay local.

## Limitations

* 100 val questions per type: confidence intervals of ±4 (P5) to ±20 (P6) points; small differences are only
  claimed when the paired bootstrap supports them. Three seeds for R1 only.
* The val split is by scene but shares nuScenes' four areas: in-distribution, not a new-city test. Qwen3-VL may
  have seen nuScenes images in pre-training; all compared models share the backbone.
* Camera ablations reuse the R1 checkpoint. Open-loop QA does not measure driving quality.
* Conventions: trailers and construction vehicles count as trucks.

## What went wrong along the way (kept on purpose, see `results/log.md`)

The first question parsers did not match the real v1.1 templates; "standing", rider labels and "Road sign" were
mis-scored (B0's P6 was inflated from 36.9 to 53.7); a run lookup scored `r1_10k` with the oversampled adapter;
smoke-test runs once leaked into the tables. Several of my own conclusions were overturned by later controls:
"flat loss → more data won't help", then "more data helps" (it was training length); "oversampling learns the
turn direction" (wrong baseline); "±5–9 points is random noise" (it came from changing the training data).
