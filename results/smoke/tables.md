# SMOKE TEST on synthetic data (not results)

All rows: same fixed val ids (scene-level split). F1 at tau = 100 px.

## Table 1: main results

| Exp | P1 F1 (%) | Halluc. (%) | P2 status m-F1 (%) | P2o observed m-F1 (%) | P3 category (%) | P4 bal. acc (%) | P5 region F1 (%) | P6 region status m-F1 (%) | Parse-fail (%) | Latency p50 (s) | VRAM (GB) | Warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_mock | 86.22 | 3.81 | 74.29 | 69.56 | 80.28 | 80.48 | 80.27 | 88.73 | 0.00 | – | – | – |
| smoke_prior | – | – | 24.02 | 37.72 | 50.70 | 72.95 | 62.71 | 34.60 | 0.00 | – | – | COLLAPSE: P2 has 1 distinct answers over 71 samples; COLLAPSE: P2o has 1 distinct answers over 71 samples; COLLAPSE: P3 has 1 distinct answers over 71 samples |

Grounding rate (% of predicted P1 tags inside some nuScenes box): smoke_mock 95.24

P1 F1 at every tau:

| Exp | tau=50 | tau=100 | tau=200 |
| --- | --- | --- | --- |
| smoke_mock | 48.89 | 86.22 | 89.78 |

## Table 2: camera ablation

| Exp | Cameras | Camera names | P1 F1 CAM_FRONT (%) | P1 F1 other cams (%) | Hallucination (%) | Image tokens |
| --- | --- | --- | --- | --- | --- | --- |
| smoke_mock | 6 | True | 73.33 | 88.21 | 3.81 | – |

## Table 3: generalisation (P1 F1 at tau = 100 px / hallucination %)

| Subset | smoke_mock |
| --- | --- |
| city=boston | 86.13 / 6.15 (n=18) |
| city=singapore | 86.36 / 0.00 (n=12) |
| condition=day | 84.78 / 6.82 (n=12) |
| condition=night | 86.36 / 0.00 (n=12) |
| condition=rain | 88.89 / 4.76 (n=6) |

## Table 4: label-noise-aware scores

Rows whose DriveLM label contradicts the nuScenes annotation (flag `nusc_disagree`, set by `make data`) are dropped.

| Exp | P2 acc all / clean (%) | P2o acc all / clean (%) | P4 bal. acc all / clean (%) | P3 vs DriveLM / vs nuScenes (%) |
| --- | --- | --- | --- | --- |
| smoke_mock | 74.65 / 75.00 | 70.42 / 70.00 | 80.48 / 80.27 | 80.28 / 80.28 |
| smoke_prior | 56.34 / 57.35 | 60.56 / 61.43 | 72.95 / 73.81 | 50.70 / 50.70 |

## Table 5: P2 (moving status) on every val P2 question

| Exp | n | turns | accuracy (%) | balanced acc (%) | macro-F1 (%) | turn recall, exact (%) | turn recall, any (%) | turn precision (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
