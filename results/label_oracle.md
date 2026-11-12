# Where do DriveLM turn labels come from? (P2, nuScenes box yaw)

Linked P2 questions: train 7540, val 1145. Val labels: {'going_ahead': 1064, 'turn_left': 38, 'turn_right': 43}.

Rule: |heading change| >= threshold -> turn (sign gives left/right), else going ahead. Threshold tuned on train (balanced accuracy), scored on val. Past and future rows of one horizon use the same questions (objects tracked h seconds both ways). For reference, R1 on all val P2: balanced accuracy 33.9, turn recall 1.2; the x8 oversampled model: 46.7 / 34.9.

| Window | best threshold (deg) | val balanced acc (%) | val turn recall, right direction (%) | val predicted-turn rate (%) | val n |
| --- | --- | --- | --- | --- | --- |
| past, 1 s | 3 | 75.0 | 74.0 | 25.8 | 1099 |
| future, 1 s | 3 | 83.8 | 85.7 | 24.2 | 1099 |
| past, 2 s | 5 | 74.9 | 75.0 | 26.1 | 835 |
| future, 2 s | 7 | 88.0 | 91.1 | 21.8 | 835 |
| past, 3 s | 8 | 73.3 | 70.8 | 23.9 | 669 |
| future, 3 s | 12 | 86.2 | 87.5 | 20.8 | 669 |

Sign check (train turn questions above threshold): share whose heading change has the sign of the label (left = counter-clockwise). Near 100% = labels use the object's own left/right; near 0% = mirrored.

- past1: 93.6%
- future1: 93.8%
- past2: 92.9%
- future2: 93.2%
- past3: 92.7%
- future3: 93.4%

Reading: compare the best past window with the best future window. A model that only sees the past (multi-frame input) cannot beat the past row; the future rows show how much of the label is about what the object will do next.
