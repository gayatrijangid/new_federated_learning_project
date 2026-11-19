## Results with identical settings, nothing tuned
10 hospitals, 3 malicious, 30 rounds, accuracy %. The 3 malicious hospitals are **chosen at random for
every run**, so each row below is the **mean ± standard deviation over 15 different random attacker sets**
(every defense faces the same 15 sets). The "none" row has no attackers, so it is a single run.
Soft Weighting uses lambda = 0.5 and the mean-based detector on BOTH datasets; nothing was tuned.

**Pima Indians Diabetes** (test set = 154 patients, 1 patient = 0.65 points)

| Attack | FedAvg | Multi-Krum | Median | Hard Excl. | Soft Wt. |
|---|---|---|---|---|---|
| none (control) | 72.1 | 72.1 | 72.7 | 73.4 | 73.4 |
| label_flipping | 69.1 ± 1.9 | 72.3 ± 1.2 | 72.0 ± 1.4 | 70.8 ± 2.6 | 72.7 ± 1.4 |
| sign_flipping | 70.0 ± 2.6 | 71.9 ± 1.3 | 71.0 ± 2.7 | 71.7 ± 1.3 | 71.6 ± 1.6 |
| byzantine | 53.0 ± 12.5 | 72.6 ± 0.7 | 72.2 ± 0.8 | 72.5 ± 0.7 | 72.4 ± 0.6 |
| gradient_scaling | 72.7 ± 0.6 | 71.9 ± 0.8 | 72.9 ± 0.6 | 71.9 ± 0.6 | 72.2 ± 0.6 |
| gaussian_noise | 69.9 ± 3.2 | 72.6 ± 0.7 | 72.5 ± 0.9 | 72.5 ± 0.7 | 72.3 ± 0.8 |
| mix label+sign | 70.3 ± 3.0 | 72.7 ± 1.6 | 69.8 ± 1.3 | 72.3 ± 1.7 | 72.7 ± 2.1 |
| mix all 10 attacks | 55.2 ± 7.7 | 72.5 ± 0.6 | 72.7 ± 1.1 | 71.7 ± 1.1 | 72.0 ± 0.6 |

**Breast Cancer Wisconsin (Diagnostic)** (test set = 114 patients, 1 patient = 0.88 points)

| Attack | FedAvg | Multi-Krum | Median | Hard Excl. | Soft Wt. |
|---|---|---|---|---|---|
| none (control) | 98.2 | 98.2 | 97.4 | 99.1 | 99.1 |
| label_flipping | 87.4 ± 4.0 | 98.1 ± 0.5 | 97.0 ± 0.6 | 95.9 ± 5.5 | 98.0 ± 0.8 |
| sign_flipping | 93.6 ± 3.1 | 97.4 ± 0.6 | 96.3 ± 1.3 | 95.7 ± 1.3 | 96.7 ± 1.2 |
| byzantine | 48.8 ± 23.7 | 98.2 ± 0.5 | 97.8 ± 0.4 | 98.2 ± 0.5 | 98.2 ± 0.5 |
| gradient_scaling | 97.8 ± 1.0 | 98.0 ± 0.7 | 98.1 ± 0.5 | 97.9 ± 0.6 | 98.0 ± 0.5 |
| gaussian_noise | 92.7 ± 2.9 | 98.1 ± 0.4 | 97.8 ± 0.8 | 98.1 ± 0.4 | 98.1 ± 0.4 |
| mix label+sign | 90.5 ± 2.4 | 97.8 ± 1.1 | 96.8 ± 1.1 | 97.1 ± 1.1 | 97.4 ± 1.3 |
| mix all 10 attacks | 71.0 ± 9.9 | 97.7 ± 0.7 | 97.7 ± 0.4 | 98.8 ± 0.4 | 98.8 ± 0.4 |

## How to read this honestly
- Strong attacks wreck plain FedAvg (byzantine: 53% on Pima and 49% on Breast Cancer; all attacks
  mixed: 55% and 71%), and its result swings by 8-24 points depending on WHICH hospitals are
  malicious. Every defense stays within about 1 point of its no-attack accuracy.
- Because the attackers are random, a single run can mislead. Always report mean ± std over many
  attacker sets, as in the tables above, and state how many.
- Among the defenses the gaps are small, mostly 1 point or less (one patient is 0.65 points on
  Pima and 0.88 on Breast Cancer). Multi-Krum has the best mean in most rows, but it is told
  the exact number of attackers (f = 3); Soft Weighting and Hard Exclusion are not. Soft Weighting
  is best on Pima label flipping and ties for best on Breast Cancer with all attacks mixed
  (98.8%), but it trails Multi-Krum by about 0.7 points on Breast Cancer sign flipping.
  Do not claim Soft Weighting is "the best"; say it is competitive without being told the
  attacker count, and far better than no defense.
- Breast Cancer is an easy dataset (98% even with no attack), so it shows "defense vs no defense"
  well but cannot separate the defenses. Pima is harder.
- Both datasets are small. Backdoor and Sybil attacks are not expected to be caught by the
  distance-based detector.

## Possible third dataset (future work)
UCI Heart Disease was collected at four hospitals (Cleveland, Hungary, Switzerland, VA Long
Beach), so it has a natural per-hospital split and is a good fit for federated learning.
