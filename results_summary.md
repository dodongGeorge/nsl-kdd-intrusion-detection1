# Results (KDDTest+)

## Overall

| Metric | Before | After |
|---|---|---|
| Accuracy | 73.97% | 82.01% |
| Macro-F1 | 48.71% | 69.59% |
| Attacks detected | 57.95% | 76.40% |
| Normal flagged as attack | 2.70% | 3.46% |

## Per class

| Class | Recall before | Recall after | F1 before | F1 after | Precision before | Precision after |
|---|---|---|---|---|---|---|
| Normal | 97.30% | 96.54% | 76.96% | 84.78% | 63.65% | 75.58% |
| DoS | 76.45% | 78.71% | 85.14% | 86.49% | 96.06% | 95.98% |
| Probe | 62.41% | 85.96% | 72.54% | 84.40% | 86.59% | 82.91% |
| R2L | 0.35% | 39.62% | 0.69% | 52.26% | 90.91% | 76.76% |
| U2R | 4.48% | 26.87% | 8.22% | 40.00% | 50.00% | 78.26% |

## R2L specialist validation

Threshold: 0.35; validation precision: 98.49%; validation recall: 98.49%.

## R2L / U2R recall on attack types seen vs unseen in training

| Class | Type | Records | Recall before | Recall after |
|---|---|---|---|---|
| R2L | seen | 2199 | 0.45% | 51.16% |
| R2L | unseen | 686 | 0.00% | 2.62% |
| U2R | seen | 37 | 5.41% | 35.14% |
| U2R | unseen | 30 | 3.33% | 16.67% |

## Best grid-search settings

| rare_n | max_n | min_samples_leaf | CV macro-F1 |
|---|---|---|---|
| 3000 | 8000 | 1 | 0.9424 |

The after metrics include the existing R2L probability boost and the validation-tuned R2L specialist. Threshold selection used a held-out split from KDDTrain+; KDDTest+ was reserved for final evaluation. Results may vary after retraining.