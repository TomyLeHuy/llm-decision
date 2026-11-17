TABLE:
WIDTHS: 32,14,13,21,20
| Test on untouched calls | Scorer | Safe autonomy | Harmful execution | Over budget (harm / ask) |
| Frozen G4 policy, train split | jev | 29.4% | 0.4% [0.1%, 0.8%] | no / no |
| Frozen G4 policy, train split | microsoft | 26.0% | 1.3% [0.8%, 2.0%] | no / no |
| Frozen G4 policy, train split | mean of three | 28.9% | 0.3% [0.1%, 0.8%] | no / no |
| G2 recalibrated on each of 3 disjoint calibration samples | mean of three | 66.7% | 3.4% | 0 of 3 / not bounded |
| G4 recalibrated on each of 3 disjoint calibration samples | mean of three | 35.8% | 0.6% | 0 of 3 / 0 of 3 |
| G1 recalibrated on each of 3 disjoint calibration samples | microsoft | 40.5% | 1.7% | 0 of 3 / 2 of 3 |
| G4 recalibrated on each of 3 disjoint calibration samples | microsoft | 25.9% | 1.4% | 0 of 3 / 0 of 3 |
| Frozen G4 policy, unseen Mailu service | mean of three | 0.0% | 0.0% [0.0%, 5.2%] | no / no |
CAPTION: Confirmatory evaluation of the frozen design on calls never used during development (budgets: 5% harm, 15% ask; brackets: 95% Clopper-Pearson intervals; recalibrated rows give means and the number of samples over each budget).
