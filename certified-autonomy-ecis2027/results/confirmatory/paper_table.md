TABLE:
WIDTHS: 32,14,13,21,20
| Test on untouched calls | Scorer | Safe autonomy | Harmful execution | Over budget (harm / ask) |
| Frozen G4 policy, train split | jev | 27.6% | 1.6% [1.0%, 2.3%] | no / no |
| Frozen G4 policy, train split | microsoft | 24.1% | 1.5% [0.9%, 2.2%] | no / no |
| Frozen G4 policy, train split | mean of three | 27.3% | 1.5% [1.0%, 2.2%] | no / no |
| G2 recalibrated on each of 4 disjoint calibration samples | mean of three | 53.2% | 5.4% | 3 of 4 / not bounded |
| G4 recalibrated on each of 4 disjoint calibration samples | mean of three | 24.6% | 1.8% | 0 of 4 / 0 of 4 |
| G1 recalibrated on each of 4 disjoint calibration samples | microsoft | 35.4% | 3.6% | 0 of 4 / 2 of 4 |
| G4 recalibrated on each of 4 disjoint calibration samples | microsoft | 19.2% | 1.1% | 0 of 4 / 0 of 4 |
| Frozen G4 policy, unseen Mailu service | mean of three | 0.0% | 0.0% [0.0%, 5.2%] | no / no |
CAPTION: Confirmatory evaluation of the frozen design on calls never used during development (budgets: 5% harm, 15% ask; brackets: 95% Clopper-Pearson intervals; recalibrated rows give means and the number of samples over each budget).
