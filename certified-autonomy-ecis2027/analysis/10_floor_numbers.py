"""Step 10. Numbers quoted with Propositions 1 and 2 (Section 4.4, 5.6, 5.8, Figure 7a).
k_max(n): largest k with BinomCDF(k; n, alpha) <= delta/m; n_min: smallest n for which any k qualifies = ceil(ln(delta/m)/ln(1-alpha)).
Usage: python analysis/10_floor_numbers.py results/r10_floor_numbers.json"""
import json, math, sys
from scipy.stats import binom
def k_max(n, a, dm):
    ks = [k for k in range(n + 1) if binom.cdf(k, n, a) <= dm]; return max(ks) if ks else None
def n_min(a, dm): return math.ceil(math.log(dm) / math.log(1 - a))
R = {"prop1_threshold": {n: n - math.ceil((n + 1) * 0.95) for n in [101, 202, 364]},
     "k_max_a5_d10_m9": {n: k_max(n, 0.05, 0.10 / 9) for n in [101, 202, 304, 364, 405]},
     "k_max_a5_d10_m3": {n: k_max(n, 0.05, 0.10 / 3) for n in [202, 364]},
     "n_min_G4_m9": {f"a{a}_conf{1 - d:.2f}": n_min(a, d / 9) for a in [0.02, 0.05, 0.10] for d in [0.20, 0.10, 0.05, 0.01]},
     "n_min_G5_simultaneous_4groups": n_min(0.05, 0.10 / 2 / (9 * 4)), "n_min_G5_per_service": n_min(0.05, 0.10 / 2 / 9)}
print(json.dumps(R, indent=1)); json.dump(R, open(sys.argv[1], "w"), indent=1)
