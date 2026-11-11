"""Pick the pilot subset: 30 single-step test calls, 10 per label, spread over the number of present fields.

Per label the 10 calls are split 3 / 3 / 4 over n_fields = 5 / 6 / 7 (all strata have enough records). Records whose
October gpt score is an error (3 provider refusals on the full input) are excluded: their full-coalition value is
undefined, so no Shapley value could be computed. The first call of each label with a different n_fields forms the
3-call pilot (direct_execute n=5, ask_human n=6, refuse n=7).
Output: results/subset_ids.json
"""
import json
import random
from collections import Counter

import pandas as pd

from common import CLASSES, DATA_PATH, MODEL, OCT_SCORES, SUBSET_PATH
from fields import CallFields

SEED = 20261110
PER_N = {5: 3, 6: 3, 7: 4}
PILOT_N = {"direct_execute": 5, "ask_human": 6, "refuse": 7}


def main():
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    oct_ = pd.read_csv(OCT_SCORES)
    oct_err = set(oct_[(oct_.model == MODEL) & oct_.error.notna()].dataset_idx)

    pool = {}   # (label, n_fields) -> [dataset_idx]
    fmt_counts = Counter()
    for i, r in enumerate(data):
        if sum(m.get("role") == "user" for m in r["prompt"]) != 1 or i in oct_err:
            continue
        c = CallFields(r)                       # asserts the round trip for every candidate
        fmt_counts[c.fmt] += 1
        pool.setdefault((r["meta"]["decision"], len(c.players)), []).append(i)

    rng = random.Random(SEED)
    chosen, pilot = [], []
    for lab in CLASSES:
        for n, k in PER_N.items():
            cand = sorted(pool[(lab, n)])
            assert len(cand) >= k, (lab, n, len(cand))
            pick = rng.sample(cand, k)
            chosen += [{"dataset_idx": i, "label": lab, "n_fields": n} for i in pick]
            if PILOT_N[lab] == n:
                pilot.append(pick[0])

    out = {"seed": SEED, "per_n_fields": PER_N, "excluded_oct_gpt_errors": sorted(int(x) for x in oct_err),
           "single_step_round_trip_formats": dict(fmt_counts),
           "pool_sizes": {f"{lab}|{n}": len(v) for (lab, n), v in sorted(pool.items())},
           "pilot_ids": pilot, "calls": chosen}
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(SUBSET_PATH, "w"), indent=2)
    print(f"single-step candidates (round trip OK): {sum(fmt_counts.values())}  formats: {dict(fmt_counts)}")
    print("pool sizes:", out["pool_sizes"])
    print("chosen:", Counter((c["label"], c["n_fields"]) for c in chosen))
    print("pilot ids:", pilot)
    print(f"written: {SUBSET_PATH}")


if __name__ == "__main__":
    main()
