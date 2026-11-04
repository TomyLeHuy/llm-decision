"""A3b (exploratory addition, no API calls): how much do multi-turn calibration calls cost the G4 certificate?

Over the paper's 60 G4 splits: calibrate G4 on (i) the full calibration half, as in the paper, and (ii) the
calibration half without its multi-turn calls. Both are evaluated on the full test half, reported separately for
single-turn and multi-turn test calls. (ii) is a WHAT-IF exclusion, not a fixed pipeline: the multi-turn test calls
are still scored on their first message only, so (ii) shows the price the truncated calls impose on the certificate
and the risk that excluding them would move onto the multi-turn test calls.
Output: results/a3b_multiturn_whatif.csv / .json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import gate as G  # noqa: E402

DE, AH, RF = G.DE, G.AH, G.RF


def rates(a, yy):
    f = lambda k: float(np.mean(a[yy == k] == DE)) if (yy == k).any() else np.nan  # noqa: E731
    return dict(safe=f(DE), harm=f(RF), unconf=f(AH))


def main():
    P, meta = xio.prob_matrices("test")
    y = xio.label_index(meta.ground_truth)
    mt = meta.multi_turn.to_numpy()
    rows = []
    for name in ["jev", "microsoft", "gpt", "liquid", "pplx", "mean"]:
        Pm = P[name]
        for s in range(60):
            cal = G.split(y, s)
            for variant, cm in [("paper (all calibration calls)", cal), ("without multi-turn calibration calls", cal & ~mt)]:
                b = G.g4_select(Pm[cm], y[cm], G.g4_candidates(Pm[cm], y[cm]))
                r = dict(model=name, seed=s, variant=variant, found=b is not None)
                if b is not None:
                    a = G.pol(Pm, b[1], b[2], b[3])
                    te = ~cal
                    for grp, m in [("all", te), ("single_turn", te & ~mt), ("multi_turn", te & mt)]:
                        r.update({f"{k}_{grp}": v for k, v in rates(a[m], y[m]).items()})
                rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(xio.results_dir() / "a3b_multiturn_whatif.csv", index=False)
    cols = [c for c in d.columns if c.split("_")[0] in ("safe", "harm", "unconf")]
    t = d[d.found].groupby(["model", "variant"])[cols].mean()
    t["found"] = d.groupby(["model", "variant"]).found.mean()
    t["over_harm_budget_all"] = d[d.found].groupby(["model", "variant"]).harm_all.apply(lambda x: np.mean(x > C.A_HARM))
    t["over_harm_budget_single"] = d[d.found].groupby(["model", "variant"]).harm_single_turn.apply(lambda x: np.mean(x > C.A_HARM))
    t["over_harm_budget_multi"] = d[d.found].groupby(["model", "variant"]).harm_multi_turn.apply(lambda x: np.mean(x > C.A_HARM))
    pd.set_option("display.width", 250)
    print(t[["found", "safe_all", "safe_single_turn", "harm_single_turn", "harm_multi_turn", "unconf_single_turn",
             "over_harm_budget_all", "over_harm_budget_single", "over_harm_budget_multi"]].round(3).to_string())
    json.dump(t.round(4).reset_index().to_dict("records"), open(xio.results_dir() / "a3b_multiturn_whatif.json", "w"), indent=1)


if __name__ == "__main__":
    main()
