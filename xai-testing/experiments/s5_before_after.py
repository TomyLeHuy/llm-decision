"""Step 5 before/after (no API calls). Before = today's paper-input scores; after = same with multi-turn calls scored
on the full conversation. Reads results/today/ and results/today_fixed/ (run the A-scripts with XAI_SOURCE first).
Output: results/s5_before_after.json and printed tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402

B, A = C.RESULTS / "today", C.RESULTS / "today_fixed"
FIVE = ["jev", "microsoft", "gpt", "liquid", "pplx", "mean"]
OUT = {}


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def scorer_table():
    rows = []
    for split in ["test", "train"]:
        for src in ["today", "today_fixed"]:
            P, meta = xio.prob_matrices(split, source=src)
            y = xio.label_index(meta.ground_truth); mt = meta.multi_turn.to_numpy()
            for m in FIVE:
                for g, sel in [("single_turn", ~mt), ("multi_turn", mt)]:
                    p, yy = P[m][sel], y[sel]
                    rows.append(dict(split=split, source=src, model=m, group=g, n=int(sel.sum()), n_refuse=int((yy == 2).sum()),
                                     accuracy=float(np.mean(p.argmax(1) == yy)),
                                     refusal_recall=float(np.mean(p[yy == 2].argmax(1) == 2)),
                                     refusals_executed=float(np.mean(p[yy == 2].argmax(1) == 0)),
                                     refusals_p_le_001=float(np.mean(p[yy == 2, 2] <= 0.01)),
                                     refusals_p0=float(np.mean(p[yy == 2, 2] == 0)),
                                     auroc_refuse=auroc(p[yy == 2, 2], p[yy != 2, 2])))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "s5_scorer_before_after.csv", index=False)
    OUT["scorers"] = t.round(4).to_dict("records")
    w = t[t.group == "multi_turn"].pivot_table(index=["split", "model"], columns="source",
                                               values=["refusal_recall", "refusals_executed", "refusals_p_le_001", "auroc_refuse", "accuracy"])
    print("multi-turn calls, before (today) vs after (today_fixed)\n", w.round(3).to_string())


def a1():
    r = {}
    for src, d in [("before", B), ("after", A)]:
        f = pd.read_csv(d / "a1_flags_test.csv")
        s = json.load(open(d / "a1_summary.json"))
        r[src] = dict(ge3_single=int(((f.n_flags >= 3) & ~f.multi_turn).sum()), ge3_multi=int(((f.n_flags >= 3) & f.multi_turn).sum()),
                      five_multi=int(((f.n_flags == 5) & f.multi_turn).sum()),
                      multi_refuse_to_execute=int(((f.n_flags >= 3) & f.multi_turn & (f.ground_truth == "refuse") & (f.suggested == "direct_execute")).sum()),
                      flag_rate_per_model_multi={m: v["True"]["rate"] for m, v in s["flags_per_model"].items()},
                      train_hist=s["train"]["agreement_hist"])
    OUT["A1"] = r
    print("A1", json.dumps(r, indent=1))


def a2():
    r = {}
    for src, d in [("before", B), ("after", A)]:
        x = pd.read_csv(d / "a2_model_service.csv")
        sel = x[x.service.isin(["atbench", "dh_app", "G5:atbench"]) & (x.subset == "all")]
        r[src] = sel[["split", "scorer", "service", "n_rf", "rf_p0", "rf_le001", "auroc", "g5_sep_lever", "g5_sep_safe_autonomy",
                      "g5_sep_extra_clean_refusals"]].round(4).to_dict("records")
        r[src + "_lever_counts"] = x[x.subset == "all"].groupby(["split", "g5_sep_lever"]).size().to_dict()
        pool = pd.read_csv(d / "a2_g4_pool.csv")
        r[src + "_g4_pool"] = pool[["split", "scorer", "certifies", "safe", "harm"] + [c for c in pool.columns if c in ("harm_atbench", "harm_dh_app")]].round(4).to_dict("records")
    OUT["A2"] = r
    b = pd.DataFrame(r["before"]); a = pd.DataFrame(r["after"])
    m = b.merge(a, on=["split", "scorer", "service"], suffixes=("_before", "_after"))
    print("A2 atbench / dh_app\n", m[["split", "scorer", "service", "n_rf_before", "rf_le001_before", "rf_le001_after", "auroc_before", "auroc_after",
                                     "g5_sep_lever_before", "g5_sep_lever_after"]].round(3).to_string())
    pb, pa = pd.DataFrame(r["before_g4_pool"]), pd.DataFrame(r["after_g4_pool"])
    print("G4 on pool\n", pb.merge(pa, on=["split", "scorer"], suffixes=("_before", "_after")).round(3).to_string())


def a3():
    r = {}
    for src, d in [("before", B), ("after", A)]:
        base = pd.read_csv(d / "a3_baseline.csv")
        h = pd.read_csv(d / "a3_boundary_holders.csv") if (d / "a3_boundary_holders.csv").exists() else pd.DataFrame()
        r[src] = dict(baseline=base[["model", "U_test", "lr", "la", "slack"]].round(4).to_dict("records"),
                      holders=h[["model", "dataset_idx", "service", "multi_turn", "a1_n_flags", "dU_test"]].round(4).to_dict("records") if len(h) else [])
    OUT["A3"] = r
    print("A3", json.dumps(r, indent=1, default=str))


def a3b():
    r = {}
    for src, d in [("before", B), ("after", A)]:
        t = pd.DataFrame(json.load(open(d / "a3b_multiturn_whatif.json")))
        r[src] = t[t.variant.str.startswith("paper")][["model", "safe_all", "safe_single_turn", "harm_all", "harm_single_turn", "harm_multi_turn",
                                                        "over_harm_budget_all", "over_harm_budget_multi"]].round(4).to_dict("records")
    OUT["A3b"] = r
    b, a = pd.DataFrame(r["before"]), pd.DataFrame(r["after"])
    print("A3b\n", b.merge(a, on="model", suffixes=("_before", "_after")).round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    scorer_table(); a1(); a2(); a3(); a3b()
    xio.dump_json(OUT, C.RESULTS / "s5_before_after.json")
