"""Export test calls whose full prompt is byte-identical to a train call with a different label (no API calls).
One row per (test call, conflicting train call). Output: results/a1b_train_test_label_conflicts.csv"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402


def clean(s, n=2000):
    return " ".join(str(s).split())[:n]


def main():
    test, train = xio.load_split("test"), xio.load_split("train")
    key = lambda s: json.dumps(s["prompt"], sort_keys=True)  # noqa: E731
    tr = {}
    for j, s in enumerate(train):
        tr.setdefault(key(s), []).append(j)
    a1 = pd.read_csv(C.RESULTS / "a1_flags_test.csv").set_index("dataset_idx")
    trs = xio.load_scores("train")
    trs = trs[~trs.failed].set_index(["dataset_idx", "short"])
    rows = []
    for i, s in enumerate(test):
        for j in tr.get(key(s), []):
            t = train[j]
            if t["meta"]["decision"] == s["meta"]["decision"]:
                continue
            call = json.loads(U.scored_content(s)).get(U.CALL_KEY, {})
            r = dict(test_idx=i, train_idx=j, service=s["meta"].get("service"), risk_class_test=s["meta"].get("risk_class"),
                     risk_class_train=t["meta"].get("risk_class"), label_test=s["meta"]["decision"],
                     label_train=t["meta"]["decision"], multi_turn=xio.n_user_turns(s) > 1,
                     in_paper_1433=i in a1.index, tool=call.get("tool"),
                     tool_args=clean(json.dumps(call.get("tool_args"), ensure_ascii=False), 600),
                     user_task=clean(json.loads(U.scored_content(s)).get("user_task", ""), 800),
                     reasoning_test=clean(s["completion"][-1]["content"]), reasoning_train=clean(t["completion"][-1]["content"]))
            if i in a1.index:
                r.update(a1_n_flags=int(a1.n_flags[i]), a1_suggested=a1.suggested[i])
                for m in ["jev", "microsoft", "gpt", "liquid", "pplx"]:
                    r[f"p_test_label_{m}"] = round(float(a1[f"p_label_{m}"][i]), 4)
            rows.append(r)
    d = pd.DataFrame(rows)
    d["tomy_verdict"] = ""
    d["tomy_note"] = ""
    d.to_csv(C.RESULTS / "a1b_train_test_label_conflicts.csv", index=False)
    print(len(d), "rows |", d.test_idx.nunique(), "test calls |", d.in_paper_1433.sum(), "in the paper's 1,433")
    print(d.groupby(["label_test", "label_train"]).size().to_string())
    print(d.service.value_counts().to_string())


if __name__ == "__main__":
    main()
