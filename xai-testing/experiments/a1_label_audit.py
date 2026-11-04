"""A1 label audit with Confident Learning (Northcutt et al., 2021), on stored scores; no API calls.

Per model: per-class threshold t_j = mean p_j over samples labelled j. A sample is 'confident' for every class j with
p_j >= t_j; if several, the one with the largest p_j. It is flagged if its confident class exists and differs from
its label (off-diagonal of the confident joint). Flags are combined over the five models.
  python experiments/a1_label_audit.py            (test split, primary; train as a replication)
Outputs: results/a1_*.
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
from src import query as Q  # noqa: E402

R = xio.results_dir()  # results/ for the paper scores, results/<XAI_SOURCE>/ otherwise
FIVE = ["jev", "microsoft", "gpt", "liquid", "pplx"]


def strkeys(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "->".join(map(str, k)) if isinstance(k, tuple) else str(k)): strkeys(v)
                for k, v in o.items()}
    if isinstance(o, list):
        return [strkeys(v) for v in o]
    return o


def confident_learning(P, y):
    t = np.array([P[y == j, j].mean() for j in range(3)])
    above = P >= t[None, :]
    masked = np.where(above, P, -np.inf)
    conf = masked.argmax(1)
    has = above.any(1)
    conf = np.where(has, conf, -1)
    flag = has & (conf != y)
    joint = pd.crosstab(pd.Series(y, name="label"), pd.Series(conf, name="confident"))
    return t, conf, flag, joint


def run(split):
    P, meta = xio.prob_matrices(split)
    y = xio.label_index(meta.ground_truth)
    out = meta[["dataset_idx", "ground_truth", "service", "service_type", "risk_class", "risk_family", "multi_turn",
                "scored_tool", "labelled_tool"]].copy()
    thr, joints = {}, {}
    for m in FIVE:
        t, conf, flag, joint = confident_learning(P[m], y)
        thr[m] = dict(zip(C.CLASSES, map(float, t)))
        joints[m] = joint.rename(index=dict(enumerate(C.CLASSES)), columns={-1: "none", **dict(enumerate(C.CLASSES))})
        out[f"conf_{m}"] = [C.CLASSES[c] if c >= 0 else "none" for c in conf]
        out[f"flag_{m}"] = flag
        out[f"p_label_{m}"] = P[m][np.arange(len(y)), y]
    out["n_flags"] = out[[f"flag_{m}" for m in FIVE]].sum(axis=1)
    # suggested label: majority among flagging models' confident classes
    def sugg(r):
        c = [r[f"conf_{m}"] for m in FIVE if r[f"flag_{m}"]]
        return pd.Series(c).mode().iloc[0] if c else ""
    out["suggested"] = out.apply(sugg, axis=1)
    out["mean_p_label"] = out[[f"p_label_{m}" for m in FIVE]].mean(axis=1)
    return out, thr, joints, P, y


def main():
    res = {}
    out, thr, joints, P, y = run("test")
    out.to_csv(R / "a1_flags_test.csv", index=False)
    res["thresholds"] = thr
    res["confident_joint"] = {m: j.to_dict() for m, j in joints.items()}
    per_model = {}
    for m in FIVE:
        per_model[m] = {str(mt): dict(n=int((out.multi_turn == mt).sum()), flagged=int(out.loc[out.multi_turn == mt, f"flag_{m}"].sum()),
                                      rate=round(float(out.loc[out.multi_turn == mt, f"flag_{m}"].mean()), 4))
                        for mt in (False, True)}
    res["flags_per_model"] = per_model
    res["agreement_hist"] = {str(mt): out[out.multi_turn == mt].n_flags.value_counts().sort_index().astype(int).to_dict()
                             for mt in (False, True)}
    # chance agreement: if the five models flagged independently at their own rates
    for mt in (False, True):
        g = out[out.multi_turn == mt]
        rates = [g[f"flag_{m}"].mean() for m in FIVE]
        # Poisson-binomial expectation of >=3 flags
        from itertools import combinations
        p3 = sum(np.prod([rates[i] if i in S else 1 - rates[i] for i in range(5)])
                 for k in range(3, 6) for S in combinations(range(5), k))
        res.setdefault("expected_ge3_if_independent", {})[str(mt)] = dict(expected=round(float(p3 * len(g)), 1),
                                                                         observed=int((g.n_flags >= 3).sum()))
    # cross-tabs
    out["agree3"] = out.n_flags >= 3
    def tab(by):
        t = out.groupby(by + ["multi_turn"]).agg(n=("dataset_idx", "size"), flagged_ge3=("agree3", "sum"),
                                                  flagged_5=("n_flags", lambda x: int((x == 5).sum())),
                                                  rate_ge3=("agree3", "mean")).reset_index()
        return t
    for by, name in [(["ground_truth"], "label"), (["ground_truth", "suggested"], "label_to_suggested"),
                     (["service"], "service"), (["risk_class"], "risk_class")]:
        t = tab(by) if name != "label_to_suggested" else out[out.agree3].groupby(["ground_truth", "suggested", "multi_turn"]).size().rename("n").reset_index()
        t.to_csv(R / f"a1_crosstab_{name}.csv", index=False)
        res[f"crosstab_{name}"] = t.round(4).astype({c: str for c in t.columns if t[c].dtype == bool}).to_dict("records")

    # independent check: identical scored input, different labels (from S1)
    sh = pd.read_csv(C.RESULTS / "p0_s1_shared_states_test.csv")
    out = out.merge(sh[["dataset_idx", "state_label_conflict"]], on="dataset_idx", how="left")
    st = xio.sample_table("test").set_index("dataset_idx")
    out["annotator"] = out.dataset_idx.map(st.annotator)
    # train/test identical full prompts with different labels (a direct label-inconsistency signal)
    test, train = xio.load_split("test"), xio.load_split("train")
    key = lambda s: json.dumps(s["prompt"], sort_keys=True)  # noqa: E731
    tr_lab = {}
    for s in train:
        tr_lab.setdefault(key(s), set()).add(s["meta"]["decision"])
    out["in_train_identical"] = [key(test[i]) in tr_lab for i in out.dataset_idx]
    out["train_label_differs"] = [bool(key(test[i]) in tr_lab and (tr_lab[key(test[i])] - {test[i]["meta"]["decision"]}))
                                  for i in out.dataset_idx]
    single = out[~out.multi_turn]
    res["train_duplicates"] = dict(
        test_samples_with_identical_prompt_in_train=int(out.in_train_identical.sum()),
        of_which_label_differs=int(out.train_label_differs.sum()),
        flag_rate_ge3_when_train_label_differs=round(float(single[single.train_label_differs].agree3.mean()), 3)
        if single.train_label_differs.any() else None,
        flag_rate_ge3_otherwise=round(float(single[~single.train_label_differs].agree3.mean()), 3),
        n_single_turn_label_differs=int(single.train_label_differs.sum()))
    out.to_csv(R / "a1_flags_test.csv", index=False)

    # export the 40 most-agreed single-turn flags (main list) and the multi-turn ones separately
    cols = ["dataset_idx", "service", "risk_class", "ground_truth", "suggested", "n_flags", "mean_p_label",
            "train_label_differs", "scored_tool", "annotator"]
    top = single[single.n_flags >= 1].sort_values(["n_flags", "mean_p_label"], ascending=[False, True]).head(40)
    top = out.set_index("dataset_idx").loc[top.dataset_idx].reset_index()
    top["annotator"] = top.annotator.str.replace(r"\s+", " ", regex=True).str[:1500]
    top[cols].assign(tomy_verdict="", tomy_note="").to_csv(R / "a1_top40_single_turn_for_review.csv", index=False)
    mt = out[out.multi_turn & (out.n_flags >= 3)].sort_values(["n_flags", "mean_p_label"], ascending=[False, True])
    mt[cols[:-1] + ["labelled_tool"]].to_csv(R / "a1_multi_turn_flags.csv", index=False)
    res["top40_summary"] = dict(n_flags=top.n_flags.value_counts().sort_index().astype(int).to_dict(),
                                label_to_suggested={f"{a}->{b}": int(n) for (a, b), n in top.groupby(["ground_truth", "suggested"]).size().items()},
                                services=top.service.value_counts().astype(int).to_dict())

    # replication on train (same procedure, own thresholds)
    out_tr, thr_tr, *_ = run("train")
    out_tr.to_csv(R / "a1_flags_train.csv", index=False)
    res["train"] = dict(thresholds=thr_tr, agreement_hist={str(mt): out_tr[out_tr.multi_turn == mt].n_flags.value_counts()
                                                            .sort_index().astype(int).to_dict() for mt in (False, True)},
                        n={str(mt): int((out_tr.multi_turn == mt).sum()) for mt in (False, True)})
    json.dump(strkeys(res), open(R / "a1_summary.json", "w"), indent=1, default=str)
    for k in ["flags_per_model", "agreement_hist", "expected_ge3_if_independent", "train_duplicates", "top40_summary"]:
        print(k, json.dumps(strkeys(res[k]), default=str))
    print("train", json.dumps(res["train"]["agreement_hist"]))
    print(pd.DataFrame(res["crosstab_label"]).to_string())
    print(pd.DataFrame(res["crosstab_label_to_suggested"]).to_string())


if __name__ == "__main__":
    main()
