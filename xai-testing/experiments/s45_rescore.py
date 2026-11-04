"""Steps 4 and 5: re-scoring (8 workers).

  python experiments/s45_rescore.py today        all 1,436 test calls x 5 models, paper input (truncated), today
                                                 -> runs/today/test_scores.jsonl
                                                 + the 733 train multi-turn calls, paper input, today (the 'before'
                                                 for step 5 on train) -> runs/today/train_multiturn_scores.jsonl
  python experiments/s45_rescore.py fixed        123 test + 733 train multi-turn calls x 5 models, full conversation
                                                 -> runs/fixed_multiturn/{test,train}_scores.jsonl
Records follow score_split.py's format plus 'outcome' (ok / provider_refusal / error) and 'input' (paper / full_conversation).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import query as Q  # noqa: E402
from src import units as U  # noqa: E402
from src.scoring_fixed import full_conversation_sample  # noqa: E402

RUNS = C.XAI / "runs"


def run(split, ids, make_sample, tag, out, input_name):
    data = xio.load_split(split)
    jobs = [dict(model=m, sample=make_sample(data[i]), tag=tag, meta=dict(split=split, dataset_idx=int(i)))
            for i in ids for m in C.MODELS]
    res = Q.Querier().run(jobs, workers=8, require_tag=tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for j, r in zip(jobs, res):
            i = j["meta"]["dataset_idx"]; meta = data[i]["meta"]; r = r or {}
            p = r.get("probabilities") or {}
            pred = r.get("choice")
            rec = dict(dataset_idx=i, service=meta.get("service", "unknown"), risk_class=meta.get("risk_class", "unknown"),
                       risk_family=meta.get("risk_family", "unknown"), ground_truth=meta.get("decision"), model=j["model"],
                       pred_choice=pred, is_match=(pred == meta.get("decision")) if pred is not None else False,
                       p_direct_execute=p.get("direct_execute", float("nan")), p_ask_human=p.get("ask_human", float("nan")),
                       p_refuse=p.get("refuse", float("nan")), prob_sum=r.get("prob_sum", float("nan")),
                       latency_ms=r.get("latency_ms", 0.0), cost_usd=r.get("cost_usd", 0.0), error=r.get("error"),
                       split=split, scored_at=r.get("queried_at"), response_model=r.get("response_model"),
                       outcome=Q.outcome(r if r else None), input=input_name)
            f.write(json.dumps(rec) + "\n")
    n_out = {}
    for j, r in zip(jobs, res):
        n_out[Q.outcome(r)] = n_out.get(Q.outcome(r), 0) + 1
    print(out, len(jobs), n_out, flush=True)


def main(step):
    st_test, st_train = xio.sample_table("test"), xio.sample_table("train")
    mt_test = st_test.dataset_idx[st_test.multi_turn].tolist()
    mt_train = st_train.dataset_idx[st_train.multi_turn].tolist()
    if step == "today":
        run("test", st_test.dataset_idx.tolist(), U.variant, "today_test", RUNS / "today/test_scores.jsonl", "paper")
        run("train", mt_train, U.variant, "today_train_mt", RUNS / "today/train_multiturn_scores.jsonl", "paper")
    elif step == "fixed":
        run("test", mt_test, full_conversation_sample, "fixed_test", RUNS / "fixed_multiturn/test_scores.jsonl", "full_conversation")
        run("train", mt_train, full_conversation_sample, "fixed_train", RUNS / "fixed_multiturn/train_scores.jsonl", "full_conversation")
    print("done", step, datetime.now(timezone.utc).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main(sys.argv[1])
