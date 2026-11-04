"""Cached, resumable, budget-checked querying through the paper's call_openrouter_decision (used unchanged).

- score_split.py is loaded from its file path with importlib; bytecode writing is disabled during the import so that
  nothing is written into the frozen package (no new __pycache__ entry).
- Cache: append-only JSONL, key = sha256(model + "\\n" + state_string), where state_string is the exact string
  call_openrouter_decision sends. Failed calls are stored too, but count as cache misses (they cost nothing and are
  retried). Records are flushed one by one, so a run can be interrupted and restarted with the same command.
- Repeats (noise floor) go to a separate file keyed by sha256(model + "\\n" + state + "\\n#repeat=" + r).
- Budget: before every batch the spent amount (sum of cached cost_usd over both files) plus a conservative estimate
  for the batch must stay below config.BUDGET_USD; at 80% spent the batch is refused and the reason reported.
- The API key is read from the repo-root .env and is never printed or logged.
"""
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def _load_score_split(name):
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, C.SCORE_SPLIT_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


ss = _load_score_split("score_split_live")          # used for real calls, never modified
_ss_capture = _load_score_split("score_split_capture")  # separate instance with a stub transport, to read the state


class _StubResponse:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return {"answers": {"safety_decision": {"choice": None, "probabilities": {}}}, "usage": {}, "model": None,
                "_payload": self._p}


_capture_lock = threading.Lock()
_last_payload = {}


def _stub_post(url, headers=None, json=None, timeout=None):  # noqa: A002
    _last_payload["url"], _last_payload["payload"] = url, json
    return _StubResponse(json)


_ss_capture.requests = type("R", (), {"post": staticmethod(_stub_post)})()


def captured_payload(sample, model_id="x/capture"):
    """The exact JSON payload (incl. the state string) that call_openrouter_decision would POST. No network."""
    with _capture_lock:
        _ss_capture.call_openrouter_decision("NO_KEY", model_id, sample, max_retries=1)
        return _last_payload["url"], _last_payload["payload"]


def state_string(sample):
    return captured_payload(sample)[1]["state"]


def cache_key(model, state, repeat=None):
    s = model + "\n" + state + ("" if repeat is None else f"\n#repeat={repeat}")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_jsonl(path):
    out = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


PROVIDER_REFUSAL = "refused to answer"


def outcome(r):
    """'ok' | 'provider_refusal' (the vendor declined to answer, e.g. gpt's HTTP 502 'OpenAI refused to answer
    question'; a final outcome, cached and never retried) | 'error' (transport or malformed; retried on the next run)"""
    if r is None:
        return "missing"
    if r.get("error"):
        return "provider_refusal" if PROVIDER_REFUSAL in str(r["error"]) else "error"
    p = r.get("probabilities") or {}
    v = [p.get(c) for c in C.CLASSES]
    if any(x is None for x in v) or min(v) < 0 or abs(sum(v) - 1) > 0.02:
        return "error"
    return "ok"


def record_failed(r):
    return outcome(r) != "ok"


def spent_usd():
    return sum(float(r.get("cost_usd") or 0) for p in (C.CACHE_FILE, C.REPEAT_CACHE_FILE) for r in _read_jsonl(p))


def spent_by_tag():
    out = {}
    for p in (C.CACHE_FILE, C.REPEAT_CACHE_FILE):
        for r in _read_jsonl(p):
            t = r.get("tag", "untagged")
            out.setdefault(t, [0, 0.0])
            out[t][0] += 1
            out[t][1] += float(r.get("cost_usd") or 0)
    return out


# conservative per-call estimate (max observed mean per model in the October/27 Oct runs, x1.5)
EST_COST = {"typesafe/jev-1.14": 6.4e-5, "microsoft/microsoft-decision-1": 5.7e-5, "openai/gpt-6-luna-decisions": 1.23e-4,
            "liquid/d1-ultra": 5.1e-5, "perplexity/pplx-decider-v1.2": 2.5e-5}


class BudgetExceeded(RuntimeError):
    pass


def check_budget(todo_models, spent=None):
    spent = spent_usd() if spent is None else spent
    est = 1.5 * sum(EST_COST.get(m, 1.5e-4) for m in todo_models)
    if spent >= C.BUDGET_WARN_FRAC * C.BUDGET_USD:
        raise BudgetExceeded(f"spent USD {spent:.4f} >= {C.BUDGET_WARN_FRAC:.0%} of the cap; stop and report")
    if spent + est > C.BUDGET_USD:
        raise BudgetExceeded(f"spent USD {spent:.4f} + batch estimate {est:.4f} > cap {C.BUDGET_USD}")
    return spent, est


def _api_key():
    try:
        from dotenv import dotenv_values
        key = dotenv_values(C.ENV_FILE).get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    except ImportError:
        key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key.endswith("DEIN_KEY_HIER"):
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")
    return key


class Querier:
    """q = Querier(); results = q.run(jobs) with jobs = [dict(model=..., sample=..., tag=..., meta={...}, repeat=None)].
    Results are cache records (or None); use outcome(rec) to tell ok / provider_refusal / error apart."""

    def __init__(self):
        C.CACHE.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.cache = {}
        self.spent = 0.0
        for path in (C.CACHE_FILE, C.REPEAT_CACHE_FILE):
            for r in _read_jsonl(path):
                self.spent += float(r.get("cost_usd") or 0)
                if outcome(r) in ("ok", "provider_refusal"):
                    self.cache[r["key"]] = r  # last final record per key

    def lookup(self, model, sample, repeat=None):
        return self.cache.get(cache_key(model, state_string(sample), repeat))

    def run(self, jobs, workers=C.MAX_WORKERS, dry_run=False, verbose=True, require_tag=None, chunk=4000):
        """require_tag: cached records from another run (other tag) count as misses, so that baseline and masked
        variants come from the same run (plan rule 5). The new record then supersedes the old one in the cache.
        The budget is re-checked before every chunk of calls."""
        prepared, todo, seen = [], [], set()
        for j in jobs:
            st = state_string(j["sample"])
            k = cache_key(j["model"], st, j.get("repeat"))
            prepared.append((j, k, st))
            hit = self.cache.get(k)
            stale = hit is not None and require_tag is not None and hit.get("tag") != require_tag
            if (hit is None or stale) and k not in seen:
                todo.append((j, k, st)); seen.add(k)
        est_all = 1.5 * sum(EST_COST.get(j["model"], 1.5e-4) for j, _, _ in todo)
        if verbose:
            print(f"[query] jobs {len(jobs)} | cached {len(jobs) - len(todo)} | to call {len(todo)} | "
                  f"spent USD {self.spent:.4f} | batch est. <= {est_all:.4f}", flush=True)
        if dry_run or not todo:
            return [self.cache.get(k) for _, k, _ in prepared]
        key = _api_key()
        t0 = time.time()
        n_err = n_ref = 0

        def call(j, k, st):
            res = ss.call_openrouter_decision(key, j["model"], j["sample"])
            rec = dict(key=k, model=j["model"], repeat=j.get("repeat"), tag=j.get("tag"), meta=j.get("meta", {}),
                       state_sha256=hashlib.sha256(st.encode("utf-8")).hexdigest(),
                       queried_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **res)
            rec["outcome"] = outcome(rec)
            return rec

        path = C.REPEAT_CACHE_FILE if any(j.get("repeat") is not None for j, _, _ in todo) else C.CACHE_FILE
        assert all((j.get("repeat") is not None) == (path == C.REPEAT_CACHE_FILE) for j, _, _ in todo), \
            "do not mix repeat and normal jobs in one batch"
        n = 0
        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            for c0 in range(0, len(todo), chunk):
                part = todo[c0:c0 + chunk]
                check_budget([j["model"] for j, _, _ in part], spent=self.spent)
                futs = [ex.submit(call, *t) for t in part]
                for fut in as_completed(futs):
                    rec = fut.result(); n += 1
                    with self._lock:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        self.spent += float(rec.get("cost_usd") or 0)
                        o = rec["outcome"]
                        n_err += o == "error"; n_ref += o == "provider_refusal"
                        if o in ("ok", "provider_refusal"):
                            self.cache[rec["key"]] = rec
                        if o == "error" and "HTTP 402" in str(rec.get("error") or ""):
                            for ff in futs: ff.cancel()
                            raise RuntimeError("OpenRouter account out of credits (HTTP 402): stopping. "
                                               "Top up, then re-run; cached answers are kept and only the missing calls are made.")
                    if verbose and (n % 1000 == 0 or n == len(todo)):
                        el = time.time() - t0
                        print(f"[query] {n}/{len(todo)} | errors {n_err} | provider_refusal {n_ref} | {el:.0f}s | "
                              f"~{el / n * (len(todo) - n) / 60:.0f} min left | spent USD {self.spent:.4f}", flush=True)
        return [self.cache.get(k) for _, k, _ in prepared]
