"""llmSHAP LLMInterface adapter for the OpenRouter decisions endpoint, with a persistent JSONL cache.

generate(state) POSTs {"model", "state", "questions": QUESTIONS_PAYLOAD} exactly like score_split.call_openrouter_decision
(same endpoint, headers, timeout and retry policy) and returns a JSON string:
    {"status": "ok", "p": {"direct_execute": .., "ask_human": .., "refuse": ..}}
    {"status": "provider_refusal", "error": "..."}   refusal / unusable answer, cached
    {"status": "transient_error", "error": "..."}    network / rate limit after all retries, NOT cached

Cache: one JSONL file per model, one line per raw answer, key = sha256 of the canonical payload (which contains the
model id, so an answer of one model can never be returned for another). A key can hold several answers (--repeats).

Repeats (non-deterministic models): the payoff of a coalition is the mean of its FIRST r answers in file order. The
value is therefore fixed once r answers exist and is the same in every later run, so the Shapley axioms hold. If any of
the r answers is a refusal, the coalition counts as provider_refusal (no imputation). With r = 1 the payoff is the
single cached answer itself (the gpt pilot).
"""
import hashlib
import json
import random
import threading
import time
from datetime import datetime, timezone
from math import sqrt

import requests
from llmSHAP.llm.llm_interface import LLMInterface

from common import CACHE_PATH, CLASSES, MAX_WORKERS, MODEL, api_key, ss

REFUSAL_MARKER = "refused to answer"   # body of gpt's HTTP 502 refusal: 'OpenAI refused to answer question "safety_decision"'
FATAL_STATUS = {401, 402}              # bad key / no credit: stop the run instead of caching anything


def payload_for(state: str, model: str = MODEL) -> dict:
    return {"model": model, "state": state, "questions": ss.QUESTIONS_PAYLOAD}


def cache_key(payload: dict) -> str:
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def valid_probs(p: dict) -> bool:
    v = [p.get(k) for k in CLASSES]
    if any(not isinstance(x, (int, float)) or x != x for x in v):
        return False
    return min(v) >= 0 and abs(sum(v) - 1) <= 0.02      # same rule as score_split.is_failed


def failure_type(rec: dict) -> str | None:
    """Coarse class of a non-ok answer (older gpt cache lines carry no failure_type field, so derive it)."""
    if rec.get("status") == "ok":
        return None
    if rec.get("failure_type"):
        return rec["failure_type"]
    err = rec.get("error") or ""
    if REFUSAL_MARKER in err:
        return f"http_{err.split()[1].rstrip(':')}_refused_to_answer" if err.startswith("HTTP ") else "refused_to_answer"
    if err.startswith("HTTP 200"):
        return "http_200_invalid_probabilities"
    if err.startswith("HTTP "):
        return f"http_{err.split()[1].rstrip(':')}"
    return "other"


def aggregate(answers: list, r: int) -> dict | None:
    """Payoff record from the first r cached answers of one key; None if fewer than r exist."""
    if len(answers) < r:
        return None
    use = answers[:r]
    if r == 1:
        return use[0]
    bad = [a for a in use if a["status"] != "ok"]
    base = {"key": use[0]["key"], "model": use[0]["model"], "n_answers": r,
            "answers": [{k: a.get(k) for k in ("status", "probabilities", "choice", "error")} for a in use]}
    if bad:
        return {**base, "status": "provider_refusal", "error": bad[0].get("error"),
                "failure_type": failure_type(bad[0]), "n_refused_answers": len(bad)}
    mean = {k: sum(a["probabilities"][k] for a in use) / r for k in CLASSES}
    std = {k: sqrt(sum((a["probabilities"][k] - mean[k]) ** 2 for a in use) / (r - 1)) for k in CLASSES}
    return {**base, "status": "ok", "probabilities": mean, "std": std,
            "choice": max(mean, key=mean.get), "cost_usd": sum(a.get("cost_usd") or 0.0 for a in use)}


def load_cache(path) -> dict:
    """key -> list of raw answers in file order."""
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.setdefault(r["key"], []).append(r)
    return out


class DecisionsLLM(LLMInterface):
    def __init__(self, model: str = MODEL, cache_path=CACHE_PATH, use_disk_cache: bool = True, repeats: int = 1):
        assert repeats >= 1
        self.model = model
        self.cache_path = cache_path
        self.use_disk_cache = use_disk_cache
        self.repeats = repeats
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(MAX_WORKERS)
        self.answers = load_cache(cache_path) if use_disk_cache else {}
        for k, v in self.answers.items():
            assert all(a["model"] == model for a in v), f"cache {cache_path} holds answers of another model"
        self.n_http = 0          # HTTP requests sent (incl. retries)
        self.n_new = 0           # answers received from the network in this process
        self.n_hits = 0
        self.cost_usd = 0.0

    @property
    def cache(self) -> dict:
        """key -> payoff record for keys that already have `repeats` answers."""
        out = {}
        for k, v in self.answers.items():
            agg = aggregate(v, self.repeats)
            if agg is not None:
                out[k] = agg
        return out

    def missing_answers(self, key: str) -> int:
        return max(0, self.repeats - len(self.answers.get(key, [])))

    # ---- llmSHAP interface -------------------------------------------------------------------------------
    def generate(self, prompt, tools=None, images=None) -> str:
        rec = self.query(prompt)
        if rec["status"] == "ok":
            return json.dumps({"status": "ok", "p": {k: rec["probabilities"][k] for k in CLASSES}})
        return json.dumps({"status": rec["status"], "error": rec.get("error")})

    # ---- cached query --------------------------------------------------------------------------------------
    def query(self, state: str, meta: dict | None = None, bypass_cache: bool = False) -> dict:
        """Payoff record of `state`. bypass_cache=True: one fresh answer, not stored (determinism probes)."""
        payload = payload_for(state, self.model)
        key = cache_key(payload)
        if bypass_cache:
            return self._fetch(payload, key, meta, store=False)
        with self._lock:
            have = len(self.answers.get(key, []))
        if have >= self.repeats:
            with self._lock:
                self.n_hits += 1
            return aggregate(self.answers[key], self.repeats)
        for _ in range(self.repeats - have):
            rec = self._fetch(payload, key, meta, store=True)
            if rec["status"] == "transient_error":
                return rec
        with self._lock:
            return aggregate(self.answers[key], self.repeats)

    def _fetch(self, payload, key, meta, store):
        with self._sem:
            res = self._post(payload)
        rec = {"key": key, "model": self.model, **res, **(meta or {}),
               "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self._lock:
            self.n_new += 1
            self.cost_usd += res.get("cost_usd") or 0.0
            if store and rec["status"] != "transient_error":
                self.answers.setdefault(key, []).append(rec)
                if self.use_disk_cache:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _post(self, payload, max_retries=5, backoff_factor=2.0) -> dict:
        """Retry policy of score_split.call_openrouter_decision, except that a refusal is final at once."""
        headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                   "X-Title": "DFKI-Agent-Governance-Conformal"}
        last_error = ""
        for attempt in range(max_retries):
            try:
                with self._lock:
                    self.n_http += 1
                resp = requests.post(ss.DECISIONS_ENDPOINT, headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    ans = data.get("answers", {}).get("safety_decision", {})
                    probs = ans.get("probabilities", {}) or {}
                    out = {"choice": ans.get("choice"), "probabilities": probs,
                           "cost_usd": data.get("usage", {}).get("cost", 0.0), "response_model": data.get("model")}
                    if valid_probs(probs):
                        return {"status": "ok", "error": None, **out}
                    return {"status": "provider_refusal", "failure_type": "http_200_invalid_probabilities",
                            "error": "HTTP 200 without a valid probability vector",
                            "example_body": json.dumps(data)[:300], **out}
                body = resp.text[:200]
                if REFUSAL_MARKER in body:
                    return {"status": "provider_refusal", "failure_type": f"http_{resp.status_code}_refused_to_answer",
                            "error": f"HTTP {resp.status_code}: {body}", "cost_usd": 0.0}
                if resp.status_code in FATAL_STATUS:
                    raise RuntimeError(f"fatal HTTP {resp.status_code} from OpenRouter: {body}")
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}: {body}"
                    time.sleep((backoff_factor ** attempt) + random.uniform(0.1, 0.5))
                    continue
                return {"status": "provider_refusal", "failure_type": f"http_{resp.status_code}",
                        "error": f"HTTP {resp.status_code}: {body}", "cost_usd": 0.0}
            except RuntimeError:
                raise
            except Exception as e:  # network errors
                last_error = str(e)
                time.sleep((backoff_factor ** attempt) + random.uniform(0.1, 0.5))
        return {"status": "transient_error", "failure_type": "transient_max_retries",
                "error": f"Max retries exceeded: {last_error}", "cost_usd": 0.0}
