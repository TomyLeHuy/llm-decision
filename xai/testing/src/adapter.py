"""llmSHAP LLMInterface adapter for the OpenRouter decisions endpoint, with a persistent JSONL cache.

generate(state) POSTs {"model", "state", "questions": QUESTIONS_PAYLOAD} exactly like score_split.call_openrouter_decision
(same endpoint, headers, timeout and retry policy) and returns a JSON string:
    {"status": "ok", "p": {"direct_execute": .., "ask_human": .., "refuse": ..}}
    {"status": "provider_refusal", "error": "..."}   deterministic refusal / unusable answer, cached
    {"status": "transient_error", "error": "..."}    network / rate limit after all retries, NOT cached
Cache key = sha256 of the canonical request payload (it contains the model id). Every request that reaches the
network is counted, so the run log can report queries and cost.
"""
import hashlib
import json
import random
import threading
import time
from datetime import datetime, timezone

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


class DecisionsLLM(LLMInterface):
    def __init__(self, model: str = MODEL, cache_path=CACHE_PATH, use_disk_cache: bool = True):
        self.model = model
        self.cache_path = cache_path
        self.use_disk_cache = use_disk_cache
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(MAX_WORKERS)
        self.cache = {}
        self.n_http = 0          # HTTP requests sent (incl. retries)
        self.n_new = 0           # distinct payloads answered by the network in this process
        self.n_hits = 0
        self.cost_usd = 0.0
        if use_disk_cache and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.cache[r["key"]] = r

    # ---- llmSHAP interface -------------------------------------------------------------------------------
    def generate(self, prompt, tools=None, images=None) -> str:
        rec = self.query(prompt)
        if rec["status"] == "ok":
            return json.dumps({"status": "ok", "p": {k: rec["probabilities"][k] for k in CLASSES}})
        return json.dumps({"status": rec["status"], "error": rec.get("error")})

    # ---- cached query --------------------------------------------------------------------------------------
    def query(self, state: str, meta: dict | None = None, bypass_cache: bool = False) -> dict:
        payload = payload_for(state, self.model)
        key = cache_key(payload)
        if not bypass_cache:
            with self._lock:
                if key in self.cache:
                    self.n_hits += 1
                    return self.cache[key]
        with self._sem:
            res = self._post(payload)
        rec = {"key": key, "model": self.model, **res, **(meta or {}),
               "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self._lock:
            self.n_new += 1
            self.cost_usd += res.get("cost_usd") or 0.0
            if not bypass_cache and rec["status"] != "transient_error":
                self.cache[key] = rec
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
                    return {"status": "provider_refusal", "error": "HTTP 200 without a valid probability vector", **out}
                body = resp.text[:200]
                if REFUSAL_MARKER in body:
                    return {"status": "provider_refusal", "error": f"HTTP {resp.status_code}: {body}", "cost_usd": 0.0}
                if resp.status_code in FATAL_STATUS:
                    raise RuntimeError(f"fatal HTTP {resp.status_code} from OpenRouter: {body}")
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}: {body}"
                    time.sleep((backoff_factor ** attempt) + random.uniform(0.1, 0.5))
                    continue
                return {"status": "provider_refusal", "error": f"HTTP {resp.status_code}: {body}", "cost_usd": 0.0}
            except RuntimeError:
                raise
            except Exception as e:  # network errors
                last_error = str(e)
                time.sleep((backoff_factor ** attempt) + random.uniform(0.1, 0.5))
        return {"status": "transient_error", "error": f"Max retries exceeded: {last_error}", "cost_usd": 0.0}
