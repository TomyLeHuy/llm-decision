"""Phase B/C field-level occlusion.

  python experiments/b_occlusion.py --pilot [--dry]     query baseline + M1 + M2 variants for the 50 pilot calls
  python experiments/b_occlusion.py --pilot --analyze   attributions, validity, noise, M1/M2 agreement -> results/b_pilot_*

Units: user_task, user_memory, prior_steps, service, tool, description, each tool_args key (units.parse).
M1 remove: the key is dropped. M2 replace: the value is taken from a direct_execute call of ANOTHER service in the test
split (scored message of the donor):
  top-level / call fields      -> the same field of a random donor that has it, with a different value  ('same_field')
  tool_args.<k>                 -> tool_args[k] of a random donor with that key and a different value    ('same_key')
                                   else a random argument value of the same JSON type from a donor        ('type_matched')
                                   else any random argument value from a donor                            ('any_arg')
Donor values must be non-empty (not null, "", [] or {}). Every draw uses rng = default_rng([SEED, dataset_idx, unit position]); donor index, key and method are logged.
Attribution of unit u for class c = p_c(baseline) - p_c(masked u); baseline and variants are queried in the same run.
microsoft is labelled 'microsoft (post-drift)' in all outputs (it changed after 23 Oct, see Phase 0).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402
from src import query as Q  # noqa: E402
from experiments.p0_sanity import pilot_ids  # noqa: E402

R = C.RESULTS
LABEL = {"microsoft": "microsoft (post-drift)"}


def jtype(v):
    if isinstance(v, bool): return "bool"
    if isinstance(v, (int, float)): return "number"
    if v is None: return "null"
    return type(v).__name__  # str, list, dict


class DonorPool:
    def __init__(self):
        data = xio.load_split("test")
        self.items = []  # (idx, service, obj)
        for i, s in enumerate(data):
            if s["meta"]["decision"] == "direct_execute":
                self.items.append((i, s["meta"].get("service"), json.loads(U.scored_content(s))))
        self.args = []  # (idx, service, key, value)
        for i, sv, o in self.items:
            for k, v in ((o.get(U.CALL_KEY) or {}).get("tool_args") or {}).items():
                self.args.append((i, sv, k, v))

    def draw(self, obj, unit, service, rng):
        orig = U.get_value(obj, unit)
        if not unit.startswith("tool_args."):
            c = [(i, U.get_value(o, unit)) for i, sv, o in self.items if sv != service and _has(o, unit)
                 and _nonempty(U.get_value(o, unit)) and U.get_value(o, unit) != orig]
            if c:
                i, v = c[rng.integers(len(c))]
                return True, v, dict(donor_idx=int(i), donor_key=unit, method="same_field")
            return False, None, dict(donor_idx=None, donor_key=None, method="no_donor")
        k = unit.split(".", 1)[1]
        for method, cond in [("same_key", lambda a: a[2] == k), ("type_matched", lambda a: jtype(a[3]) == jtype(orig)),
                             ("any_arg", lambda a: True)]:
            c = [a for a in self.args if a[1] != service and _nonempty(a[3]) and a[3] != orig and cond(a)]
            if c:
                i, _, dk, v = c[rng.integers(len(c))]
                return True, v, dict(donor_idx=int(i), donor_key=dk, method=method)
        return False, None, dict(donor_idx=None, donor_key=None, method="no_donor")


def _nonempty(v):
    """donor values must carry content: null, "" and empty lists/objects would amount to a removal"""
    return not (v is None or v == "" or v == [] or v == {})


def _has(o, unit):
    try:
        U.get_value(o, unit); return True
    except (KeyError, TypeError):
        return False


def build_variants(ids):
    """-> list of dict(dataset_idx, unit, unit_type, strategy, sample, donor...) incl. the baseline (unit None)."""
    data = xio.load_split("test")
    pool = DonorPool()
    out = []
    for i in ids:
        s = data[i]
        obj, fmt, units = U.parse(s)
        svc = s["meta"].get("service")
        out.append(dict(dataset_idx=i, unit=None, unit_type="baseline", strategy="baseline", sample=U.variant(s)))
        for pos, u in enumerate(units):
            out.append(dict(dataset_idx=i, unit=u, unit_type=U.unit_type(u), strategy="M1_remove",
                            sample=U.variant(s, u, "remove")))
            rng = np.random.default_rng([C.SEED, int(i), pos])
            ok, v, info = pool.draw(obj, u, svc, rng)
            if not ok:
                out.append(dict(dataset_idx=i, unit=u, unit_type=U.unit_type(u), strategy="M2_replace", sample=None, **info))
                continue
            out.append(dict(dataset_idx=i, unit=u, unit_type=U.unit_type(u), strategy="M2_replace",
                            sample=U.variant(s, u, "replace", v), **info))
    return out


def query(ids, tag, dry):
    variants = build_variants(ids)
    jobs, keys = [], []
    for v in variants:
        if v["sample"] is None:
            continue
        for m in C.MODELS:
            jobs.append(dict(model=m, sample=v["sample"], tag=tag,
                             meta=dict(split="test", dataset_idx=int(v["dataset_idx"]), unit=v["unit"], strategy=v["strategy"],
                                       donor_idx=v.get("donor_idx"), donor_key=v.get("donor_key"), method=v.get("method"))))
    q = Q.Querier()
    # rule 5: baselines must come from this run, not from earlier phases -> report any pre-existing cache hits
    pre = sum(q.lookup(j["model"], j["sample"]) is not None for j in jobs)
    print(f"variants {len(variants)} | jobs {len(jobs)} | already cached before this run: {pre}", flush=True)
    t0 = time.time()
    q.run(jobs, dry_run=dry, require_tag=tag)
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


def noise_thresholds():
    s3 = json.load(open(R / "p0_s3_summary.json"))["rms_within_sd_by_model"]
    # 2 * sqrt(2) * SD (difference of two noisy queries, ~95%); floor 0.005 for the deterministic models
    return {m: {c: max(2 * np.sqrt(2) * s3[m][c], 0.005) for c in C.CLASSES} for m in s3}


def collect(ids, tag):
    variants = build_variants(ids)
    q = Q.Querier()
    st = xio.sample_table("test").set_index("dataset_idx")
    rows = []
    for v in variants:
        for m in C.MODELS:
            r = q.lookup(m, v["sample"]) if v["sample"] is not None else None
            p = (r or {}).get("probabilities") or {}
            rows.append(dict(dataset_idx=v["dataset_idx"], model=C.SHORT[m], unit=v["unit"], unit_type=v["unit_type"],
                             strategy=v["strategy"], donor_idx=v.get("donor_idx"), donor_key=v.get("donor_key"),
                             method=v.get("method"), json_valid=_valid(v["sample"]), got=Q.outcome(r) == "ok",
                             outcome=Q.outcome(r) if v["sample"] is not None else "no_donor",
                             choice=(r or {}).get("choice"), response_model=(r or {}).get("response_model"),
                             cost_usd=(r or {}).get("cost_usd"), latency_ms=(r or {}).get("latency_ms"),
                             queried_at=(r or {}).get("queried_at"), cache_tag=(r or {}).get("tag"),
                             **{f"p_{c}": p.get(c) for c in C.CLASSES}))
    d = pd.DataFrame(rows)
    d["multi_turn"] = d.dataset_idx.map(st.multi_turn)
    d["ground_truth"] = d.dataset_idx.map(st.ground_truth)
    d["service"] = d.dataset_idx.map(st.service)
    d["service_type"] = d.dataset_idx.map(st.service_type)
    d["model_label"] = d.model.map(lambda m: LABEL.get(m, m))
    base = d[d.strategy == "baseline"].set_index(["dataset_idx", "model"])
    for c in C.CLASSES:
        d[f"base_{c}"] = [base.loc[(i, m), f"p_{c}"] for i, m in zip(d.dataset_idx, d.model)]
        d[f"attr_{c}"] = d[f"base_{c}"] - d[f"p_{c}"]
    d["base_choice"] = [base.loc[(i, m), "choice"] for i, m in zip(d.dataset_idx, d.model)]
    d["flip"] = d.choice != d.base_choice
    d["tv"] = d[[f"attr_{c}" for c in C.CLASSES]].abs().sum(axis=1) / 2  # total-variation size of the change
    thr = noise_thresholds()
    for c in C.CLASSES:
        d[f"above_{c}"] = d[f"attr_{c}"].abs() > d.model.map(lambda m: thr[m][c])
    d["above_any"] = d[[f"above_{c}" for c in C.CLASSES]].any(axis=1)
    return d, thr


def _valid(sample):
    if sample is None:
        return False
    try:
        json.loads(U.scored_content(sample)); return True
    except Exception:
        return False


def malformed(row):
    v = [row[f"p_{c}"] for c in C.CLASSES]
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v): return True
    return min(v) < 0 or abs(sum(v) - 1) > 0.02


def analyze(ids, tag, prefix):
    d, thr = collect(ids, tag)
    d.to_csv(R / f"{prefix}_long.csv", index=False)
    res = dict(n_samples=len(ids), thresholds=thr, ids=[int(i) for i in ids], seed=C.SEED)
    v = d[d.strategy != "baseline"]
    vv = v[v.got]
    # ---- validity
    val = v.groupby(["strategy", "multi_turn"]).agg(variants=("unit", "size"), json_valid=("json_valid", "mean"),
                                                    answered=("got", "mean"),
                                                    provider_refusal=("outcome", lambda x: int((x == "provider_refusal").sum()))).reset_index()
    pr = v[v.outcome == "provider_refusal"]
    res["provider_refusal"] = dict(by_unit_type={f"{k[0]}|{k[1]}": int(n) for k, n in pr.groupby(["strategy", "unit_type"]).size().items()},
                                   by_label={f"{k[0]}|{k[1]}": int(n) for k, n in pr.groupby(["strategy", "ground_truth"]).size().items()},
                                   by_model=pr.model.value_counts().astype(int).to_dict())
    val["malformed"] = [int(sum(malformed(r) for _, r in vv[(vv.strategy == s) & (vv.multi_turn == mt)].iterrows()))
                        for s, mt in zip(val.strategy, val.multi_turn)]
    res["validity"] = val.astype({"multi_turn": str}).to_dict("records")
    res["m2_methods"] = {f"{k[0]}|{k[1]}": int(n) for k, n in
                         v[(v.strategy == "M2_replace") & (v.model == "jev")].groupby(["unit_type", "method"]).size().items()}
    res["pre_existing_cache_hits"] = int((d.cache_tag.notna() & (d.cache_tag != tag)).sum())
    res["response_models"] = {m: sorted(set(map(str, g.response_model.dropna()))) for m, g in d.groupby("model")}
    # ---- effects vs noise floor
    agg = dict(n=("unit", "size"), above_any=("above_any", "mean"), above_refuse=("above_refuse", "mean"),
               above_ask=("above_ask_human", "mean"), above_exec=("above_direct_execute", "mean"),
               flip=("flip", "mean"), median_tv=("tv", "median"), mean_tv=("tv", "mean"), max_tv=("tv", "max"))
    eff = vv.groupby(["model_label", "strategy", "multi_turn"]).agg(**agg).reset_index()
    eff.to_csv(R / f"{prefix}_effects.csv", index=False)
    res["effects"] = eff.astype({"multi_turn": str}).round(4).to_dict("records")
    # per unit type (preview of D1)
    ut = vv.groupby(["model_label", "strategy", "multi_turn", "unit_type"]).agg(n=("unit", "size"), mean_tv=("tv", "mean"),
                                                                             above_any=("above_any", "mean"), flip=("flip", "mean")).reset_index()
    ut.to_csv(R / f"{prefix}_unit_types.csv", index=False)
    # ---- M1 vs M2 agreement per sample x model
    ag = []
    for (i, m), g in vv.groupby(["dataset_idx", "model"]):
        a = g[g.strategy == "M1_remove"].set_index("unit")
        b = g[g.strategy == "M2_replace"].set_index("unit")
        common = a.index.intersection(b.index)
        r = dict(dataset_idx=i, model=m, multi_turn=bool(g.multi_turn.iloc[0]), n_units=len(common))
        if len(common) >= 3:
            for name, col in [("tv", "tv"), ("refuse", "attr_refuse"), ("ask", "attr_ask_human"), ("exec", "attr_direct_execute")]:
                x, y = a.loc[common, col].to_numpy(float), b.loc[common, col].to_numpy(float)
                r[f"rho_{name}"] = spearmanr(x, y).statistic if (np.ptp(x) > 0 and np.ptp(y) > 0) else np.nan
            r["top_same"] = a.loc[common, "tv"].idxmax() == b.loc[common, "tv"].idxmax() if (a.loc[common, "tv"].max() > 0
                                                                                            and b.loc[common, "tv"].max() > 0) else np.nan
            ab_a, ab_b = a.loc[common, "above_any"], b.loc[common, "above_any"]
            r["above_both"] = int((ab_a & ab_b).sum()); r["above_m1_only"] = int((ab_a & ~ab_b).sum())
            r["above_m2_only"] = int((~ab_a & ab_b).sum())
        ag.append(r)
    ag = pd.DataFrame(ag)
    ag["model_label"] = ag.model.map(lambda m: LABEL.get(m, m))
    ag.to_csv(R / f"{prefix}_m1m2_agreement.csv", index=False)
    aga = ag.groupby(["model_label", "multi_turn"]).agg(
        n=("dataset_idx", "size"), rho_tv_defined=("rho_tv", lambda x: int(x.notna().sum())),
        rho_tv_median=("rho_tv", "median"), rho_refuse_median=("rho_refuse", "median"),
        top_same=("top_same", "mean"), above_both=("above_both", "sum"), above_m1_only=("above_m1_only", "sum"),
        above_m2_only=("above_m2_only", "sum")).reset_index()
    res["agreement"] = aga.astype({"multi_turn": str}).round(4).to_dict("records")
    # ---- cost and runtime
    cr = d[d.cache_tag == tag]
    res["cost"] = {("multi_turn" if mt else "single_turn"): dict(calls=int(len(g)), usd=round(float(g.cost_usd.sum()), 5),
                                                                 samples=int(g.dataset_idx.nunique()),
                                                                 usd_per_sample=round(float(g.cost_usd.sum() / max(g.dataset_idx.nunique(), 1)), 5))
                   for mt, g in cr.groupby("multi_turn")}
    if len(cr):
        ts = pd.to_datetime(cr.queried_at)
        res["wall_clock_min"] = round((ts.max() - ts.min()).total_seconds() / 60, 1)
    # ---- worked examples: per category the pilot call with the largest summed M1 effect over models
    ex = {}
    m1 = vv[vv.strategy == "M1_remove"]
    for name, sel in [("refusal_single_turn", (m1.ground_truth == "refuse") & ~m1.multi_turn),
                      ("ask_single_turn", (m1.ground_truth == "ask_human") & ~m1.multi_turn),
                      ("multi_turn_refusal", (m1.ground_truth == "refuse") & m1.multi_turn)]:
        g = m1[sel]
        if g.empty: continue
        i = int(g.groupby("dataset_idx").above_any.sum().idxmax())
        e = vv[vv.dataset_idx == i]
        base_rows = d[(d.dataset_idx == i) & (d.strategy == "baseline")]
        ex[name] = dict(dataset_idx=i, service=e.service.iloc[0], label=e.ground_truth.iloc[0],
                        baseline={r.model_label: [round(r[f"p_{c}"], 3) for c in C.CLASSES] for _, r in base_rows.iterrows()},
                        top_units={ml: [(r.unit, r.strategy, round(r.attr_refuse, 3), round(r.attr_ask_human, 3),
                                         round(r.attr_direct_execute, 3), bool(r.above_any), bool(r.flip))
                                        for _, r in g2.sort_values("tv", ascending=False).head(4).iterrows()]
                                   for ml, g2 in e.groupby("model_label")})
    res["examples"] = ex
    json.dump(res, open(R / f"{prefix}_summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print("validity\n", val.to_string())
    print("\nM2 methods (per unit, jev rows)\n", res["m2_methods"])
    print("\npre-existing cache hits:", res["pre_existing_cache_hits"], "| versions:", res["response_models"])
    print("\neffects\n", eff.round(3).to_string())
    print("\nagreement\n", aga.round(3).to_string())
    print("\ncost", res["cost"], res.get("wall_clock_min"))
    print("\nunit types (M1)\n", ut[ut.strategy == "M1_remove"].pivot_table(index=["model_label", "multi_turn"], columns="unit_type",
                                                                        values="mean_tv").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    assert a.pilot, "only --pilot is implemented until Phase C is confirmed"
    ids = pilot_ids()
    if a.analyze:
        analyze(ids, "b_pilot", "b_pilot")
    else:
        query(ids, "b_pilot", a.dry)
