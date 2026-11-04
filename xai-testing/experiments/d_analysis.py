"""Phase D: analyses of the Phase C occlusion results (no API calls). Input: results/c_long.csv.gz.
Every table is split by single-turn / multi-turn. Outputs: results/d_*.csv, results/d_summary.json.

D0 outcomes  provider_refusal per unit type and label (gpt), baseline and masked
D1 reliance  mean TV and share above noise per unit type x model x label x turn
D2 plausibility  top-attributed unit type vs (i) the UNREVIEWED risk-class map, (ii) units the annotator reasoning names
D3 shortcuts  effect of removing `description` and `service`; refusals whose p_refuse RISES when description is removed
D4 truncation  label-free and label-based diagnostics on multi-turn calls; false-positive rate on single-turn calls
D5 stability  test-retest of single-query attributions (repeats), agreement across models, noise floor
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402

R = C.RESULTS
MODELS = ["jev", "microsoft", "gpt", "liquid", "pplx"]
LABEL = {"microsoft": "microsoft (post-drift)"}
TURN = {False: "single_turn", True: "multi_turn"}
RES = {}


def load():
    d = pd.read_csv(R / "c_long.csv.gz", low_memory=False)
    d["turn"] = d.multi_turn.map(TURN)
    return d


def d0(d):
    g = d[d.model == "gpt"]
    t = g.groupby(["turn", "unit_type"]).agg(n=("outcome", "size"), provider_refusal=("outcome", lambda x: int((x == "provider_refusal").sum())))
    t["rate"] = t.provider_refusal / t.n
    tl = g.groupby(["turn", "ground_truth"]).agg(n=("outcome", "size"), provider_refusal=("outcome", lambda x: int((x == "provider_refusal").sum())))
    tl["rate"] = tl.provider_refusal / tl.n
    calls = g.groupby("dataset_idx").outcome.apply(lambda x: (x == "provider_refusal").any())
    allm = d.groupby(["model", "outcome"]).size().unstack(fill_value=0)
    t.to_csv(R / "d0_provider_refusal_by_unit.csv"); tl.to_csv(R / "d0_provider_refusal_by_label.csv")
    RES["D0"] = dict(outcomes_all_models={m: r.to_dict() for m, r in allm.iterrows()},
                     by_unit=t.reset_index().round(4).to_dict("records"), by_label=tl.reset_index().round(4).to_dict("records"),
                     calls_with_any_provider_refusal=int(calls.sum()),
                     baseline_provider_refusal_calls=int(((g.unit_type == "baseline") & (g.outcome == "provider_refusal")).sum()))
    print("D0 gpt provider_refusal by unit\n", t.round(4).to_string(), "\nby label\n", tl.round(4).to_string())


def valid(d):
    v = d[(d.unit_type != "baseline") & (d.outcome == "ok") & (d.base_outcome == "ok")].copy()
    return v


def noise_section(v):
    t = v.groupby(["model", "turn"]).agg(n=("tv", "size"), above=("above", "mean"), flip=("flip", "mean"),
                                         median_tv=("tv", "median"), mean_tv=("tv", "mean"), thr=("thr_tv", "first"))
    RES["noise_and_effects"] = t.reset_index().round(4).to_dict("records")
    print("effects vs noise\n", t.round(3).to_string())


def d1(v):
    t = v.groupby(["model", "turn", "ground_truth", "unit_type"]).agg(n=("tv", "size"), mean_tv=("tv", "mean"),
                                                                       above=("above", "mean"), flip=("flip", "mean"),
                                                                       mean_attr_refuse=("attr_refuse", "mean"),
                                                                       mean_attr_exec=("attr_direct_execute", "mean")).reset_index()
    t.to_csv(R / "d1_reliance.csv", index=False)
    s = v.groupby(["model", "turn", "unit_type"]).agg(n=("tv", "size"), mean_tv=("tv", "mean"), above=("above", "mean"),
                                                      flip=("flip", "mean")).reset_index()
    s.to_csv(R / "d1_reliance_overall.csv", index=False)
    RES["D1"] = s.round(4).to_dict("records")
    print("D1 mean TV (share above noise) by unit type\n",
          s.assign(x=s.mean_tv.round(3).astype(str) + " (" + (100 * s.above).round(0).astype(int).astype(str) + "%)")
          .pivot_table(index=["model", "turn"], columns="unit_type", values="x", aggfunc="first").to_string())


def mentioned_units(sample, reasoning):
    obj, _, units = U.parse(sample)
    txt = reasoning.lower()
    out = set()
    words = {"user_task": ["user_task", "task", "request", "user asked", "user asks"], "user_memory": ["memory", "user_memory"],
             "prior_steps": ["prior step", "prior_steps", "previous step", "earlier step"],
             "description": ["description"], "service": []}
    for u in units:
        ut = U.unit_type(u)
        if ut in words and any(w in txt for w in words[ut]):
            out.add(u)
        if ut == "service" and str(obj.get("service", "")).lower() in txt:
            out.add(u)
        if ut == "tool":
            t = str(obj[U.CALL_KEY].get("tool", "")).lower()
            if t and (t in txt or t.replace("_", " ") in txt):
                out.add(u)
        if ut == "tool_args":
            k = u.split(".", 1)[1]
            val = obj[U.CALL_KEY]["tool_args"][k]
            vs = str(val).lower() if not isinstance(val, (dict, list)) else ""
            if (len(k) >= 4 and (k.lower() in txt or k.lower().replace("_", " ") in txt)) or (len(vs) >= 4 and vs in txt):
                out.add(u)
    return out, units


def d2(v):
    mp = pd.read_csv(R / "riskclass_unit_map.csv")
    exp = {r.risk_class: set(r.expected_unit_types.split(";")) for r in mp.itertuples()}
    data = xio.load_split("test")
    st = xio.sample_table("test").set_index("dataset_idx")
    ment = {}
    for i in v.dataset_idx.unique():
        ment[i] = mentioned_units(data[i], st.annotator[i])
    rows = []
    for (i, m), g in v.groupby(["dataset_idx", "model"]):
        if g.tv.max() <= 0 or g.above.sum() == 0:
            top = None
        else:
            top = g.loc[g.tv.idxmax()]
        rc = st.risk_class[i]
        mset, units = ment[i]
        r = dict(dataset_idx=i, model=m, turn=g.turn.iloc[0], risk_class=rc, label=st.ground_truth[i],
                 has_top=top is not None, top_unit=None if top is None else top.unit,
                 top_type=None if top is None else top.unit_type, n_units=len(units))
        if rc in exp:
            r["map_hit"] = (top is not None) and top.unit_type in exp[rc]
            r["map_chance"] = np.mean([U.unit_type(u) in exp[rc] for u in units])
        r["n_mentioned"] = len(mset)
        if mset:
            r["ment_hit"] = (top is not None) and top.unit in mset
            r["ment_chance"] = len(mset) / len(units)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(R / "d2_plausibility.csv", index=False)
    a = t[t.map_chance.notna() & t.has_top].groupby(["model", "turn"]).agg(n=("map_hit", "size"), hit=("map_hit", "mean"),
                                                                           chance=("map_chance", "mean")).reset_index()
    b = t[t.ment_chance.notna() & t.has_top].groupby(["model", "turn"]).agg(n=("ment_hit", "size"), hit=("ment_hit", "mean"),
                                                                            chance=("ment_chance", "mean")).reset_index()
    c = t[t.map_chance.notna() & t.has_top & (t.turn == "single_turn")].groupby(["risk_class"]).agg(
        n=("map_hit", "size"), hit=("map_hit", "mean"), chance=("map_chance", "mean")).reset_index()
    tops = t[t.has_top].groupby(["model", "turn", "top_type"]).size().unstack(fill_value=0)
    RES["D2"] = dict(map_status="UNREVIEWED (results/riskclass_unit_map.csv) - provisional",
                     vs_map=a.round(4).to_dict("records"), vs_annotator=b.round(4).to_dict("records"),
                     by_risk_class_single=c.round(4).to_dict("records"),
                     calls_without_above_noise_unit=t.groupby("model").has_top.apply(lambda x: int((~x).sum())).to_dict(),
                     top_type_counts={f"{k[0]}|{k[1]}": r.to_dict() for k, r in tops.iterrows()},
                     mention_rate=float(np.mean([len(ment[i][0]) > 0 for i in ment])))
    print("D2 vs map (UNREVIEWED)\n", a.round(3).to_string(), "\nD2 vs annotator\n", b.round(3).to_string())
    print(c.round(3).to_string())


def d3(v):
    out = {}
    for ut in ["description", "service"]:
        g = v[v.unit_type == ut]
        t = g.groupby(["model", "turn", "ground_truth"]).agg(
            n=("tv", "size"), above=("above", "mean"), flip=("flip", "mean"), mean_attr_refuse=("attr_refuse", "mean"),
            mean_attr_exec=("attr_direct_execute", "mean"),
            ).reset_index()
        # removing the unit RAISES p_refuse (attr_refuse < 0) by more than noise
        g = g.assign(raise_rf=(g.attr_refuse < 0) & g.above, lower_rf=(g.attr_refuse > 0) & g.above,
                     raise_ex=(g.attr_direct_execute < 0) & g.above)
        t = g.groupby(["model", "turn", "ground_truth"]).agg(n=("tv", "size"), above=("above", "mean"), flip=("flip", "mean"),
                                                             mean_attr_refuse=("attr_refuse", "mean"),
                                                             mean_attr_exec=("attr_direct_execute", "mean"),
                                                             removal_raises_refuse=("raise_rf", "mean"),
                                                             removal_lowers_refuse=("lower_rf", "mean"),
                                                             removal_raises_exec=("raise_ex", "mean")).reset_index()
        t.to_csv(R / f"d3_{ut}.csv", index=False)
        out[ut] = t.round(4).to_dict("records")
        print(f"D3 {ut}\n", t.round(3).to_string())
    g = v[(v.unit_type == "description") & (v.ground_truth == "refuse") & (v.attr_refuse < 0) & v.above]
    cnt = g.groupby("dataset_idx").model.nunique()
    out["refusals_where_description_suppresses_refusal"] = dict(
        by_n_models=cnt.value_counts().sort_index().astype(int).to_dict(),
        n_refusals_single=int((v[(v.ground_truth == "refuse") & ~v.multi_turn].dataset_idx.nunique())),
        n_refusals_multi=int((v[(v.ground_truth == "refuse") & v.multi_turn].dataset_idx.nunique())),
        ge3_models_ids=cnt[cnt >= 3].index.astype(int).tolist(),
        ge3_turn={TURN[k]: int(n) for k, n in xio.sample_table("test").set_index("dataset_idx").multi_turn[cnt[cnt >= 3].index].value_counts().items()})
    ex = g[g.dataset_idx.isin(cnt[cnt >= 3].index)].sort_values("attr_refuse").head(15)
    ex[["dataset_idx", "model", "service", "turn", "attr_refuse", "base_refuse", "p_refuse"]].to_csv(R / "d3_description_suppresses_refusal_examples.csv", index=False)
    RES["D3"] = out
    print("description suppresses refusal (>= n models):", out["refusals_where_description_suppresses_refusal"]["by_n_models"])


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def d4(v, d):
    st = xio.sample_table("test").set_index("dataset_idx")
    call = v.unit_type.isin(["tool", "tool_args"])
    rows = []
    for (i, m), g in v.groupby(["dataset_idx", "model"]):
        c = g[g.unit_type.isin(["tool", "tool_args"])]
        tot = g.tv.sum()
        rows.append(dict(dataset_idx=i, model=m, multi_turn=bool(st.multi_turn[i]), label=st.ground_truth[i],
                         call_any_above=bool(c.above.any()), call_share=float(c.tv.sum() / tot) if tot > 0 else np.nan,
                         task_share=float(g[g.unit_type == "user_task"].tv.sum() / tot) if tot > 0 else np.nan,
                         base_refuse=g.base_refuse.iloc[0]))
    t = pd.DataFrame(rows)
    t.to_csv(R / "d4_call_reliance.csv", index=False)
    per = []
    for m, g in t.groupby("model"):
        mt, s = g[g.multi_turn], g[~g.multi_turn]
        per.append(dict(model=m, n_multi=len(mt), n_single=len(s),
                        flag_call_insensitive_multi=float((~mt.call_any_above).mean()),
                        flag_call_insensitive_single=float((~s.call_any_above).mean()),
                        call_share_median_multi=float(mt.call_share.median()), call_share_median_single=float(s.call_share.median()),
                        auroc_low_call_share=auroc(-mt.call_share.dropna(), -s.call_share.dropna()),
                        auroc_low_call_share_refusals=auroc(-mt[mt.label == "refuse"].call_share.dropna(), -s[s.label == "refuse"].call_share.dropna())))
    per = pd.DataFrame(per)
    agree = t.assign(ins=~t.call_any_above).groupby("dataset_idx").agg(k=("ins", "sum"), mt=("multi_turn", "first"))
    struct_mt = int((st.multi_turn & (st.scored_tool != st.labelled_tool)).sum())
    RES["D4"] = dict(
        label_based=dict(rule="labelled tool not among the scored units (scored_tool != labelled_tool)",
                         flagged_multi=struct_mt, n_multi=int(st.multi_turn.sum()), false_positive_single=0,
                         note="exact by construction: single-turn calls score the labelled call; needs the last turn, i.e. the full conversation"),
        label_free=per.round(4).to_dict("records"),
        ge3_models_call_insensitive=dict(multi=int(((agree.k >= 3) & agree.mt).sum()), n_multi=int(agree.mt.sum()),
                                         single=int(((agree.k >= 3) & ~agree.mt).sum()), n_single=int((~agree.mt).sum())))
    print("D4 label-free\n", per.round(3).to_string(), "\n", RES["D4"]["ge3_models_call_insensitive"])


def d5(v):
    # (a) test-retest of single-query attributions: rep c0 vs rep c1 (noisy models)
    # computed from the long table with per-repeat columns
    rel = []
    for m in ["jev", "microsoft", "liquid", "pplx"]:  # pplx: 3 repeats since the 3 Nov drift
        g = v[v.model == m]
        base = _base_reps(m)
        for (i, turn), h in g.groupby(["dataset_idx", "turn"]):
            if len(h) < 3 or i not in base.index: continue
            tvs = []
            for r in ["c0", "c1", "c2"]:
                b = base.loc[i, [f"p_{c}_{r}" for c in C.CLASSES]].to_numpy(float)
                x = h[[f"p_{c}_{r}" for c in C.CLASSES]].to_numpy(float)
                tvs.append(np.abs(b[None, :] - x).sum(1) / 2)
            if any(np.isnan(t).any() for t in tvs) or min(np.ptp(t) for t in tvs[:2]) == 0: continue
            rho = spearmanr(tvs[0], tvs[1]).statistic
            rel.append(dict(model=m, dataset_idx=i, turn=turn, rho_single_query=rho))
    rel = pd.DataFrame(rel)
    rr = rel.groupby(["model", "turn"]).rho_single_query.median()
    sb = (3 * rr) / (1 + 2 * rr)  # Spearman-Brown for the mean of 3
    # (b) agreement across models: per call, Spearman of TV vectors over units for each model pair
    pairs = []
    piv = v.pivot_table(index=["dataset_idx", "unit"], columns="model", values="tv")
    turn = v.drop_duplicates("dataset_idx").set_index("dataset_idx").turn
    for i, h in piv.groupby(level=0):
        if len(h) < 3: continue
        for a, b in combinations(MODELS, 2):
            x, y = h[a].to_numpy(), h[b].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0: continue
            pairs.append(dict(dataset_idx=i, turn=turn[i], pair=f"{a}-{b}", rho=spearmanr(x[ok], y[ok]).statistic,
                              top_same=h.index[np.nanargmax(x)] == h.index[np.nanargmax(y)]))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(R / "d5_model_pairs.csv", index=False)
    pt = pairs.groupby(["pair", "turn"]).agg(n=("rho", "size"), rho_median=("rho", "median"), top_same=("top_same", "mean")).reset_index()
    RES["D5"] = dict(test_retest_single_query_median_rho=rr.round(4).reset_index().to_dict("records"),
                     spearman_brown_mean_of_3=sb.round(4).reset_index().to_dict("records"),
                     model_pairs=pt.round(4).to_dict("records"),
                     model_pairs_overall=pairs.groupby("turn").agg(rho_median=("rho", "median"), top_same=("top_same", "mean")).round(4).reset_index().to_dict("records"),
                     strategy_agreement_pilot="see b_pilot_m1m2_agreement.csv: median rho 0.0-0.6, top unit same 9-54%")
    print("D5 test-retest (single query)\n", rr.round(3).to_string(), "\nSpearman-Brown mean of 3\n", sb.round(3).to_string())
    print(pt.round(3).to_string())


_BASE = {}


def _base_reps(m):
    if m not in _BASE:
        d = pd.read_csv(R / "c_long.csv.gz", low_memory=False, usecols=lambda c: c in {"dataset_idx", "model", "unit_type"} or "_c" in c)
        _BASE[m] = d[(d.model == m) & (d.unit_type == "baseline")].set_index("dataset_idx")
    return _BASE[m]


def main():
    d = load()
    d0(d)
    v = valid(d)
    noise_section(v)
    d1(v); d2(v); d3(v); d4(v, d); d5(v)
    xio.dump_json(RES, R / "d_summary.json")


if __name__ == "__main__":
    main()
