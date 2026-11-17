"""Step D: cross-model comparison of the field-level Shapley values (no API calls).

Reads results/[<slug>/]shapley_values<suffix>.csv, checks<suffix>.json, determinism.json and run_log.txt of the five
models and writes results/compare/ and plots/compare/:
  mean_abs_phi<suffix>.csv/.png     mean |phi| per field x model, one panel per class
  mean_phi_own_class<suffix>.csv/.png  mean phi for refuse-labelled calls on P(refuse) and execute-labelled calls on
                                    P(direct_execute), per field x model
  baselines<suffix>.csv             mean v(empty) and v(full) per model x label x class
  agreement<suffix>.csv/.png        per model pair and class: median Spearman of the per-call phi vectors over the
                                    calls defined for both models, and the share of calls with the same top-|phi| field
  model_table.csv / .md             per model: repeats, determinism, complete calls per design, refused coalitions,
                                    full-input diff vs October, answers and billed cost
Designs: main (__fixed-tool), second view (__fixed-description+tool+tool_args), and the 7-player design for the
models where it was run on all 30 calls (gpt: undefined, not included).

  .venv/bin/python src/compare.py
"""
import itertools
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import CLASSES, PLOTS, RESULTS, SLUGS, ModelPaths  # noqa: E402
from fields import ALL_FIELDS  # noqa: E402
from plots import CLASS_TITLE, GRID, INK, INK2, SURFACE  # noqa: E402  (also sets the rcParams)

OUT, POUT = RESULTS / "compare", PLOTS / "compare"
DESIGNS = {"__fixed-tool": "main design (tool fixed)",
           "__fixed-description+tool+tool_args": "second view (whole tool call fixed)",
           "": "7-player design (no fixed field)"}
ORDER = list(SLUGS.values())                       # gpt, jev, microsoft, liquid, pplx
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a8302f"])


def load(sfx):
    """Long table of all models with a complete 30-call run of this design."""
    frames = []
    for model, slug in SLUGS.items():
        mp = ModelPaths(model)
        f = mp.results / f"shapley_values{sfx}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if d.dataset_idx.nunique() < 30:           # pilot-only run (7-player gpt): not comparable
            continue
        frames.append(d.assign(model=slug))
    return pd.concat(frames) if frames else None


def heat(ax, mat, cmap, norm, fmt="{:.3f}"):
    ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(mat.values):
        if np.isnan(v):
            ax.text(c, r, "–", ha="center", va="center", fontsize=8, color=INK2)
            continue
        rgb = cmap(norm(v))[:3]
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8, color=INK if lum > 0.55 else "#ffffff")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def mean_abs_phi(df, sfx):
    ok = df[df.status == "ok"].assign(a=lambda d: d.phi.abs())
    m = ok.groupby(["class", "field", "model"]).a.mean().unstack("model")
    models = [x for x in ORDER if x in m.columns]
    m = m[models]
    m.to_csv(OUT / f"mean_abs_phi{sfx}.csv")
    fields = [f for f in ALL_FIELDS if f in m.index.get_level_values("field")]
    vmax = float(np.nanmax(m.values))
    fig, axes = plt.subplots(1, 3, figsize=(3.0 * len(models) / 1.6 + 2.5, 0.5 * len(fields) + 2.0), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        heat(ax, m.loc[cls].reindex(fields), SEQ, plt.Normalize(0, vmax))
        ax.set_title(CLASS_TITLE[cls], loc="left", fontsize=10)
    fig.suptitle(f"Mean |φ| per field and model — {DESIGNS[sfx]}, 30 calls (complete calls only)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(POUT / f"mean_abs_phi{sfx}.png", dpi=160); plt.close(fig)
    return m


def mean_phi_own(df, sfx):
    ok = df[df.status == "ok"]
    parts = {"refuse-labelled calls, φ on P(refuse)": ok[(ok.label == "refuse") & (ok["class"] == "refuse")],
             "execute-labelled calls, φ on P(direct_execute)":
                 ok[(ok.label == "direct_execute") & (ok["class"] == "direct_execute")]}
    tabs = {k: v.groupby(["field", "model"]).phi.mean().unstack("model") for k, v in parts.items()}
    models = [x for x in ORDER if any(x in t.columns for t in tabs.values())]
    fields = [f for f in ALL_FIELDS if f in set(ok.field)]
    out = pd.concat({k: t.reindex(index=fields, columns=models) for k, t in tabs.items()}, names=["subset"])
    out.to_csv(OUT / f"mean_phi_own_class{sfx}.csv")
    lim = float(np.nanmax(np.abs(out.values)))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.9 * len(models) + 1.2) + 1.5, 0.5 * len(fields) + 2.0), sharey=True)
    for ax, (k, t) in zip(axes, tabs.items()):
        heat(ax, t.reindex(index=fields, columns=models), DIV, TwoSlopeNorm(0, -lim, lim), fmt="{:+.3f}")
        ax.set_title(k, loc="left", fontsize=10)
    fig.suptitle(f"Mean φ on the label's own class — {DESIGNS[sfx]} (red raises, blue lowers)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(POUT / f"mean_phi_own_class{sfx}.png", dpi=160); plt.close(fig)
    return out


def baselines(df, sfx):
    ok = df[df.status == "ok"].drop_duplicates(["model", "dataset_idx", "class"])
    b = ok.groupby(["model", "label", "class"])[["v_empty", "v_full"]].mean().round(4).reset_index()
    b["model"] = pd.Categorical(b.model, ORDER, ordered=True)
    b = b.sort_values(["model", "label", "class"])
    b.to_csv(OUT / f"baselines{sfx}.csv", index=False)
    return b


def agreement(df, sfx):
    ok = df[df.status == "ok"]
    models = [x for x in ORDER if x in set(ok.model)]
    rows = []
    for cls in CLASSES:
        d = ok[ok["class"] == cls]
        vec = {(m, i): g.set_index("field").phi for (m, i), g in d.groupby(["model", "dataset_idx"])}
        for a, b in itertools.combinations(models, 2):
            ids = sorted({i for (m, i) in vec if m == a} & {i for (m, i) in vec if m == b})
            rhos, same = [], []
            for i in ids:
                x, y = vec[(a, i)], vec[(b, i)].reindex(vec[(a, i)].index)
                same.append(x.abs().idxmax() == y.abs().idxmax())
                if x.nunique() > 1 and y.nunique() > 1:
                    rhos.append(spearmanr(x, y).statistic)
            rows.append({"class": cls, "model_a": a, "model_b": b, "n_calls": len(ids),
                         "median_spearman": float(np.median(rhos)) if rhos else np.nan,
                         "mean_spearman": float(np.mean(rhos)) if rhos else np.nan,
                         "n_calls_with_rho": len(rhos), "share_same_top_field": float(np.mean(same)) if same else np.nan})
    ag = pd.DataFrame(rows)
    ag.to_csv(OUT / f"agreement{sfx}.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for col, cls in enumerate(CLASSES):
        for row, (metric, title, cmap, norm, fmt) in enumerate([
                ("median_spearman", "median Spearman ρ of φ", DIV, TwoSlopeNorm(0, -1, 1), "{:.2f}"),
                ("share_same_top_field", "share same top field", SEQ, plt.Normalize(0, 1),
                 "{:.2f}")]):
            mat = pd.DataFrame(np.nan, index=models, columns=models)
            for _, r in ag[ag["class"] == cls].iterrows():
                mat.loc[r.model_a, r.model_b] = mat.loc[r.model_b, r.model_a] = r[metric]
            ax = axes[row, col]
            heat(ax, mat, cmap, norm, fmt=fmt)
            ax.set_title(f"{CLASS_TITLE[cls]} · {title}", loc="left", fontsize=9)
    fig.suptitle(f"Agreement between models — {DESIGNS[sfx]} (calls defined for both models)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(POUT / f"agreement{sfx}.png", dpi=160); plt.close(fig)
    return ag


def billed_cost(run_log):
    if not run_log.exists():
        return 0.0
    return sum(float(x) for x in re.findall(r"billed cost now \(API usage\.cost\): USD ([0-9.]+)", run_log.read_text()))


def model_table():
    rows = []
    for model, slug in SLUGS.items():
        mp = ModelPaths(model)
        det = json.load(open(mp.determinism)) if mp.determinism.exists() else None
        if isinstance(det, dict):
            determinism = f"no (probe max diff {det['max_abs_diff']:.3g}, mean std {det['mean_std_per_class_and_input']:.3g})" \
                if not det["deterministic"] else "yes (probe max diff 0)"
            probe_cost = det["billed_cost_usd"]
        else:
            determinism = f"yes (max diff {max(d['max_abs_diff'] for d in det)} on 3 re-queries)" if det else "?"
            probe_cost = 0.0
        n_answers = sum(1 for line in open(mp.cache) if line.strip()) if mp.cache.exists() else 0
        row = {"model": slug, "model_id": model, "repeats": mp.repeats(), "deterministic": determinism,
               "cache_answers": n_answers, "billed_cost_usd": round(billed_cost(mp.run_log) + probe_cost, 4)}
        for sfx, name in [("__fixed-tool", "main"), ("__fixed-description+tool+tool_args", "second"), ("", "7p")]:
            f = mp.results / f"checks{sfx}.json"
            if not f.exists():
                continue
            c = json.load(open(f))
            row[f"{name}_calls"] = c["n_calls"]
            row[f"{name}_complete"] = c["n_complete"]
            row[f"{name}_refused_coalitions"] = sum(c["provider_refusals"]["refused_coalitions_by_size"].values())
            if name == "main":
                row["oct_max_abs_diff"] = c["agreement_with_october"]["max_abs_diff"]
                row["oct_mean_abs_diff"] = c.get("agreement_with_october_mean_abs_diff")
                row["failure_types"] = "; ".join(f"{k}: {v['count']}" for k, v in c["failure_types_in_cache"].items()) \
                    or "none"
                row["efficiency_max_err"] = c["efficiency_max_abs_error"]
                row["indep_shapley_max_diff"] = c["independent_shapley_max_abs_diff"]
                nz = c.get("noise")
                row["noise_mean_std"] = nz["mean_std_all_classes"] if nz else None
                row["share_abs_phi_below_noise"] = (np.mean([v["share_abs_phi_below_mean_std"]
                                                             for v in nz["per_class"].values()]) if nz else None)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "model_table.csv", index=False)
    try:
        md = t.to_markdown(index=False)            # needs the optional 'tabulate' package
    except ImportError:
        cols = list(t.columns)
        md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                       ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in t.itertuples(index=False)])
    (OUT / "model_table.md").write_text(md + "\n")
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    POUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sfx in DESIGNS:
        df = load(sfx)
        if df is None:
            continue
        m = mean_abs_phi(df, sfx)
        own = mean_phi_own(df, sfx)
        baselines(df, sfx)
        ag = agreement(df, sfx)
        summary[sfx or "__7player"] = {
            "models": sorted(set(df.model), key=ORDER.index),
            "complete_calls": df[df.status == "ok"].groupby("model").dataset_idx.nunique().to_dict(),
            "top_field_by_mean_abs_phi": {cls: m.loc[cls].idxmax().to_dict() for cls in CLASSES},
            "agreement_median_spearman_by_class": ag.groupby("class").median_spearman.median().to_dict(),
            "agreement_same_top_by_class": ag.groupby("class").share_same_top_field.mean().to_dict(),
            "agreement_pairs_with_gpt_vs_without": {
                name: {"mean_of_median_spearman": float(sel.median_spearman.mean()),
                       "mean_share_same_top_field": float(sel.share_same_top_field.mean()), "n_pair_class_rows": len(sel)}
                for name, sel in [("with_gpt", ag[(ag.model_a == "gpt") | (ag.model_b == "gpt")]),
                                  ("without_gpt", ag[(ag.model_a != "gpt") & (ag.model_b != "gpt")])] if len(sel)}}
        print(f"\n== {DESIGNS[sfx]}: models {summary[sfx or '__7player']['models']}")
        print(m.round(3).to_string())
        print(own.round(3).to_string())
        print(ag.round(2).to_string(index=False))
    t = model_table()
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
