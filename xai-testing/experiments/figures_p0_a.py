"""Figures for the Phase 0 and Phase A sections of FINDINGS.md (static PNGs, from results/ only).
Palette: validated two-slot categorical (blue #2a78d6, orange #eb6834) on surface #fcfcfb; text in ink, not series color."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

R = C.RESULTS
S1, S2, SURF, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
ORDER = ["gpt", "pplx", "jev", "microsoft", "liquid"]


def bars(ax, labels, a, b, la, lb, fmt, loc="upper left"):
    x = np.arange(len(labels)); w = 0.36
    for off, v, c, lab in [(-w / 2 - 0.01, a, S1, la), (w / 2 + 0.01, b, S2, lb)]:
        ax.bar(x + off, v, w, color=c, label=lab, edgecolor=SURF, linewidth=1)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi, fmt(vi), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels); ax.legend(frameon=False, loc=loc)


def fig_noise_vs_drift():
    s3 = json.load(open(R / "p0_s3_summary.json"))["rms_within_sd_by_model"]
    s2 = pd.read_csv(R / "p0_s2_drift.csv")
    noise = [np.mean(list(s3[m].values())) for m in ORDER]
    drift = [s2[s2.model == m][[f"absdiff_{c}" for c in C.CLASSES]].to_numpy().mean() for m in ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, ORDER, noise, drift, "noise: within-call SD today (10 calls x 3 repeats)",
         "drift: mean |today - October| (30 calls)", lambda v: f"{v:.3f}")
    ax.set_ylabel("probability units (mean over 3 classes)")
    ax.set_title("Phase 0: microsoft drifted far above its noise floor; gpt and pplx are deterministic", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "fig_p0_noise_vs_drift.png", dpi=170); plt.close(fig)


def fig_a1_agreement():
    a = pd.read_csv(R / "a1_flags_test.csv")
    k = np.arange(6)
    st = a[~a.multi_turn].n_flags.value_counts().reindex(k, fill_value=0) / (~a.multi_turn).sum()
    mt = a[a.multi_turn].n_flags.value_counts().reindex(k, fill_value=0) / a.multi_turn.sum()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, [str(i) for i in k], st.values, mt.values, f"single-turn (n = {(~a.multi_turn).sum()})",
         f"multi-turn (n = {a.multi_turn.sum()})", lambda v: f"{v:.0%}", loc="upper right")
    ax.set_xlabel("number of the five models that flag the label (Confident Learning)")
    ax.set_ylabel("share of test calls")
    ax.set_title("A1: label flags by model agreement, test split", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "fig_a1_agreement.png", dpi=170); plt.close(fig)


def fig_a3b():
    t = pd.DataFrame(json.load(open(R / "a3b_multiturn_whatif.json")))
    t = t[t.variant.str.startswith("paper")].set_index("model")
    ms = ["jev", "microsoft", "liquid", "pplx", "mean"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, ms, t.loc[ms, "harm_single_turn"].values, t.loc[ms, "harm_multi_turn"].values,
         "single-turn test refusals", "multi-turn test refusals", lambda v: f"{v:.1%}", loc="upper right")
    ax.axhline(C.A_HARM, color=INK2, lw=1, ls="--"); ax.text(-0.55, C.A_HARM + 0.001, "5% budget", va="bottom", ha="left", fontsize=7.5, color=INK2)
    ax.set_ylabel("harmful execution (mean over 60 splits)")
    ax.set_title("A3b: the G4 harm budget is spent on multi-turn (truncated) refusals", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "fig_a3b_harm_by_turn.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_noise_vs_drift(); fig_a1_agreement(); fig_a3b(); print("ok")


def fig_b_pilot():
    """Phase B pilot: share of units above the noise threshold (M1, single-turn) vs. the same threshold on pure noise."""
    eff = pd.read_csv(R / "b_pilot_effects.csv")
    nz = pd.read_csv(R / "b_pilot_noise_check.csv")
    sys.path.insert(0, str(ROOT))
    from experiments.b_occlusion import noise_thresholds
    thr = noise_thresholds()
    nz["fp"] = [any(r[f"d_{c}"] > thr[r.model][c] for c in C.CLASSES) for _, r in nz.iterrows()]
    fp = nz.groupby("model").fp.mean()
    lab = {"microsoft": "microsoft (post-drift)"}
    obs = [eff[(eff.model_label == lab.get(m, m)) & (eff.strategy == "M1_remove") & (~eff.multi_turn)].above_any.iloc[0] for m in ORDER]
    ORD2 = ["gpt", "pplx", "jev", "microsoft\n(post-drift)", "liquid"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, ORD2, obs, [fp[m] for m in ORDER], "M1 units above threshold (single-turn pilot)",
         "same threshold on pure repeat noise (40 pairs)", lambda v: f"{v:.0%}", loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of units (any class above threshold)")
    ax.set_title("Phase B pilot: occlusion effects vs. the noise floor", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "fig_b_pilot_vs_noise.png", dpi=170); plt.close(fig)


def fig_b_unit_types():
    d = pd.read_csv(R / "b_pilot_long.csv")
    d = d[d.got & (d.strategy != "baseline") & ~d.multi_turn]
    order = ["user_task", "prior_steps", "tool_args", "user_memory", "tool", "service", "description"]
    t = d.groupby(["unit_type", "strategy"]).tv.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, order, t.loc[order, "M1_remove"].values, t.loc[order, "M2_replace"].values, "M1 remove", "M2 replace",
         lambda v: f"{v:.2f}", loc="upper right")
    ax.set_ylabel("mean total-variation change")
    ax.set_title("Phase B pilot: effect size by unit type, single-turn, all 5 models", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "fig_b_unit_types.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_b_pilot(); fig_b_unit_types()
