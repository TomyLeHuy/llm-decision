"""Plots for the Shapley table (called from analysis.py --plots).

  beeswarm_<class><suffix>.png  one row per field, one dot per call at its phi, coloured (and shaped) by the TRUE label.
                                Fields are text, so there is no feature-value colouring.
  mean_abs_phi<suffix>.png      mean |phi| per field, one panel per class.
Undefined calls (provider refusal) are left out and counted in the subtitle.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CLASSES, PLOTS
from fields import ALL_FIELDS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
LABEL_STYLE = {"direct_execute": ("#2a78d6", "o"), "ask_human": ("#eb6834", "s"), "refuse": ("#1baf7a", "^")}
CLASS_TITLE = {"direct_execute": "P(direct_execute)", "ask_human": "P(ask_human)", "refuse": "P(refuse)"}
BAR = "#2a78d6"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def swarm_offsets(x, width, step=0.09, max_off=0.4):
    """Deterministic beeswarm: dots whose x values fall in the same bin are stacked alternately above/below the row."""
    order = np.argsort(x)
    off = np.zeros(len(x))
    bins = np.floor(np.asarray(x) / width)
    count = {}
    for j in order:
        k = count.get(bins[j], 0)
        count[bins[j]] = k + 1
        off[j] = min(max_off, step * ((k + 1) // 2)) * (1 if k % 2 else -1) if k else 0.0
    return off


def _fields(df):
    return [f for f in ALL_FIELDS if f in set(df.field)]


def beeswarm(df, cls, sfx, design):
    d = df[(df["class"] == cls) & (df.status == "ok")]
    fields = _fields(d)
    n_calls = d.dataset_idx.nunique()
    n_undef = df[(df["class"] == cls) & (df.status != "ok")].dataset_idx.nunique()
    lim = max(0.05, float(np.nanmax(np.abs(d.phi)))) * 1.08
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * len(fields) + 2.3))
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for r, f in enumerate(fields):
        g = d[d.field == f]
        off = swarm_offsets(g.phi.values, width=2 * lim / 60)
        for lab, (col, mk) in LABEL_STYLE.items():
            m = (g.label == lab).values
            ax.scatter(g.phi.values[m], r + off[m], s=40, c=col, marker=mk, edgecolors=SURFACE, linewidths=0.8,
                       zorder=3, label=lab if r == 0 else None)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([f"{f}  (n={d[d.field == f].dataset_idx.nunique()})" for f in fields])
    ax.invert_yaxis()
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(f"Shapley value φ on {CLASS_TITLE[cls]}  (← lowers · raises →)")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Field Shapley values for {CLASS_TITLE[cls]} — gpt, {design}", x=0.02, ha="left", fontsize=11,
                 fontweight="bold")
    ax.set_title(f"{n_calls} calls, one dot per call, colour/shape = true label"
                 + (f"\n{n_undef} call(s) undefined (provider refusal) left out" if n_undef else ""),
                 loc="left", fontsize=9, color=INK2)
    leg = ax.legend(title="true label", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
                    fontsize=9, title_fontsize=9)
    leg.get_title().set_color(INK2)
    fig.tight_layout()
    path = PLOTS / f"beeswarm_{cls}{sfx}.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def mean_abs(df, sfx, design):
    d = df[df.status == "ok"]
    fields = _fields(d)
    m = d.assign(a=d.phi.abs()).groupby(["class", "field"]).a.mean()
    xmax = float(m.max()) * 1.18
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.45 * len(fields) + 1.7), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        vals = [m.get((cls, f), np.nan) for f in fields]
        y = np.arange(len(fields))
        ax.barh(y, vals, height=0.62, color=BAR, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.015, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(CLASS_TITLE[cls], loc="left", fontsize=10)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("mean |φ|")
    axes[0].set_yticks(range(len(fields)))
    axes[0].set_yticklabels(fields)
    axes[0].invert_yaxis()
    fig.suptitle(f"Mean |φ| per field and class — gpt, {design}, {d.dataset_idx.nunique()} calls "
                 "(mean over calls where the field is present)", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = PLOTS / f"mean_abs_phi{sfx}.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def make_all(df, sfx):
    PLOTS.mkdir(exist_ok=True)
    design = {"__fixed-tool": "tool fixed (main design)",
              "__fixed-description+tool+tool_args": "whole tool call fixed"}.get(sfx, sfx or "7 players")
    paths = [beeswarm(df, c, sfx, design) for c in CLASSES] + [mean_abs(df, sfx, design)]
    for p in paths:
        print(f"plot: {p}")
