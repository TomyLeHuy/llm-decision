"""Figures 1 to 7 of the paper. Usage (from the repository root): python figures/make_figs.py results figures
One style for all figures: Inter, 7.5 pt at print size, figures drawn at 15.5 cm width (1:1 in the DOCX),
one fixed colour and marker per decision model in every figure, recessive grid, direct labels.
Data: raw scores (cp/full/raw_full.csv) and the saved experiment outputs (results*.json, per-split CSVs).
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FULL = sys.argv[1]; OUT = sys.argv[2]   # results directory, output directory
W = 15.5 / 2.54  # inches, full text width used in the DOCX

# ---------- style
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
C = {"mean": "#2a78d6", "microsoft": "#eb6834", "gpt": "#1baf7a", "jev": "#6250d6"}
MK = {"mean": "o", "microsoft": "s", "gpt": "^", "jev": "D"}
LAB = {"mean": "mean of three", "microsoft": "microsoft", "gpt": "gpt", "jev": "jev"}
STATUS = {"execute": "#0ca30c", "ask": "#fab219", "refuse": "#d03b3b"}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 600, "lines.linewidth": 1.6,
    "legend.frameon": False, "mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold", "axes.titlelocation": "left", "axes.titleweight": "semibold", "axes.titlepad": 6,
})
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=600, facecolor=SURF)
    plt.close(fig); print("saved", name)
def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", color=INK)
def pct(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)
def budget_v(ax, x, text, y=None, ha="left"):
    ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    if text: ax.text(x + (0.4 if ha == "left" else -0.4), ax.get_ylim()[1] if y is None else y, text, color=INK2, fontsize=6.5, va="top", ha=ha)
def budget_h(ax, y, text, x=None):
    ax.axhline(y, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    if text: ax.text(ax.get_xlim()[1] if x is None else x, y + 0.4, text, color=INK2, fontsize=6.5, va="bottom", ha="right")

# ---------- data
ORDER = ["direct_execute", "ask_human", "refuse"]; DE, AH, RF = 0, 1, 2; PC = [f"p_{c}" for c in ORDER]
d = pd.read_csv(os.path.join(FULL, "scores_test_1436.csv")); d = d[d.error.isna()].copy(); d["short"] = d.model.str.split("/").str[-1]
common = set.intersection(*[set(g.dataset_idx) for _, g in d.groupby("short")]); d = d[d.dataset_idx.isin(common)]
meta = d.drop_duplicates("dataset_idx").set_index("dataset_idx").sort_index(); ids = meta.index.to_numpy()
y = meta.ground_truth.map({c: i for i, c in enumerate(ORDER)}).to_numpy(); src = meta.service.to_numpy()
def mat(m):
    g = d[d.short == m].set_index("dataset_idx").loc[ids, PC].to_numpy(float); g = np.clip(g, 0, None); return g / g.sum(1, keepdims=True)
def quant(P, grid=0.01):
    Q = np.round(P / grid) * grid; return Q / Q.sum(1, keepdims=True)
M = {"jev": mat("jev-1.14"), "microsoft": mat("microsoft-decision-1"), "gpt": mat("gpt-6-luna-decisions")}
M["mean"] = np.mean([M["jev"], M["microsoft"], M["gpt"]], 0)
M["microsoft_r"] = quant(M["microsoft"])
R1 = json.load(open(os.path.join(FULL, "r02_core.json"))); R2 = json.load(open(os.path.join(FULL, "r03_policies_groups.json"))); R3 = json.load(open(os.path.join(FULL, "r04_calibration_size.json")))
KEY = {"ensemble (mean of 3)": "mean", "microsoft-decision-1": "microsoft", "gpt-6-luna-decisions": "gpt", "jev-1.14": "jev",
       "microsoft, rounded to 0.01": "microsoft_r"}
facts = {}

# =====================================================================================
# Figure 1: architecture
def fig_arch():
    fig = plt.figure(figsize=(W, 2.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#f6f6f4", ec="#c9c8c2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.2, weight="semibold", color=INK)
        if body: ax.text(x + w / 2, y + h - 5.9, body, ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.4)
    def tag(x, y, text):
        ax.text(x, y, text, ha="right", va="center", fontsize=5.8, weight="semibold", color="white",
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc=C["mean"], ec="none"), zorder=6)
    def arrow(x0, y0, x1, y1, col=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, shrinkA=0, shrinkB=0, zorder=4))
    for yy, h, lab in [(25.0, 22.2, "at run time"), (0.8, 22.2, "before deployment")]:
        ax.add_patch(FancyBboxPatch((0.5, yy), 99.0, h, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=GRID, lw=0.8))
        ax.text(2.3, yy + h / 2, lab, rotation=90, fontsize=6.5, color=MUTED, va="center", ha="center", style="italic")
    # run time
    box(5.0, 27.5, 13.5, 16.5, "Proposed tool call", "task, service,\ncall and\narguments")
    box(21.5, 27.5, 17.0, 16.5, "Decision models", "")
    for k, (m, yy) in enumerate([("jev", 36.4), ("microsoft", 32.4), ("gpt", 28.4)]):
        ax.add_patch(FancyBboxPatch((23.0, yy), 14.0, 3.2, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=C[m], lw=0.9, zorder=3))
        ax.text(30.0, yy + 1.6, f"vendor {k + 1}: {m}", ha="center", va="center", fontsize=6.0, color=INK, zorder=4)
    box(41.5, 27.5, 13.5, 16.5, "Combine", "average of\nthe vendors'\nvectors")
    box(58.0, 27.5, 18.5, 16.5, "Delegation gate", "execute if $p_{refuse} \\leq \\lambda_r$\nand $p_{ask} \\leq \\lambda_a$; block\nif $p_{refuse} \\geq \\tau$; else ask")
    arrow(18.5, 35.75, 21.5, 35.75); arrow(38.5, 35.75, 41.5, 35.75); arrow(55.0, 35.75, 58.0, 35.75)
    for k, (lab, col) in enumerate([("execute autonomously", STATUS["execute"]), ("ask a human", "#b07d00"), ("block the call", STATUS["refuse"])]):
        yy = 40.2 - k * 4.45
        ax.add_patch(FancyBboxPatch((80.0, yy - 1.65), 18.0, 3.3, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=col, lw=1.0))
        ax.text(89.0, yy, lab, ha="center", va="center", fontsize=6.3, color=INK)
        arrow(76.5, 35.75, 80.0, yy)
    # before deployment
    box(5.0, 3.0, 15.5, 16.5, "Labelled past calls", "a label per call\nand target\nservice")
    box(24.0, 3.0, 18.5, 16.5, "Certifiability check", "critical calls at the\nlowest expressible\nprobabilities, per class")
    box(46.0, 3.0, 26.0, 16.5, "Calibration and certification", "per class and service; binomial\ntests (Learn-then-Test, $1-\\delta$);\ncheapest certified policy")
    box(75.5, 3.0, 22.5, 16.5, "Budgets and costs", "harm and confirmation\nbudgets, confidence; costs\nof harm, review, blocking")
    arrow(20.5, 11.25, 24.0, 11.25); arrow(42.5, 11.25, 46.0, 11.25); arrow(75.5, 11.25, 72.0, 11.25)
    arrow(33.25, 19.5, 30.0, 27.5, col=INK2); ax.text(31.0, 23.5, "admit vendor?", fontsize=6.0, color=INK2, ha="right", va="center")
    arrow(59.0, 19.5, 67.25, 27.5, col=C["mean"], lw=1.0); ax.text(64.6, 23.5, "thresholds $\\lambda_r, \\lambda_a, \\tau$", fontsize=6.0, color=C["mean"], ha="left", va="center")
    for x, yy, t in [(38.2, 44.0, "DP3"), (54.7, 44.0, "DP3"), (76.2, 44.0, "DP5"), (20.2, 19.5, "DP4"), (42.2, 19.5, "DP2"), (71.7, 19.5, "DP1  DP4  DP5")]:
        tag(x, yy, t)
    save(fig, "fig2_architecture.png")

# =====================================================================================
# Figure 2: the certifiability floor
def fig_floor():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw=dict(wspace=0.32))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.19)
    series = [("microsoft_r", "microsoft rounded to 0.01"), ("jev", "jev"), ("gpt", "gpt"), ("microsoft", "microsoft"), ("mean", "mean of three")]
    def sty(k):
        base = "microsoft" if k == "microsoft_r" else k
        return dict(color=C[base], ls=(0, (3, 1.5)) if k == "microsoft_r" else "-", marker=MK[base],
                    mfc="white" if k == "microsoft_r" else C[base], mec=C[base])
    # (a) ECDF of p_refuse on true refusals
    ax = axs[0]; xs = np.concatenate([[0], np.logspace(-4, np.log10(0.05), 300)])
    for k, lab in series:
        p = M[k][y == RF, RF]; e = np.array([np.mean(p <= x) for x in xs]) * 100; s = sty(k)
        ax.step(xs, e, where="post", color=s["color"], ls=s["ls"], lw=1.4)
        ax.plot([0], [e[0]], marker=s["marker"], ms=4.5, mfc=s["mfc"], mec=s["mec"], mew=1.0, ls="none", zorder=5)
        facts[f"zero_{k}"] = float(e[0])
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.35); ax.set_xlim(-0.00002, 0.05)
    ax.set_xticks([0, 1e-4, 1e-3, 1e-2, 0.05]); ax.set_xticklabels(["0", "0.0001", "0.001", "0.01", "0.05"])
    ax.minorticks_off(); ax.set_ylim(0, 35); pct(ax)
    budget_h(ax, 5, "budget 5%", x=0.048)
    ax.set_xlabel("probability of refuse returned by the model"); ax.set_ylabel("true refusals at or below")
    panel(ax, "a", "Critical calls scored as almost surely safe")
    ax.annotate("17.0% exactly 0", xy=(0, 17.0), xytext=(0.00006, 26), fontsize=6.3, color=INK2, arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax.annotate("7.9%", xy=(0, 7.9), xytext=(0.00006, 11.5), fontsize=6.3, color=INK2, arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    # (b) safe autonomy vs harm budget (G2), from E9
    ax = axs[1]; E9 = pd.DataFrame(R1["E9"]); E9["k"] = E9.model.map(KEY)
    for k, lab in series:
        g = E9[E9.k == k].sort_values("alpha_rf"); s = sty(k)
        ax.plot(g.alpha_rf * 100, g.de_exec * 100, color=s["color"], ls=s["ls"], lw=1.4, marker=s["marker"], ms=3.6, mfc=s["mfc"], mec=s["mec"], mew=0.9)
    ax.set_xlim(1.5, 20.5); ax.set_ylim(-2, 85); pct(ax); pct(ax, "x"); ax.set_xticks([2, 5, 7.5, 10, 15, 20]); ax.set_xticklabels(["2%", "5%", "7.5%", "10%", "15%", "20%"])
    budget_v(ax, 5, "")
    ax.set_xlabel("budget for executing a call that should be blocked"); ax.set_ylabel("safe calls executed without a human")
    panel(ax, "b", "Autonomy starts above the floor (gate G2)")
    hand = [Line2D([], [], color=sty(k)["color"], ls=sty(k)["ls"], marker=sty(k)["marker"], ms=4, mfc=sty(k)["mfc"], mec=sty(k)["mec"], lw=1.4, label=lab) for k, lab in series[::-1]]
    fig.legend(handles=hand, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), handlelength=2.4, columnspacing=1.6)
    save(fig, "fig3_floor.png")

# =====================================================================================
# Figure 3: vendor complementarity and the shared blind spot
def fig_vendors():
    fig = plt.figure(figsize=(W, 2.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 0.95, 0.30, 1.05], wspace=0.06, left=0.03, right=0.985, top=0.85, bottom=0.19)
    vend = ["jev", "microsoft", "gpt"]; low = {m: M[m][y == RF, RF] <= 0.01 for m in vend}; n = int((y == RF).sum())
    combos = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    cnt = []
    for c in combos:
        m = np.ones(n, bool)
        for v, on in zip(vend, c): m &= low[v] if on else ~low[v]
        cnt.append(int(m.sum()))
    facts["upset"] = dict(zip([str(c) for c in combos], cnt)); facts["any_low"] = int(np.sum(low["jev"] | low["microsoft"] | low["gpt"]))
    facts["mean_low"] = int(np.sum(M["mean"][y == RF, RF] <= 0.01)); facts["n_ref"] = n
    # dot matrix
    axm = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=axm)
    ypos = np.arange(len(combos))[::-1]
    for i, c in enumerate(combos):
        for j, on in enumerate(c):
            axm.plot(j, ypos[i], "o", ms=5.2, color=C[vend[j]] if on else GRID, zorder=3)
        on_idx = [j for j, o in enumerate(c) if o]
        if len(on_idx) > 1: axm.plot([min(on_idx), max(on_idx)], [ypos[i]] * 2, color=INK2, lw=1.0, zorder=2)
    axm.set_xlim(-0.6, 2.6); axm.set_xticks(range(3)); axm.set_xticklabels(vend, rotation=0); axm.tick_params(axis="x", length=0, pad=4)
    axm.set_yticks([]); axm.grid(False); [axm.spines[s].set_visible(False) for s in ["left", "bottom"]]
    axm.set_title("a   Which vendors give p(refuse) ≤ 0.01", color=INK)
    axb.barh(ypos, np.array(cnt) / n * 100, height=0.62, color=[C["mean"] if sum(c) == 3 else "#b9b8b2" for c in combos], zorder=3)
    for yy, v in zip(ypos, cnt): axb.text(v / n * 100 + 0.25, yy, f"{v} ({v / n * 100:.1f}%)", va="center", fontsize=6.5, color=INK2)
    axb.set_xlim(0, 12.5); pct(axb, "x"); axb.tick_params(axis="y", left=False, labelleft=False); axb.grid(axis="y", visible=False)
    axb.set_xlabel(f"share of the {n} true refusals"); axb.spines["left"].set_visible(False)
    axb.set_title(" ", color=INK)
    # (b) blind spot on dh_app: p_refuse of refusals versus safe calls, per vendor and mean
    ax = fig.add_subplot(gs[0, 3])
    cols = ["jev", "microsoft", "gpt", "mean"]; rng = np.random.default_rng(3)
    for j, m in enumerate(cols):
        for lab, cls, dx, filled in [("refuse", RF, -0.17, True), ("execute", DE, 0.17, False)]:
            sel = (src == "dh_app") & (y == cls); p = M[m][sel, RF]
            xx = j + dx + rng.uniform(-0.06, 0.06, sel.sum())
            ax.plot(xx, np.maximum(p, 1e-4), MK[m], ms=3.6, mfc=C[m] if filled else "white", mec=C[m], mew=0.8, ls="none", alpha=0.9)
        facts[f"dh_{m}"] = dict(ref_med=float(np.median(M[m][(src == "dh_app") & (y == RF), RF])), safe_med=float(np.median(M[m][(src == "dh_app") & (y == DE), RF])))
    ax.set_yscale("log"); ax.set_ylim(7e-5, 1.5); ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["≤0.0001", "0.001", "0.01", "0.1", "1"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["jev", "microsoft", "gpt", "mean"]); ax.grid(axis="x", visible=False); ax.minorticks_off()
    ax.set_ylabel("probability of refuse")
    ax.set_title("b   Shared blind spot on one service", color=INK)
    hand = [Line2D([], [], marker="o", ls="none", ms=4, mfc=INK2, mec=INK2, label="true refusal (18)"),
            Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=INK2, label="true execute (8)")]
    ax.legend(handles=hand, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig4_vendors.png")

# =====================================================================================
# Figure 4: service-conditional calibration (dumbbells)
def fig_services():
    E10 = pd.DataFrame(R2["E10"]); E10["k"] = E10.model.map(KEY)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 0.92]))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.775, bottom=0.19)
    groups = [("atbench", "atbench (54)"), ("agentharm", "agentharm (49)"), ("tsbench", "tsbench (124)"), ("other", "other services (178)")]
    scorers = ["microsoft", "gpt", "mean"]; off = {"microsoft": 0.22, "gpt": 0.0, "mean": -0.22}
    for ax, pref, xmax, title, xl in [(axs[0], "cat_", 19, "a   Harmful execution per service", "calls that should be blocked but run"),
                                       (axs[1], "auto_", 80, "b   Safe autonomy per service", "safe calls executed without a human")]:
        gl = groups if pref == "cat_" else [g for g in groups if g[0] != "tsbench"]
        for gi, (g, glab) in enumerate(gl):
            base = len(gl) - 1 - gi
            for s in scorers:
                a = E10[(E10.k == s) & (E10.variant == "class-conditional")][pref + g].iloc[0] * 100
                b = E10[(E10.k == s) & (E10.variant == "class x source")][pref + g].iloc[0] * 100
                yy = base + off[s]
                ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-", color=C[s], lw=1.1, alpha=0.55))
                ax.plot(a, yy, MK[s], ms=4.2, mfc="white", mec=C[s], mew=0.9, zorder=4)
                ax.plot(b, yy, MK[s], ms=4.2, mfc=C[s], mec=C[s], zorder=5)
        ax.set_yticks(range(len(gl))); ax.set_yticklabels([gl[len(gl) - 1 - i][1] for i in range(len(gl))])
        ax.set_ylim(-0.6, len(gl) - 0.4); ax.grid(axis="y", visible=False); ax.set_xlim(0 if pref == "cat_" else 0, xmax); pct(ax, "x")
        ax.set_xlabel(xl); panel(ax, "", title.split("   ")[1]); ax.set_title(title, color=INK)
        if pref == "cat_": budget_v(ax, 5, "budget 5%", y=len(gl) - 0.45)
    axs[1].set_yticklabels([lab.split(" (")[0] if lab.startswith("all") is False else "all other" for lab in [t.get_text() for t in axs[1].get_yticklabels()]])
    hand = [Line2D([], [], marker=MK[s], ls="none", ms=4.5, mfc=C[s], mec=C[s], label=LAB[s]) for s in scorers] + \
           [Line2D([], [], marker="o", ls="none", ms=4.5, mfc="white", mec=INK2, label="one refusal threshold (G2)"),
            Line2D([], [], marker="o", ls="none", ms=4.5, mfc=INK2, mec=INK2, label="threshold per service (G3)")]
    fig.legend(handles=hand, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.0), handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig5_services.png")

# =====================================================================================
# Figure 5: per-split distributions, average versus confidence
def fig_splits():
    e35 = pd.read_csv(os.path.join(FULL, "r02_core_E35.csv")); e35["k"] = e35.model.map(KEY)
    e6 = pd.read_csv(os.path.join(FULL, "r03_policies_groups_E6b.csv")); e6["k"] = e6.model.map(KEY); e6 = e6[(e6.harm == 10) & (e6.found)]
    def pick(k, kind):
        if kind == "G1": return e35[(e35.k == k) & (e35.rule == "conservative")]
        if kind == "G2": return e35[(e35.k == k) & (e35.rule == "execute if refuse excluded")]
        return e6[(e6.k == k) & (e6["mode"] == ("tuned" if kind == "tuned" else "certified"))]
    NAMES = {"G1": "G1 prediction sets", "G2": "G2 refuse excluded", "tuned": "tuned, no test", "G4": "G4 certified"}
    rows = [(k, NAMES[g], pick(k, g)) for k, gl in [("microsoft", ["G1", "G2", "G4"]), ("mean", ["G1", "G2", "tuned", "G4"]), ("jev", ["tuned", "G4"])] for g in gl]
    starts = {0: "microsoft", 3: "mean", 7: "jev"}
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.05), sharey=True, gridspec_kw=dict(wspace=0.42))
    fig.subplots_adjust(left=0.19, right=0.90, top=0.875, bottom=0.185)
    rng = np.random.default_rng(7); n = len(rows)
    gap = {i: (1.0 if i < 3 else 0.5 if i < 7 else 0.0) for i in range(n)}; ypos = {i: (n - 1 - i) + gap[i] for i in range(n)}
    for ax, col, bud, xmax, title in [(axs[0], "rf_exec", 5, 10, "a   Harmful execution in each test split"),
                                      (axs[1], "ah_exec", 15, 35, "b   Unconfirmed execution in each test split")]:
        for i, (s, gname, g) in enumerate(rows):
            v = g[col].to_numpy() * 100; yy = ypos[i] + rng.uniform(-0.2, 0.2, len(v))
            over = v > bud + 1e-9
            ax.plot(v[~over], yy[~over], MK[s], ms=2.4, mfc=C[s], mec="none", alpha=0.55, ls="none")
            ax.plot(v[over], yy[over], MK[s], ms=2.6, mfc="white", mec=C[s], mew=0.6, alpha=0.9, ls="none")
            ax.plot([v.mean()] * 2, [ypos[i] - 0.33, ypos[i] + 0.33], color=INK, lw=1.3, zorder=5)
            bounded = not (col == "ah_exec" and gname.startswith("G2"))
            txt = f"{over.mean() * 100:.0f}%" if bounded else "n/a"
            ax.text(xmax * 1.01, ypos[i], txt, va="center", fontsize=6.5, color=STATUS["refuse"] if (bounded and over.mean() > 0.1) else INK2, clip_on=False)
            facts[f"split_{s}_{gname}_{col}"] = dict(mean=float(v.mean()), over=float(over.mean()), n=int(len(v)))
        ax.set_xlim(0, xmax); pct(ax, "x"); ax.grid(axis="y", visible=False)
        ax.axvline(bud, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        top = max(ypos.values()) + 0.55
        ax.text(bud, top, f"budget {bud}%", ha="center", va="bottom", fontsize=6.5, color=INK2, bbox=dict(fc="white", ec="none", pad=0.6))
        ax.text(xmax * 1.01, top, "above", ha="left", va="bottom", fontsize=6.3, color=INK2, clip_on=False)
        ax.set_title(title, color=INK, pad=10); ax.set_xlabel("observed rate on the test half", labelpad=2)
    axs[0].set_yticks([ypos[i] for i in range(n)]); axs[0].set_yticklabels([r[1] for r in rows]); axs[0].set_ylim(-0.7, max(ypos.values()) + 1.0)
    for i0, s in starts.items():
        axs[0].text(-0.44, ypos[i0] + 0.48, LAB[s], transform=axs[0].get_yaxis_transform(), fontsize=7, weight="semibold", color=C[s], ha="left", va="bottom")
    hand = [Line2D([], [], marker="o", ls="none", ms=3.5, mfc=INK2, mec="none", label="one split within budget"),
            Line2D([], [], marker="o", ls="none", ms=3.5, mfc="white", mec=INK2, label="one split above budget"),
            Line2D([], [], color=INK, lw=1.3, label="mean over splits")]
    fig.legend(handles=hand, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0), handletextpad=0.4, columnspacing=1.6)
    save(fig, "fig6_splits.png")

# =====================================================================================
# Figure 6: confidence, budget and calibration size
def fig_frontier():
    import math
    R5 = json.load(open(os.path.join(FULL, "r06_revision.json")))
    RAMP = {0.20: "#86b6ef", 0.10: "#3987e5", 0.05: "#1c5cab", 0.01: "#0d366b"}
    CONF = {0.20: "80%", 0.10: "90%", 0.05: "95%", 0.01: "99%"}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw=dict(wspace=0.34))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.76, bottom=0.18)
    ax = axs[0]; al = np.linspace(0.01, 0.15, 200)
    for dl, col in RAMP.items():
        nmin = np.log(dl / 9) / np.log(1 - al); ax.plot(al * 100, nmin, color=col, lw=1.6)
        v = math.ceil(math.log(dl / 9) / math.log(0.95)); ax.plot(5, v, "o", ms=3.8, color=col, zorder=5)
        facts[f"nmin_5_{dl}"] = v
    for yv, lab, xl, ha in [(405, "all refusals in the corpus (405)", 15, "right"), (202, "half of them (202)", 15, "right"), (54, "atbench, all its calls (54)", 1.2, "left")]:
        ax.axhline(yv, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.text(xl, yv * (1.07 if yv > 100 else 0.93), lab, ha=ha, va="bottom" if yv > 100 else "top", fontsize=6.3, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(20, 1200); ax.set_xlim(1, 15); pct(ax, "x")
    ax.set_yticks([25, 50, 100, 200, 400, 800]); ax.set_yticklabels(["25", "50", "100", "200", "400", "800"]); ax.minorticks_off()
    ax.set_xlabel("budget for harmful execution"); ax.set_ylabel("labelled refusals needed")
    ax.set_title("a   Minimum calibration size for G4", color=INK)
    ax.text(4.55, 66, "75", fontsize=6.2, color=INK2, va="center", ha="right"); ax.text(5.4, 140, "133", fontsize=6.2, color=INK2, va="center")
    ax = axs[1]; R4 = pd.DataFrame(R5["R4"]); R4 = R4[R4.model == "mean"]
    for dl, col in RAMP.items():
        g = R4[R4.delta == dl].sort_values("frac"); ok = g[g.certifiable > 0]; no = g[g.certifiable == 0]
        ax.plot(ok.n_cal_rf, ok.safe * 100, color=col, marker="o", ms=3.8, lw=1.6)
        ax.plot(no.n_cal_rf, np.zeros(len(no)), "x", color=col, ms=4.5, mew=1.2, ls="none", clip_on=False, zorder=5)
    ax.text(108, 2.2, "not certifiable at 95% and 99%", fontsize=6.3, color=INK2, va="bottom")
    ax.set_xlim(85, 380); ax.set_ylim(0, 42); pct(ax); ax.set_xticks([101, 202, 304, 364])
    ax.set_xlabel("refusals in the calibration data"); ax.set_ylabel("safe calls executed (G4)")
    ax.set_title("b   Certified autonomy of the mean of three", color=INK)
    hand = [Line2D([], [], color=RAMP[dl], lw=1.6, marker="o", ms=3.8, label=f"{CONF[dl]} confidence") for dl in [0.20, 0.10, 0.05, 0.01]]
    fig.legend(handles=hand, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), columnspacing=1.6)
    save(fig, "fig7_frontier.png")

# =====================================================================================
# Figure 0 (opening figure): what goes in, what comes out
def fig_teaser():
    fig = plt.figure(figsize=(W, 2.35)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 38.5); ax.axis("off")
    def rbox(x, y, w, h, ec="#c9c8c2", fc="#f6f6f4", lw=0.7, r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    def arrow(x0, y0, x1, y1, col=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, shrinkA=0, shrinkB=0, zorder=4))
    def num(x, y, k, col):
        ax.text(x, y, str(k), ha="center", va="center", fontsize=6.0, weight="semibold", color="white", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="none"))
    ax.text(1.5, 37.4, "In: proposed tool calls", fontsize=7.2, weight="semibold", color=INK, va="top")
    ax.text(73.5, 37.4, "Out: one action per call", fontsize=7.2, weight="semibold", color=INK, va="top")
    calls = [("ticketing", "close a resolved ticket and\nnotify the customer"),
             ("database", "update a customer record;\na required field is missing"),
             ("password vault", "export all entries to an\nexternal e-mail address")]
    outs = [("Execute autonomously", STATUS["execute"], "15% of all calls"), ("Ask a human", "#b07d00", "71% of all calls"), ("Block the call", STATUS["refuse"], "14% of all calls")]
    H = 8.6; ys = [25.0, 15.6, 6.2]; mid = 6.2 + (25.0 + H - 6.2) / 2; top = 25.0 + H
    for k, ((svc, txt), y) in enumerate(zip(calls, ys)):
        rbox(1.5, y, 24.0, H, fc="white")
        ax.text(5.6, y + H - 1.4, svc, fontsize=6.4, weight="semibold", color=INK, va="top")
        ax.text(5.6, y + H - 3.7, txt, fontsize=6.0, color=INK2, va="top", linespacing=1.2)
        num(3.4, y + H - 2.3, k + 1, INK2)
        arrow(25.5, y + H / 2, 30.0, mid)
    rbox(30.0, 6.2, 15.5, top - 6.2)
    ax.text(37.75, top - 1.8, "Decision models", ha="center", va="top", fontsize=7.0, weight="semibold")
    ax.text(37.75, top - 5.6, "three vendors\neach return\np(execute)\np(ask)\np(refuse)", ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.35)
    arrow(45.5, mid, 49.0, mid)
    rbox(49.0, 6.2, 21.0, top - 6.2, ec=C["mean"], fc="#eef4fc", lw=1.0)
    ax.text(59.5, top - 1.8, "Certified delegation gate", ha="center", va="top", fontsize=7.0, weight="semibold")
    ax.text(59.5, top - 5.6, "executes a call only if its\naveraged p(refuse) and p(ask)\nlie below thresholds certified\non labelled past calls", ha="center", va="top", fontsize=6.1, color=INK2, linespacing=1.35)
    rbox(51.0, 7.8, 17.0, 6.8, ec=C["mean"], fc="white", lw=0.8, r=0.9)
    ax.text(59.5, 11.2, "at most 5% of calls that\nshould be blocked run,\nwith 90% confidence", ha="center", va="center", fontsize=5.9, color=C["mean"], weight="semibold", linespacing=1.25)
    for k, ((lab, col, share), y) in enumerate(zip(outs, ys)):
        rbox(73.5, y + 0.6, 25.0, H - 1.2, ec=col, fc="white", lw=1.0)
        ax.text(78.4, y + H / 2 + 1.1, lab, fontsize=6.6, color=INK, va="center", weight="semibold")
        ax.text(78.4, y + H / 2 - 1.6, share, fontsize=6.0, color=INK2, va="center")
        num(75.9, y + H / 2, k + 1, col)
        arrow(70.0, mid, 73.5, y + H / 2)
    ax.add_patch(FancyBboxPatch((1.5, 0.5), 97.0, 4.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#f6f6f4", ec="none"))
    ax.text(3.0, 2.7, "Set by the organisation:  harm budget 5%  ·  confirmation budget 15%  ·  confidence 90%  ·  cost of one harmful execution = 10 reviews",
            fontsize=6.1, color=INK2, va="center")
    save(fig, "fig1_overview.png")

for f in [fig_teaser, fig_arch, fig_floor, fig_vendors, fig_services, fig_splits, fig_frontier]: f()
json.dump(facts, open(os.path.join(OUT, "fig_facts.json"), "w"), indent=1, default=float); print(json.dumps(facts, indent=0, default=float)[:3000])
