"""Stage 10. Render the six dissertation figures from the banked CSVs.
Styled to the reference paper (Nutricati et al., QST 2025): serif STIX
typography, blue primary, red reference, white background, no grids.
Reads only data in data/ and writes PDF and PNG pairs to figures/."""

from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = str(ROOT / "figures") + "/"
DATA = str(ROOT / "data") + "/"

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

BLUE = "#3070b3"
LIGHTBLUE = "#a8c8e8"
RED = "#c02020"
GOLD = "#e0a020"
GREY = "#707070"


def save(name):
    plt.savefig(FIG + name + ".pdf", bbox_inches="tight")
    plt.savefig(FIG + name + ".png", bbox_inches="tight", dpi=300)
    plt.close()
    print(name + ".pdf / .png")


# figure 1: embedding cliffs heatmap
df = pd.read_csv(DATA + "embedding_baseline.csv")
grid = df.pivot_table(index="d", columns="n", values="success",
                      aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(6.5, 2.6))
im = ax.imshow(grid.values, aspect="auto", cmap="RdBu",
               vmin=0, vmax=100, origin="lower")
fig.colorbar(im, ax=ax, label="embedding success rate (%)")
ax.set_yticks(range(len(grid.index)), grid.index)
ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=45)
ax.set_xlabel("problem size $n$ (variables)")
ax.set_ylabel("degree $d$")
for i in range(len(grid.index)):
    for j in range(len(grid.columns)):
        v = grid.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if (v < 25 or v > 75) else "black")
save("fig1_embedding_cliffs")


# figure 2: embedding validation, paired per seed
df = pd.read_csv(DATA + "embedding_validation.csv")
p = df.pivot(index="seed", columns="method", values="longest_chain")
top = p["defaults"].max() + 3
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for seed, row in p.iterrows():
    d, t = row["defaults"], row["tuned"]
    if pd.isna(t):
        ax.plot([0, 1], [d, top], color=RED, alpha=0.85, marker="o",
                markersize=4, linestyle=":", linewidth=1.1)
        ax.text(1.04, top, "failed", fontsize=7, color=RED, va="center")
    else:
        colour = BLUE if t < d else RED
        ax.plot([0, 1], [d, t], color=colour, alpha=0.85, marker="o",
                markersize=4, linewidth=1.1)
ax.set_xticks([0, 1], ["defaults", "tuned"])
ax.set_xlim(-0.25, 1.45)
ax.set_ylabel("longest chain (10 unseen seeds)")
save("fig2_embedding_validation")


# figure 3: decomposition baseline panels
df = pd.read_csv(DATA + "decomposition_baseline.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
markers = {3: "o", 4: "s", 5: "^"}
shades = {3: LIGHTBLUE, 4: BLUE, 5: "#1a4a80"}
for ax, col, sub in [(axes[0], "ours_gap", "(a) one-pass stitch"),
                     (axes[1], "qb_gap", "(b) iterative rival")]:
    for d in sorted(df["d"].unique()):
        s = df[df["d"] == d].groupby("n")[col].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=shades[d],
                    marker=markers[d], markersize=4, capsize=2,
                    linewidth=1.1, label=f"$d$ = {d}")
    ax.set_xlabel("problem size $n$ (variables)")
    ax.text(0.5, -0.32, sub, transform=ax.transAxes, ha="center",
            fontsize=8.5)
axes[0].set_ylabel("gap to exact optimum (edges)")
axes[0].legend(frameon=True, edgecolor="black", fancybox=False)
save("fig3_decomposition_baseline")


# figure 4: joint experiment, chain-aware scoring
order = ["defaults", "cut_only", "embed_only", "joint"]
labels4 = ["defaults", "decomp.\nonly", "embed.\nonly", "joint"]

ca = pd.read_csv(DATA + "joint_chain_aware.csv")    # chain-aware run
mask = ca["arm"] == "defaults"                        # one penalty scale
ca.loc[mask, "score"] = (ca.loc[mask, "score"]
                         - 1.5 * ca.loc[mask, "max_chain"]
                         - 5.0 * ca.loc[mask, "failed"])

fig, ax = plt.subplots(figsize=(4.8, 3.2))
means = [ca[ca["arm"] == a]["score"].mean() for a in order]
colours = [LIGHTBLUE] * 4
colours[3] = GOLD
ax.bar(range(4), means, color=colours, edgecolor=BLUE, linewidth=1.0)
for i, a in enumerate(order):
    ys = ca[ca["arm"] == a]["score"]
    ax.scatter([i + 0.12] * len(ys), ys, color="black", s=6, zorder=3)
    ax.text(i, means[i] + 0.8, f"{means[i]:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(range(4), labels4, fontsize=8)
ax.set_ylabel("chain-aware score")
ax.set_ylim(ca["score"].min() - 3, ca["score"].max() + 4)
save("fig4_joint_chain_aware")

# figure 5: budget-quality curve
def quality_curve(path):
    ev = pd.read_csv(path)
    ceil = ev["cold_bo"].mean()
    pts = {0: 100 * ev["defaults"].mean() / ceil}
    guess = ev[["feature_pred", "gnn_pred"]].max(axis=1)
    pts[1] = 100 * guess.mean() / ceil
    for b in (6, 11):
        col = f"warm_bo_{b}"
        if col in ev:
            pts[b] = 100 * ev[col].mean() / ceil
    pts[20] = 100.0
    return pts


fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.axhline(100, color=RED, linewidth=0.9, zorder=1)
ax.text(0.3, 100.7, "cold search", color=RED, fontsize=8, ha="left")
for path, label, marker, ls, colour in [
        (DATA + "pipeline_evaluation_scale.csv",
         "at scale ($n$ = 100 to 300, realistic solver)", "o", "-", BLUE),
        (DATA + "pipeline_evaluation_smallscale.csv",
         "small scale ($n$ = 20 to 60)", "s", "--", GREY)]:
    pts = quality_curve(path)
    xs = sorted(pts)
    ys = [pts[x] for x in xs]
    ax.plot(xs, ys, marker=marker, linestyle=ls, color=colour,
            linewidth=1.2, markersize=5,
            markerfacecolor="white" if marker == "s" else colour,
            label=label, zorder=3)
ax.set_xlabel("pipeline evaluations spent selecting the configuration")
ax.set_ylabel("share of cold-search quality (%)")
ax.set_xticks([0, 1, 6, 11, 20])
ax.set_ylim(80, 104)
ax.legend(loc="lower right", frameon=True, edgecolor="black",
          fancybox=False)
save("fig5_budget_quality")

# figure 6: scale bars with wall-clock
ev = pd.read_csv(DATA + "pipeline_evaluation_scale.csv")
strategies = ["defaults", "feature_pred", "gnn_pred",
              "warm_bo_6", "warm_bo_11", "cold_bo"]
labels6 = ["defaults", "feature\nguess", "GNN\nguess",
           "guess +\npolish (6)", "guess +\npolish (11)",
           "full search\n(20)"]
means = [ev[s].mean() for s in strategies]
times = [ev[s + "_t"].mean() for s in strategies]
colours = [LIGHTBLUE] * 6
colours[3] = GOLD
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(range(6), means, color=colours, edgecolor=BLUE, linewidth=1.0)
for i, (m, t) in enumerate(zip(means, times)):
    ax.text(i, m + 4, f"{m:.0f}", ha="center", fontsize=8.5)
    ax.text(i, m * 0.5, f"{t:.0f} s", ha="center", fontsize=8)
ax.set_xticks(range(6), labels6, fontsize=8)
ax.set_ylabel("satisfied interactions (mean)")
ax.set_ylim(0, max(means) * 1.14)
save("fig6_scale_headline")

print()
print("6 figures written to figures/ as PDF and PNG")
