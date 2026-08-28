"""Stage 9. Protein folding case study: a real HP-lattice instance
through the finished system. A 20-residue benchmark chain is placed on a
2D lattice and encoded as an interaction graph, where variables are
residue-position choices and edges connect interacting choices. The
graph is naturally clustered, the structure this pipeline targets. Runs
the same five strategies as the scale evaluation. Writes
data/protein_case_study.csv."""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import torch
import dwave_networkx as dnx
from skopt import Optimizer
from skopt.space import Integer
from decomposition import full_pipeline_real
from predictors import (to_data, from01, SettingsGNN, SETTINGS,
                        tr_df, mu, sd, FEATS)

DATA = Path(__file__).resolve().parent.parent / "data"

SEQUENCE = "HPHPPHHPHPPHPHHPPHPH"      # 20-residue HP benchmark chain
GRID = 5                               # 5x5 lattice
TIMEOUT = 2
COLD_BUDGET, WARM_BUDGETS = 20, [6, 11]


def build_protein_graph(seq, grid):
    """HP-lattice interaction graph. A variable = (residue i, cell c):
    'residue i sits at cell c'. Edges couple decisions that interact:
    consecutive residues at adjacent cells (chain connectivity) and
    H-H pairs at adjacent cells (folding energy). Cells limited to a
    band around the diagonal to keep the instance ~200 variables."""
    n = len(seq)
    cells = [(r, c) for r in range(grid) for c in range(grid)]

    def near_diag(i, cell):
        r, c = cell
        return abs((r * grid + c) - (i * grid * grid) // n) <= grid + 2

    variables = [(i, cell) for i in range(n) for cell in cells
                 if near_diag(i, cell)]
    idx = {v: k for k, v in enumerate(variables)}
    g = nx.Graph()
    g.add_nodes_from(range(len(variables)))

    def adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    for (i, ci), ki in idx.items():
        for (j, cj), kj in idx.items():
            if kj <= ki:
                continue
            if j == i + 1 and adjacent(ci, cj):
                g.add_edge(ki, kj)                 # chain link
            elif seq[i] == "H" and seq[j] == "H" \
                    and j > i + 1 and adjacent(ci, cj):
                g.add_edge(ki, kj)                 # H-H contact
    g.remove_nodes_from(list(nx.isolates(g)))
    return nx.convert_node_labels_to_integers(g)


problem = build_protein_graph(SEQUENCE, GRID)
n, e = problem.number_of_nodes(), problem.number_of_edges()
feats = pd.Series({"n": n, "density": nx.density(problem),
                   "clustering": nx.average_clustering(problem)})
print(f"protein instance: {n} variables, {e} interactions, "
      f"clustering {feats['clustering']:.3f}")

hardware = dnx.chimera_graph(16)
model = SettingsGNN()
model.load_state_dict(torch.load(DATA / "gnn_model.pt"))
model.eval()

space = [Integer(10, 28, name="piece_size"),
         Integer(0, 2, name="overlap"),
         Integer(1, 100, name="tries"),
         Integer(1, 100, name="max_no_improvement"),
         Integer(1, 100, name="chainlength_patience")]


def run_settings(k, seed=0):
    cut_k = {"piece_size": int(np.clip(k["piece_size"], 6, 28)),
             "overlap": int(np.clip(k["overlap"], 0, 2)),
             "order_seed": None, "ordering": "spectral"}
    embed_k = {nm: int(np.clip(k[nm], 1, 100)) for nm in
               ("tries", "max_no_improvement", "chainlength_patience")}
    score, _ = full_pipeline_real(problem, hardware, cut_k, embed_k,
                                  embed_seed=seed, timeout=TIMEOUT)
    return score


def neighbour_settings(k=2):
    z = ((tr_df[FEATS] - mu) / sd
         - ((feats[FEATS] - mu) / sd)).pow(2).sum(axis=1)
    nearest = tr_df.loc[z.astype(float).nsmallest(k).index]
    return [dict(r[SETTINGS].astype(int)) for _, r in nearest.iterrows()]


def rival_from_feats(k=5):
    z = ((tr_df[FEATS] - mu) / sd
         - ((feats[FEATS] - mu) / sd)).pow(2).sum(axis=1)
    nearest = tr_df.loc[z.astype(float).nsmallest(k).index]
    return {s: int(round(nearest[s].mean())) for s in SETTINGS}


def warm_bo(budget, tells):
    opt = Optimizer(space, random_state=0)
    best, used = -np.inf, 0
    for t in tells[:budget]:
        s = run_settings(t, seed=used)
        opt.tell([int(np.clip(t[d.name], d.low, d.high)) for d in space], -s)
        best, used = max(best, s), used + 1
    for attempt in range(used, budget):
        x = opt.ask()
        k = {d.name: int(v) for d, v in zip(space, x)}
        if k["overlap"] >= k["piece_size"]:
            opt.tell(x, 1000.0)
            continue
        s = run_settings(k, seed=attempt)
        opt.tell(x, -s)
        best = max(best, s)
    return best


DEFAULTS = {"piece_size": 20, "overlap": 1, "tries": 10,
            "max_no_improvement": 10, "chainlength_patience": 10}

gnn_guess = from01(model(to_data(problem))[0].detach().numpy())
tells = [gnn_guess] + neighbour_settings(k=2)
print("GNN predicted settings:", gnn_guess)

results = {"defaults": run_settings(DEFAULTS),
           "cold_bo": warm_bo(COLD_BUDGET, tells=[]),
           "feature_pred": run_settings(rival_from_feats()),
           "gnn_pred": run_settings(gnn_guess)}
for b in WARM_BUDGETS:
    results[f"warm_bo_{b}"] = warm_bo(b, tells=tells)

df = pd.DataFrame([{"strategy": k, "satisfied_interactions": v,
                    "of_total": e} for k, v in results.items()])
df.to_csv(DATA / "protein_case_study.csv", index=False)
print()
print(df.to_string(index=False))
gap = results["cold_bo"] - results["defaults"]
if gap > 0:
    for b in WARM_BUDGETS:
        rec = 100 * (results[f"warm_bo_{b}"] - results["defaults"]) / gap
        print(f"warm_bo_{b}: {rec:.0f}% of the tuning benefit at "
              f"{b}/{COLD_BUDGET} the cost")
