"""Settings predictors shared by the evaluation stages. Loads the
combined training set, rebuilds each problem graph, and provides the
GNN model class, the graph-to-tensor conversion, and the nearest
neighbour rival. Training lives in run_train_predictors.py; this module
only prepares data and defines the pieces, it does not train or write files."""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

torch.manual_seed(0)
np.random.seed(0)

DATA = Path(__file__).resolve().parent.parent / "data"

SETTINGS = ["piece_size", "overlap", "tries",
            "max_no_improvement", "chainlength_patience"]
RANGES = {"piece_size": (6, 28), "overlap": (0, 2), "tries": (1, 100),
          "max_no_improvement": (1, 100), "chainlength_patience": (1, 100)}

small = pd.read_csv(DATA / "gnn_training_set.csv")
small["scale"] = False
big = pd.read_csv(DATA / "gnn_training_set_scale.csv")
big["scale"] = True
df = pd.concat([small, big], ignore_index=True)


def rebuild(kind, seed, scale):
    if not scale:
        n = 20 + (seed * 7) % 41
        if kind == "random":
            d = 4 + (seed * 3) % 7
            if (n * d) % 2 == 1:
                n += 1
            return nx.random_regular_graph(d, n, seed=seed)
        n_clusters = 3 + seed % 4
        size = max(4, n // n_clusters)
        g = nx.random_partition_graph([size] * n_clusters, 0.9, 0.06,
                                      seed=seed)
        return nx.Graph(g)
    n = 100 + (seed * 13) % 201
    if kind == "random":
        d = 4 + (seed * 3) % 5
        if (n * d) % 2 == 1:
            n += 1
        return nx.random_regular_graph(d, n, seed=seed)
    n_clusters = max(4, n // 20)
    size = n // n_clusters
    g = nx.random_partition_graph([size] * n_clusters, 0.35, 0.02,
                                  seed=seed)
    return nx.Graph(g)


# every fifth row is held out for testing; same split the training uses
df["is_test"] = df.index % 5 == 0


def to01(row):
    return [(row[s] - RANGES[s][0]) / (RANGES[s][1] - RANGES[s][0])
            for s in SETTINGS]


def from01(vec):
    out = {}
    for s, v in zip(SETTINGS, vec):
        lo, hi = RANGES[s]
        out[s] = int(round(lo + float(np.clip(v, 0, 1)) * (hi - lo)))
    return out


def to_data(g, label01=None):
    nodes = sorted(g.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    deg = dict(g.degree)
    clu = nx.clustering(g)
    x = torch.tensor([[deg[v] / 10.0, clu[v]] for v in nodes],
                     dtype=torch.float)
    edges = [[idx[u], idx[v]] for u, v in g.edges]
    edge_index = torch.tensor(edges + [[b, a] for a, b in edges],
                              dtype=torch.long).t().contiguous()
    d = Data(x=x, edge_index=edge_index)
    d.gfeats = torch.tensor([[g.number_of_nodes() / 300.0,
                              nx.density(g),
                              nx.average_clustering(g)]],
                            dtype=torch.float)
    if label01 is not None:
        d.y = torch.tensor([label01], dtype=torch.float)
    return d


class SettingsGNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.c1 = GCNConv(2, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 5), nn.Sigmoid())

    def forward(self, d):
        h = torch.relu(self.c1(d.x, d.edge_index))
        h = torch.relu(self.c2(h, d.edge_index))
        batch = getattr(d, "batch",
                        torch.zeros(d.x.size(0), dtype=torch.long))
        pooled = global_mean_pool(h, batch)
        return self.head(torch.cat([pooled, d.gfeats], dim=1))


FEATS = ["n", "density", "clustering"]
tr_df, te_df = df[~df["is_test"]], df[df["is_test"]]
mu, sd = tr_df[FEATS].mean(), tr_df[FEATS].std()


def rival_predict(row, k=5):
    z = ((tr_df[FEATS] - mu) / sd
         - ((row[FEATS] - mu) / sd)).pow(2).sum(axis=1)
    nearest = tr_df.loc[z.astype(float).nsmallest(k).index]
    return {s: int(round(nearest[s].mean())) for s in SETTINGS}
