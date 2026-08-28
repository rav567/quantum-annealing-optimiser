"""Stage 6. Build the scale training set: 120 problems of 100-300
variables on the full Chimera C16, tuned through the realistic pipeline
(annealed through embeddings, chains can break). Saves per problem and
is resumable. Writes data/gnn_training_set_scale.csv."""

import os
from pathlib import Path
import networkx as nx
import pandas as pd
import dwave_networkx as dnx
from skopt import Optimizer
from skopt.space import Integer
from decomposition import full_pipeline_real

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "gnn_training_set_scale.csv"
ATTEMPTS = 15
TIMEOUT = 2
N_RANDOM, N_CLUSTERED = 40, 40

hardware = dnx.chimera_graph(16)

space = [Integer(10, 28, name="piece_size"),
         Integer(0, 2, name="overlap"),
         Integer(1, 100, name="tries"),
         Integer(1, 100, name="max_no_improvement"),
         Integer(1, 100, name="chainlength_patience")]


def make_problem(kind, seed):
    n = 100 + (seed * 13) % 201            # sizes 100-300
    if kind == "random":
        d = 4 + (seed * 3) % 5             # degrees 4-8
        if (n * d) % 2 == 1:
            n += 1
        return nx.random_regular_graph(d, n, seed=seed), n, d
    n_clusters = max(4, n // 20)           # clusters of ~20
    size = n // n_clusters
    g = nx.random_partition_graph([size] * n_clusters, 0.35, 0.02,
                                  seed=seed)
    return nx.Graph(g), size * n_clusters, round(
        2 * g.number_of_edges() / g.number_of_nodes(), 2)


def tune(problem):
    opt = Optimizer(space, random_state=0)
    best, best_knobs = -float("inf"), None
    for attempt in range(ATTEMPTS):
        x = opt.ask()
        k = {dim.name: int(v) for dim, v in zip(space, x)}
        if k["overlap"] >= k["piece_size"]:
            opt.tell(x, 1000.0)
            continue
        cut_k = {"piece_size": k["piece_size"], "overlap": k["overlap"],
                 "order_seed": None, "ordering": "spectral"}
        embed_k = {n: k[n] for n in
                   ("tries", "max_no_improvement", "chainlength_patience")}
        score, diag = full_pipeline_real(problem, hardware, cut_k, embed_k,
                                         embed_seed=attempt,
                                         timeout=TIMEOUT)
        opt.tell(x, -score)
        if score > best:
            best, best_knobs = score, k
    return best_knobs, best


done, rows = set(), []
if os.path.exists(OUT):
    old = pd.read_csv(OUT)
    rows = old.to_dict("records")
    done = {(r["kind"], r["seed"]) for r in rows}
    print(f"resuming: {len(rows)} already done")

jobs = ([("random", s) for s in range(N_RANDOM)]
        + [("clustered", s) for s in range(N_CLUSTERED)])

for kind, seed in jobs:
    if (kind, seed) in done:
        continue
    g, n, d = make_problem(kind, seed)
    knobs, score = tune(g)
    rows.append({"kind": kind, "seed": seed, "n": n, "d": d,
                 "edges": g.number_of_edges(),
                 "density": round(nx.density(g), 4),
                 "clustering": round(nx.average_clustering(g), 4),
                 "best_score": round(score, 2), **knobs})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)}/120  {kind} seed={seed} n={n}  -> {knobs}")
