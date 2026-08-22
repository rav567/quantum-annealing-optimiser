"""Stage 4. Tune the cut. Knobs are piece_size, overlap, and order_seed.
The objective is mean gap to exact truth over three fixed training
instances. Runs random search and Bayesian optimisation under an equal
budget, writing data/decomposition_search_random.csv and
data/decomposition_search_bayesian.csv."""

import random
from pathlib import Path
import networkx as nx
import pandas as pd
from skopt import Optimizer
from skopt.space import Integer
from decomposition import solve_exact, stitch_solve

DATA = Path(__file__).resolve().parent.parent / "data"

ATTEMPTS = 50
TRAIN_INSTANCES = [(16, 5, s) for s in (0, 1, 2)]

space = [Integer(2, 6, name="piece_size"),
         Integer(0, 2, name="overlap"),
         Integer(0, 99, name="order_seed")]

problems = []
for n, d, s in TRAIN_INSTANCES:
    g = nx.random_regular_graph(d, n, seed=s)
    truth, _ = solve_exact(g)
    problems.append((g, truth))


def evaluate(knobs):
    """Mean gap to truth across the training instances. Invalid combos
    (overlap >= piece_size) get a large penalty."""
    if knobs["overlap"] >= knobs["piece_size"]:
        return 99.0
    gaps = []
    for g, truth in problems:
        s, _, _ = stitch_solve(g, knobs["piece_size"],
                               knobs["overlap"], knobs["order_seed"])
        gaps.append(truth - s)
    return sum(gaps) / len(gaps)


for METHOD in ("random", "bayesian"):
    random.seed(0)
    opt = Optimizer(space, random_state=0) if METHOD == "bayesian" else None
    rows, best = [], None
    for attempt in range(ATTEMPTS):
        if METHOD == "bayesian":
            x = opt.ask()
            knobs = {dim.name: int(v) for dim, v in zip(space, x)}
        else:
            knobs = {"piece_size": random.randint(2, 6),
                     "overlap": random.randint(0, 2),
                     "order_seed": random.randint(0, 99)}
        obj = evaluate(knobs)
        if METHOD == "bayesian":
            opt.tell(x, obj)
        if best is None or obj < best:
            best = obj
        rows.append({"attempt": attempt, **knobs,
                     "objective": obj, "best_so_far": best})
        print(f"[{METHOD}] attempt {attempt:2d}: obj={obj:.2f}  best={best:.2f}")
        pd.DataFrame(rows).to_csv(
            DATA / f"decomposition_search_{METHOD}.csv", index=False)
