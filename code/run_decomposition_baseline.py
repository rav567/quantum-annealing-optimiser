"""Stage 3. Decomposition baseline sweep. 450 random problems solved
three ways: exact brute-force truth, our one-pass stitch, and a
QBSolv-style iterative rival. The metric is gap to truth in satisfied
edges. Sizes stay small so the exact answer exists to compare against.
Writes data/decomposition_baseline.csv."""

import time
from pathlib import Path
import networkx as nx
import pandas as pd
from decomposition import solve_exact, stitch_solve, qbsolv_style_solve

DATA = Path(__file__).resolve().parent.parent / "data"

sizes = [8, 10, 12, 14, 16]
degrees = [3, 4, 5]
batch = 30
PIECE_SIZE = 6

rows = []
for n in sizes:
    for d in degrees:
        for i in range(batch):
            problem = nx.random_regular_graph(d, n, seed=i)
            edges = problem.number_of_edges()

            true_score, _ = solve_exact(problem)

            t0 = time.time()
            ours, _, pieces = stitch_solve(problem, PIECE_SIZE)
            t_ours = time.time() - t0

            t0 = time.time()
            qb, _, rounds = qbsolv_style_solve(problem, PIECE_SIZE, seed=i)
            t_qb = time.time() - t0

            rows.append({"n": n, "d": d, "seed": i, "edges": edges,
                         "true_score": true_score,
                         "ours_score": ours, "ours_gap": true_score - ours,
                         "ours_time": t_ours, "pieces": len(pieces),
                         "qb_score": qb, "qb_gap": true_score - qb,
                         "qb_time": t_qb, "qb_rounds": rounds})
        pd.DataFrame(rows).to_csv(DATA / "decomposition_baseline.csv",
                                  index=False)
        sub = rows[-batch:]
        print(f"n={n} d={d}: mean gap ours="
              f"{sum(r['ours_gap'] for r in sub)/batch:.2f}  "
              f"qb={sum(r['qb_gap'] for r in sub)/batch:.2f}")

print(len(rows), "instances recorded")
