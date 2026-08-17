"""Stage 1. The minorminer baseline sweep: 1,560 problems across sizes
and degrees on Chimera C16. Writes data/embedding_baseline.csv.
This takes hours to rerun; do not run it casually."""

from pathlib import Path
import pandas as pd
import dwave_networkx as dnx
from embedding import make_problem, run_one, score

DATA = Path(__file__).resolve().parent.parent / "data"

hardware = dnx.chimera_graph(16)

sizes = range(10, 251, 20)      # 10, 30, ..., 250
degrees = [3, 4, 5, 6]
batch = 30

rows = []
for n in sizes:
    for d in degrees:
        for i in range(batch):
            problem = make_problem(n, d, seed=i)
            emb, runtime = run_one(problem, hardware, seed=i)
            row = {"n": n, "d": d, "seed": i,
                   "runtime": runtime, "success": len(emb) > 0}
            if row["success"]:
                row.update(score(emb))
            rows.append(row)
        pd.DataFrame(rows).to_csv(DATA / "embedding_baseline.csv", index=False)
        wins = sum(r["success"] for r in rows[-batch:])
        print(f"n={n} d={d}: {wins}/{batch} succeeded")

print(len(rows), "attempts recorded")
