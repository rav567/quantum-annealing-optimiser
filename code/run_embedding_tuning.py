"""Stage 2. Tune minorminer's knobs on one frozen hard problem, then
validate. Bayesian optimisation minimises the robust objective and
writes data/embedding_bayesian_robust.csv; the winning knobs are then
tested on ten unseen seeds, written to data/embedding_validation.csv."""

from pathlib import Path
import pandas as pd
import dwave_networkx as dnx
from skopt import Optimizer
from skopt.space import Integer
from embedding import make_problem, evaluate_knobs_robust, run_one, score

DATA = Path(__file__).resolve().parent.parent / "data"

hardware = dnx.chimera_graph(16)
problem = make_problem(150, 5, 0)

# Bayesian search over the knobs
ATTEMPTS = 50
space = [Integer(1, 100, name="tries"),
         Integer(1, 100, name="max_no_improvement"),
         Integer(1, 100, name="chainlength_patience")]

opt = Optimizer(space, random_state=0)
rows, best = [], None
for attempt in range(ATTEMPTS):
    x = opt.ask()
    knobs = {dim.name: int(v) for dim, v in zip(space, x)}
    obj, avg, fails, runtime = evaluate_knobs_robust(problem, hardware, knobs)
    opt.tell(x, obj)
    if best is None or obj < best:
        best = obj
    rows.append({"attempt": attempt, **knobs, "objective": obj,
                 "avg_longest_chain": avg, "fails": fails,
                 "runtime": runtime, "best_so_far": best})
    print(f"attempt {attempt:2d}: obj={obj:.2f} fails={fails}  best={best:.2f}")
    pd.DataFrame(rows).to_csv(DATA / "embedding_bayesian_robust.csv", index=False)

# Validate the winning knobs on unseen seeds
TEST_SEEDS = range(100, 110)
bay = pd.read_csv(DATA / "embedding_bayesian_robust.csv")
best_row = bay.loc[bay["objective"].idxmin()]
best_knobs = {k: int(best_row[k])
              for k in ["tries", "max_no_improvement", "chainlength_patience"]}
print("winning knobs:", best_knobs)
print("training objective:", round(best_row["objective"], 2),
      "fails in training:", int(best_row["fails"]))

rows = []
for s in TEST_SEEDS:
    for label, knobs in [("defaults", {}), ("tuned", best_knobs)]:
        emb, rt = run_one(problem, hardware, seed=s, knobs=knobs)
        row = {"seed": s, "method": label, "success": len(emb) > 0}
        if row["success"]:
            row.update(score(emb))
        rows.append(row)
        print(f"seed {s} {label}: longest={row.get('longest_chain')}")

df = pd.DataFrame(rows)
df.to_csv(DATA / "embedding_validation.csv", index=False)
print()
print(df.groupby("method")["success"].mean().rename("success_rate"))
print()
print(df.groupby("method")["longest_chain"].agg(["mean", "std"]))
