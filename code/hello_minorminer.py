import time
import networkx as nx
import dwave_networkx as dnx
import minorminer
import pandas as pd

def score(embedding):
    lengths = [len(chain) for chain in embedding.values()]
    return {"total_qubits": sum(lengths),
            "longest_chain": max(lengths),
            "avg_chain": sum(lengths) / len(lengths)}

def run_one(problem, hardware, seed):
    start = time.time()
    emb = minorminer.find_embedding(
        list(problem.edges), list(hardware.edges),
        timeout=60, random_seed=seed)
    return emb, time.time() - start

def make_problem(n, d, seed):
    return nx.random_regular_graph(d, n, seed=seed)

# --- the chip map, built once ---
hardware = dnx.chimera_graph(16)

# --- step 6: the real sweep ---
sizes = range(10, 251, 20)     # 10, 30, 50, ..., 250
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
        pd.DataFrame(rows).to_csv("results.csv", index=False)   # save after every batch
        wins = sum(r["success"] for r in rows[-batch:])
        print(f"n={n} d={d}: {wins}/{batch} succeeded")

print(len(rows), "attempts recorded, saved to results.csv")