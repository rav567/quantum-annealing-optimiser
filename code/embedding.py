"""Minor-embedding toolbox. Builds random problem graphs, runs one
minorminer embedding attempt, and scores the result by chain length.
Used by the embedding baseline and tuning stages."""

import time
import networkx as nx
import minorminer


def make_problem(n, d, seed):
    """Random regular problem graph: n variables, degree d, reproducible from seed."""
    return nx.random_regular_graph(d, n, seed=seed)


def run_one(problem, hardware, seed, knobs=None):
    """One embedding attempt. Returns (embedding dict, runtime in seconds)."""
    knobs = knobs or {}
    start = time.time()
    emb = minorminer.find_embedding(
        list(problem.edges), list(hardware.edges),
        timeout=60, random_seed=seed, **knobs)
    return emb, time.time() - start


def score(embedding):
    """Three quality metrics from an embedding dict."""
    lengths = [len(chain) for chain in embedding.values()]
    return {"total_qubits": sum(lengths),
            "longest_chain": max(lengths),
            "avg_chain": sum(lengths) / len(lengths)}


def evaluate_knobs_robust(problem, hardware, knobs,
                          seeds=(0, 1, 2, 3, 4), fail_chain=40):
    """Penalised objective: mean over seeds, each failure counted as a
    chain of fail_chain. Rewards short chains and reliability.
    Returns (objective, avg over successes or None, fails, runtime)."""
    scores, fails, total = [], 0, 0.0
    for s in seeds:
        emb, rt = run_one(problem, hardware, seed=s, knobs=knobs)
        total += rt
        if emb:
            scores.append(score(emb)["longest_chain"])
        else:
            fails += 1
    objective = (sum(scores) + fail_chain * fails) / len(seeds)
    avg = sum(scores) / len(scores) if scores else None
    return objective, avg, fails, total
