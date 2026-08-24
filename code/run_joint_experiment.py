"""Stage 5. Joint tuning under embedding pressure. Degree-8 problems on
a 32-qubit cell with large pieces, so cuts genuinely trade severed edges
against embeddability. Four arms: defaults, cut-only, embed-only, and
joint. Writes data/joint_chain_aware.csv."""

from pathlib import Path
import networkx as nx
import pandas as pd
import dwave_networkx as dnx
from skopt import Optimizer
from skopt.space import Integer
from decomposition import full_pipeline

DATA = Path(__file__).resolve().parent.parent / "data"

INSTANCES = [(40, 8, s) for s in range(200, 230)]
BUDGET = 20
DEFAULT_CUT = {"piece_size": 12, "overlap": 1, "order_seed": None}
DEFAULT_EMBED = {}
TIMEOUT = 0.2

hardware = dnx.chimera_graph(2)          # 32 qubits

CUT_DIMS = [Integer(14, 32, name="piece_size"),
            Integer(0, 2, name="overlap"),
            Integer(0, 99, name="order_seed")]
EMBED_DIMS = [Integer(1, 100, name="tries"),
              Integer(1, 100, name="max_no_improvement"),
              Integer(1, 100, name="chainlength_patience")]

ARMS = {"defaults": [], "cut_only": CUT_DIMS,
        "embed_only": EMBED_DIMS, "joint": CUT_DIMS + EMBED_DIMS}


def knobs_from(dims, x):
    d = {dim.name: int(v) for dim, v in zip(dims, x)}
    cut_k = {"piece_size": d.get("piece_size", DEFAULT_CUT["piece_size"]),
             "overlap": d.get("overlap", DEFAULT_CUT["overlap"]),
             "order_seed": d.get("order_seed", DEFAULT_CUT["order_seed"])}
    embed_k = {k: d[k] for k in
               ("tries", "max_no_improvement", "chainlength_patience") if k in d}
    return cut_k, embed_k


rows = []
for n, deg, s in INSTANCES:
    g = nx.random_regular_graph(deg, n, seed=s)

    for arm, dims in ARMS.items():
        if not dims:
            raw_score, diag = full_pipeline(g, hardware, DEFAULT_CUT,
                                            DEFAULT_EMBED, embed_seed=0,
                                            timeout=TIMEOUT)
            embedding_penalty = (0.5 * diag["max_chain"] +
                                 5.0 * diag["failed"])
            score = raw_score - embedding_penalty
            best = score
            best_diag = diag
            best_knobs = {}
        else:
            opt = Optimizer(dims, random_state=0)
            best = -float("inf")
            best_diag = None
            best_knobs = None
            for attempt in range(BUDGET):
                x = opt.ask()
                cut_k, embed_k = knobs_from(dims, x)

                raw_score, diag = full_pipeline(g, hardware, cut_k, embed_k,
                                                embed_seed=attempt,
                                                timeout=TIMEOUT)

                embedding_penalty = (2.0 * diag["max_chain"] +
                                     10.0 * diag["failed"])
                score = raw_score - embedding_penalty

                opt.tell(x, -score)
                if score > best:
                    best = score
                    best_diag = diag
                    best_knobs = {**cut_k, **embed_k}

        rows.append({"seed": s, "arm": arm, "score": best,
                     **best_diag, "knobs": str(best_knobs)})
        print(f"seed {s} {arm}: score={best} "
              f"(failed={best_diag['failed']}, "
              f"max_chain={best_diag['max_chain']}, "
              f"severed={best_diag['severed']})")

    pd.DataFrame(rows).to_csv(DATA / "joint_chain_aware.csv", index=False)

df = pd.DataFrame(rows)
print()
print(df.groupby("arm")["score"].agg(["mean", "std"]))
