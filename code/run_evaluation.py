"""Stage 8. Scale evaluation: five configuration strategies on the
held-out scale instances (100-300 variables, full Chimera C16), through
the realistic pipeline (annealed through embeddings, chains can break).
Scores are raw satisfied edges; physics does the penalising. Wall-clock
per strategy is recorded. Writes data/pipeline_evaluation_scale.csv."""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import dwave_networkx as dnx
from skopt import Optimizer
from skopt.space import Integer
from decomposition import full_pipeline_real
from predictors import (rebuild, to_data, from01, rival_predict,
                        SettingsGNN, SETTINGS, df as train_df,
                        tr_df, mu, sd, FEATS)
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

TIMEOUT = 2
COLD_BUDGET = 20
WARM_BUDGETS = [6, 11]

hardware = dnx.chimera_graph(16)
model = SettingsGNN()
model.load_state_dict(torch.load(DATA / "gnn_model.pt"))
model.eval()

space = [Integer(10, 28, name="piece_size"),
         Integer(0, 2, name="overlap"),
         Integer(1, 100, name="tries"),
         Integer(1, 100, name="max_no_improvement"),
         Integer(1, 100, name="chainlength_patience")]


def run_settings(g, k, seed=0):
    cut_k = {"piece_size": int(np.clip(k["piece_size"], 6, 28)),
             "overlap": int(np.clip(k["overlap"], 0, 2)),
             "order_seed": None, "ordering": "spectral"}
    embed_k = {n: int(np.clip(k[n], 1, 100)) for n in
               ("tries", "max_no_improvement", "chainlength_patience")}
    score, diag = full_pipeline_real(g, hardware, cut_k, embed_k,
                                     embed_seed=seed, timeout=TIMEOUT)
    return score


def neighbour_settings(row, k=2):
    z = ((tr_df[FEATS] - mu) / sd
         - ((row[FEATS] - mu) / sd)).pow(2).sum(axis=1)
    nearest = tr_df.loc[z.astype(float).nsmallest(k).index]
    return [dict(r[SETTINGS].astype(int)) for _, r in nearest.iterrows()]


def warm_bo(g, budget, tells):
    opt = Optimizer(space, random_state=0)
    best, used = -np.inf, 0
    for t in tells[:budget]:
        s = run_settings(g, t, seed=used)
        opt.tell([int(np.clip(t[d.name], d.low, d.high)) for d in space], -s)
        best = max(best, s)
        used += 1
    for attempt in range(used, budget):
        x = opt.ask()
        k = {d.name: int(v) for d, v in zip(space, x)}
        if k["overlap"] >= k["piece_size"]:
            opt.tell(x, 1000.0)
            continue
        s = run_settings(g, k, seed=attempt)
        opt.tell(x, -s)
        best = max(best, s)
    return best


DEFAULTS = {"piece_size": 20, "overlap": 1, "tries": 10,
            "max_no_improvement": 10, "chainlength_patience": 10}

tests = pd.read_csv(DATA / "test_seeds.csv")
tests = tests[tests["scale"] == True]
print(f"{len(tests)} held-out scale instances")

rows = []
for _, t in tests.iterrows():
    g = rebuild(t["kind"], t["seed"], True)
    row_df = train_df[(train_df["kind"] == t["kind"])
                      & (train_df["seed"] == t["seed"])
                      & (train_df["scale"] == True)].iloc[0]

    gnn_guess = from01(model(to_data(g))[0].detach().numpy())
    rival_guess = rival_predict(row_df)
    tells = [gnn_guess] + neighbour_settings(row_df, k=2)

    r = {"kind": t["kind"], "seed": t["seed"],
         "n": g.number_of_nodes(), "edges": g.number_of_edges()}
    t0 = time.time(); r["defaults"] = run_settings(g, DEFAULTS)
    r["defaults_t"] = round(time.time() - t0, 1)
    t0 = time.time(); r["cold_bo"] = warm_bo(g, COLD_BUDGET, tells=[])
    r["cold_bo_t"] = round(time.time() - t0, 1)
    t0 = time.time(); r["feature_pred"] = run_settings(g, rival_guess)
    r["feature_pred_t"] = round(time.time() - t0, 1)
    t0 = time.time(); r["gnn_pred"] = run_settings(g, gnn_guess)
    r["gnn_pred_t"] = round(time.time() - t0, 1)
    for b in WARM_BUDGETS:
        t0 = time.time(); r[f"warm_bo_{b}"] = warm_bo(g, b, tells=tells)
        r[f"warm_bo_{b}_t"] = round(time.time() - t0, 1)
    rows.append(r)
    print(f"{t['kind']} {t['seed']} n={r['n']}: def={r['defaults']} "
          f"cold={r['cold_bo']} feat={r['feature_pred']} "
          f"gnn={r['gnn_pred']} " +
          " ".join(f"w{b}={r[f'warm_bo_{b}']}" for b in WARM_BUDGETS),
          flush=True)
    pd.DataFrame(rows).to_csv(DATA / "pipeline_evaluation_scale.csv",
                              index=False)

df = pd.DataFrame(rows)
cols = ["defaults", "cold_bo", "feature_pred", "gnn_pred",
        "warm_bo_6", "warm_bo_11"]
print()
print(df[cols].mean().round(2))
gap = df["cold_bo"] - df["defaults"]
hard = df[gap >= 10]
print(f"\ninstances with tuning headroom (>=10): {len(hard)}/{len(df)}")
if len(hard):
    print(hard[cols].mean().round(2))
for b in WARM_BUDGETS:
    pct = 100 * df[f"warm_bo_{b}"].mean() / df["cold_bo"].mean()
    print(f"warm_bo_{b}: {pct:.1f}% of cold-search quality at "
          f"{b}/{COLD_BUDGET} the cost")
wins = (df["gnn_pred"] > df["feature_pred"]).sum()
ties = (df["gnn_pred"] == df["feature_pred"]).sum()
print(f"GNN vs feature prediction: {wins} wins, {ties} ties, "
      f"{len(df) - wins - ties} losses")
print(f"\nmean wall-clock per instance: cold search "
      f"{df['cold_bo_t'].mean():.0f}s vs warm(6) "
      f"{df['warm_bo_6_t'].mean():.0f}s vs single guess "
      f"{df['gnn_pred_t'].mean():.0f}s")
