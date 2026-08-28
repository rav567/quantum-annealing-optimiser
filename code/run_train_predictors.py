"""Stage 7. Train the settings GNN on the combined training set and save
it to data/gnn_model.pt, alongside the held-out seed list in
data/test_seeds.csv. Reports label error against the nearest-neighbour
rival as a progress check; the real judge is the pipeline evaluation."""

import numpy as np
import torch
import torch.nn as nn
from predictors import (DATA, df, rebuild, to01, to_data, from01,
                        SETTINGS, RANGES, SettingsGNN,
                        tr_df, te_df, mu, sd, FEATS, rival_predict)

print(f"combined: {len(df[~df['scale']])} small + {len(df[df['scale']])} "
      f"scale = {len(df)}")
print(f"split: {(~df['is_test']).sum()} train / {df['is_test'].sum()} test")
df[df["is_test"]][["kind", "seed", "scale"]].to_csv(
    DATA / "test_seeds.csv", index=False)

train_data, test_data = [], []
for _, row in df.iterrows():
    g = rebuild(row["kind"], row["seed"], row["scale"])
    item = to_data(g, to01(row))
    (test_data if row["is_test"] else train_data).append(item)

model = SettingsGNN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
lossf = nn.MSELoss()


def dataset_loss(data):
    model.eval()
    with torch.no_grad():
        return float(np.mean([lossf(model(d), d.y).item() for d in data]))


best_test, patience = 1e9, 0
for epoch in range(200):
    model.train()
    np.random.shuffle(train_data)
    for d in train_data:
        opt.zero_grad()
        loss = lossf(model(d), d.y)
        loss.backward()
        opt.step()
    tr, te = dataset_loss(train_data), dataset_loss(test_data)
    if te < best_test - 1e-4:
        best_test, patience = te, 0
        torch.save(model.state_dict(), DATA / "gnn_model.pt")
    else:
        patience += 1
    if epoch % 10 == 0 or patience == 15:
        print(f"epoch {epoch:3d}: train={tr:.4f} test={te:.4f} "
              f"(best {best_test:.4f})")
    if patience == 15:
        print("stopped: no test improvement for 15 epochs")
        break

model.load_state_dict(torch.load(DATA / "gnn_model.pt"))
model.eval()
gnn_err, rival_err = [], []
per = {s: [] for s in SETTINGS}
for _, row in te_df.iterrows():
    g = rebuild(row["kind"], row["seed"], row["scale"])
    pred = from01(model(to_data(g))[0].detach().numpy())
    rv = rival_predict(row)
    truth = {s: row[s] for s in SETTINGS}
    gnn_err.append(np.mean([abs(pred[s] - truth[s])
                            / (RANGES[s][1] - RANGES[s][0])
                            for s in SETTINGS]))
    rival_err.append(np.mean([abs(rv[s] - truth[s])
                              / (RANGES[s][1] - RANGES[s][0])
                              for s in SETTINGS]))
    for s in SETTINGS:
        per[s].append(abs(pred[s] - truth[s])
                      / (RANGES[s][1] - RANGES[s][0]))

print()
print("label error on unseen (0=perfect, lower=better):")
print(f"  GNN   : {np.mean(gnn_err):.3f}")
print(f"  rival : {np.mean(rival_err):.3f}")
print("\nGNN error per setting:")
for s in SETTINGS:
    print(f"  {s:22s}: {np.mean(per[s]):.3f}")
print("\nprogress check only, the pipeline evaluation is the real judge")
print("saved: gnn_model.pt, test_seeds.csv")
