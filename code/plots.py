import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

metrics = ["longest_chain", "avg_chain", "total_qubits", "runtime", "success"]
labels = {
    "longest_chain": "Longest chain",
    "avg_chain": "Average chain length",
    "total_qubits": "Total physical qubits",
    "runtime": "Runtime (seconds)",
    "success": "Success rate",
}

for metric in metrics:
    plt.figure()
    for d in sorted(df["d"].unique()):
        sub = df[df["d"] == d]
        stats = sub.groupby("n")[metric].agg(["mean", "std"])
        if metric == "success":
            plt.plot(stats.index, stats["mean"], marker="o", label=f"d={d}")
        else:
            plt.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                         marker="o", capsize=3, label=f"d={d}")
    plt.xlabel("Problem size n (variables)")
    plt.ylabel(labels[metric])
    plt.title(f"{labels[metric]} vs problem size (Chimera C16, minorminer)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_{metric}.png", dpi=150)
    plt.close()

print("5 plots saved")
