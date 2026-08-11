"""
Task 1: Two qubits, one coupler, three knobs (h1, h2, J).
Sweeps each knob while holding the others fixed, runs on D-Wave hardware,
and plots the probability of each state (up-up, up-down, down-up, down-down).

Setup (once):
    pip install dwave-ocean-sdk matplotlib
    dwave setup   (paste your Leap API token when asked)

Run:
    python two_qubit_experiment.py
"""

import dimod
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite

NUM_READS = 1000
SWEEP = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

sampler = EmbeddingComposite(DWaveSampler())


def run(h1, h2, J):
    """Run one (h1, h2, J) setting, return probability of each spin state."""
    bqm = dimod.BinaryQuadraticModel.from_ising({"s1": h1, "s2": h2},
                                                {("s1", "s2"): J})
    result = sampler.sample(bqm, num_reads=NUM_READS,
                            label=f"2-qubit h1={h1} h2={h2} J={J}")
    counts = {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0}
    for sample, _, occurrences, _ in result.data():
        counts[(sample["s1"], sample["s2"])] += occurrences
    return {state: n / NUM_READS for state, n in counts.items()}


STATES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
LABELS = {(1, 1): "up,up", (1, -1): "up,down",
          (-1, 1): "down,up", (-1, -1): "down,down"}

# Three sweeps: (name, fixed values, function mapping swept value -> (h1,h2,J))
sweeps = [
    ("Sweep J  (h1=0, h2=0)",        lambda x: (0.0, 0.0, x)),
    ("Sweep h1 (h2=0, J=-1)",        lambda x: (x, 0.0, -1.0)),
    ("Sweep h2 (h1=0.5, J=-1)",      lambda x: (0.5, x, -1.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (title, knobs) in zip(axes, sweeps):
    probs = {s: [] for s in STATES}
    for x in SWEEP:
        p = run(*knobs(x))
        print(f"{title}  x={x:+.2f}  " +
              "  ".join(f"{LABELS[s]}={p[s]:.2f}" for s in STATES))
        for s in STATES:
            probs[s].append(p[s])
    for s in STATES:
        ax.plot(SWEEP, probs[s], marker="o", label=LABELS[s])
    ax.set_title(title)
    ax.set_xlabel("swept value")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("probability")
axes[0].legend()
plt.tight_layout()
plt.savefig("two_qubit_results.png", dpi=150)
print("Saved plot to two_qubit_results.png")