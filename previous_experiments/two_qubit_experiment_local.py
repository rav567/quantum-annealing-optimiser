import dimod
import matplotlib.pyplot as plt

from dwave.samplers import SimulatedAnnealingSampler

# Every experiment is repeated 1000 times so we get percentages, not a single answer. SWEEP is the list of knob positions we'll try, 
# from -1 to +1, because that's the allowed range on D-Wave hardware.

NUM_READS = 1000
SWEEP = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

sa_sampler = SimulatedAnnealingSampler()
exact_solver = dimod.ExactSolver()

STATES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
LABELS = {(1, 1): "up,up", (1, -1): "up,down",
          (-1, 1): "down,up", (-1, -1): "down,down"}


def make_bqm(h1, h2, J):
    return dimod.BinaryQuadraticModel.from_ising({"s1": h1, "s2": h2},
                                                 {("s1", "s2"): J})


def exact_energies(h1, h2, J):
    """Exact energy of each of the 4 states."""
    result = exact_solver.sample(make_bqm(h1, h2, J))
    return {(s["s1"], s["s2"]): e for s, e in
            zip(result.samples(), result.record.energy)}


def run(h1, h2, J):
    """Sample with simulated annealing, return probability of each state."""
    result = sa_sampler.sample(make_bqm(h1, h2, J), num_reads=NUM_READS)
    counts = {s: 0 for s in STATES}
    for sample, _, occurrences in result.data(["sample", "energy",
                                               "num_occurrences"]):
        counts[(sample["s1"], sample["s2"])] += occurrences
    return {s: n / NUM_READS for s, n in counts.items()}


# --- Part 1: exact energies for a few key settings -------------------------
print("EXACT ENERGIES (check your predictions against these)")
for h1, h2, J in [(0, 0, -1), (0, 0, 1), (-0.5, 0, -1), (0.5, -0.5, -1)]:
    energies = exact_energies(h1, h2, J)
    line = "  ".join(f"{LABELS[s]}: {energies[s]:+.2f}" for s in STATES)
    print(f"h1={h1:+.1f} h2={h2:+.1f} J={J:+.1f}  ->  {line}")
print()

# --- Part 2: the three knob sweeps ------------------------------------------
sweeps = [
    ("Sweep J  (h1=0, h2=0)",   lambda x: (0.0, 0.0, x)),
    ("Sweep h1 (h2=0, J=-1)",   lambda x: (x, 0.0, -1.0)),
    ("Sweep h2 (h1=0.5, J=-1)", lambda x: (0.5, x, -1.0)),
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
plt.savefig("two_qubit_results_local.png", dpi=150)
print("\nSaved plot to two_qubit_results_local.png")