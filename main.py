"""Run the whole pipeline end to end. The ten stages run in order and
the run stops if any stage fails. Use --skip-slow to skip the multi-hour
stages so the wiring can be checked in a few minutes.

    python main.py              run every stage
    python main.py --skip-slow  skip the slow stages
"""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"

# (script, description, slow) in pipeline order
STAGES = [
    ("run_embedding_baseline.py",    "embedding baseline sweep",       True),
    ("run_embedding_tuning.py",      "embedding knob tuning",          True),
    ("run_decomposition_baseline.py", "decomposition baseline",        False),
    ("run_decomposition_tuning.py",  "decomposition cut tuning",       False),
    ("run_joint_experiment.py",      "joint chain-aware experiment",   True),
    ("run_generate_training.py",     "generate GNN training data",     True),
    ("run_train_predictors.py",      "train settings predictors",      True),
    ("run_evaluation.py",            "scale pipeline evaluation",       True),
    ("run_protein_case_study.py",    "protein folding case study",      True),
    ("make_figures.py",              "render the six figures",         False),
]


def main():
    skip_slow = "--skip-slow" in sys.argv[1:]
    for n, (script, desc, slow) in enumerate(STAGES, start=1):
        if skip_slow and slow:
            print(f"Stage {n}: {desc} (skipped, slow)")
            continue
        print(f"Stage {n}: {desc}")
        result = subprocess.run([sys.executable, str(CODE / script)], cwd=CODE)
        if result.returncode != 0:
            print(f"Stage {n} failed: {script}. Stopping.")
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
