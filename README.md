# quantum-annealing-optimiser

Learning to configure the two steps a problem needs before it runs on a
quantum annealer: decomposition (cutting a large problem into pieces) and
minor embedding (mapping each piece onto the hardware graph).

## Claim

A small model reads a problem's graph and predicts good decomposition and
embedding settings. A single learned guess, or that guess plus a few
optimisation steps, recovers most of the quality of a full configuration
search at a fraction of the cost. This holds at scale (100 to 300
variables) and on a protein folding case study.

## Layout

- `code/` the modules and the ten stage scripts
- `data/` banked results, training sets, and the trained model
- `figures/` the six figures used in the write-up, as PDF and PNG

## Running

Install the dependencies, then run everything from the repository root:

    pip install -r requirements.txt
    python main.py

`main.py` runs the ten stages in order and stops if any stage fails. The
full run takes hours. To check the wiring quickly, skip the multi-hour
stages:

    python main.py --skip-slow

The `data/` and `figures/` folders are already populated, so the final
stage (`make_figures.py`) regenerates the six figures from the banked
CSVs without rerunning the experiments.
