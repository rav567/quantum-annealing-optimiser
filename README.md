# Machine Learning for Optimising Decomposition and Minor Embedding in Quantum Annealing

MSc dissertation project, UCL Department of Electronic and Electrical Engineering.

A learned system that chooses the settings of the quantum annealing preparation pipeline for each new problem, instead of leaving them at their defaults or searching from scratch.

## Highlights

- Maps where the standard embedding tool fails on Chimera hardware, and shows it gives up while roughly a third of the chip is still unused
- Shows that almost all recoverable solution quality comes from the decomposition settings
- A graph neural network predicts pipeline settings for unseen problems, recovering 98.5 percent of full-search quality at 30 percent of the evaluations
- Cuts selection time per problem from 43 seconds to 10, and to 2 seconds for the prediction alone
- On a protein folding problem excluded from all training, a single prediction captured 95 percent of the available benefit

## Overview

Problems that run on a quantum annealer must first be prepared. Problems with too many variables are decomposed into pieces, and pieces whose connectivity exceeds the hardware are minor embedded onto the qubit topology. Both stages have settings, and in practice these are left at default values.

This project treats those five settings as decision variables. It measures what the defaults lose, searches for better settings on each problem, tests whether the two stages should be chosen together, and finally trains a graph neural network to predict good settings for problems it has never seen. All experiments run through a hardware-realistic solver in which qubit chains can break.

## Requirements

- Python 3.10 or later
- No system dependencies and no GPU required

## Installation

git clone https://github.com/rav567/quantum-annealing-optimiser.git

cd quantum-annealing-optimiser

pip install -r requirements.txt

## Usage

Run every stage in order from the repository root:

python main.py


Some stages take hours. To check setup quickly, skip the slow stages:

python main.py --skip-slow


Each stage prints a line when it starts and stops if anything fails.

## Structure

main.py entry point running all ten stages
code/ toolboxes, stage scripts and the figure script
data/ results of every experiment as csv files
figures/ the six dissertation figures as pdf and png


## Data and figures

The data folder ships with the saved results of every experiment, so the figures and every number quoted in the dissertation regenerate without rerunning the experiments. Rerunning a stage reproduces its results from fixed seeds. The figure script reads only from the data folder:

cd code
python make_figures.py

To regenerate the data behind a result, run its stage script from `code/`. The rerun column marks the stages that take hours, the same ones `--skip-slow` bypasses.

| Paper result | Data file | Stage script | Rerun |
|---|---|---|---|
| Embedding cliffs | `data/embedding_baseline.csv` | `run_embedding_baseline.py` | slow |
| Decomposition baseline | `data/decomposition_baseline.csv` | `run_decomposition_baseline.py` | fast |
| Embedding validation | `data/embedding_validation.csv` | `run_embedding_tuning.py` | slow |
| Joint comparison | `data/joint_chain_aware.csv` | `run_joint_experiment.py` | slow |
| Budget and scale figures | `data/pipeline_evaluation_scale.csv`, `data/pipeline_evaluation_smallscale.csv` | `run_evaluation.py` | slow |
| Protein table | `data/protein_case_study.csv` | `run_protein_case_study.py` | slow |

Every seed is either a constant or a function of the problem index, so a rerun reproduces the same numbers.

## Use of generative AI

GenAI (Chat GPT and Claude) was used during the preparation of this manuscript in a writing-support capacity only, consistent with UCL's Academic Manual.

First, generative AI was used to assist with writing parts of the experimental code. Prompts were used to draft and debug functions within the pipeline, which were then reviewed, corrected, and integrated by the author. The experimental design, the choice of methods, and the interpretation of all results were the author's own, and every line of assisted code was checked and understood before use.

Second, generative AI was used for language and writing review. This included grammar checking, improving sentence clarity, and suggesting structural adjustments to section ordering and flow. The technical content, analysis, and arguments were written by the author.

No AI tools were used to generate citations, fabricate data, or produce the results. All experimental design, execution, data analysis, and scientific claims are the author's own. Every suggested edit was reviewed and either accepted, modified, or rejected by the author.