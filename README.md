# Machine Learning for Jointly Optimising Decomposition and Minor Embedding in Quantum Annealing

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

## Installation

Requires Python 3.10 or later.

pip install -r requirements.txt


## Usage

Run every stage in order from the repository root:

python main.py


Some stages take hours. To check the wiring quickly, skip the slow stages:

python main.py --skip-slow


Each stage prints a line when it starts and stops if anything fails.

## Structure

main.py entry point running all ten stages
code/ toolboxes, stage scripts and the figure script
data/ results of every experiment as csv files
figures/ the six dissertation figures as pdf and png


## Data and figures

The data folder ships with the banked results of every experiment, so the figures and every number quoted in the dissertation regenerate without rerunning the experiments. Rerunning a stage reproduces its results from fixed seeds. The figure script reads only from the data folder:

cd code
python make_figures.py