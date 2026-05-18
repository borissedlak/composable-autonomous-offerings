
# Composite Autonomous Offerings 

This README describes the notebooks for the ICSOC submission. Together, they allow to replicate the experiments and create the 
figures used in the paper. The repository contains the following contents:

## E1: Experiment #1

In E1, we analyze the convergence of the Gaussian Process (GP) that is trained
as the agent explores the execution environment randomly.

First, in [1_1_extract_candidates.ipynb](notebooks/1_1_extract_candidates.ipynb)
we train the GPs with increasing amounts of data and extract the potential candidate solutions
that could be offered.

The candidates are then executed empirically in the execution environment; we collect the results
and follow with the next step. 

Second, in [1_2_evaluate_candidates.ipynb](notebooks/1_2_evaluate_candidates.ipynb),
we then results we collected and compare them against the expectations of the agent. 

## E2: Experiment #2

[2_create_deepGP_chunk_training.ipynb](notebooks/2_create_deepGP_chunk_training.ipynb)

## Miscellaneous

### Execution Environment

