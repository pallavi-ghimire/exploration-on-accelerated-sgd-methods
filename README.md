# Exploration of Accelerated Stochastic Gradient Methods
This repository contains the implementation and experiments for my MSc Data Science thesis on Exploration of Accelerated Stochastic Gradient Methods for Least Squares Problems. It includes Python scripts for different optimizers (SGD, SVRG, ASG), synthetic dataset generation, real-world data experiments, and utilities for configuration, analysis, and plotting.

## requirements.txt
Python dependencies required for running the experiments. Replaced venv/ with this file to keep the repo lightweight.

## Dataset Generation
There already is a sample real-world dataset obtained from Kaggle: https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data 
This is saved in the dataset folder as SPX_clean.csv. Additionally, synthetic datasets can be generated using the file named synthetic_dataset_generator.py inside dataset/synthetic folder by specifying the desired file name. 

### synthetic_dataset_generator.py
WIP

## Optimizer Implementations
All of the optimizers are implemented in the file names starting with the name of the optimizer. 
- sgd – Stochastic Gradient Descent (SGD) for ridge regression with noise analysis and convergence logging.
- svrg – Stochastic Variance Reduced Gradient (SVRG), following Johnson & Zhang (2013).
- asg – Accelerated Stochastic Gradient (ASG), inspired by Assran & Rabbat (2020). Includes spectral radius and Pareto analysis
There are two versions for each optimizer, one is designed for 
