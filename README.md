# Project 2 Machine Learning (CS-433)
# ML4Science: Data-driven surrogate modeling of the hemodynamics by prediction of POD coefficients

Authors: Francesca Bettinelli, Maddalena Detti, Antonio Tirotta

### Project description

The aim of this project is to develop a fully connected neural network to find a data-driven reduced-order solution for the time-dependent Navier-Stokes equations within a blood vessel. In particular, the network learns the map between the physical parameters $\boldsymbol{\mu}$ and the Proper Orthogonal Decomposition (POD) space-time coefficients for both velocity and pressure.

The baseline models for both velocity and pressure are inspired by the implementation proposed in [Hesthaven et al.](https://doi.org/10.1016/j.jcp.2018.02.037) for the steady-state Navier-Stokes equations in a 2D domain. In particular, we start from a fully connected neural network (MLP) with two hidden layers and experiment with different architectures by varying the number of layers, the number of neurons per layer, and the activation functions.

For the choice of the loss function, we adopt two different approaches: (1) a standard MSE loss, and (2) a weighted MSE loss embedding the physical information contained in the singular values computed during the POD procedure. The first approach significantly outperforms the second one.

Our best models are the following:
- Velocity: 4 hidden layers, 256 neurons per layer, GELU activation function, layer normalization;
- Pressure: 3 hidden layers, 256 neurons per layer, GELU activation function, layer normalization.

### Project structure

This repository is structured as follows:
- [load_data.py](load_data.py): Python file containing a function to load the dataset;
- [models.py](models.py): Python file containing the classes for the final models (of velocity and pressure);
- [training.py](training.py): Python file containing the main training routines both for the standard and the weighted MSE losses;
- [run.ipynb](run.ipynb): Jupyter notebook to train the final models with the standard loss;
- [run_weighted.ipynb](run_weighted.ipynb): Jupyter notebook to train the final models with the weighted loss;
- [run_cross_validation.ipynb](run_cross_validation.ipynb): Jupyter notebook to perform a 5-fold cross-validation with the standard loss;
- [run_cross_validation_weighted.ipynb](run_cross_validation_weighted.ipynb): Jupyter notebook to perform a 5-fold cross-validation with the weighted loss;
- [requirements.txt](requirements.txt): file containing the dependencies needed to run the notebooks.

### How to run

You can access the dataset at this [SWITCHDrive link](https://drive.switch.ch/index.php/s/FT3uQF4gNXrtgaO). Download the ```dataset``` folder, unzip it, and place it in the repository.

run locale vs colab (reproducibility su colab)
tempi di training 



[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13272957&assignment_repo_type=AssignmentRepo)
