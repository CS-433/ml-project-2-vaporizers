# Project 2 Machine Learning (CS-433)
# ML4Science: Data-driven surrogate modeling of the hemodynamics by prediction of POD coefficients

Authors: Francesca Bettinelli, Maddalena Detti, Antonio Tirotta

### Project description

The aim of this project is to develop a fully connected neural network to find a data-driven reduced-order solution for the time-dependent Navier-Stokes equations within a blood vessel. In particular, the network learns the map between the physical parameters $\boldsymbol{\mu}$ and the Proper Orthogonal Decomposition (POD) space-time coefficients for both velocity and pressure.

The baseline models for both velocity and pressure are inspired by the implementation proposed in [Hesthaven et al.](https://doi.org/10.1016/j.jcp.2018.02.037) for the steady-state Navier-Stokes equations in a 2D domain. In particular, we start from a fully connected neural network (MLP) with two hidden layers and experiment with different architectures by varying the number of layers, the number of neurons per layer, and the activation functions.

For the choice of the loss function, we adopt two different approaches: (1) a standard MSE loss, and (2) a weighted MSE loss embedding the physical information contained in the singular values computed during the POD procedure. The first approach significantly outperforms the second one.

We train two separate models for velocity and pressure. In both cases, the best model contains 4 hidden layers, 256 neurons per layer, the GELU activation function, and layer normalization. We achieve a test error of $1.6\\%$ and $1.3\\%$ respectively.

### Project structure

This repository contains the following files:
- [load_data.py](load_data.py): Python file containing a function to load the dataset;
- [models.py](models.py): Python file containing the classes for the final models of velocity and pressure;
- [training.py](training.py): Python file containing the main training routines both for the standard and the weighted MSE losses;
- [run.ipynb](run.ipynb): Jupyter Notebook to train the final models with the standard loss;
- [run_weighted.ipynb](run_weighted.ipynb): Jupyter Notebook to train the final models with the weighted loss;
- [run_cross_validation.ipynb](run_cross_validation.ipynb): Jupyter Notebook to perform a 5-fold cross-validation with the standard loss;
- [run_cross_validation_weighted.ipynb](run_cross_validation_weighted.ipynb): Jupyter Notebook to perform a 5-fold cross-validation with the weighted loss;
- [requirements.txt](requirements.txt): file containing the dependencies needed to run the notebooks.

### How to run

You can access the dataset at this [SWITCHDrive link](https://drive.switch.ch/index.php/s/FT3uQF4gNXrtgaO). Download the ```dataset``` folder, unzip it, and place it in the repository. Make sure that ```dataset``` contains the folders ```basis``` and ```RB_data``` (WITHOUT another ```dataset``` nested folder).

If you want to run the notebooks on your local machine, we recommend creating a conda environment with Jupyter Notebook installed. Then, install the required libraries in the environment by running:
```
pip install -r requirements.txt
```

If you want to run the notebooks on Google Colab, copy the repository on your Google Drive and, at the beginning of the notebook, allow Colab to access the other files in the repository by adding the following cell:
```
from google.colab import drive
drive.mount('/content/gdrive')
```

Our implementation supports the usage of a GPU to accelerate the training phase. In this setting, you can expect a training time for a single model between 1 and 2 minutes. Instead, if you use a CPU, you can expect a training time between 3 and 5 minutes.

**Disclaimer**: all our experiments have been performed with the NVIDIA Tesla T4 GPU provided by Google Colab. Reproducibility is not guaranteed on other hardware.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13272957&assignment_repo_type=AssignmentRepo)
