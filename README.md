# Invertible Neural Network for 1D Resistivity

This repository demonstrates a basic workflow for building, training, and using an invertible neural network (INN) to invert 1D resistivity data from electromagnetic (EM) measurements.

## Contents

1. **`TrainINN_1D.py`**  
   - Main script for training the INN on resistivity and EM data.

2. **`utilities.py`**  
   - Helper functions for data preparation, metrics, and splitting datasets.

3. **`network.py`**  
   - Defines the INN architecture and handles training loops, losses, and optimizers.

4. **`SolverComparison.py`**  
   - Loads the trained network and compares its inversion results to other inversion methods (e.g., LMA, Occam’s).

5. **`plotting.py`**  
   - Various plotting routines for visualizing true vs. inverted resistivity, uncertainty, and misfit metrics.
