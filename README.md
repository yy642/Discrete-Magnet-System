# Discrete-Magnet-System
## Installation
To install `cvxpy` (if you have Anaconda),
```
conda install -c conda-forge lapack
conda install -c cvxgrp cvxpy
```
You can test the installation of `cvxpy` with `nose`:
```
conda install nose
nosetests cvxpy
```
##
This project provides methods to model the interaction between two arrays of magnetic dipole with {-1,1} representing the magnetic orientation for each dipole (-1 means north, 1 means south). 


## For users:
**example.ipynb**        : a sample code showing how to use:
* compute the force and energy between given two magnetic arrays with {-1,1} representing the magnetic dipole orientations*
* enumerate all possible interactions between two magnetic arrays using vectorization computing *
* plot the number of extrema in configuration space *
* using inverse_solver to back out the magnetic array that satisify the target interaction *

**NN_forward_inverse.py**: Using NN in keras/Tensorflow to model both foward and inverse problem.
**backtrack.py**         : Using backtrack/DFS to enumerate all possible interactions between two magnetic arrays.
**inverse_solver.py**    : Given the target interaction (force or energy), compute the configuration of magnetic arrays.
**Analyze.py**           : Given the computed interaction, extract useful infomation, such as number of extremas.
**visualization.py**     : Given magnet arrays, plot the configurations.
