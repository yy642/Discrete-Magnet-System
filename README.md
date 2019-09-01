# Discrete-Magnet-System
## Installation
to install cvxpy
```
pip install cvxpy
```
##
This project provides methods to model the interaction between two arrays of magnetic dipole with {-1,1} representing the magnetic orientation for each dipole (-1 means north, 1 means south). 

(1) compute the force and energy between given two magnetic arrays with {-1,1} representing the magnetic dipole orientations.

(2) Using backtrack/DFS to enumerate all possible interactions between two magnetic arrays, see backtrack.py.

(2) Given the target interaction (force or energy), compute the configuration of magnetic arrays, see inverse_solver.py

(3) Using NN in keras/Tensorflow to model both foward and inverse problem, see NN_forward_inverse.py

The example.ipynb contains the example of how to use core funcitons.
