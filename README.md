# Discrete-Magnet-System
This project provides methods to model the interaction between two arrays of magnetic dipole with {-1,1} representing the magnetic orientation for each dipole (-1 means north, 1 means south). 

(1) compute the force and energy between given two magnetic arrays with {-1,1} representing the magnetic dipole orientations.

(2) Using backtrack/DFS to enumerate all possible interactions between two magnetic arrays.

(2) Given the target interaction (force or energy), compute the configuration of magnetic arrays.

(3) Using NN in keras/Tensorflow to model both foward and inverse problem

The example.ipynb contains the example of how to use core funcitons.

utils.py contains the core function, including compute the force/energy based on the magneitc array configuration and coordinates, enumerate all combinations of magnetic array, compute energy/force for a large batch of magbetic arrays, etc...
