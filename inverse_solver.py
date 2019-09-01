"""
This code provides a method to solve the unknown x, y staisify the following set of equations:
x^T A^k y = b^k (1)
with k=1,2,...,N
A_k, b_k are known matrices, and x, y are two vectors with vector elements ∈ {-1, 1}.
The above quadratic problem can be linearized as:
(a) convert X, Y to to binary variable:
    let pi, qi ∈{0,1}, then the element of x and y can be represented using p and q
    xi = 2pi - 1
    yi = 2qi - 1
(b) rewrite equation (1) as:
   \sum_{i,j} x_i y_j A_{i,j}^k = \sum_{i, j} (1-2(pi xor qi)) A_{i,j}^k = b^k
   Let w_{i,j} = pi xor qi, then the above equation can be linearzed as:
   \sum_{i,j} (1-2w_{i,j})A_{i,j}^k = b^k
   with a set of constriants:
   w_{i,j} <= pi+qj
   w_{i,j} >= pi-qj
   w_{i,j} >= qj-pi
   w_{i,j} <= 2-pi-qj
"""



import numpy as np
import time
from l2distance import l2distance
import numpy.random as random
from utils import *
from cvxopt.modeling import *
from cvxpy import *


def opt(target, N, A, eps=100, maxiter=1):
    """
    inverse solver
    
    INPUT:
    target, 1D array of pes
    N, the length of first dimenson of the square pad
    A, the interaction tensor, the last dimension is the distance
    eps, the error threshold
    maxiter: the maximum number of iteration
    
    OUTPUT:
    two 1D array, represting two flattened pads
    error, float number.
    """
    N2 = N * N
    P = Variable((N2, 1), boolean=True)
    Q = Variable((N2, 1), boolean=True)
    W = Variable((N2, N2))
    
    P_T_matrix = vstack([P.T] * N2)
    Q_matrix = hstack([Q] * N2)
    
    P_add_Q = P_T_matrix + Q_matrix
    p_sub_Q = P_T_matrix - Q_matrix
    
    constraint = []
    
    constraint.append(W <=  P_add_Q)
    constraint.append(W >=  p_sub_Q)
    constraint.append(W >= -p_sub_Q)
    constraint.append(W <= -P_add_Q + 2)
     
    objective = 0
    
    W_tmp = 1 - 2 * W
    
    for d in range(len(target)):
        objective += sum_squares(sum(multiply(W_tmp, A[:,:,d])) - target[d])
        
    prob = Problem(Minimize(objective), constraint)
    err = 1000

    
    prob.solve()
    err = prob.value
    besterr = prob.value
    bestQ = Q.value.round(1)
    bestP = P.value.round(1)
    iteration = 1
    while (err > eps):
        iteration += 1
        if (iteration > maxiter): break
        prob.solve(warm_start=True)
        err = prob.value
        if err > besterr:
            besterr = err
            bestQ = Q.value.round(1)
            bestP = P.value.round(1)

    print("err=",err, "iteration=", iteration)
    return bestP.reshape(-1)*2-1, bestQ.reshape(-1)*2-1, besterr



