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



def gen_problem(target, N, A, addnoise=True,sd=0.1):
    """
    INPUT: 
    target, a list of target values 
    N, integer, the size of the target
    A, a list of interaction tensors 
    
    OUTPUT:
    Pout, a N by 1 array
    Qout, a N by 1 array
    value, float, the difference between target and optimized results
    """
    N2 = N * N
    P = Variable((N2, 1),boolean=True)
    Q = Variable((N2, 1),boolean=True)
    
    W = []
    constraint = []
    

    for i in range(N2):
        W.append(Variable((N2, 1)))
        constraint.append(W[i] <= P[i] + Q)
        constraint.append(W[i] >= P[i] - Q)
        constraint.append(W[i] >= Q - P[i])
        constraint.append(W[i] <= 2 - P[i] - Q)
        
        
    objective = 0                   
    for d in range(len(target)):
        O=0.0
        for i in range(N2):

            O += multiply(( 1 - 2 * W[i]), A[d][i].reshape(N2,1))
           
            
        objective += sum_squares(sum(O) - target[d])

                       
    prob = Problem(Minimize(objective), constraint)
    prob.solve()
    Qout = Q.value.round(1)
    Pout = P.value.round(1)
    value = prob.value
    return Pout.reshape(-1)*2-1, Qout.reshape(-1)*2-1, value
