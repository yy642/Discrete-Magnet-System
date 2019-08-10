import numpy as np
def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    def innerproduct(X,Z=None):
        # function innerproduct(X,Z)
        #
        # Computes the inner-product matrix.
        # Syntax:
        # D=innerproduct(X,Z)
        # Input:
        # X: nxd data matrix with n vectors (rows) of dimensionality d
        # Z: mxd data matrix with m vectors (rows) of dimensionality d
        #
        # Output:
        # Matrix G of size nxm
        # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
        #
        # call with only one input:
        # innerproduct(X)=innerproduct(X,X)
        #
        if Z is None: # case when there is only one input (X)
            Z=X;
        
        G = np.dot(X, Z.T)
     
        return G
    
    preS = np.diag(innerproduct(X)).reshape(-1,1)
    preR = np.diag(innerproduct(Z)).reshape(1,-1)
    D2   = preS - 2 * innerproduct(X,Z) + preR
    D    = np.sqrt(D2)
    
    return D2

