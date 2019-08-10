import numpy as np

def initialize_buckets(interaction_tensor_2D):
    """
    generate an array of unique dx^2 + dy^2 value, and corresponding mask array for given interaction_tensor_2D.
    
    INPUT:
    interaction_tensor_2D, a 2D array, the pairwise dx^2 + dy^2
    
    OUTPUT:
    unique_x2y2: a 1D array, contains the sorted unique value of dx^2 + dy^2 in the interaction_tensor_2D.
    mask_arrays: a 3D array, first dimension is len(unique_x2y2), each dimension corresponding to a dx^2+dy^2 value,
    each dimension contains a 2D binary array that has the same shape as interaction_tensor_2D, indicating the position 
    where interaction_tensor_2D has the dx^2+dy^2 value.
    """
    unique_x2y2 = np.unique(interaction_tensor_2D)
    mask_arrays = np.empty([len(unique_x2y2),interaction_tensor_2D.shape[0], interaction_tensor_2D.shape[1]])
    i=0
    for x2y2 in unique_x2y2:
        mask_arrays[i] = (interaction_tensor_2D == x2y2)
        i+=1
    return unique_x2y2, mask_arrays

def single_dipole_buckets(mask_arrays, outer_product):
    """
    given two pads, pads[i] and pads[j], and the idx_unique_x2y2, fill in the buckets.
        
    INPUT:
    idx_unique_x2y2, a 3D array, computed from initialize_buckets function
    pads: a list of pads
    i, integer, the index of the first pad
    j, integer, the index of the second pad
    
    OUTPUT:
    buckets: a 1D array, same shape as idx_unique_x2y2, k-th element represents the net number of k-th dx^2+dy^2
    appears in given pads interactions. 
    """
    buckets = np.zeros([len(mask_arrays)])
    buckets[0] = np.sum(np.diag(outer_product))
    
    outer_product = outer_product.reshape(1, len(outer_product), len(outer_product))
    
    buckets[1:] = np.sum(np.sum(outer_product * mask_arrays[1:], axis=1), axis=1)
    
    return buckets 

def single_dipole_energy(x2y2, z):
    """
    Given the dx^2 + dy^2 and z for two dipoles, calculate the pes for those two
    INPUT:
    x2y2, integer, the dx^2+dy^2 between two dipoles
    z, a 1D array of float number, the distance in z direction for two dipoles
    
    OUTPUT:
    1D array of pes
    """
    return -(2.0 * z**2 - x2y2) / ((x2y2 + z**2)**(2.5))

def single_dipole_force(x2y2, z):
    """
    Given the dx^2 + dy^2 and z for two dipoles, calculate the dE/dD for those two
    INPUT:
    x2y2, integer, the dx^2+dy^2 between two dipoles
    z, a 1D array of float number, the distance in z direction for two dipoles
    
    OUTPUT:
    1D array of pes
    """    
    return (9.0 * x2y2 * z - 6.0 * z ** 3) / ((x2y2 + z**2)**(3.5))

def combination_of_single_dipole(unique_x2y2, buckets, dlist, myfunc):
    """
    Given the unique dx^2+dy^2 list and the filled buckets, and the distance list, return the linear combination
    of single dipole-dipole pes or force.
    INPUT:
    unique_x2y2: a 1D array, contains the sorted unique value of dx^2 + dy^2 in the interaction_tensor_2D.
    buckets: a 1D array, same shape as idx_unique_x2y2, k-th element represents the net number of k-th dx^2+dy^2
    appears in given pads interactions. 
    dlist, a 1D array of distances.
    myfunc: single_dipole_force or single_dipole_energy
    
    OUTPUT:
    1D array of pes for the given buckets.
    """
    res = np.zeros(len(dlist))
    for i in range(len(unique_x2y2)):
        res += buckets[i] * myfunc(unique_x2y2[i], dlist)
    return res
