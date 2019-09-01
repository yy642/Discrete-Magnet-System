import numpy as np
from scipy.signal import argrelextrema

def find_extrema(pes,prec=10):
    """
    INPUT:
    pes, a M1 by M2 by D array
    
    OUTPUT:
    extremas, a M1 by M2 array, with extremas[i,j] is the number of extremas found for ith pad1 with jth pad2
    
    """
    
    pes = np.round(pes, prec)
    extremas = np.zeros([pes.shape[0], pes.shape[1]])
    
    minimas=np.array(argrelextrema(pes, np.less, axis=2)).T
    maximas=np.array(argrelextrema(pes, np.greater, axis=2)).T
    
    if minimas.shape[0] != 0:
        unique_minima_index, unique_minima_count =np.unique(minimas[:,:2], return_counts=True, axis=0)
        extremas[unique_minima_index[:,0],unique_minima_index[:,1]] += unique_minima_count
    if maximas.shape[0] != 0:
        unique_maxima_index, unique_maxima_count = np.unique(maximas[:,:2], return_counts=True, axis=0) 
        extremas[unique_maxima_index[:,0],unique_maxima_index[:,1]] += unique_maxima_count
    return extremas

def extrema_info(pes,dlist,prec=10,precD=1,precE=3):
    """
    given a 1d array of pes, find all the extremas as  (distance, energy) tuple, stored in a tuple
    input: 
    pes: 1d array of pes
    dlist: 1d array of distance, same dimension as pes
    output:
    a tuple of tuple
    """

    assert len(pes) == len(dlist)
    info=[]
    pes = np.round(pes,prec)
    for i in range(1, len(pes)-1):
        if ((pes[i]-pes[i - 1]) > 0 and (pes[i] - pes[i + 1]) > 0) or ((pes[i]-pes[i - 1]) < 0 and (pes[i] - pes[i + 1]) < 0):
            info.append(tuple([np.round(dlist[i],precD),np.round(pes[i],precE)]))
    return tuple(info)

def find_number_of_nodes(pes):
    """
    INPUT:
    pes, a M-D array, last dimension is the distance
    OUTPUT:
    a (M-1)-D array,  each element is the number of nodes   
    """
    mask = pes > 0
    if len(pes.shape) == 3:
        return np.sum(mask[:,:,:-1] ^ mask[:,:,1:], axis = -1)
    elif len(pes.shape) == 2:
        return np.sum(mask[:,:-1] ^ mask[:,1:], axis = -1)
    elif len(pes.shape) == 1:
        return np.sum(mask[:-1] ^ mask[1:], axis = -1)

def find_unique_index(pes,prec=10):
    """
    precision....
    find the unique pes
    INPUT:
    pes, a 3d array, with shape of M x N x D
    OUTPUT:
    (1) a 2d array, shape of g x D, where g is the number of unique pes   
    (2) a 2d array, pair index of the pads that generate the unique pes.
    """
    pes_copy = np.round(pes,prec)
    if (len(pes.shape) == 3):
        m, n ,D = pes.shape
        _,unique_index = np.unique(pes_copy.reshape(m * n, D), axis = 0, return_index = True)
    elif (len(pes.shape) == 2):
        m, D = pes.shape
        _,unique_index = np.unique(pes_copy, axis=0, return_index = True)
    else:
        print("wrong dimension", pes.shape)
        return
    unique_index = np.array([unique_index // m, unique_index % m])
    unique_index = np.swapaxes(unique_index, 0, 1)
    return unique_index

