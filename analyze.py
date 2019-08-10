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













def get_data(pads1,extrema,pads2=None):
    if pads2 is None:
        pads2 = pads1
    L = extrema.shape[0]
    pair = np.meshgrid(np.arange(L), np.arange(L))
    data = np.zeros([L * L,6])
    data[:,0] = pair[1].flatten()
    data[:,1] = pair[0].flatten()
    data[:,2] = np.sum(pads1[pair[1].flatten()],axis=1)
    data[:,3] = np.sum(pads2[pair[0].flatten()],axis=1)
    data[:,4] = np.dot(pads1,pads2.T).flatten()
    data[:,5] = extrema.flatten()
    return data

def get_statistics_large_input(count, restriction, data,mag,target_list):
    for i in range(len(mag)):
        for j in range(len(mag)):
            for k in range(len(mag)):
                restriction_idx = (data[:,2] == mag[i]) * (data[:,3] == mag[j]) * (data[:,4] == mag[k])
                restriction[i,j,k] += np.sum(restriction_idx)

                for t in range(len(target_list)):
                    count[t,i,j,k] += np.sum(restriction_idx * (data[:,5] == target_list[t]))

    
    return 

def partition(target_list, N, group_size):
    global interaction_tensor_3D
    N2 = N*N
    mag = np.arange(-N2, N2+2,2)
    count = np.zeros([len(target_list),len(mag), len(mag),len(mag)])
    restriction = np.zeros([len(mag), len(mag),len(mag)])
    n_group = int(2**(N2) / group_size)
    for i in range(n_group):
        s1 = i * group_size
        e1 = (i + 1) * group_size
        pads1 = gen_magnet_pads(N, start=s1, end=e1)
        
        for j in range(n_group):
            s2 = j * group_size
            e2 = (j + 1) * group_size
            pads2 = gen_magnet_pads(N, start=s2, end=e2)
            pes = tensor(pads1, interaction_tensor_3D,pads2)
            extrema = find_extrema(pes)
            data = get_data(pads1, extrema, pads2)
            get_statistics_large_input(count, restriction, data, mag, target_list)
    return count, restriction

def gen_pattern_map(N, PESmap, pattern_map={}, pattern_pad_map={}):
    """
    go through PESmap, get the pattern of the PES, defined as
    np.argsort(PES)
    INPUT:
    PESmap   : dict, key<string, Pattern>, value<list(tuple(idx1, idx2))>
    N        : integer, size of the pad
    pattern_map: 
    dict, key<string, Pattern>, value<list of arrays PES>
    defalut: an empty dict
    pattern_pad_map:
    dict, key<string, Pattern>, value<list of tuple(idx1, idx2)>
    defalut: an empty dict
    OUTPUT:
    pattern_map, pattern_pad_map
    """
    for key in PESmap.keys():
        indexes = PESmap[key]
        l=key.split("_")
        l2=[float(n) for n in l if n != '' ]
        pattern = str(np.argsort(l2))
        if pattern not in pattern_map:
            pattern_map[pattern] = list()
            pattern_pad_map[pattern]=list()
        pattern_map[pattern].append(l2)
        pattern_pad_map[pattern].append(indexes)
    return pattern_map, pattern_pad_map

def rotate_pad(pad, N):
    """
    INPUT:
    pad: 
    N: Integer
    OUTPUT:
    a list of rotated pad, with rotation angle = 90, 180 and 270.
    
    """
    if len(pad) != N:     
        pad = pad.reshape(N,N)
    pad1 = np.rot90(pad, 1).flatten()
    pad2 = np.rot90(pad, 2).flatten()
    pad3 = np.rot90(pad, 3).flatten()
    return [pad, pad1, pad2, pad3]
