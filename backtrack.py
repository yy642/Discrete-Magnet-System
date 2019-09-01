from Buckets import *

def combination_unique_pads(N, idxlist, three_states=False):
    """
    Given a set of force, find a subset set S of it,
    such that sum(abs(sign_i F[S_i])) < eps, and sum(sign_i E[S_i]) < targetE)
    
    return:
    (1) a list of selected index : S_i
    (2) a list of corresponding sign for S_i: sign_i

   
    force: a 1-D array
    energy: a 1-D array
    
    targetF: the target value of net force
    targetE: the upper bound of sum of energy
    
    three_states: boolean
    
    eps: the maxium difference between the solution sum with the target.
    
    """
    pad1 = np.zeros([N * N])
    pad2 = np.zeros([N * N])
    global base_bucket
    base_bucket = outerToBucket(np.outer(np.ones([N*N]),np.ones([N*N])),idxlist)
    res=set()
    curlist = []
    curF = 0
    curE = 0
    dfs_unique_pads(0, pad1, pad2, res, np.zeros([N*N, N*N])) 
    
    return res


def dfs_unique_pads(idx, pad1, pad2, res, outerproduct):
    global idxlist,base_bucket

    if (idx == len(pad1)):
        bucket = outerToBucket(outerproduct,idxlist)
        res.add(hash_bucket(bucket, base_bucket ))
        return
   
    
    pre = outerproduct
    
    pad1[idx]=1
    pad2[idx]=1
    outerproduct[idx] = pad1[idx] * pad2
    outerproduct[:,idx] = pad2[idx] * pad1
    outerproduct[idx][idx]=1
    dfs_unique_pads(idx+1,pad1,pad2, res, outerproduct)
    
    pad2[idx]=-1
    outerproduct[idx] = pad1[idx] * pad2
    outerproduct[:,idx] = pad2[idx] * pad1
    outerproduct[idx][idx]=-1
    dfs_unique_pads(idx+1,pad1,pad2,res, outerproduct)
    
    pad1[idx]=-1
    outerproduct[idx] = pad1[idx] * pad2
    outerproduct[:,idx] = pad2[idx] * pad1
    outerproduct[idx][idx]=1
    dfs_unique_pads(idx+1,pad1,pad2,res, outerproduct)    
    
    pad2[idx]=1
    outerproduct[idx] = pad1[idx] * pad2
    outerproduct[:,idx] = pad2[idx] * pad1
    outerproduct[idx][idx]=-1
    dfs_unique_pads(idx+1,pad1,pad2,res, outerproduct)
    
    pad1[idx]=0
    pad2[idx]=0
    outerproduct = pre


