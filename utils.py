import numpy as np
from l2distance import l2distance
import itertools
def get_magnet_pad(N, m, M=None):
    """
	generate the mth pad in 2**(N*M) configurations
	e.g.:
	N=2
	m=1,arr = [-1,-1,-1,1]
    INPUT:
    N, integer, the first dimension of the pad
    M, integer, the second dimension of the pad

	OUTPUT:
	arr, a 1D numpy array
    """
    if M is None:
        M = N
    assert m < 2<<(N*M)
    arr = np.asarray(list([m >> i & 1 for i in range(N * M - 1,-1,-1)] )) * 2 - 1
    return arr

def gen_magnet_pads(N, M=None, start=0, end=None):
    """
	generate the all possible pad in 2**N*M configurations

    INPUT:
    N, integer, the first dimension of the pad
    M, integer, the second dimension of the pad
    start, integer, the start index of the pads, defalut 0
    end, integer, the end index of the pads, defalut 2**(N*M)
	OUTPUT:
	arr, an num x (N*M) array, where num = end - start
    """
    if M is None:
        M = N

    assert(N*M < 25)

    if end is None:
        end = 2**(N*M)
    index = np.arange(start, end, 1).astype('int32') 
    arr = (np.asarray([index >> i & 1 for i in range(N * M - 1,-1,-1)] )*2-1).astype('int8')
    arr = np.swapaxes(arr, 0, 1)
    return arr    

def gen_xyz(N,M=None,dx=0,dy=0):
    """
    generate x and y coordinates given N and M on a square lattice
    INPUT:
    N, integer, the first dimension of the pad
    M, integer, the second dimension of the pad, defalt is N
    dx, integer, the displacement on x direction
    dy, integer, the displacement on y direction
    OUTPUT:
    an array, N*M by 2
    """

    if M is None:
        M = N
    meshx = np.arange(N) + dx
    meshy = np.arange(M) + dy
    x,y = np.meshgrid(meshy,meshx)    
    pos = np.dstack((y  ,x )) 
    return pos.reshape(-1,2)

def gen_2D_pos(N1, M1 = None, N2=None, M2=None, dx = 0, dy = 0):
    """
    generate pariwise Rxi^2 + Ryj^2 between two pads 
    pos = [[(0,0),(0,1)],[(1,0),(1,1)]]
    input:
    N1, integer, the first dimension of pad1
    M1, integer, the second dimension of pad1, default: N1
    
    N2, integer, the first dimension of pad2, default: N1
    M2, integer, the second dimension of pad2, default: N1
    
    dx: integer, the displacement on x direction, default: 0
    dy: integer, the displacement on y direction, default: 0
       
    output:
    N x M by N x M array, with pairwise RX^2+RY^2 as array element 
    """    
    if M1 is None:
        M1 = N1
    if N2 is None:
        N2 = N1
    if M2 is None:
        M2 = M1
        
    num1 = N1 * M1
    num2 = N2 * M2
    
    pos1 = gen_xyz(N1,M1,dx=0,dy=0) 
    pos2 = gen_xyz(N2,M2,dx,dy)

    return l2distance(pos1.reshape(num1, 2) ,pos2.reshape(num2, 2) )


def gen_rectangular_xyz(N,da,db,M=None):
    """
    generate x and y coordinates given N and M on a rectangular lattice
    INPUT:
    N, integer, the first dimension of the pad
    M, integer, the second dimension of the pad, defalt is N
    da, stretching factor along x
    db, stretching factor along y
    OUTPUT:
    an array, N*M by 2
    """

    if M is None:
        M = N
    meshx = np.arange(N) * (1 + da)
    meshy = np.arange(M) * (1 + db)
    x,y = np.meshgrid(meshy,meshx)    
    pos = np.dstack((y  ,x )) 
    return pos.reshape(-1,2)


def gen_trianglar_xyz(N,theta=np.pi/6,M=None):
    """
    generate x and y coordinates given N and M on a triangular lattice
    INPUT:
    N, integer, the first dimension of the pad
    M, integer, the second dimension of the pad, defalt is N
    theta: angle, default is np.pi/6
    OUTPUT:
    an array, N*M by 2
    """

    if M is None:
        M = N
    xy=gen_xyz(M,N)
    move=np.copy(xy[:,0])
    xy[:,0] = 0
    newxy=np.dot(xy,np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))
    newxy[:,0] += move
 
    return newxy.reshape(-1,2)

def gen_2D_pos_general(pos1,pos2):
    """
    generate pariwise Rxi^2 + Ryj^2 between two poses 
    
    input:
    pos1 : M by 2 array, the x,y coordinates for pad1
    pos2 : N by 2 array, the x,y coordinates for pad2
    output:
    M by N array, with pairwise RX^2+RY^2 as array element 
    """ 
    return l2distance(pos1.reshape(-1, 2) ,pos2.reshape(-1, 2) )





def face_to_face_tensor(pos_2D, dlist):
    """
    given the pos_2D, generate the 3D interaction tensor
    INPUT:
    pos_2D: an 2D array of Rx^2 + Ry^2, the output of calling gen_2D_pos function
    dlist: a 1D array of distances
    
    OUTPUT:
    interaction_tensor: the interaction tensor
    """
    L1, L2 = pos_2D.shape
    pos_list = np.zeros([L1, L2, len(dlist)])
    for i in range(len(dlist)): 
        d_square = dlist[i] * dlist[i]
        R_square = pos_2D + d_square 
        pos_list[:,:,i] = ((R_square - 3 * d_square) / np.sqrt(R_square ** 5))
    return np.asfarray(pos_list)


def hinge_tensor(N, angle_list, radius):
    """
    generate pariwise Rxi^2 + Ryj^2 + d^2 between two N by N pads 
    INPUT:
    N, integer
    angle_list: a list of angles
    OUTPUT:
    pos_list: a list of N^2 by N^2 array, ith array is pos + ith distance in dlist. 
    """
    
    # generate N^2 by N^2 array, with coordinates on the plane
    N2 = N * N    
    mesh = np.arange(N)
    x,y = np.meshgrid(mesh,mesh)
    pos = np.dstack((y,x)).reshape(N2, 2)
    
    # pos_list is a list of N^2 by N^2 arrays as pairwise distance in space
    pos_list = np.zeros([N2, N2, len(angle_list)])
    for i in range(len(angle_list)): 
        pos_array = np.zeros([N2, N2])
        for j in range(len(pos)):
            for k in range(len(pos)):
                x1, y1 = pos[j]
                x2, y2 = pos[k]
                
                x1 = N - 1 - x1
                cos_ = math.cos(angle_list[i])
                sin_ = math.sin(angle_list[i])
                
                rx = (x1 + radius) * cos_ + x2 + radius
                ry = y2 - y1
                rz = -(x1 + radius) * sin_
                        
                r_squared = rx * rx + ry * ry + rz * rz
                numerator = 3 * (sin_ * rx + cos_ * rz) * rz / r_squared - cos_
                pos_array[j][k] = -1 * numerator / np.sqrt(r_squared ** 3)
        pos_list[:,:,i] = pos_array
    return np.asfarray(pos_list,'float64')


def face_to_face_force_tensor(pos_2D, dlist):
    """
    INPUT:
    generate pariwise d(Rxi^2 + Ryj^2 + D^2)/dD between two identical N by N pads 
    pos: N^2 by N^2 array
    dlist: a list of distances
    OUTPUT:
    pos_list: a list of N^2 by N^2 array, ith array is pos + ith distance in dlist. 
    """
    L1, L2 = pos_2D.shape
    pos_list = np.zeros([L1, L2, len(dlist)])
    for i in range(len(dlist)):
        d_square = dlist[i] * dlist[i]
        R_square = pos_2D + d_square
        pos_array = dlist[i] * (6.0 * d_square - 9.0 * pos_2D) / (R_square) ** 3.5
        pos_list[:,:,i] = pos_array
    return pos_list 


def hinge_force_explicit(x1,y1,x2,y2,theta):
    """
    check!!!
    given initial x1,y1 and x2,y2 of two dipoles, compute the force as a function of theta

    """
    cos_ = np.cos(theta)
    cos2_ = np.cos(2*theta)
    sin_ = np.sin(theta)
    de=(-(x1**2+x2**2+(y1-y2)**2)**2+0.5*x1*x2*(-10*(x1**2+x2**2+(y1-y2)**2)*cos_-x1*x2*(-29+cos2_)))*sin_
    nu=(x1**2+x2**2+(y1-y2)**2-2*x1*x2*cos_)**3.5
    return de/nu


def hinge_force_tensor(pos1, pos2, angle_list):
    """
    generate force tensor with respect to theta 
    INPUT:
    N, integer
    angle_list: a list of angles
    OUTPUT:
    pos_list: a list of N^2 by N^2 array, ith array is pos + ith distance in dlist. 
    
    """
    N1=len(pos1)
    N2=len(pos2)
    pos_list = np.zeros([N1, N2, len(angle_list)])
    for i in range(len(angle_list)): 
        pos_array = np.zeros([N1, N2])
        for j in range(len(pos1)):
            for k in range(len(pos2)):
                x1, y1 = pos1[j]                
                x2, y2 = pos2[k]
                pos_array[j][k] =  hinge_force_explicit(x1,y1,x2,y2,angle_list[i])
        pos_list[:,:,i] = pos_array

    return np.asfarray(pos_list,'float64')


def compute(pad1, poslist, pad2):
    """
    INPUT:
    pad1: a 1D array
    pad2: a 1D array
    poslist: a 3D array, M  by N by D, where D is the dimension for distances 
    OUTPUT:
    1d array
    """    
    return np.dot((np.dot(pad1, poslist).T),pad2)


def computeAll(pads1, poslist, pads2 = None):
    """
    INPUT:
    pads1: a 2D array,  shape of M1 by N1*M1, all combinations of N1 by M1 magnet pad
    pads2: a 2D array,  shape of M2 by N2*M2, all combinations of N2 by M2 magnet pad, default is pads1
    
    interaction_tensor_3D: a 3D array, M1 by M2 by D, where D is the dimension for distances
                           one should get this array from calling interaction_tensor function  
    OUTPUT:
    B, a 3D array, M1 by M2 by D
    """
    if pads2 is None:
        pads2 = pads1
    A=np.tensordot(pads1, poslist,axes=([1],[0]))
    B=np.tensordot(A, pads2.T,axes=([1],[0]))
    B=np.swapaxes(B,1,2)
    B=np.swapaxes(B,0,1)
    return B

def ComputeAll_2(pads1, poslist, pads2):
    """
    given two equal length pads, compute pes
    INPUT:
    pads1: a 2D array, shape of M by N^2
    pads2: a 2D array, shape of M by N^2
    poslist: a 3D array, M by M by D, where D is the dimension for distances 
    OUTPUT:
    B, a 3D array, M by D
    """
    assert len(pads1)==len(pads2)
    _,_,D=poslist.shape
    A = np.dot(pads1, poslist[:,:])
    B = np.repeat(pads2[:, :, np.newaxis], D, axis=2)
    B = np.sum(A * B, axis=1)
    return B




def gen_3D_dipole(u):
    """
    generate [ux,uy,uz]
    """
    dipole_3D = np.zeros([len(u), 3])
    dipole_3D[:,2] = u
    return dipole_3D

def gen_3D_position(pos):
    pos_3D = np.zeros([len(pos), 3])
    pos_3D[:,:2] = pos
    return pos_3D

def single_dipole_interaction(uA, uB, RA, RB):
    """
    compute the dipole-dipole interation energy
    E(A, B) = - \sum_{alpha, beta} \mu_{alpha}^A * T_alpha_beta * \mu_{beta}^B
    INPUT:
    uA, a 1D array, length of 3, the dipole moment vector of A
    uB, a 1D array, length of 3, the dipole moment vector of B
    RA, a 1D array, length of 3, the xyz coordinate of A    
    RB, a 1D array, length of 3, the xyz coordinate of B
    
    OUTPUT:
    energy, double
    """
    R = RA - RB
    energy = 0.0
    for alpha in range(3):
        for beta in range(3):
            energy += uA[alpha] * T_alpha_beta(R, alpha, beta) * uB[beta]
    return -1.0 * energy


def Tensor_alpha_beta(pos1, pos2, theta_list):
    """
    first dimension corresponds to theta_list
    last dimension are corresponding to xx, yy, zz, xy, xz, yz
    """
    
    tensor_list = np.zeros([len(theta_list),len(pos1), len(pos2), 9])
    for i_theta in range(len(theta_list)):
        R_xyzlist = []

        pos2_rotated = rotate(pos2, theta_list[i_theta])

        
        R_xyzlist.append(pos1[:,0:1] - pos2_rotated[:,0:1].T)
        R_xyzlist.append(pos1[:,1:2] - pos2_rotated[:,1:2].T)
        R_xyzlist.append(pos1[:,2:3] - pos2_rotated[:,2:3].T)
       
  
        R_squared = l2distance(pos1, pos2_rotated) 
        R_fifth =  R_squared  ** 2.5
    
        tensor_list[i_theta,:,:,0] = (3 * R_xyzlist[0] * R_xyzlist[0] - R_squared) / R_fifth #xx
        tensor_list[i_theta,:,:,1] = (3 * R_xyzlist[0] * R_xyzlist[1] ) / R_fifth #xy
        tensor_list[i_theta,:,:,2] = (3 * R_xyzlist[0] * R_xyzlist[2] ) / R_fifth #xz
    
        tensor_list[i_theta,:,:,3] = (3 * R_xyzlist[1] * R_xyzlist[0] ) / R_fifth #yx
        tensor_list[i_theta,:,:,4] = (3 * R_xyzlist[1] * R_xyzlist[1] - R_squared) / R_fifth #yy
        tensor_list[i_theta,:,:,5] = (3 * R_xyzlist[1] * R_xyzlist[2] ) / R_fifth#zz
        
        tensor_list[i_theta,:,:,6] = (3 * R_xyzlist[2] * R_xyzlist[0] ) / R_fifth #za
        tensor_list[i_theta,:,:,7] = (3 * R_xyzlist[2] * R_xyzlist[1] ) / R_fifth #zy
        tensor_list[i_theta,:,:,8] = (3 * R_xyzlist[2] * R_xyzlist[2] - R_squared) / R_fifth #zz
    
    return tensor_list


