import numpy as np
from numpy import linalg as LA
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
    #assert m < 2<<(N*M)
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

def gen_magnet_pads_from_index(N, index, M=None):
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
    #index = np.arange(start, end, 1).astype('int32') 
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
    return pos_list.astype('float32') 

def face_to_face_force_derivative_tensor(pos_2D, dlist):
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
        pos_list[:,:,i] = -(9 * pos_2D * pos_2D - 72 * pos_2D * d_square + 24 * d_square**2) / (R_square) ** 4.5 

    return pos_list.astype('float32') 

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

def Compute(pad1, Tensor, pad2):
    """ 
    INPUT:
    pad1: a 1D array
    pad2: a 1D array
    poslist: a 3D array, M  by N by D, where D is the dimension for distances 
    OUTPUT:
    1d array
    """   
    return np.dot(np.dot(pad2,Tensor).T,pad1)

def compute(pad1, poslist, pad2):
    """
    disactiviated, see Compute(pad1, Tensor, pad2)
    """    
    print("disactiviated, see Compute(pad1, Tensor, pad2)")
    return


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
            energy += uA[alpha] * Tensor_alpha_beta(RA,RB, alpha, beta) * uB[beta]
    return -1.0 * energy


def T_alpha_beta(pos1, pos2, theta_list):
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

def get_rotate_matrix(theta,alpha):
    """
    the rotation matrix along alpha axis
    alpha: 0:x, 1:y, or 2:z
    """
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    if alpha == 0:
        return np.array([[1,0,0],
                        [0,cos_,-sin_],
                        [0,sin_,cos_]])
    elif alpha == 1:
        return np.array([[cos_, 0, -sin_],
                    [0, 1, 0],
                    [sin_, 0, cos_]])
    elif alpha == 2:
        return np.array([[cos_,-sin_,0],
                       [sin_,cos_,0],
                       [0,0,1]])
    else :
        print("invalid alpha")


def get_dipole_force(r1, r2, m1, m2, mu0 = 1):
    """
    compute magnetic force felt by m1 located on r1
    INPUT:
    r1: 1D array with len 3, the (x,y,z) for magnet 1
    r2: 1D array with len 3, the (x,y,z) for magnet 2
    m1: 1D array with len 3, the (mu_x,mu_y,mu_z) for magnet 1
    m2: 1D array with len 3, the (mu_x,mu_y,mu_z) for magnet 2
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    OUTPUT:
    force flet by m1, 1D array with len 3
    """        
    r = r1 - r2
    rr = LA.norm(r) # the length of r
    dr = r / rr # unit direction of r  
    r_x_m1 = np.cross(dr, m1)
    r_x_m2 = np.cross(dr, m2)
    F = 3*mu0 /(4*np.pi*rr**4) * (np.cross(r_x_m1, m2) + np.cross(r_x_m2, m1) - 2 * dr * np.dot(m1, m2) + 5 * dr *(np.dot(r_x_m1, r_x_m2)))
    return F

def get_dp_dp_energy(r1, r2, m1, m2, mu0=1):
    """
    Energy between m1(located at r1) and m2(located at r2)
    INPUT:
    r1: 1D array with len 3, the (x,y,z) for magnet 1
    r2: 1D array with len 3, the (x,y,z) for magnet 2
    m1: 1D array with len 3, the (mu_x,mu_y,mu_z) for magnet 1
    m2: 1D array with len 3, the (mu_x,mu_y,mu_z) for magnet 2
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    OUTPUT:
    interaction energy, 1D array with len 3

    """
    r = r1 - r2
    rr = LA.norm(r) # the length of r
    dr = r / rr # unit direction of r    
    return -mu0/(4 * np.pi * rr**3) *(3 * np.dot(m1, dr) * np.dot(m2, dr) - np.dot(m1, m2))

def B(r1, r_origin, m, mu0=1):
    """
    magnetic field at r1, generated by m located at r_origin, 
    INPUT:
    r1: 1D array with len 3, the (x,y,z) for magnet 1
    r_origin: 1D array with len 3, the (x,y,z) for the test dipole
    m: 1D array with len 3, the (mu_x,mu_y,mu_z) for magnet at r_origin 
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    OUTPUT:
    magnetic field at r1, 1D array with len 3

    """
    r = r1 - r_origin
    rr = LA.norm(r) # the length of r
    return mu0 / (4 * np.pi) * (3*r*np.dot(m, r)/rr**5 - m / rr**3)


def get_torque(r, F):
    return np.cross(r, F)

def gen_hinge_xyz(N, M, d, dx=0, dy=0):
    """
    generate (x,y,z) coordinate for N by M grids (with z=0)
    dx: absolute shift in x direction 
    dy: absolute shift in y direction
    d: grid space between adjacent magnets
    """
    xy=gen_xyz(N,M)
    xyz = np.zeros([len(xy),3])
    xyz[:,:2]=xy
    xyz = (xyz * d )+ np.array([dx, dy, 0])
    return xyz

def hinge_energy_tensor(xyz1, xyz2, angles, mu0=1, Mdipole=1):
    """
    interaction energy tensor between two arrays of magnets located at xyz1, and xyz2
    INPUT:
    xyz1: 2D array (x,y,z) for pad1
    xyz2: 2D array (x,y,z) for pad2
    angles: 1D array, angle between two pads, by defulat, pad1 is rotated about y axis, and angle is how
    much the pad1 is rotated
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    Mdipole: the dipole strength of the magnet, default is 1, if compared with experiment, reset.
    OUTPUT:
    3d ARRAY.

    """
    tensor = np.zeros([len(xyz1),len(xyz2),len(angles)])
    for i in range(len(angles)):
        theta = angles[i] 
        rot_y = get_rotate_matrix(theta, alpha=1) 
 
        for j in range(len(xyz1)):
            mu1 =  np.array([0,0,1]) * Mdipole
            r1 = xyz1[j]
            new_r1 = np.dot(r1, rot_y)
            new_mu1 = np.dot(mu1, rot_y)
            
            for k in range(len(xyz2)):
                mu2 =  np.array([0,0,1]) * Mdipole
                r2 = xyz2[k]                                
                tensor[j][k][i] = get_dp_dp_energy(new_r1, r2, new_mu1, mu2, mu0=mu0)
    return tensor

                
def hinge_Fz_tensor(xyz1, xyz2, angles, mu0=1,Mdipole=1):
    """
    interaction force (in z direction) tensor between two arrays of magnets located at xyz1, and xyz2
    INPUT:
    xyz1: 2D array (x,y,z) for pad1
    xyz2: 2D array (x,y,z) for pad2
    angles: 1D array, angle between two pads, by defulat, pad1 is rotated about y axis, and angle is how
    much the pad1 is rotated
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    Mdipole: the dipole strength of the magnet, default is 1, if compared with experiment, reset.
    OUTPUT:
    3d ARRAY.

    """

    tensor = np.zeros([len(xyz1),len(xyz2),len(angles)])
    for i in range(len(angles)):
        theta = angles[i] 
        rot_y = get_rotate_matrix(theta, alpha=1) 
        for j in range(len(xyz1)):
            mu1 =  np.array([0,0,1]) * Mdipole
            r1 = xyz1[j]
            new_r1 = np.dot(r1, rot_y)
            new_mu1 = np.dot(mu1, rot_y)
            for k in range(len(xyz2)):
                mu2 =  np.array([0,0,1]) * Mdipole
                r2 = xyz2[k]   
                F2 =  get_dipole_force(r2, new_r1, mu2, new_mu1, mu0=mu0) # force on r2
                tensor[j][k][i] = F2[-1]
    return tensor

                
def hinge_tau_tensor(xyz1, xyz2, angles, mu0=1,Mdipole=1):
    """
    total tau
    interaction torque tensor between two arrays of magnets located at xyz1, and xyz2
    tau includes two terms: rxF and mxB
    INPUT:
    xyz1: 2D array (x,y,z) for pad1
    xyz2: 2D array (x,y,z) for pad2
    angles: 1D array, angle between two pads, by defulat, pad1 is rotated about y axis, and angle is how
    much the pad1 is rotated
    mu0: Vacuum permeability, default is set mu0 to be 1. 
    if compared with experiment, reset mu0
    Mdipole: the dipole strength of the magnet, default is 1, if compared with experiment, reset.
    OUTPUT:
    3d ARRAY.

    """
    tensor = np.zeros([len(xyz1),len(xyz2),len(angles)])
    for i in range(len(angles)):
        theta = angles[i] 
        rot_y = get_rotate_matrix(theta, alpha=1) 
        for j in range(len(xyz1)):
            mu1 =  np.array([0,0,1]) * Mdipole
            r1 = xyz1[j]
            new_r1 = np.dot(r1, rot_y)
            new_mu1 = np.dot(mu1, rot_y)
            for k in range(len(xyz2)):
                mu2 =  np.array([0,0,1]) * Mdipole
                r2 = xyz2[k]   
                F1 =  get_dipole_force(new_r1, r2, new_mu1, mu2, mu0=mu0) # force on r1
                t1 = get_torque(new_r1, F1) # torque on r1
                B_ = B(new_r1, r2, mu2, mu0=mu0)# magnetic field generated by mu2 at new_r1:        
                t2 = get_torque(new_mu1, B_)# torque on r1 from field 
                tensor[j][k][i] = t1[1] + t2[1]
    return tensor

