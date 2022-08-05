import numpy as np
import scipy as sp
from sympy import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker

from scipy import optimize


mu0 = 4*np.pi*1e-7 

#-------------------------------------------------------------elastic truss functions------------------------------------------------#

def rotation(ang, points):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in radian).
        points: positions of points need rotating, 2D arrary [num_pts, 3].
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    rot = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    points_rot = np.matmul(rot, points.T)
    return points_rot.T

def get_adj(connectivity):
    """Get adjacency matrix from connectivity.

    PARAMETERS
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        
    RETURN
        adj: 2D array, [num_pts, num_pts].
    """
    num_pts = len(connectivity)
    adj = np.zeros([num_pts, num_pts])
    for pair in connectivity:
        adj[pair[0], pair[1]] = 1
        adj[pair[1], pair[0]] = 1
    return adj

def get_edge_vector(points, connectivity):
    """Get edge vectors from connectivity and point positions.

    PARAMETERS
        points: positions of points, 2D arrary [num_pts, 3].
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        
    RETURN
        edge_vec: 2D array, [num_edges, 3].
    """
    num_edges = len(connectivity)
    edge_vec = np.zeros([num_edges, 3])
    i = 0
    for pair in connectivity:
        this_edge = points[pair[1]] - points[pair[0]]
        edge_vec[i] = this_edge
        i += 1
    return edge_vec

def update_pts(o_points, w):
    """Update points positions by phi and u. Assumptions here are the upper layer is a rigid polygon
       and points are on the same height. Can add another angle to simulate cases with pts not of the
       same height. But the rigid polygon assumption should be strictly kept.

    PARAMETERS
        o_points: points need updating, 2D arrary [num_pts, 3] (in mm).
        w: [phi, u]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
    RETURN
        all_points: 2D array, [num_pts, 3].
    """
    phi = w[1]
    u = w[0]
    n = len(o_points)//2
    base = o_points.copy()[:n]
    points = o_points.copy()[n:]
    points[:,2] = points[:,2] + np.ones(len(points))*u
    points =  rotation(phi, points)
    all_points = np.vstack([base, points])
    return all_points

def get_length(points, connectivity):
    """Get length of each truss element/edge based on current geometry (extension positive). 

    PARAMETERS
        points: current positions of points, 2D arrary [num_pts, 3].
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        
    RETURN
        length: 1D array, [num_edges, ] (in mm).
    """
    n = len(points)//2
    edge_vec_new = get_edge_vector(points, connectivity)
    length = np.linalg.norm(edge_vec_new, axis=1)
    return length

def get_Qx(points_ref, points, connectivity, ks):
    """Get basic forces of each truss element/edge based on current geometry (tension positive). 
       Here, use rotated engineering deformation (RE, Ln-L). Nonlinear kinematics is considered.

    PARAMETERS
        points_ref: initial/reference positions of points, 2D arrary [num_pts, 3].
        points: current positions of points, 2D arrary [num_pts, 3].
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        ks: spring stiffness (in kN/m).
        
    RETURN
        Qx: 1D array, [num_edges, ] (in kN).
    """
    num_edges = len(connectivity)
    edge_vec0 = get_edge_vector(points_ref, connectivity)
    length0 = np.linalg.norm(edge_vec0, axis=1)
    edge_vec_new = get_edge_vector(points, connectivity)
    length = np.linalg.norm(edge_vec_new, axis=1)
    Qx = (length - length0)*ks*0.001
    Qx = Qx.reshape([num_edges, 1])
    return Qx

def get_Bx(points, connectivity):
    """Get force influence matrix B based on current geometry. Nonliner statics is considered.
    
    PARAMETERS
        points: current positions of points, 2D arrary [num_pts, 3].
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
    
    RETURN
        Bx: 2D array, [num_pts*3, num_edges].
    """
    num_pts = len(connectivity)
    num_edges = len(connectivity)
    temp = np.linalg.norm(get_edge_vector(points, connectivity), axis=1)
    b_vec = get_edge_vector(points, connectivity)/temp.reshape([len(temp),1])
    b_vec = np.hstack([-b_vec, b_vec])
    dof_id = np.array(range(num_pts*3)).reshape([num_pts, 3])
    id_vec = np.zeros([num_edges, 3*2])
    id_vec = id_vec.astype(int) 
    for bar in range(num_edges):
        id_vec[bar] = np.hstack([dof_id[connectivity[bar][0]], dof_id[connectivity[bar][1]]])
    Bx = np.zeros([num_edges, num_pts*3])
    for ele in range(num_edges):
        Bx[ele][id_vec[ele]] = b_vec[ele]
    Bx = Bx.T
    return Bx

def create_KTgeometry(n, R0, h0, theta1, ks, mag_m, mag_arrange):
    """Create necessary variables for the KT skeleton.
    
    PARAMETERS
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: inital height (in mm).
        theta1: initial rotation angle, counter-clockwise positive (in radian).
        ks: spring stiffness (in kN/m).
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets, 3].
    
    RETURN: 
        dic: dictionary.
    """
    dic = {}
    num_pts = 2*n
    num_edges = 2*n
    theta0 = theta1-np.pi/n # in degree, counter-clockwise
    phi_max = (np.pi-np.pi/n)-theta1
    phi_min = -((np.pi-np.pi/n)+theta1)

    # initial config.
    the = np.linspace(0,2*np.pi,n+1)
    x = R0*np.cos(the)
    y = R0*np.sin(the)
    # lower layer
    z1 = np.ones(len(x))*0
    points1 = np.column_stack([x, y, z1])
    points1 = points1[:-1]    # n x 3, fixed base pts
    # upper layer
    z2 = np.ones(len(x))*h0
    points2 = np.column_stack([x, y, z2])
    points2 = points2[:-1]    
    points2 = rotation(theta0, points2) # n x 3
    points_ref = np.vstack([points1, points2])
    # connectivity, initial edge vec.
    temp = np.array(range(n)) + np.ones(n)*n
    connectivity_mountain = np.vstack([np.array(range(n)), temp])
    connectivity_valley = np.vstack([np.array(range(n)), np.append(temp[1:], temp[0])])
    connectivity = np.hstack([connectivity_mountain, connectivity_valley])
    connectivity = connectivity.T
    connectivity = connectivity.astype(int)
    edge_vec0 = get_edge_vector(points_ref, connectivity)
    adj = get_adj(connectivity)
    
    dic['n'] = n
    dic['R0'] = R0 
    dic['h0'] = h0
    dic['theta1'] = theta1
    dic['points_ref'] = points_ref
    dic['adj'] = adj
    dic['connectivity'] = connectivity
    dic['points_ref'] = points_ref
    dic['ks'] = ks
    dic['mag_m'] = mag_m
    dic['mag_arrange'] = mag_arrange
    dic['phi_max'] = phi_max
    dic['phi_min'] = phi_min

    return dic

def get_current_force(o_points, w, connectivity, ks):
    """Get truss forces at DOFs based on current geometry (positive force in positive direction). 
       Current geometry is obtained from old positions and displacements. Nonlinear statics is considered. 
       
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        ks: spring stiffness (in kN/m).
    
    RETURN
        Q: 2D array, [num_pts, 3] (in kN).
    """
    num_pts = len(o_points)
    n_points = update_pts(o_points, w)
    Qx = get_Qx(o_points, n_points, connectivity, ks)
    Bx = get_Bx(n_points, connectivity)
    Q = np.matmul(Bx, Qx)
    Q = Q.reshape([num_pts, 3])
    return Q

def upperlayer_force(o_points, w, connectivity, ks):
    """Get upper layer polygon truss forces and moments based on reference pts and displcement vector, w. 
    
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        ks: spring stiffness (in kN/m).
    
    RETURN
        Fx: resultant force of the upper layer/ upper layer force output in X-dir, float (in kN).
        Fy: resultant force of the upper layer/ upper layer force output in Y-dir, float (in kN).
        Fz: resultant force of the upper layer/ upper layer force output in Z-dir, float (in kN).
        Mx: resultant moment of the upper layer/ upper layer moment output around pos. X-axis, float (in kN*m).
        My: resultant moment of the upper layer/ upper layer moment output around pos. Y-axis, float (in kN*m).
        Mz: resultant torque of the upper layer/ upper layer moment output around pos. Z-axis, float (in kN*m).
    """
    n = len(o_points)//2
    n_points = update_pts(o_points, w)
    Q = get_current_force(o_points, w, connectivity, ks)
    
    Fx = np.sum(Q[n:,0])
    Fy = np.sum(Q[n:,1])
    Fz = np.sum(Q[n:,2])
    
    Mx = np.sum(n_points[n:, 1]*Q[n:, 2])*0.001
    My = -np.sum(n_points[n:, 0]*Q[n:, 2])*0.001
    Mz = np.sum(n_points[n:, 0]*Q[n:, 1])*0.001 - np.sum(n_points[n:, 1]*Q[n:, 0])*0.001
    
    return np.array([Fx, Fy, Fz, Mx, My, Mz])

def elastic_U(w, n, R0, h0, theta1, ks, norm=True):
    """Get truss elastic energy based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in rad).
            u: vertival displacement, 'downward' positive, float (in mm).
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: initial height (in mm).
        theta1: initial rotation angle (radian).
        ks: spring stiffness (in kN/m).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        U_norm: float (dimensionless or in J).
    """
    u0, phi0 = [0, 0]
    a0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2+theta1/2-np.pi/2/n))**2)
    b0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2+theta1/2+np.pi/2/n))**2)
    
    u, phi = w
    a = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2+theta1/2-np.pi/2/n))**2)    # mm
    b = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2+theta1/2+np.pi/2/n))**2)    # mm
    
    U = 0.5*n*ks*(a-a0)**2 + 0.5*n*ks*(b-b0)**2    # in kN/m*mm2  (or mJ)
    U = U*0.001 # in J
    if norm:
        U = U/(ks*h0**2*0.001)    # dimensionless
    
    return U 

def elastic_FT(w, n, R0, h0, theta1, ks):

    """Get truss force and torque based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in rad).
            u: vertival displacement, 'downward' positive, float (in mm).
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: initial height (in mm).
        theta1: initial rotation angle (radian).
        ks: spring stiffness (in kN/m).
    
    RETURN
        (Fz, Tz): in kN and kN*m.
    """
    
    u0, phi0 = [0, 0]
    u, phi = w
    
    a0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2+theta1/2-np.pi/2/n))**2)
    b0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2+theta1/2+np.pi/2/n))**2)
    
    a = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2+theta1/2-np.pi/2/n))**2)
    b = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2+theta1/2+np.pi/2/n))**2) 

    FF = n*ks*(h0+u)*(2-a0/a-b0/b)*0.001    # in kN
    TT = 0.001**2*n*ks*R0**2*((1-a0/a)*np.sin(phi+theta1-np.pi/n)+
                           (1-b0/b)*np.sin(phi+theta1+np.pi/n))    # in kN*m
    
    return FF, TT

#-------------------------------------------------------------magnet functions--------------------------------------------------#

def Force_from_vec(mag_m, n1, n2, r_vec12):
    """Get magnet force from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        Orientation parameters:
            n1: m1 dipole moment direction vector, 1D array [3,] (in A*m2).
            n2: m2 dipole moment direction vector, 1D array [3,] (in A*m2).
            r_vec: distance vector of two magnets, from point dipole 1 to point dipole 2 (pt2_vec - pt1_vec), 
                   1D array [3,] (in mm).
       
    RETURN
        F_mag: magnet force output (negative magnet force) on point dipole 2, 
               also the magnet force on point dipole 1. float (in kN).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    
    F_from_vec = -3*mu0/(np.pi*4*np.linalg.norm(r_vec)**5)*(np.inner(m1, r_vec)*m2 + np.inner(m2, r_vec)*m1 
        + np.inner(m1, m2)*r_vec - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec/np.linalg.norm(r_vec)**2)
    return F_from_vec*0.001

def get_mag_force(mag_m, mag_points, mag_arrange):
    """Get magnets force output at magnet centers based on position and dipole arrangement.
       Assume point dipoles has fixed dipole arrangement. Sliding and overturning are prohibited.
       If dipole tilts, need modification (include in-plane magnet-magnet interactions of each polygon).
       
    PARAMETERS
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        mag_points: points of magnetic dipoles, 2D arrary [num_magnets, 3].
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets, 3].
    
    RETURN
        forces: lower level magnets follwed by upper layer's, 2D array, [num_pts, 3] (in kN).
    """
    n = len(mag_points)//2
    forces = np.zeros([2*n, 3])
    for j in range(n,2*n):
        force_j = 0
        for i in range(n):
            force_j = force_j+Force_from_vec(mag_m, mag_arrange[i], mag_arrange[j], 
                                             mag_points[j]-mag_points[i])
        forces[j] = force_j
    for j in range(n):
        force_j = 0
        for i in range(n, 2*n):
            force_j = force_j+Force_from_vec(mag_m, mag_arrange[i], mag_arrange[j], 
                                             mag_points[j]-mag_points[i])
        forces[j] = force_j
    return forces

def Energy_from_vec(mag_m, n1, n2, r_vec12, h0, ks, norm=True):
    """Get magnet force from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        n1: m1 dipole moment direction vector, 1D array [3,] (in A*m2).
        n2: m2 dipole moment direction vector, 1D array [3,] (in A*m2).
        r_vec12: distance vector of two magnets, from point dipole 1 to point dipole 2 (pt2_vec - pt1_vej), 
               1D array [3,] (in mm).
        h0: initial height (mm).
        ks: spring stiffness kN/m.
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J or dimensionless).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    
    E_from_vec = mu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec)**3  
                    - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/np.linalg.norm(r_vec)**5)
    
    if norm:
        E_from_vec = E_from_vec/(ks*h0**2*0.001)
    return E_from_vec

def this_magnet_energy(mag_m, mag_points, mag_arrange, h0, ks, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        mag_points: points of magnetic dipoles, 2D arrary [num_magnets, 3].
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets, 3].
        h0: initial height (mm).
        ks: spring stiffness kN/m.
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless or in J).
    """
    temp = 0
    n = len(mag_points)//2
    for i in range(n):
        for j in range(n,2*n):
            temp = temp + Energy_from_vec(mag_m, mag_arrange[i], mag_arrange[j], mag_points[j]-mag_points[i], h0, ks, norm=norm)
    return temp

def magnet_potential(KTdic, u, phi, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        phi: twisting angle, counter-clockwise positive, float (in degree).
        u: vertival displacement, upward positive, float (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless or in J).
    """
    h0 = KTdic['h0']
    points_ref = KTdic['points_ref']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']
    mag_points = update_pts(points_ref, [u, phi])
    magnet_E = this_magnet_energy(mag_m, mag_points, mag_arrange, h0, ks, norm=norm)
    return magnet_E


#-----------------------------------------------------------total (elastic + magnets) functions----------------------------------------------#

def get_total_force(KTdic, w):
    """Get magnets force output at magnet centers based on position and dipole arrangement.
       Assume point dipoles with fixed dipole arrangement. Sliding and overturning are prohibited. 
       
    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in rad).
            u: vertival displacement, upward positive, float (in mm).
    
    RETURN
        forces: lower level magnets follwed by upper layer's, 2D array, [num_pts, 3] (in kN).
    """
    o_points = KTdic['points_ref']
    ks = KTdic['ks']
    connectivity = KTdic['connectivity']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']
    mag_points = update_pts(o_points, w)
    tot_force =  get_current_force(o_points, w, connectivity, ks) + get_mag_force(mag_m, mag_points, mag_arrange)
    return tot_force

def upperlayer_total_force(KTdic, w):
    """Get upper layer polygon truss and magnetic forces and moments based on reference pts and displcement vector, w. 
    
    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in rad).
            u: vertival displacement, upward positive, float (in mm).
    
    RETURN
        Fx: resultant force of the upper layer/ upper layer force output in X-dir, float (in kN).
        Fy: resultant force of the upper layer/ upper layer force output in Y-dir, float (in kN).
        Fz: resultant force of the upper layer/ upper layer force output in Z-dir, float (in kN).
        Mx: resultant moment of the upper layer/ upper layer moment output around pos. X-axis, float (in kN*m).
        My: resultant moment of the upper layer/ upper layer moment output around pos. Y-axis, float (in kN*m).
        Mz: resultant torque of the upper layer/ upper layer moment output around pos. Z-axis, float (in kN*m).
    """
    n = KTdic['n']
    o_points = KTdic['points_ref']
    ks = KTdic['ks']
    connectivity = KTdic['connectivity']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    n_points = update_pts(o_points, w)
    Q = get_total_force(KTdic, w)
    
    Fx = np.sum(Q[n:,0])
    Fy = np.sum(Q[n:,1])
    Fz = np.sum(Q[n:,2])
    
    Mx = np.sum(n_points[n:, 1]*Q[n:, 2])*0.001
    My = -np.sum(n_points[n:, 0]*Q[n:, 2])*0.001
    Mz = np.sum(n_points[n:, 0]*Q[n:, 1])*0.001 - np.sum(n_points[n:, 1]*Q[n:, 0])*0.001
#     Mz2 = np.sum(n_points[:n, 0]*Q[:n, 1]) - np.sum(n_points[:n, 1]*Q[:n, 0])*0.001
    
    return np.array([Fx, Fy, Fz, Mx, My, Mz])

def total_potential(KTdic, u, phi, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        phi: twisting angle, counter-clockwise positive, float (in degree).
        u: vertival displacement, upward positive, float (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless).
    """
    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']
    mag_points = update_pts(points_ref, [u, phi])
    total = elastic_U([u, phi], n, R0, h0, theta1, ks, norm=norm) + this_magnet_energy(mag_m, mag_points, mag_arrange, h0, ks, norm=norm)
    return total

#-------------------------------------------------------------parallel calc functions---------------------------------------#

@np.vectorize
def rotation_parallel(ang, size1=N, size2=N):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    # ang in degree
    points = points_ref_inner
    global special_i
    rot = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    points_rot = np.matmul(rot, points.T)
    points_rot = points_rot.T
    if special_i<size1*size2:
        points_rot_set[special_i] = points_rot
    special_i += 1
    pass

@np.vectorize
def update_pts_parallel(u, phi, size1=N, size2=N):
    """Update points positions by phi and u. Assumptions here are the upper layer is a rigid polygon
       and points are on the same height. Can add another angle to simulate cases with pts not of the
       same height. But the rigid polygon assumption should be strictly kept.

    PARAMETERS
        u: vertival displacement, upward positive, float (in mm).
        phi: twisting angle, counter-clockwise positive, float (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        all_points: 2D array, [num_pts, 3].
    """
    points = points_ref_inner
    # print(points_ref)
    n = len(points)//2
    global special_i   # vectorize loop has already started from here (u is not in [N,N], u is a scalar)
    rot = np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    points_rot = np.matmul(rot, points.T)
    points_rot = points_rot.T
    points_rot[:,2] = points_rot[:,2] + u
    points_rot[:n] = points[:n].copy()
    if special_i<size1*size2:
        points_updated_set[special_i] = points_rot
    special_i += 1
    pass

def magnet_potential_parallel(KTdic, u, phi, size1=N, size2=N, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless or in J).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for ii in range(n):
        for jj in range(n,2*n):
            r_vec12 = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec12 = r_vec12/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]

            this = mu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec12, axis=1)**3  
                         - 3*np.matmul(r_vec12,m1)*np.matmul(r_vec12,m2)/np.linalg.norm(r_vec12, axis=1)**5)
            if norm:
            	temp = temp + this/(ks*h0**2*0.001)
            else:
            	temp = temp + this
    temp = temp.reshape([size1, size2])
    return temp

def total_potential_parallel(KTdic, u, phi, size1=N, size2=N, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless or in J).
    """

    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for ii in range(n):
        for jj in range(n,2*n):
            r_vec12 = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec12 = r_vec12/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]

            this = mu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec12, axis=1)**3  
                         - 3*np.matmul(r_vec12,m1)*np.matmul(r_vec12,m2)/np.linalg.norm(r_vec12, axis=1)**5)
            if norm:
            	temp = temp + this/(ks*h0**2*0.001)
            else:
            	temp = temp + this
    temp = temp.reshape([size1, size2])
    total = elastic_U([u, phi], n, R0, h0, theta1, ks, norm=norm) + temp
    return total

def magnet_Zforce_parallel(KT_dic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for jj in range(n,2*n):
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[2] + np.inner(m2, r_vec)*m1[2] + np.inner(m1, m2)*r_vec[:,2] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,2]/np.linalg.norm(r_vec,axis=1)**2)
            temp = temp + this*0.001
    temp = temp.reshape([size1, size2])
    return temp



def magnet_torque_parallel(KT_dic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN*m).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)
    
    arm1 = np.zeros(size1*size2)
    arm2 = np.zeros(size1*size2)
    for jj in range(n, 2*n):
        temp1 = np.zeros(size1*size2)
        temp2 = np.zeros(size1*size2)
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this1 = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[0] + np.inner(m2, r_vec)*m1[0] + np.inner(m1, m2)*r_vec[:,0] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,0]/np.linalg.norm(r_vec,axis=1)**2)
            this2 = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[1] + np.inner(m2, r_vec)*m1[1] + np.inner(m1, m2)*r_vec[:,1] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,1]/np.linalg.norm(r_vec,axis=1)**2)
            temp1 = temp1 + this1*0.001
            temp2 = temp2 + this2*0.001
        arm1 = arm1 + temp1*points_updated_set[:, jj][:,1]*0.001
        arm2 = arm2 + temp2*points_updated_set[:, jj][:,0]*0.001
    torque = (arm2 - arm1).reshape([size1, size2])
    return torque

def truss_Zforce_parallel(KTdic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)

    # truss force
    temp = np.zeros(size1*size2)            
    for conn in connectivity:
        conn_vec = points_updated_set[:, conn[1]] - points_updated_set[:, conn[0]]    
        current_len = np.linalg.norm(conn_vec, axis=1) 
        elong = current_len - np.linalg.norm(points_ref[conn[1]]-points_ref[conn[0]])   
        ele_force = elong * ks 
        ele_force = ele_force.reshape([size1*size2,1]) * conn_vec/current_len.reshape([size1*size2,1])
        temp = temp + ele_force[:,2]*0.001
    temp = temp.reshape([size1, size2])

    return temp

def truss_torque_parallel(KTdic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN*m).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)
    
    # truss torque
    torque2 = np.zeros(size1*size2)
    for conn in connectivity:
        conn_vec = points_updated_set[:, conn[1]] - points_updated_set[:, conn[0]]     
        current_len = np.linalg.norm(conn_vec, axis=1)  
        elong = current_len - np.linalg.norm(points_ref[conn[1]]-points_ref[conn[0]]) 
        ele_force = elong * ks 
        ele_force = ele_force.reshape([size1*size2,1]) * conn_vec/current_len.reshape([size1*size2,1])
        torque2 = torque2 + ele_force[:,1]*0.001*points_updated_set[:, conn[1]][:, 0]*0.001 - ele_force[:,0]*0.001*points_updated_set[:, conn[1]][:, 1]*0.001
    torque2 = torque2.reshape([size1, size2])

    return torque2


def total_Zforce_parallel(KTdic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)

    # magnet force
    temp = np.zeros(size1*size2)
    for jj in range(n,2*n):
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[2] + np.inner(m2, r_vec)*m1[2] + np.inner(m1, m2)*r_vec[:,2] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,2]/np.linalg.norm(r_vec,axis=1)**2)
            temp = temp + this*0.001
            
    # truss force
    for conn in connectivity:
        conn_vec = points_updated_set[:, conn[1]] - points_updated_set[:, conn[0]]     
        current_len = np.linalg.norm(conn_vec, axis=1)  
        elong = current_len - np.linalg.norm(points_ref[conn[1]]-points_ref[conn[0]]) 
        ele_force = elong * ks 
        ele_force = ele_force.reshape([size1*size2,1]) * conn_vec/current_len.reshape([size1*size2,1])
        temp = temp + ele_force[:,2]*0.001
    temp = temp.reshape([size1, size2])

    return temp

def total_torque_parallel(KTdic, u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        KTdic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        u: vertival displacement, upward positive, [size1, size2] (in mm).
        phi: twisting angle, counter-clockwise positive, meshed [size1, size2] (in radian).
        size1: [keyword] mesh grid size.
        size2: [keyword] mesh grid size.
    
    RETURN
        total: float (kN*m).
    """
    global special_i
    global points_updated_set
    global points_ref_inner

    n = KTdic['n']
    R0 = KTdic['R0'] 
    h0 = KTdic['h0']
    theta1 = KTdic['theta1']
    points_ref = KTdic['points_ref']
    points_ref_inner = points_ref.copy()
    connectivity = KTdic['connectivity']
    ks = KTdic['ks']
    mag_m = KTdic['mag_m']
    mag_arrange = KTdic['mag_arrange']

    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_parallel(u, phi, size1, size2)
    
    part1 = np.zeros(size1*size2)
    part2 = np.zeros(size1*size2)
    
    # magnet torque
    for jj in range(n, 2*n):
        temp1 = np.zeros(size1*size2)
        temp2 = np.zeros(size1*size2)
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this1 = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[0] + np.inner(m2, r_vec)*m1[0] + np.inner(m1, m2)*r_vec[:,0] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,0]/np.linalg.norm(r_vec,axis=1)**2)
            this2 = -3*mu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[1] + np.inner(m2, r_vec)*m1[1] + np.inner(m1, m2)*r_vec[:,1] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,1]/np.linalg.norm(r_vec,axis=1)**2)
            temp1 = temp1 + this1*0.001
            temp2 = temp2 + this2*0.001
        part1 = part1 + temp1*points_updated_set[:, jj][:,1]*0.001
        part2 = part2 + temp2*points_updated_set[:, jj][:,0]*0.001
        torque1 = (part2 - part1).reshape([size1, size2])
    
    # truss torque
    torque2 = np.zeros(size1*size2)
    for conn in connectivity:
        conn_vec = points_updated_set[:, conn[1]] - points_updated_set[:, conn[0]]     
        current_len = np.linalg.norm(conn_vec, axis=1)  
        elong = current_len - np.linalg.norm(points_ref[conn[1]]-points_ref[conn[0]]) 
        ele_force = elong * ks 
        ele_force = ele_force.reshape([size1*size2,1]) * conn_vec/current_len.reshape([size1*size2,1])
        torque2 = torque2 + ele_force[:,1]*0.001*points_updated_set[:, conn[1]][:, 0]*0.001 - ele_force[:,0]*0.001*points_updated_set[:, conn[1]][:, 1]*0.001
    torque2 = torque2.reshape([size1, size2])

    return torque1 + torque2


#-------------------------------------------------------------symbolic calc functions-----------------------------------------------------#
# cannot use np otherwise the return is float instaead of symbols

def elastic_U_sym(n, R0, h0, theta1, ks, w, norm=True):
    """Get truss normalized elastic energy based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: inital height (in mm).
        theta1: initial rotation angle, counter-clockwise positive (in radian).
        ks: spring stiffness (in kN/m).
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in radian).
            u: vertival displacement, 'downward' positive, float (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).

    RETURN
        U_norm: float (dimensionless if norm=True).
    """
    u0, phi0 = [0, 0]
    a0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2+theta1/2-np.pi/2/n))**2)
    b0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2+theta1/2+np.pi/2/n))**2)
    
    u, phi = w
    a = sqrt((h0+u)**2+4*R0**2*(sin(phi/2+theta1/2-np.pi/2/n))**2)    # mm
    b = sqrt((h0+u)**2+4*R0**2*(sin(phi/2+theta1/2+np.pi/2/n))**2)    # mm
    
    U = 0.5*n*ks*(a-a0)**2 + 0.5*n*ks*(b-b0)**2    # in kN/m*mm2  (or mJ)
    U = U*0.001
    if norm:
        U = U/(ks*h0**2*0.001)    # dimensionless
    
    return U

def elastic_FT_sym(w, n, R0, h0, theta1, ks):

    """Get truss force and torque based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in rad).
            u: vertival displacement, 'downward' positive, float (in mm).
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: initial height (in mm).
        theta1: initial rotation angle (radian).
        ks: spring stiffness (in kN/m).
    
    RETURN
        (Fz, Tz): in kN and kN*m.
    """
    
    u0, phi0 = [0, 0]
    u, phi = w
    
    a0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2+theta1/2-np.pi/2/n))**2)
    b0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2+theta1/2+np.pi/2/n))**2)
    
    a = sqrt((h0+u)**2+4*R0**2*(sin(phi/2+theta1/2-np.pi/2/n))**2)
    b = sqrt((h0+u)**2+4*R0**2*(sin(phi/2+theta1/2+np.pi/2/n))**2) 

    FF = n*ks*(h0+u)*(2-a0/a-b0/b)*0.001    # in kN
    TT = 0.001**2*n*ks*R0**2*((1-a0/a)*sin(phi+theta1-np.pi/n)+
                           (1-b0/b)*sin(phi+theta1+np.pi/n))    # in kN*m
    
    return FF, TT

def rotation_sym(ang, points):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in degree).
        points: positions of points need rotating, 2D arrary [num_pts, 3].
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    rot = np.array([[cos(ang), -sin(ang), 0],[sin(ang), cos(ang), 0], [0, 0, 1]])
    points_rot = rot @ points.T
    return points_rot.T

def update_pts_sym(o_points, w):
    """Update points positions by phi and u. Assumptions here are the upper layer is a rigid polygon
       and points are on the same height. Can add another angle to simulate cases with pts not of the
       same height. But the rigid polygon assumption should be strictly kept.

    PARAMETERS
        o_points: points need updating, 2D arrary [num_pts, 3] (in mm).
        w: [phi, u]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
    RETURN
        all_points: 2D array, [num_pts, 3].
    """
    phi = w[1]
    u = w[0]
    n = len(o_points)//2
    base = o_points.copy()[:n]
    points = o_points.copy()[n:]
    points = np.hstack([points[:,0], points[:,1], np.ones(len(points))*u + points[:,2]]).reshape([3,n]).T
    points =  rotation_sym(phi, points)
    all_points = np.vstack([base, points])
    return all_points

def Energy_from_vec_sym(mag_m, n1, n2, r_vec12, ks, h0, norm=True):
    """Get magnet force from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        Magnets parameters:
            m: magnet moment, float (in A*m2).
        Orientation parameters:
            n1: m1 dipole moment direction vector, 1D array [3,] (in A*m2).
            n2: m2 dipole moment direction vector, 1D array [3,] (in A*m2).
            r_vec12: distance vector of two magnets, from point dipole 1 to point dipole 2 (pt2_vec - pt1_vej), 
                   1D array [3,] (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J or fimensionless).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    r_vec_norm = sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    
    E_from_vec = mu0/(np.pi*4)*(np.inner(m1, m2)/r_vec_norm**3  
                    - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/r_vec_norm**5)
    
    if norm:
        E_from_vec = E_from_vec/(ks*h0**2*0.001)
    return E_from_vec

def this_magnet_energy_sym(mag_m, mag_points, mag_arrange, ks, h0, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        mag_points: positions of magnetic point dipoles, 2D arrary [num_pts, 3] (in mm).
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets, 3].
        ks: spring stiffness (in kN/m).
        h0: inital height of the truss skeleton (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).
    
    RETURN
        total: float (dimensionless).
    """
    temp = 0
    n = len(mag_points)//2
    for i in range(n):
        for j in range(n,2*n):
            temp = temp + Energy_from_vec_sym(mag_m, mag_arrange[i], mag_arrange[j], mag_points[j]-mag_points[i], ks, h0, norm=norm)
    return temp

def total_potential_sym(o_points, n, R0, h0, theta1, ks, mag_m, mag_arrange, w, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        o_points: initial/reference positions of points, 2D arrary [num_pts, 3] (in mm).
        n: number of nodes in each polygon.
        R0: polygon radius (in mm).
        h0: inital height (in mm).
        theta1: initial rotation angle, counter-clockwise positive (in radian).
        ks: spring stiffness (in kN/m).
        mag_m: magnitude of magnetic dipole (in A*m2), float.
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets, 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in radian).
            u: vertival displacement, 'downward' positive, float (in mm).
        norm: [keyword] if normalize energy by kh2, boolean (default is True).

    
    RETURN
        total: float (dimensionless if norm=True).
    """
    mag_points = update_pts_sym(o_points, w)
    total = elastic_U_sym(n, R0, h0, theta1, ks, w, norm=norm) + this_magnet_energy_sym(mag_m, mag_points, mag_arrange, ks, h0, norm=norm)
    return total

#-------------------------------------------------------------plotting -----------------------------------------------------#

def plot_TCO(dic, w, in_F=False, out_F=False, nsize=16, disp=True, gap=5, offset=60, arrow_coeff=20, 
            grid=True, axis=True, tick=False, ind0=False,
            xoff=0, yoff=0, zoff=0, xoff2=0, yoff2=0, zoff2=0):
    """Make 3D plots. Only works for TCO.
    
    PARAMETERS
        dic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        w:[u, phi]
            u: vertival displacement, upward positive, float (in mm).
            phi: twisting angle, counter-clockwise positive, float (in rad).
        in_F: decide if plotting input force or not, boolean (default True).
        out_F: decide if plotting output force or not, boolean (default True).
        nsize: font size.
        disp_F: decide if plotting displacements phi and u or not, boolean (default True).
        gap: margin between the plotting panel and the plotting object max/min. 
        arrow_coeff: arrow size, float.
        offset: distance between arrow tail and upper layer center, float.
        grid: decide if showing grid or not, boolean (default True).
        axis: decide if showing axes force or not, boolean (default True).
        tick: decide if showing ticks or not, boolean (default False).
        ind0: decied if the numbering of nodes starts from 0, bollean (default True, if Flase, starts from 1).
        xoff: bottom layer node number position offset in x-dir.
        yoff: bottom layer node number position offset in y-dir.
        zoff: bottom layer node number position offset in z-dir.
        xoff2: top layer node number position offset in x-dir.
        yoff2: top layer node number position offset in y-dir.
        zoff2: top layer node number position offset in z-dir.
        
    NO RETURN
    """
    n = dic['n']
    R0 = dic['R0'] 
    h0 = dic['h0']
    theta1 = dic['theta1']
    o_points = dic['points_ref']
    connectivity = dic['connectivity']
    ks = dic['ks']
    mag_m = dic['mag_m']
    mag_arrange = dic['mag_arrange']
    
    phi = w[1]
    u = w[0]
    points = update_pts(o_points, w)
    xs, ys, zs = points[:,0], points[:,1], points[:,2] 
    fig = plt.figure(figsize=[6,5])
    ax = fig.add_subplot(111, projection='3d')

    verts1 = [list(zip(xs[:n], ys[:n], zs[:n]))]
    ax.add_collection3d(Poly3DCollection(verts1, zorder=1, alpha = 0.4, color = 'gray', linewidth=1, edgecolor = 'k'))

    plt.plot(np.hstack([xs[:n],xs[0]]), np.hstack([ys[:n],ys[0]]), np.hstack([zs[:n],zs[0]]), 
             '--', zorder=2, c='k',  linewidth=2.5)
    plt.plot(np.hstack([xs[n:],xs[n]]), np.hstack([ys[n:],ys[n]]), np.hstack([zs[n:],zs[n]]), 
             c='k', zorder=3, linewidth=2.5)
    
#     a = np.linspace(0.5, 1, n)
    for ii in range(n):
        this_edge = np.vstack([points[connectivity[ii][0]], points[connectivity[ii][1]]])
        plt.plot(this_edge[:,0], this_edge[:,1], this_edge[:,2], 'b', alpha=1.0, linewidth=2.5)
    a = np.linspace(0.5, 1, n)
    for ii in range(n,2*n):
        this_edge = np.vstack([points[connectivity[ii][0]], points[connectivity[ii][1]]])
        plt.plot(this_edge[:,0], this_edge[:,1], this_edge[:,2], 'r', alpha=a[ii-n], linewidth=2.5)
    
    verts2 = [list(zip(xs[n:], ys[n:], zs[n:]))]
    ax.add_collection3d(Poly3DCollection(verts2, zorder=2*n+4, alpha = 0.8, color = 'gray', 
                                         linewidth=1, edgecolor = 'k'))

    ax.scatter(xs[:n], ys[:n], zs[:n], zorder=2*n+5, c='k', s=30)
    ax.scatter(xs[n:], ys[n:], zs[n:], zorder=2*n+6, c='k', s=30)
    ax.scatter(0, 0, 0, zorder=2*n+7, c='k', s=10)
    ax.scatter(0, 0, zs[-1], zorder=2*n+8, c='k', s=10)
    for no in range(n):
        if not ind0:
            ax.text(xs[no]+xoff, ys[no]+yoff, zs[no]+zoff, f'{no+1}', size=nsize, zorder=2*n+9+no, color='k', 
                    horizontalalignment='left',
                    verticalalignment='bottom') 
        else: 
            ax.text(xs[no]+xoff, ys[no]+yoff, zs[no]+zoff, f'{no}', size=nsize, zorder=2*n+9+no, color='k', 
                    horizontalalignment='left',
                    verticalalignment='bottom') 
    for no in range(n, 2*n):
        ax.text(xs[no]+xoff2, ys[no]+yoff2, zs[no]+zoff2, f'{no+1}', size=nsize, zorder=2*n+9+no, color='k', 
                horizontalalignment='left',
                verticalalignment='bottom') 
    ax.text(0, 0, 0,  'O', size=nsize, zorder=4*n+10, color='k', horizontalalignment='left',
                verticalalignment='bottom') 
    ax.text(0, 0, zs[-1],  "O`", size=nsize, zorder=4*n+11, color='k', horizontalalignment='left',
                verticalalignment='bottom') 
    
    if mag_m > 0:
        for pt,arrange in zip(points, mag_arrange):
            ax.quiver(pt[0]-arrange[0]*arrow_coeff*0.5,pt[1]-arrange[1]*arrow_coeff*0.5,pt[2]-arrange[2]*arrow_coeff*0.5, # <-- starting point of vector
            arrange[0]*arrow_coeff,arrange[1]*arrow_coeff,arrange[2]*arrow_coeff, # <-- directions of vector
            color = 'limegreen', alpha = 1.0, lw = 1.5, arrow_length_ratio=0.4)
    
    if in_F:
        input_F = [upperlayer_total_force(dic, w)[2], 
                    upperlayer_total_force(dic, w)[-1]]
#         output_F = [total_Zforce_vec(w[0], w[1], 1, 1)[0][0], total_torque_vec(w[0], w[1], 1, 1)[0][0]]
#         print(output_F)
        props2 = dict(boxstyle='round', facecolor='cornflowerblue', alpha=.5)
        # place a text box in upper left in axes coords
        ax.text2D(0.0, 0.6, f'Force-in:\nF = {input_F[0]:.3f} kN\nT = {input_F[1]:.3f} kN\u2022m',
                transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props2)
    if out_F:
        output_F = [upperlayer_total_force(dic, w)[2], 
                    upperlayer_total_force(dic, w)[-1]]
        props2 = dict(boxstyle='round', facecolor='cornflowerblue', alpha=.5)
        # place a text box in upper left in axes coords
        ax.text2D(0.0, 0.6, f'Force-out:\nF = {output_F[0]:.3f} kN\nT = {output_F[1]:.3f} kN\u2022m',
                transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props2)
        
    if disp:
        props3 = dict(boxstyle='round', facecolor='bisque', alpha=.5)
        # place a text box in upper left in axes coords
        ax.text2D(0.0, 0.35, f'$\phi$={phi:.2f} rad\n u={u:.2f} mm',
                transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props3)

    loc = plticker.MultipleLocator(base=40.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    
    if h0>100:
        ax.zaxis.set_major_locator(loc)
    
    ax.set_xlim(-R0, R0)
    ax.set_ylim(-R0, R0)
    ax.set_zlim(-gap, h0+gap)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    # Hide grid lines
    if not grid:
        ax.grid(False)
    
    if not axis:
        plt.axis('off')

    # Hide axes ticks
    if not tick:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
#     plt.tight_layout()
    plt.show()
    pass


#------------------------------------------------------Path searching functions-----------------------------------------------------#
def find_F0_path(dic, startphi, endphi, spacing=0.5/180*np.pi, startu=0.1, endu=-0.66):
    """Calculate displacement and rotation on the rotation-controlled path (F=0 loading) by finding u for each given phi. 

    PARAMETERS
        dic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        startphi: beginning of rotation angle in the path, float (in radian).
        endphi: ending of rotation angle in the path, float (in radian).
        spacing: [keyword] spacing of rotation from startphi to endphi, float (in radian).
        startu: [keyword] beginning of displacement field for searching F=0 solution (normalize by h0), float (dimensionless).
        endu: [keyword] end of displacement field for searching F=0 solution (normalize by h0), float (dimensionless).
    
    RETURN
        uset_tot11: searched u along the path, 1D array [num_of_phi_along_path,] (in mm).  
        phi_set1: phi along the path, 1D array [num_of_phi_along_path,] (in radian).  
        U_minset_tot11: energy of searched solution (u, phi) along the path, 1D array [num_of_phi_along_path,] (dimensionless).  
        err11: error between F=0 and F of searched solution (u, phi) along the path, 1D array [num_of_phi_along_path,] (in N).  
    """
    
    n = dic['n']
    R0 = dic['R0'] 
    h0 = dic['h0']
    theta1 = dic['theta1']
    o_points = dic['points_ref']
    connectivity = dic['connectivity']
    ks = dic['ks']
    mag_m = dic['mag_m']
    mag_arrange = dic['mag_arrange']
    
    # estimate path
    phi_set1 = np.arange(startphi, endphi, spacing)
    phi_set1[-1] = endphi
    U_minset_tot1 = np.zeros(len(phi_set1))
    uset_tot1= np.zeros(len(phi_set1))
    N = 10000

    i = 0
    err1=[]
    for  fixphi in phi_set1: 
        x, y = np.mgrid[startu*h0:endu*h0:complex(0, N), fixphi:fixphi:complex(0, 1)]
        F_fixphi = total_Zforce_parallel(dic, x, y, N, 1)
        uset_tot1[i] = (np.linspace(startu*h0, endu*h0, N))[np.argmin(np.abs(F_fixphi))]
        U_minset_tot1[i] = total_potential_parallel(dic, uset_tot1[i], fixphi, 1, 1)
        def f_num(x):
            return [total_Zforce_parallel(dic, [x[0]], [fixphi], 1, 1)[0][0]]
        err1.append(f_num([uset_tot1[i]]))
        i += 1
        
    # optimize path by exactly finding F=0 pts
    U_minset_tot11 = np.zeros(len(phi_set1))
    uset_tot11 = np.zeros(len(phi_set1))
    N = 10000

    i = 0
    err11=[]
    for  fixphi in phi_set1: 
        def f_num(x):
            return [total_Zforce_parallel(dic, [x[0]], [fixphi], 1, 1)[0][0]]
        if i==0:
            sol = optimize.root(f_num, [uset_tot1[0]], method='hybr')
        else:
            sol = optimize.root(f_num, [uset_tot11[i-1]], method='hybr')
        if sol.success:
            uset_tot11[i] = sol.x
            U_minset_tot11[i] = total_potential_parallel(dic, sol.x, fixphi, 1, 1)
            err11.append(f_num([uset_tot11[i]]))
        else:
            print('Optimization failed. Return estimated path.')
            return (uset_tot1, phi_set1, U_minset_tot1, err1)
            break
        i += 1
    return (uset_tot11, phi_set1, U_minset_tot11, err11)

def find_T0_path(dic, startu, endu, startphi, endphi, spacing=0.01):
    """Calculate displacement and rotation on the rotation-controlled path (T=0 loading) by finding phi for each given u. 

    PARAMETERS
        dic: a dictionary with info of the KT system, 
                (keys must include: n, R0, h0, theta1, points_ref, ks, mag_m, mag_arrange, connectivity).
        startphi: beginning of rotation field for searching T=0 solution, float (in radian).
        endphi: end of rotation field for searching T=0 solution, float (in radian).
        startu: beginning of displacement in the path, float (dimensionless). 
        endu: ending of displacement angle in the path, float (dimensionless).
        spacing: [keyword] spacing of displacement from startphi to endphi, float (dimensionless).
    
    RETURN
        u_set1*h0: searched u along the path, 1D array [num_of_phi_along_path,] (in mm).  
        phiset_tot11: phi along the path, 1D array [num_of_phi_along_path,] (in radian).  
        U_minset_tot11: energy of searched solution (u, phi) along the path, 1D array [num_of_phi_along_path,] (dimensionless).  
        err11: error between T=0 and T of searched solution (u, phi) along the path, 1D array [num_of_phi_along_path,] (in kN*m).  
    """
    n = dic['n']
    R0 = dic['R0'] 
    h0 = dic['h0']
    theta1 = dic['theta1']
    o_points = dic['points_ref']
    connectivity = dic['connectivity']
    ks = dic['ks']
    mag_m = dic['mag_m']
    mag_arrange = dic['mag_arrange']
    
    # estimate path
    u_set1 = np.arange(startu, endu, spacing)
    u_set1[-1] = endu
    U_minset_tot1 = np.zeros(len(u_set1))
    phiset_tot1= np.zeros(len(u_set1))
    N = 10000

    i = 0
    err1=[]
    for fixu in u_set1: 
        x, y = np.mgrid[fixu*h0:fixu*h0:complex(0, 1), startphi:endphi:complex(0, N)]
        T_fixu = total_torque_parallel(dic, x, y, 1, N)
        phiset_tot1[i] = (np.linspace(startphi, endphi, N))[np.argmin(np.abs(T_fixu))]
        U_minset_tot1[i] = total_potential_parallel(dic, fixu*h0, phiset_tot1[i], 1, 1)
        def t_num(x):
            return [total_torque_parallel(dic, [fixu*h0], [x[0]], 1, 1)[0][0]]
        err1.append(t_num([phiset_tot1[i]]))
        i += 1
        
    # optimize path by exactly finding T=0 pts
    U_minset_tot11 = np.zeros(len(u_set1))
    phiset_tot11 = np.zeros(len(u_set1))
    N = 10000

    i = 0
    err11=[]
    for  fixu in u_set1: 
        def t_num(x):
            return [total_torque_parallel(dic, [fixu*h0], [x[0]], 1, 1)[0][0]]
        if i==0:
            sol = optimize.root(t_num, [phiset_tot1[0]], method='hybr')
        else:
            sol = optimize.root(t_num, [phiset_tot11[i-1]], method='hybr')
        if sol.success:
            phiset_tot11[i] = sol.x
            U_minset_tot11[i] = total_potential_parallel(dic, fixu*h0, sol.x, 1, 1)
            err11.append(t_num([phiset_tot11[i]]))
        else:
            print('Optimization failed. Return estimated path.')
            return (u_set1*h0, phiset_tot1, U_minset_tot1, err1)
            break
        i += 1
    return (u_set1*h0, phiset_tot11, U_minset_tot11, err11)