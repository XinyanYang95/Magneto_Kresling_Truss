import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import colors
import time
from sympy import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sympy as sym
from scipy.optimize import minimize

from mpi4py import MPI
from scipy import optimize
import pandas as pd
import sympy as sym

#----------------------------------------general function---------------------------------------------
def magnet_force4(Kd, Rm, tm, x, r, upper_bound=200, interaction='repulsive'):
    """Get magnet force from current config. 

    PARAMETERS
        Magnets parameters:
            Kd: 2D arrary [num_edges, 2].
            Rm: magnet radius, float (in mm).
            tm: magnet thickness, float (in mm).
        Orientation parameters:
            x: magnets vertical gap (face-to-face distance), float (in mm).
            r: magnets horizontal distance (center-to-center distance), float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
        
    RETURN
        F_mag: magnet force output (negative magnet force), float (in kN).
    """
    sign = 1
    if interaction=='repulsive':
        sign = -sign
        
    Z_dist = x + tm
    cosi = Z_dist/Rm
    
    F_mag = sp.integrate.quad_vec(lambda q: sign*8*np.pi*Kd*(Rm/1000)**2*sp.special.jv(0, (r/1000)*q/(Rm/1000))*(sp.special.jv(1, q))**2/q*np.sinh(q*tao1)*np.sinh(q*tao2)*np.e**(-q*cosi), 
                 0, upper_bound)[0]
    
    return F_mag*0.001

def Force_from_vec(mag_m, n1, n2, r_vec12):
    """Get magnet force from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        Magnets parameters:
            Kd: 2D arrary [num_edges, 2].
            Rm: magnet radius, float (in mm).
            tm: magnet thickness, float (in mm).
        Orientation parameters:
            n1: m1 dipole moment direction vector, 1D array [3,] (in A*m2).
            n2: m2 dipole moment direction vector, 1D array [3,] (in A*m2).
            r_vec: distance vector of two magnets, from point dipole 1 to point dipole 2 (pt2_vec - pt1_vej), 
                   1D array [3,] (in mm).
       
    RETURN
        F_mag: magnet force output (negative magnet force) on point dipole 2, float (in kN).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    
    F_from_vec = -3*miu0/(np.pi*4*np.linalg.norm(r_vec)**5)*(np.inner(m1, r_vec)*m2 + np.inner(m2, r_vec)*m1 
        + np.inner(m1, m2)*r_vec - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec/np.linalg.norm(r_vec)**2)
    return F_from_vec*0.001
    
    # F_from_vec = -3*miu0/(np.pi*4*np.linalg.norm(r_vec)**5)*(np.inner(m1, r_vec)*m2 + np.inner(m2, r_vec)*m1 
    #     + np.inner(m1, m2)*r_vec - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec/np.linalg.norm(r_vec)**2)
    # return F_from_vec*0.001

def get_mag_force(mag_m, mag_points, mag_arrange):
    """Get magnets force output at magnet centers based on position and dipole arrangement.
       Assume point dipoles has fixed dipole arrangement. Sliding and overturning are prohibited.
       If dipole tilts, need modification (include in-plane magnet-magnet interactions of each polygon).
       
    PARAMETERS
        Magnets parameters:
            m: magnet moment, float (in A*m2).
        mag_points: points of magnetic dipoles, 2D arrary [num_magnets x 3].
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets x 3].
    
    RETURN
        forces: lower level magnets follwed by upper layer's, 2D array, [num_pts, 3] (in kN).
    """
    forces = np.zeros([2*n, 3])
    for j in range(n,2*n):
        force_j = 0
        for i in range(n):
            force_j = force_j+Force_from_vec(mag_m, mag_arrange[i], mag_arrange[j], 
                                             r_vec12=mag_points[j]-mag_points[i])
        forces[j] = force_j
    for j in range(n):
        force_j = 0
        for i in range(n, 2*n):
            force_j = force_j+Force_from_vec(mag_m, mag_arrange[i], mag_arrange[j], 
                                             r_vec12=mag_points[j]-mag_points[i])
        forces[j] = force_j
    return forces

def get_total_force(o_points, w, mag_m, mag_arrange):
    """Get magnets force output at magnet centers based on position and dipole arrangement.
       Assume point dipoles with fixed dipole arrangement. Sliding and overturning are prohibited. 
       
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        Kd: 2D arrary [num_edges, 2].
        Rm: magnet radius, float (in mm).
        tm: magnet thickness, float (in mm).
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets x 3].
    
    RETURN
        forces: lower level magnets follwed by upper layer's, 2D array, [num_pts, 3] (in kN).
    """
    mag_points = update_pts(o_points, w)
    tot_force =  get_current_force(o_points, w) + get_mag_force(mag_m, mag_points, mag_arrange)
    return tot_force

def upperlayer_total_force(o_points, w, mag_m, mag_arrange):
    """Get upper layer polygon truss and magnetic forces and moments based on reference pts and displcement vector, w. 
    
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        Kd: 2D arrary [num_edges, 2].
        Rm: magnet radius, float (in mm).
        tm: magnet thickness, float (in mm).
        mag_arrange: magnetic dipole arrangements (pointing from N to S), 2D arrary [num_magnets x 3].
    
    RETURN
        Fx: resultant force of the upper layer/ upper layer force output in X-dir, float (in kN).
        Fy: resultant force of the upper layer/ upper layer force output in Y-dir, float (in kN).
        Fz: resultant force of the upper layer/ upper layer force output in Z-dir, float (in kN).
        Mx: resultant moment of the upper layer/ upper layer moment output around pos. X-axis, float (in kN*m).
        My: resultant moment of the upper layer/ upper layer moment output around pos. Y-axis, float (in kN*m).
        Mz: resultant torque of the upper layer/ upper layer moment output around pos. Z-axis, float (in kN*m).
    """
    n_points = update_pts(o_points, w)
    Q = get_total_force(o_points, w, mag_m, mag_arrange)
    
    Fx = np.sum(Q[n:,0])
    Fy = np.sum(Q[n:,1])
    Fz = np.sum(Q[n:,2])
    
    Mx = np.sum(n_points[n:, 1]*Q[n:, 2])*0.001
    My = -np.sum(n_points[n:, 0]*Q[n:, 2])*0.001
    Mz = np.sum(n_points[n:, 0]*Q[n:, 1])*0.001 - np.sum(n_points[n:, 1]*Q[n:, 0])*0.001
#     Mz2 = np.sum(n_points[:n, 0]*Q[:n, 1]) - np.sum(n_points[:n, 1]*Q[:n, 0])*0.001
    
    return np.array([Fx, Fy, Fz, Mx, My, Mz])

def magnet_potential(Z_dist, r=0, upper_bound=200, norm=True):
    """Get magnetostatic interaction energy from current config. 

    PARAMETERS
        Z_dist: magnets center-to-center distance, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
        norm: if turned on, devide the potential by ks*h0**2, boolean.
        
    RETURN
        E_mag: magnetostatic interaction energy, float (in J or dimensionless).
    """
    cosi = Z_dist/Rm
    E_mag = sp.integrate.quad_vec(lambda q: 8*np.pi*Kd*(Rm/1000)**3*sp.special.jv(0, (r/1000)*q/(Rm/1000))*(sp.special.jv(1, q))**2/q**2*np.sinh(q*tao1)*np.sinh(q*tao2)*np.e**(-q*cosi), 
             0, upper_bound)[0]        
    if norm:
        E_mag = E_mag/(ks*h0**2*0.001)
    return E_mag

def Energy_from_vec(mag_m, n1, n2, r_vec12, norm=True):
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
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J or fimensionless).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    
    E_from_vec = miu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec)**3  
                    - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/np.linalg.norm(r_vec)**5)
    
    if norm:
        E_from_vec = E_from_vec/(ks*h0**2*0.001)
    return E_from_vec

@np.vectorize
def update_pts_vec(u, phi, size1=N, size2=N):
    """Update points positions by phi and u. Assumptions here are the upper layer is a rigid polygon
       and points are on the same height. Can add another angle to simulate cases with pts not of the
       same height. But the rigid polygon assumption should be strictly kept.

    PARAMETERS
        o_points: points need updating, 2D arrary [num_pts, 3] (in mm).
        w: [u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
    RETURN
        all_points: 2D array, [num_pts, 3].
    """
    points = points_ref
    global special_i   # vectorize loop has already started from here (u is not in [N,N], u is a scalar)
    phi = phi/180*np.pi
    rot = np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
#     print(rot)
    points_rot = np.matmul(rot, points.T)
    points_rot = points_rot.T
    points_rot[:,2] = points_rot[:,2] + u
#     print(points)
    points_rot[:n] = points[:n].copy()
    if special_i<size1*size2:
        points_updated_set[special_i] = points_rot
#         print(special_i)
    special_i += 1
    pass

# rotation against z-axis, about the origin
def rotation(ang, points):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in degree).
        points: positions of points need rotating, 2D arrary [num_pts, 3].
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    # ang in degree
    ang = ang/180*np.pi
    rot = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    points_rot = np.matmul(rot, points.T)
    return points_rot.T

# rotation against z-axis, about the origin
@np.vectorize
def rotation_vec(ang, size1=N, size2=N):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in degree).
        points: positions of points need rotating, 2D arrary [num_pts, 3].
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    # ang in degree
    points = points_ref
    global special_i
    ang = ang/180*np.pi
    rot = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    points_rot = np.matmul(rot, points.T)
    points_rot = points_rot.T
    if special_i<size1*size2:
        points_rot_set[special_i] = points_rot
#         print(special_i)
    special_i += 1
    pass

def get_adj(connectivity):
    """Get adjacency matrix from connectivity.

    PARAMETERS
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        
    RETURN
        adj: 2D array, [num_pts, num_pts].
    """
    adj = np.zeros([num_pts, num_pts])
    for pair in connectivity:
        adj[pair[0], pair[1]] = 1
        adj[pair[1], pair[0]] = 1
    return adj

def get_edge_vector(connectivity, points):
    """Get edge vectors from connectivity and point positions.

    PARAMETERS
        points: positions of points, 2D arrary [num_pts, 3].
        connectivity: point i and point j of each truss/edge, 2D arrary [num_edges, 2].
        
    RETURN
        edge_vec: 2D array, [num_edges, 3].
    """
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
    base = o_points.copy()[:n]
    points = o_points.copy()[n:]
    points[:,2] = points[:,2] + np.ones(len(points))*u
    points =  rotation(phi, points)
    all_points = np.vstack([base, points])
    return all_points

def get_length(n_points):
    """Get length of each truss element/edge based on current geometry (extension positive). 

    PARAMETERS
        n_points: current positions of points, 2D arrary [num_pts, 3].
        
    RETURN
        length: 1D array, [num_edges, ] (in mm).
    """
    edge_vec_new = get_edge_vector(connectivity, n_points)
    length = np.linalg.norm(edge_vec_new, axis=1)
    return length

def get_Qx(n_points):
    """Get basic forces of each truss element/edge based on current geometry (tension positive). 
       Here, use rotated engineering deformation (RE, Ln-L). Nonlinear kinematics is considered.

    PARAMETERS
        n_points: current positions of points, 2D arrary [num_pts, 3].
        
    RETURN
        Qx: 1D array, [num_edges, ] (in kN).
    """
    length0 = np.linalg.norm(edge_vec0, axis=1)
    edge_vec_new = get_edge_vector(connectivity, n_points)
    length = np.linalg.norm(edge_vec_new, axis=1)
    Qx = (length - length0)*ks*0.001
    Qx = Qx.reshape([num_edges, 1])
    return Qx

def get_Bx(n_points):
    """Get force influence matrix B based on current geometry. Nonliner statics is considered.
    
    PARAMETERS
        n_points: current positions of points, 2D arrary [num_pts, 3].
    
    RETUTN
        Bx: 2D array, [num_pts*3, num_edges].
    """
    temp = np.linalg.norm(get_edge_vector(connectivity, n_points), axis=1)
    b_vec = get_edge_vector(connectivity, n_points)/temp.reshape([len(temp),1])
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

def get_current_force(o_points, w):
    """Get truss forces at DOFs based on current geometry (positive force in positive direction). 
       Current geometry is obtained from old positions and displacements. Nonlinear statics is considered. 
       
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
    
    RETURN
        Q: 2D array, [num_pts, 3] (in kN).
    """
    n_points = update_pts(o_points, w)
    Qx = get_Qx(n_points)
    Bx = get_Bx(n_points)
    Q = np.matmul(Bx, Qx)
    Q = Q.reshape([num_pts, 3])
    return Q

def upperlayer_force(o_points, w):
    """Get upper layer polygon truss forces and moments based on reference pts and displcement vector, w. 
    
    PARAMETERS
        o_points: points before updating, 2D arrary [num_pts x 3].
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
    
    RETURN
        Fx: resultant force of the upper layer/ upper layer force output in X-dir, float (in kN).
        Fy: resultant force of the upper layer/ upper layer force output in Y-dir, float (in kN).
        Fz: resultant force of the upper layer/ upper layer force output in Z-dir, float (in kN).
        Mx: resultant moment of the upper layer/ upper layer moment output around pos. X-axis, float (in kN*m).
        My: resultant moment of the upper layer/ upper layer moment output around pos. Y-axis, float (in kN*m).
        Mz: resultant torque of the upper layer/ upper layer moment output around pos. Z-axis, float (in kN*m).
    """
    n_points = update_pts(o_points, w)
    Q = get_current_force(o_points, w)
    
    Fx = np.sum(Q[n:,0])
    Fy = np.sum(Q[n:,1])
    Fz = np.sum(Q[n:,2])
    
    Mx = np.sum(n_points[n:, 1]*Q[n:, 2])*0.001
    My = -np.sum(n_points[n:, 0]*Q[n:, 2])*0.001
    Mz = np.sum(n_points[n:, 0]*Q[n:, 1])*0.001 - np.sum(n_points[n:, 1]*Q[n:, 0])*0.001
#     Mz2 = np.sum(n_points[:n, 0]*Q[:n, 1]) - np.sum(n_points[:n, 1]*Q[:n, 0])*0.001
    
    return np.array([Fx, Fy, Fz, Mx, My, Mz])

def get_R(Q):
    """Get reaction force from current force. Current force Q current structure response. 
       First n entries are lower layer polygon points (fixed), then n entries for upper layer points.
       
    PARAMETERS
        Q: current force, 2D array [num_pts, 3].
        
    RETURN
        R: reaction force, 1D array [3,].
    """
    R = sum(Q[:n])
    return R

def get_P(Q):
    """Get applied force from current force. Current force Q is current structure response. 
       First n entries are lower layer polygon points (fixed), then n entries for upper layer points.
       
    PARAMETERS
        Q: current force, 2D array [num_pts, 3].
        
    RETURN
        P: applied force, 1D array [3,].
    """
    P = sum(Q[n:])
    return P

def elastic_U(w):
    """Get truss normalized elastic energy based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, 'upward' positive, float (in mm).
    
    RETUTN
        U_norm: float (dimensionless).
    """
    u0, phi0 = [0, 0]
    a0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2/180*np.pi+theta1/2/180*np.pi-np.pi/2/n))**2)
    b0 = np.sqrt((h0+u0)**2+4*R0**2*(np.sin(phi0/2/180*np.pi+theta1/2/180*np.pi+np.pi/2/n))**2)
    
    u, phi = w
    a = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2/180*np.pi+theta1/2/180*np.pi-np.pi/2/n))**2)    # mm
    b = np.sqrt((h0+u)**2+4*R0**2*(np.sin(phi/2/180*np.pi+theta1/2/180*np.pi+np.pi/2/n))**2)    # mm
    
    U = 0.5*n*ks*(a-a0)**2 + 0.5*n*ks*(b-b0)**2    # in kN/m*mm2  (or 1e-3J)
    U_norm = U/ks/h0**2    # dimensionless
    
    return U_norm

def Energy_from_vec(mag_m, n1, n2, r_vec12, norm=True):
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
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J or fimensionless).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    
    E_from_vec = miu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec)**3  
                    - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/np.linalg.norm(r_vec)**5)
    
    if norm:
        E_from_vec = E_from_vec/(ks*h0**2*0.001)
    return E_from_vec

def this_magent_energy(mag_m, mag_points, mag_arrange, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    temp = 0
    for i in range(n):
        for j in range(n,2*n):
            temp = temp + Energy_from_vec(mag_m, mag_arrange[i], mag_arrange[j], r_vec12=mag_points[j]-mag_points[i], norm=norm)
    return temp

def magnet_potential(o_points, w, mag_m, mag_arrange):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    mag_points = update_pts(o_points, w)
    magnet_E = this_magent_energy(mag_m, mag_points, mag_arrange, norm=True)
    return magnet_E

def total_potential(o_points, w, mag_m, mag_arrange):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    mag_points = update_pts(o_points, w)
    total = elastic_U(w) + this_magent_energy(mag_m, mag_points, mag_arrange, norm=True)
    return total

def magnet_potential_vec(u, phi, size1=N, size2=N, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for ii in range(n):
        for jj in range(n,2*n):
            r_vec12 = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec12 = r_vec12/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]

            this = miu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec12, axis=1)**3  
                         - 3*np.matmul(r_vec12,m1)*np.matmul(r_vec12,m2)/np.linalg.norm(r_vec12, axis=1)**5)
            temp = temp + this/(ks*h0**2*0.001)
    #         print(points_updated_set[:,ii])
    #         print(points_updated_set[:,jj])
    #         print(r_vec12)
    #         print(this)
    #         print(temp)
    #         print('\n')
    temp = temp.reshape([size1, size2])
#     total = elastic_U([u, phi]) + temp
    return temp

def total_potential_vec(u, phi, size1=N, size2=N, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for ii in range(n):
        for jj in range(n,2*n):
            r_vec12 = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec12 = r_vec12/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]

            this = miu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec12, axis=1)**3  
                         - 3*np.matmul(r_vec12,m1)*np.matmul(r_vec12,m2)/np.linalg.norm(r_vec12, axis=1)**5)
            temp = temp + this/(ks*h0**2*0.001)
    temp = temp.reshape([size1, size2])
    total = elastic_U([u, phi]) + temp
    return total

def magnet_Zforce_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    temp = np.zeros(size1*size2)
    for jj in range(n,2*n):
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[2] + np.inner(m2, r_vec)*m1[2] + np.inner(m1, m2)*r_vec[:,2] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,2]/np.linalg.norm(r_vec,axis=1)**2)
            temp = temp + this*0.001
    temp = temp.reshape([size1, size2])
#     total = elastic_U([u, phi]) + temp
    return temp

def magnet_torque_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    
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
            this1 = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[0] + np.inner(m2, r_vec)*m1[0] + np.inner(m1, m2)*r_vec[:,0] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,0]/np.linalg.norm(r_vec,axis=1)**2)
            this2 = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[1] + np.inner(m2, r_vec)*m1[1] + np.inner(m1, m2)*r_vec[:,1] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,1]/np.linalg.norm(r_vec,axis=1)**2)
            temp1 = temp1 + this1*0.001
            temp2 = temp2 + this2*0.001
        arm1 = arm1 + temp1*points_updated_set[:, jj][:,1]*0.001
        arm2 = arm2 + temp2*points_updated_set[:, jj][:,0]*0.001
    torque = (arm2 - arm1).reshape([size1, size2])
#     total = elastic_U([u, phi]) + temp
    return torque

def truss_Zforce_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

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

    return temp    # kN

def truss_torque_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)
    
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

    return torque2    # kN*m


def total_Zforce_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    # magnet force
    temp = np.zeros(size1*size2)
    for jj in range(n,2*n):
        for ii in range(n):
            r_vec = points_updated_set[:, jj]-points_updated_set[:, ii]
            r_vec = r_vec/1000
            m1 = mag_m*mag_arrange[ii]
            m2 = mag_m*mag_arrange[jj]
            this = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[2] + np.inner(m2, r_vec)*m1[2] + np.inner(m1, m2)*r_vec[:,2] 
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

    return temp    # kN

def total_torque_vec(u, phi, size1=N, size2=N):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    global special_i
    global points_updated_set
    special_i = -1
    points_updated_set = np.zeros([size1*size2, 2*n, 3])
    void = update_pts_vec(u, phi, size1, size2)

    
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
            this1 = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[0] + np.inner(m2, r_vec)*m1[0] + np.inner(m1, m2)*r_vec[:,0] 
                    - 5*np.inner(m1, r_vec)*np.inner(m2, r_vec)*r_vec[:,0]/np.linalg.norm(r_vec,axis=1)**2)
            this2 = -3*miu0/(np.pi*4*np.linalg.norm(r_vec, axis=1)**5)*(np.inner(m1, r_vec)*m2[1] + np.inner(m2, r_vec)*m1[1] + np.inner(m1, m2)*r_vec[:,1] 
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

    return torque1 + torque2    # kN*m

def elastic_U_sym(w):
    """Get truss normalized elastic energy based on current geometry. 
       Analytical expression for truss length is used. [FAST]
       Invalid if introducing out-of-plane rotation angle.
    
    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, 'downward' positive, float (in mm).
    
    RETUTN
        U_norm: float (dimensionless).
    """
    u0, phi0 = [0, 0]
    a0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2/180*np.pi+theta1/2/180*np.pi-np.pi/2/n))**2)
    b0 = sqrt((h0+u0)**2+4*R0**2*(sin(phi0/2/180*np.pi+theta1/2/180*np.pi+np.pi/2/n))**2)
    
    u, phi = w
    a = sqrt((h0+u)**2+4*R0**2*(sin(phi/2/180*np.pi+theta1/2/180*np.pi-np.pi/2/n))**2)    # mm
    b = sqrt((h0+u)**2+4*R0**2*(sin(phi/2/180*np.pi+theta1/2/180*np.pi+np.pi/2/n))**2)    # mm
    
    U = 0.5*n*ks*(a-a0)**2 + 0.5*n*ks*(b-b0)**2    # in kN/m*mm2  (or μJ)
    U_norm = U/ks/h0**2    # dimensionless
    
    return U_norm

def rotation_sym(ang, points):
    """Get points positions after rotating counter-clockwisely.
    
    PARAMETERS
        ang: rotation angle, counter-clockwise positive, float (in degree).
        points: positions of points need rotating, 2D arrary [num_pts, 3].
    
    RETURN: 
        positions of points after rotation, 2D arrary [num_pts, 3].
    """
    # ang in degree
    ang = ang/180*np.pi
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
    base = o_points.copy()[:n]
    points = o_points.copy()[n:]
    points = np.hstack([points[:,0], points[:,1], np.ones(len(points))*Uz + points[:,2]]).reshape([3,n]).T
    points =  rotation_sym(phi, points)
    all_points = np.vstack([base, points])
    return all_points

def Energy_from_vec_sym(mag_m, n1, n2, r_vec12, norm=True):
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
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J or fimensionless).
    """
    r_vec = r_vec12/1000
    m1 = mag_m*n1
    m2 = mag_m*n2
    r_vec_norm = sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    
    E_from_vec = miu0/(np.pi*4)*(np.inner(m1, m2)/r_vec_norm**3  
                    - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/r_vec_norm**5)
    
    if norm:
        E_from_vec = E_from_vec/(ks*h0**2*0.001)
    return E_from_vec

def this_magent_energy_sym(mag_m, mag_points, mag_arrange, norm=True):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    temp = 0
    for i in range(n):
        for j in range(n,2*n):
            temp = temp + Energy_from_vec_sym(mag_m, mag_arrange[i], mag_arrange[j], r_vec12=mag_points[j]-mag_points[i], norm=norm)
    return temp

def total_potential_sym(o_points, w, mag_m, mag_arrange):
    """Calculate normalized total potential energy from current config. 

    PARAMETERS
        w:[u, phi]
            phi: twisting angle, counter-clockwise positive, float (in degree).
            u: vertival displacement, upward positive, float (in mm).
        upper_bound: integral upper_bound, if too large, sinh(x) overflow.
    
    RETUTN
        total: float (dimensionless).
    """
    mag_points = update_pts_sym(o_points, w)
    total = elastic_U_sym(w) + this_magent_energy_sym(mag_m, mag_points, mag_arrange, norm=True)
    return total

#-------------------------------------INITIALIZATION----------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    start = time.time()

n=4       # n-sided polygons
R0 = 90     # in mm
# w0 = [134.4, 50]      # initial configuration
Kd=8*1e5    # magnetization in J/m3


P1 = 100
P2 = 100
ratio_set = np.linspace(0.8, 1.8, P1)
angle_set = np.linspace(0,100, P2)
ratio_set , angle_set = np.meshgrid(ratio_set , angle_set)
config_set = [[i, j] for i,j in zip((ratio_set*R0).flatten(), angle_set.flatten())]
config_set1 = config_set[0:250]

jjj = 0
for w0 in config_set1:
    if rank==0:
        print(f'Starting config {jjj} ( {w0[0]:.3f} , {w0[1]:.3} ):')
# for w0 in [[162, 100]]:

    # GLOBAL VARIABLES. DO NOT UPDATE!
    num_pts = 2*n
    num_edges = 2*n
    h0 = w0[0] # in mm
    theta1 = w0[1] # in degree, counter-clockwise
    theta0 = w0[1]-180/n # in degree, counter-clockwise
    phi_max = (180-180/n)-theta1
    phi_min = -(180-180/n+theta1)

    Rm = 5    # in mm
    tm = 10    # in mm
    miu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
    M = np.sqrt(2*Kd/miu0)    # magnetization (A/m)
    V = (tm/1000)*np.pi*(Rm/1000)**2    # bar magnet volume (m3)
    mag_m = M*V    # magnet moment (A*m2)

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
    edge_vec0 = get_edge_vector(connectivity, points_ref)
    adj = get_adj(connectivity)

    # ks = 3.32       # element stiffness kN/m
    ks = 0.02656
    # magnet moment direction vectors of lower and upper polygons
    tempa = np.repeat(np.array([[1],[-1]]), n//2 ,axis=1).T.flatten()
    tempb = np.repeat(np.array([[-1],[1]]), n//2 ,axis=1).T.flatten()
    mag_arrange = np.hstack([tempa, tempb]) 
    mag_arrange = np.vstack([np.zeros(len(mag_arrange)), np.zeros(len(mag_arrange)), mag_arrange]).T
    miu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
    M = np.sqrt(2*Kd/miu0)    # magnetization (A/m)
    V = (tm/1000)*np.pi*(Rm/1000)**2    # bar magnet volume (m3)
    mag_m = M*V    # magnet moment (A*m2)
    # magnet dipole positions
    mag_points = points_ref.copy()

    #---------------------------------DO SOMETHING HERE------------------------------------------------------
    #-------------------START FINDING MINIMUM AND MAXIMUM----------------------------------------------------

    Uz = sym.Symbol('Uz')
    Ang = sym.Symbol('Ang')

    # analytical equation for total potential energy
    total_sym = total_potential_sym(points_ref, [Uz, Ang], mag_m, mag_arrange)
    # force in kN (ks*h0^2 is the factor used to normalize potention, Uz is in mm)
    force_sym = sym.diff(total_sym, Uz)*ks*h0**2/1000
    # torque in kN*m (Ang is in degree, need to convert to rad)
    torque_sym = sym.diff(total_sym, Ang)*ks*h0**2/1e6*(180/np.pi)
    # Hessian
    h11_sym = sym.diff(force_sym, Uz)*1e3    # in kN/m
    h12_sym = sym.diff(force_sym, Ang)*(180/np.pi)    # in kN
    h21_sym = sym.diff(torque_sym, Uz)*1e3    # in kN
    h22_sym = sym.diff(torque_sym, Ang)*(180/np.pi)    # in kN*m

    def f_num(x):
        return [total_Zforce_vec([x[0]], [x[1]], 1, 1)[0][0], total_torque_vec([x[0]], [x[1]], 1, 1)[0][0]]

    def f_sym(x):
        f = force_sym.evalf(subs={Uz:x[0], Ang:x[1]})
        t = torque_sym.evalf(subs={Uz:x[0], Ang:x[1]})
        return [f, t]

    # search grid ---------- change search grid value here --------
    N1=30     
    N2=30
    maxdown = -0.8
    maxup = 0.8
    XX = np.linspace(maxdown*h0, maxup*h0, N1)
    YY = np.linspace(phi_min, phi_max, N2)

    # an array of pts to start for finding Fz=0 and Tz=0
    xy_pair = [[0, 0]]
    for ii in XX:
        for jj in YY:
            xy_pair.append([ii, jj])
    xy_pair = np.array(xy_pair)
    num_pair = len(xy_pair)

    #---------------------------------------------------------------Minimize in parallel---------------------------------------------------------
    comm.Barrier()

    # CASE - number of starting point is larger than number of cores
    # specify the number of starting pts for each core 
    if rank==size-1:
        this_length = num_pair - num_pair//(size-1)*rank
        bound = range(num_pair//(size-1)*rank, num_pair)
    else:
        this_length = num_pair//(size-1)
        bound = range(this_length*rank, this_length*(rank+1))

    # find roots on each core    
    sendbuf = np.empty([this_length, 2], dtype='f')

    # no magnet, use the symbolic eqn
    if Kd==0:
        f_this = f_sym
        eps_this = 1e-6
    # with magnet, use numerical sol to save time
    else:
        f_this = f_num
        eps_this = None

    j = 0
    for i in bound:
        start_pt = xy_pair[i]
        temp3 = optimize.root(f_this, start_pt, method='hybr', options=dict(eps=eps_this))
        if temp3.success:
            sendbuf[j][0], sendbuf[j][1] = temp3.x # sol.x is the root [x,y]
        else:
            sendbuf[j][0], sendbuf[j][1] = np.array([None, None]) # if unsucess,  return NA (will be further checked using Hessian)
            print(f'Failed when starting from ( {start_pt[0]:.3f} , {start_pt[1]:.3f} ), found ( {temp3.x[0]:.3f} , {temp3.x[1]:.3f} ), (F,T)=( {temp3.fun[0]:.3e} , {temp3.fun[1]:.3e} ).')
        j += 1
    sendbuf = sendbuf[~np.isnan(sendbuf)]
    sendbuf = sendbuf.reshape([len(sendbuf)//2, 2])
    sendcounts = np.array(comm.gather(len(sendbuf)*2, 0)) # rank0 gather lengths to be sent from other ranks

    if rank == 0:
        recvbuf = np.empty(sum(sendcounts), dtype='f')
    else:
        recvbuf = None
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)

    #-------------------------------------------------------------Rank 0 write files and plot---------------------------------------------------------
    if rank==0:

        succ_sol = recvbuf.reshape([len(recvbuf)//2, 2])
        result = np.unique(np.around(succ_sol, decimals=2), axis=0) # filter close pts
        result_df = pd.DataFrame(data=result)
        # remove pts not in domain
        result_df_sel = result_df[(result_df[0]<=maxup*h0) & (result_df[0]>=maxdown*h0) & (result_df[1]<=phi_max) & (result_df[1]>=phi_min) ]
        result_arr_sel = result_df_sel.to_numpy()      

        # check hessian
        hess_sign = []
        Etot_set = []
        for check_pt in result_arr_sel:
            # Hessian matrix
            a = h11_sym.evalf(subs={Uz:check_pt[0], Ang:check_pt[1]})
            b = h12_sym.evalf(subs={Uz:check_pt[0], Ang:check_pt[1]})
            c = h21_sym.evalf(subs={Uz:check_pt[0], Ang:check_pt[1]})
            d = h22_sym.evalf(subs={Uz:check_pt[0], Ang:check_pt[1]})
            H = np.array([[a, b],[c, d]])
            Etot = total_sym.evalf(subs={Uz:check_pt[0], Ang:check_pt[1]})
            # convert object to float
            H = H.astype(float)

            if np.all(np.linalg.eig(H)[0]>=0):
                hess_sign.append(1)
            elif np.all(np.linalg.eig(H)[0]<=0):
                hess_sign.append(-1)
            else:
                hess_sign.append(0)
            Etot_set.append(Etot)
                
        hess_sign = np.array(hess_sign)
        Etot_set = np.array(Etot_set)
        Etot_set = Etot_set.astype(float)

        # final result
        result_df2 = pd.DataFrame(data=np.hstack([result_arr_sel, hess_sign.reshape(len(hess_sign),1), Etot_set.reshape(len(Etot_set),1)]), columns=['u','phi','hess sign', 'E'])
        result_df2 = result_df2.sort_values(by=['hess sign'], ascending=False)
        with open(f'n{n}_Kd{Kd/1e6}M_result.cvs', 'a') as f:
            f.write(f'#(h0,theta0)=( {w0[0]:.3f} {w0[1]:3f} )\n')
            result_df2.to_csv(f, header=['u', 'phi', 'hessian_sign', 'E'], index=None, sep=' ', mode='a')
            f.write('\n')
        print(f'Finished config {jjj}.\n')

    comm.Barrier()
    jjj += 1

if rank==0:
    print(f'\nTime elasped: {(time.time() - start):.3f} sec.')