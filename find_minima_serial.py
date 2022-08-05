import numpy as np
import pandas as pd
import scipy as sp
import sympy as sym
from scipy import spatial
from scipy import optimize
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

import sys

from funcs import *

#-------------------------------------INITIALIZATION----------------------------------------------

scaleup = 1.0
fscale = 1.0

n = 4       # n-sided polygons
R0 = 90*scaleup     # in mm

# magnet
Kd = 0.8*10**6
Rm = 5    # in mm
tm = 10   # in mm
miu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
M = np.sqrt(2*Kd/miu0)    # magnetization (A/m)
V = (tm/1000)*np.pi*(Rm/1000)**2    # bar magnet volume (m3)
mag_m = M*V    # magnet moment (A*m2)
mag_m = mag_m*(scaleup**2.5)*fscale**0.5    # magnet moment (A*m2)

mag_m = 0**0.5
# tempa = np.repeat(np.array([[1],[-1]]), n//2 ,axis=1).T.flatten()
# tempb = np.repeat(np.array([[-1],[1]]), n//2 ,axis=1).T.flatten()
tempa = np.repeat(np.array([[1],[-1]]), n//2 ,axis=1).T.flatten()
tempb = np.repeat(np.array([[-1],[1]]), n//2 ,axis=1).T.flatten()
mag_arrange = np.hstack([tempa, tempb])
mag_arrange = np.vstack([np.zeros(len(mag_arrange)), np.zeros(len(mag_arrange)), mag_arrange]).T

# truss
ks = 0.02656*fscale/scaleup**5

# geometery 
# w0 = [107.794937*scaleup, 29.655172/180*np.pi]
#w0 = [81.9497487437186*scaleup, 96.4824120603015/180*np.pi]
# w0 = [140.182*scaleup, 60.60606/180*np.pi] #bistable n=4
w0 = [140*scaleup, 60/180*np.pi]

dic = create_KTgeometry(n, R0, w0[0], w0[1], ks, mag_m, mag_arrange)


# symbolic expression
Uz = sym.Symbol('Uz')
Ang = sym.Symbol('Ang')
# analytical equation for total potential energy
total_sym = total_potential_sym(dic['points_ref'], dic['n'], dic['R0'], dic['h0'], 
                                dic['theta1'], dic['ks'], dic['mag_m'], dic['mag_arrange'], [Uz, Ang])
# force in kN (ks*h0^2 is the factor used to normalize potention, Uz is in mm)
force_sym = sym.diff(total_sym, Uz)*dic['ks']*dic['h0']**2/1000
# torque in kN*m (Ang is in degree, need to convert to rad)
torque_sym = sym.diff(total_sym, Ang)*dic['ks']*dic['h0']**2/1e6
# Hessian
h11_sym = sym.diff(force_sym, Uz)*1e3    # in kN/m
h12_sym = sym.diff(force_sym, Ang)    # in kN
h21_sym = sym.diff(torque_sym, Uz)*1e3    # in kN
h22_sym = sym.diff(torque_sym, Ang)   # in kN*m

def f_num(x):
    return [total_Zforce_parallel(dic, [x[0]], [x[1]], 1, 1)[0][0], 
            total_torque_parallel(dic, [x[0]], [x[1]], 1, 1)[0][0]]

def f_sym(x):
    f = force_sym.evalf(subs={Uz:x[0], Ang:x[1]})
    t = torque_sym.evalf(subs={Uz:x[0], Ang:x[1]})
    return [f, t]

if mag_m==0:
    f_this = f_sym
    eps_this = 1e-6
# with magnet, use numerical sol to save time
else:
    f_this = f_num
    eps_this = None


#-------------------------------------SOLVER------------------------------------------------------
maxdown = -0.8
maxup = 0.8
N1 = 30
N2 = 30

XX = np.linspace(maxdown*dic['h0'], maxup*dic['h0'], N1)
YY = np.linspace(dic['phi_min'], dic['phi_max'], N2)

succ_sol = []
for i in XX:
    for j in YY:
        sol = optimize.root(f_this, [i, j], method='hybr', options=dict(eps=eps_this))
        if sol.success:
            succ_sol.append(sol.x)
        else:
            succ_sol.append([None, None])

sol = optimize.root(f_num, [0, 0], method='hybr')
if sol.success:
    succ_sol.append(sol.x)
else:
    succ_sol.append([None, None])
    
# remove pts not in domain
succ_sol = np.array(succ_sol).astype('f')
succ_sol = succ_sol[~np.isnan(succ_sol)]
succ_sol = succ_sol.reshape([len(succ_sol)//2, 2])

if len(succ_sol)==0:
    print('All failed.')

result = np.unique(np.around(succ_sol, decimals=2), axis=0)
result_df = pd.DataFrame(data=result)
result_df_sel = result_df[(result_df[0]<=maxup*dic['h0']) & (result_df[0]>=maxdown*dic['h0']) & (result_df[1]<=dic['phi_max']) & (result_df[1]>=dic['phi_min']) ]
result_arr_sel = result_df_sel.to_numpy()      


# check hessian
pt_found = []
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
        pt_found.append(check_pt)
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
with open(f'n{n}_m{mag_m:.4f}_result.csv', 'a') as f:
    h0 = dic['h0']
    theta1 = dic['theta1']
    f.write(f'#(h0,theta0)=( {h0:.3f} {theta1:3f} )\n')
    result_df2.to_csv(f, header=['u', 'phi', 'hessian_sign', 'E'], index=None, sep=' ', mode='a')
    f.write('\n')


#--------------------------------------------------------PLOTTING------------------------------------------------------

NN = 400
px, py = np.mgrid[maxdown*dic['h0']:maxup*dic['h0']:complex(0, NN), dic['phi_min']:dic['phi_max']:complex(0, NN)]
# force and total energy calc
Zforce_tot = total_Zforce_parallel(dic, px, py, NN, NN)
torque_tot = total_torque_parallel(dic, px, py, NN, NN)
energy_tot = total_potential_parallel(dic, px, py, NN, NN, norm=True)

# plots
pt_found = np.array(pt_found)


fig1 = plt.figure(figsize=[4.25,3])
ax1 = fig1.add_subplot(111)
# plt.imshow(np.flipud((Zforce_tot.T)), vmin=-0.00001, vmax=0.00001, extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=0.9, cmap=plt.cm.bwr)
# plt.colorbar()
# plt.imshow(np.flipud((torque_tot.T)), vmin=-np.max(np.abs(torque_tot)), vmax=np.max(np.abs(torque_tot)), extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=1.0, cmap=plt.cm.PuOr)
plt.imshow(np.flipud((torque_tot.T)), vmin=-0.00001, vmax=0.00001, extent=[maxdown, maxup, dic['phi_min'], dic['phi_max']], 
    aspect="auto", alpha=1.0, cmap=plt.cm.PuOr)
# plt.scatter(XX[np.where(minima!=None)]/h0, YY[np.where(minima!=None)], s=5, c='gold', alpha=0.05)
# plt.scatter((XX[num_pair]/h0), (YY[num_pair]), c='lime', s=10)  # pts satisfying Hessian condition
plt.colorbar()
plt.scatter(pt_found[:,0]/dic['h0'], pt_found[:,1], s=15, c='k', marker='x')
ax1.set_ylabel('$\phi$ (rad)')
ax1.set_xlabel('$u/h_0$')
h0 = dic['h0']
theta1 = dic['theta1']
ax1.set_title(f'Torque\n{h0:.2f}mm_{theta1:.1f}(rad)', fontsize=12)
plt.ylim(dic['phi_min'], dic['phi_max'])
plt.xlim(maxdown, maxup)
plt.tight_layout()
plt.show()


fig2 = plt.figure(figsize=[4.25,3])
ax2 = fig2.add_subplot(111)
# plt.imshow(np.flipud((Zforce_tot.T)), vmin=-0.00001, vmax=0.00001, extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=0.9, cmap=plt.cm.bwr)
# plt.imshow(np.flipud((Zforce_tot.T)), vmin=-np.max(np.abs(Zforce_tot)), vmax=np.max(np.abs(Zforce_tot)), extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=1.0, cmap=plt.cm.bwr)
plt.imshow(np.flipud((Zforce_tot.T)), vmin=-0.00001, vmax=0.00001, extent=[maxdown, maxup, dic['phi_min'], dic['phi_max']], 
    aspect="auto", alpha=1.0, cmap=plt.cm.bwr)
plt.colorbar()
# plt.scatter(XX[np.where(minima!=None)]/h0, YY[np.where(minima!=None)], s=5, c='gold', alpha=0.05)
# plt.scatter((XX[num_pair]/h0), (YY[num_pair]), c='lime', s=10)  # pts satisfying Hessian condition
plt.scatter(pt_found[:,0]/dic['h0'], pt_found[:,1], s=15, c='k', marker='x')
ax2.set_ylabel('$\phi$ (rad)')
ax2.set_xlabel('$u/h_0$')
h0 = dic['h0']
theta1 = dic['theta1']
ax2.set_title(f'Force\n{h0:.2f}mm_{theta1:.1f}(rad)', fontsize=12)
plt.ylim(dic['phi_min'], dic['phi_max'])
plt.xlim(maxdown, maxup)
plt.tight_layout()
plt.show()


fig3 = plt.figure(figsize=[4.25,3])
ax3 = fig3.add_subplot(111)
# plt.imshow(np.flipud((Zforce_tot.T)), vmin=-0.00001, vmax=0.00001, extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=0.9, cmap=plt.cm.bwr)
# plt.imshow(np.flipud((energy_tot.T)), extent=[maxdown, maxup, phi_min, phi_max], aspect="auto", alpha=1.0, cmap=plt.cm.copper)
temp3 = np.flipud((energy_tot.T)).copy()
# temp3[np.where(temp3>0.4)] = None
plt.imshow(temp3, extent=[maxdown, maxup, dic['phi_min'], dic['phi_max']], aspect="auto", alpha=1.0, cmap=plt.cm.copper)
plt.colorbar()
# plt.scatter(XX[np.where(minima!=None)]/h0, YY[np.where(minima!=None)], s=5, c='gold', alpha=0.05)
# plt.scatter((XX[num_pair]/h0), (YY[num_pair]), c='lime', s=10)  # pts satisfying Hessian condition
plt.scatter(pt_found[:,0]/dic['h0'], pt_found[:,1], s=15, c='white', marker='x')
ax3.set_ylabel('$\phi$ (rad)')
ax3.set_xlabel('$u/h_0$')
h0 = dic['h0']
theta1 = dic['theta1']
ax3.set_title(f'Energy\n{h0:.2f}mm_{theta1:.1f}rad', fontsize=12)
plt.ylim(dic['phi_min'], dic['phi_max'])
plt.xlim(maxdown, maxup)
plt.tight_layout()
plt.show()
