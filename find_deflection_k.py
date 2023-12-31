from tkinter import N
import numpy as np
from scipy.optimize import least_squares
import kinematics

'''
This files uses least squares optimizaion to find
gearbox spring constants which minimize the position error
from a dataset.
'''

dataset = 9
th_data = np.load('theta_data{}.npy'.format(dataset), allow_pickle=True) # n x 5, each row is a set of theta values
pose_data = np.load('pose_data{}.npy'.format(dataset), allow_pickle=True) # 4x4xn each 4x4 is a transformation matrix

dh_params = np.array([[0, -np.pi/2, 0.10391, 0],
                      [.20573, np.pi, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])
M = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0.42415],
              [1, 0, 0, 0.30391],
              [0, 0, 0, 1]])
s_lst = np.array([[0, 0, 1, 0, 0, 0],
                  [-1, 0, 0, 0, -0.10391, 0],
                  [1, 0, 0, 0, 0.30391, -0.05],
                  [1, 0, 0, 0, 0.30391, -0.25],
                  [0, 1, 0, -0.30391, 0, 0]])

def residual(k, theta, pose):
    n = theta.shape[0]
    err = np.zeros(n)
    for i in range(n):
        grav_torques = kinematics.get_grav(theta[i], s_lst)
        corrections = k*grav_torques
        T_corr = kinematics.FK_dh(dh_params, theta[i]-corrections, 5)
        err[i] = np.linalg.norm(pose[:3,3,i] - T_corr[:3,3])
    return err

k0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
opt_soln = least_squares(residual, k0, args=(th_data,pose_data), bounds=(0,1), verbose=2)
print(opt_soln.x)
