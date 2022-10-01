import numpy as np
import kinematics
import modern_robotics as mr

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

# tol = 1e-4

# for i in range(100):
#     print('--------- 5DOF Test: ', i, '------------')
#     joint_angles = np.random.uniform(-3.14, 3.14, 5)
#     print('joint angles', joint_angles)
#     T = kinematics.FK_dh(dh_params, joint_angles, 5)
#     theta = kinematics.IK_geometric(dh_params, T)
#     err = np.abs((theta.T - joint_angles).T)
#     best_soln_idx = np.argmin(np.nanmax(err, axis=0))
#     print(best_soln_idx, theta[:,best_soln_idx])
#     print('theta', theta.T)
#     print('max err', np.nanmax(err, axis=0))
#     print('T fk\n',T)
#     print('T ik\n', kinematics.FK_dh(dh_params, theta[:,best_soln_idx], 5))

#     assert(np.max(err[:,best_soln_idx]) < tol)



# UNIT TESTING FOR 6 DOF IK
# dh_params = np.array([[0, np.pi/2, 0.10391, 0],
#                       [.20573, 0, 0, 0],
#                       [0, np.pi/2, 0, 0],
#                       [0,-np.pi/2, .26895, 0],
#                       [0, np.pi/2, 0, 0],
#                       [0, 0, .17636, 0]])

# joint_limits = np.array([[-np.pi, np.pi],
#                         [-1.5, np.pi/2],
#                         [-np.pi/2, 1.6],
#                         [-np.pi, np.pi],
#                         [-np.pi/2, np.pi/2],
#                         [-np.pi, np.pi]])

# tol = 1e-4

# for i in range(100):
#     print('--------- 6DOF Test: ', i, '------------')
#     joint_angles = np.random.uniform(-3.14, 3.14, 6)
#     print('joint angles', joint_angles)
#     T = kinematics.FK_dh(dh_params, joint_angles, 6)
#     theta = kinematics.IK_6dof(dh_params, T)
#     err = np.abs((theta.T - joint_angles).T)
#     best_soln_idx = np.argmin(np.nanmax(err, axis=0))
#     print(best_soln_idx, theta[:,best_soln_idx])
#     # print('theta', theta.T)
#     # print('max err', np.nanmax(err, axis=0))
#     print('T fk\n',T)
#     print('T ik\n', kinematics.FK_dh(dh_params, theta[:,best_soln_idx], 6))

#     assert(np.max(err[:,best_soln_idx]) < tol)

# pose = kinematics.FK_dh(dh_params, [0.0, 0.0, 0.0, 0.0, 0.0], 5)

# theta = kinematics.IK_6dof(dh_params, pose)
# print(theta)
# kinematics.pick_6dof_soln(theta, joint_limits)

# M01 = np.array([[0, 0, -1, 0],
#                 [1, 0, 0, 0],
#                 [0, -1, 0, .10391],
#                 [0, 0, 0, 1]])
# M12 = np.array([[0, -1, 0, 0.05],
#                 [-1, 0, 0, -0.2],
#                 [0, 0, -1, 0],
#                 [0, 0, 0, 1]])
# M23 = np.array([[0, 1, 0, 0],
#                 [-1, 0, 0, -0.2],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]])
# M34 = np.array([[0, 0, 1, 0],
#                 [1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 0, 1]])
# M45 = np.array([[1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, 0.131],
#                 [0, 0, 0, 1]])
# M56 = np.array([[1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, 0.104315],
#                 [0, 0, 0, 1]])                
# Mlist = np.array([M01,M12,M23,M34,M45,M56])
# m1 = 0.257774
# m2 = 0.297782
# m3 = 0.258863
# m4 = 0.084957
# m5 = 0.072885
# G1 = np.diag([0.0,0.0,0.0,m1,m1,m1])
# G2 = np.diag([0.0,0.0,0.0,m2,m2,m2])
# G3 = np.diag([0.0,0.0,0.0,m3,m3,m3])
# G4 = np.diag([0.0,0.0,0.0,m4,m4,m4])
# G5 = np.diag([0.0,0.0,0.0,m5,m5,m5])
# Glist = np.array([G1,G2,G3,G4,G5])
# spring_constants = np.array([0.0, 0.02, 0.045, 0.015, 0.01])

# theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# g = np.array([0, 0, -9.8])
# g_forces = mr.GravityForces(theta, g, Mlist, Glist, s_lst.T)
# print("g_force", g_forces)
# corrections = spring_constants*g_forces
# print("theta: ", theta)
# print("corrected theta: ", theta-corrections)
# print("original pose:\n", pose)
# print("achieved pose:\n", kinematics.FK_dh(dh_params, theta-corrections, 5))

# Verifying g correction
# gearbox_k = np.array([0.0, 0.0194, 0.0819, 0.0, 0.0])
# theta = np.array([0.0, 0.5, 0.0, 0.0, 0.0])
# g_forces = kinematics.get_grav(theta, s_lst)
# corrections = gearbox_k*g_forces
# T_dh = kinematics.FK_dh(dh_params, theta, 5)
# T_corr = kinematics.FK_dh(dh_params, theta + corrections, 5)
# print(corrections)
# print("DH:\n", T_dh)
# print("Corr:\n", T_corr)
# print("diff\n", T_dh-T_corr)

pose = np.load('pose_data3.npy')
theta = np.load('theta_data3.npy')
for i in range(theta.shape[0]):
    print('theta:\n', theta[i])
    T_dh = kinematics.FK_dh(dh_params, theta[i], 5)
    print("DH:\n", T_dh)
    print("Actual:\n", pose[:,:,i])