import numpy as np
import kinematics

dh_params = np.array([[0, np.pi/2, 0.10391, 0],
                      [.20573, 0, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])
M = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0.42415],
              [1, 0, 0, 0.30391],
              [0, 0, 0, 1]])
s_lst = np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0.10391, 0],
                  [1, 0, 0, 0, 0.30391, -0.05],
                  [1, 0, 0, 0, 0.30391, -0.25],
                  [0, 1, 0, -0.30391, 0, 0]])

# 5 dof testing
# joint_angles = np.array([0.8,0.4,-0.8,1,0.5],dtype=np.float64)

# T = kinematics.FK_dh(dh_params, joint_angles, 5)
# Tpox = kinematics.FK_pox(joint_angles, M, s_lst)

# print("DH\n", T)
# print("PoX\n", Tpox)
# print("error")
# print(Tpox-T)
# theta = kinematics.IK_geometric(dh_params, T)
# for i in range(4):
#     print('theta', i)
#     print(theta[:,i].T)

dh_params = np.array([[0, np.pi/2, 0.10391, 0],
                      [.20573, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0,-np.pi/2, .26895, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17636, 0]])

# test individual points for 6dof ik
joint_angles = np.array([0.71709059, 0.81935725, 1.43205951, 0.96752293, 0.83272404, 0.90726165],dtype=np.float64)
T = kinematics.FK_dh(dh_params, joint_angles, 6)
print(T)
theta = kinematics.IK_6dof(dh_params, T)
print('joint angles', joint_angles)
for i in range(8):
    print('theta', i)
    print(theta[:,i].T)
    print(T)
    print(kinematics.FK_dh(dh_params, theta[:,i], 6))

# tol = 1e-4

# for i in range(10):
#     print('--------- Test: ', i, '------------')
#     joint_angles = np.random.uniform(-1.56, 1.56, 6)
#     print('joint angles', joint_angles)
#     T = kinematics.FK_dh(dh_params, joint_angles, 6)
#     theta = kinematics.IK_6dof(dh_params, T)
#     err = np.abs((theta.T - joint_angles).T)
#     best_soln_idx = np.argmin(np.nanmax(err, axis=0))
#     print(best_soln_idx, theta[:,best_soln_idx])
#     print('theta', theta.T)
#     print('max err', np.nanmax(err, axis=0))
#     assert(np.max(err[:,best_soln_idx]) < tol)
#     # print(T)
    # print(kinematics.FK_dh(dh_params, theta[:,0], 6))