import numpy as np
import kinematics
import modern_robotics as mr

def IK_5DOF_unit_tests(dh_params):
    tol = 1e-4

    for i in range(100):
        print('--------- 5DOF Test: ', i, '------------')
        joint_angles = np.random.uniform(-3.14, 3.14, 5)
        print('joint angles', joint_angles)
        T = kinematics.FK_dh(dh_params, joint_angles, 5)
        theta = kinematics.IK_geometric(dh_params, T)
        err = np.abs((theta.T - joint_angles).T)
        best_soln_idx = np.argmin(np.nanmax(err, axis=0))
        print(best_soln_idx, theta[:,best_soln_idx])
        print('theta', theta.T)
        print('max err', np.nanmax(err, axis=0))
        print('T fk\n',T)
        print('T ik\n', kinematics.FK_dh(dh_params, theta[:,best_soln_idx], 5))

        assert(np.max(err[:,best_soln_idx]) < tol)



# UNIT TESTING FOR 6 DOF IK
# def IK_6DOF_unit_tests():
#     dh_params = np.array([[0, np.pi/2, 0.10391, 0],
#                         [.20573, 0, 0, 0],
#                         [0, np.pi/2, 0, 0],
#                         [0,-np.pi/2, .26895, 0],
#                         [0, np.pi/2, 0, 0],
#                         [0, 0, .17636, 0]])

#     joint_limits = np.array([[-np.pi, np.pi],
#                             [-1.5, np.pi/2],
#                             [-np.pi/2, 1.6],
#                             [-np.pi, np.pi],
#                             [-np.pi/2, np.pi/2],
#                             [-np.pi, np.pi]])

#     tol = 1e-4

#     for i in range(100):
#         print('--------- 6DOF Test: ', i, '------------')
#         joint_angles = np.random.uniform(-3.14, 3.14, 6)
#         print('joint angles', joint_angles)
#         T = kinematics.FK_dh(dh_params, joint_angles, 6)
#         theta = kinematics.IK_6dof(dh_params, T)
#         err = np.abs((theta.T - joint_angles).T)
#         best_soln_idx = np.argmin(np.nanmax(err, axis=0))
#         print(best_soln_idx, theta[:,best_soln_idx])
#         # print('theta', theta.T)
#         # print('max err', np.nanmax(err, axis=0))
#         print('T fk\n',T)
#         print('T ik\n', kinematics.FK_dh(dh_params, theta[:,best_soln_idx], 6))

#         assert(np.max(err[:,best_soln_idx]) < tol)

def calculate_position_error(dataset, s_lst):
    gearbox_k = np.array([0.0, 0.0194, 0.0619, 0.0, 0.0])
    pose = np.load('pose_data{}.npy'.format(dataset))
    theta = np.load('theta_data{}.npy'.format(dataset))
    n = theta.shape[0]
    err = np.zeros(n)
    err_corr = np.zeros(n)
    for i in range(n):
        T_dh = kinematics.FK_dh(dh_params, theta[i], 5)
        g_forces = kinematics.get_grav(theta[i], s_lst)
        corrections = gearbox_k*g_forces
        T_corr = kinematics.FK_dh(dh_params, theta[i] - corrections, 5)
        err_corr[i] = np.linalg.norm(pose[:3,3,i]-T_corr[:3,3])
        err[i] = np.linalg.norm(pose[:3,3,i]-T_dh[:3,3])
    print(err)
    print('mean err:', np.mean(err))
    print(err_corr)
    print('mean err w/ corrections:', np.mean(err_corr))

if __name__ == '__main__':
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
    
    calculate_position_error(4, s_lst)

    # IK_5DOF_unit_tests(dh_params)