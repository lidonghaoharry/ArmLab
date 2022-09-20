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

joint_angles = np.array([0.8,0.4,-0.8,1,0.5],dtype=np.float64)

T = kinematics.FK_dh(dh_params, joint_angles, 5)
Tpox = kinematics.FK_pox(joint_angles, M, s_lst)

print("DH\n", T)
print("PoX\n", Tpox)
print("error")
print(Tpox-T)
theta = kinematics.IK_geometric(dh_params, T)
for i in range(4):
    print('theta', i)
    print(theta[:,i].T)