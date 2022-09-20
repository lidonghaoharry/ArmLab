import numpy as np
import kinematics

dh_params = np.array([[0, np.pi/2, 0.10391, 0],
                      [.20573, 0, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])
joint_angles = np.array([1,0,0.4,-0.5,0],dtype=np.float64)

T = kinematics.FK_dh(dh_params, joint_angles, 5)

print(T)
theta = kinematics.IK_geometric(dh_params, T)
for i in range(4):
    print('theta', i)
    print(theta[:,i].T)