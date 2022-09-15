import numpy as np
import kinematics

dh_params = np.array([[0, np.pi/2, 0.10391, 0],
                      [.20573, 0, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])
joint_angles = np.array([0,0,0,0,0],dtype=np.float64)

T = kinematics.FK_dh(dh_params, joint_angles, 5)


print(T)
print(T.shape)