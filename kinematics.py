"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    T = np.eye(4)
    theta = np.copy(joint_angles)
    theta[0] += np.pi/2
    theta[1] += np.arctan2(200,50)
    theta[2] -= np.arctan2(200,50)
    theta[3] += np.pi/2


    for i in range(0,link):

        Ti = get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], theta[i])
        T = np.matmul(T,Ti)

    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    
    T = np.array([[np.cos(theta),       -np.sin(theta)*np.cos(alpha),       np.sin(theta)*np.sin(alpha),        a*np.cos(theta)],
                  [np.sin(theta),       np.cos(theta)*np.cos(alpha),        -np.cos(theta)*np.sin(alpha),       a*np.sin(theta)],
                  [0,                   np.sin(alpha),                      np.cos(alpha),                      d],
                  [0,                   0,                                  0,                                   1]])
    
    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    rot = R.from_matrix(T[:3,:3])
    return rot.as_euler('zyz')


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    T = np.eye(4)
    for i in range(len(joint_angles)):
        w = s_lst[i][:3]
        v = s_lst[i][3:]
        s_mat = to_s_matrix(w,v)
        e_s = expm(s_mat*joint_angles[i])
        T = np.matmul(T,e_s)
    T = np.matmul(T,m_mat)
    return T


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    S = np.zeros((4,4))
    S[:3,:3] = to_skew_symetric(w)
    S[:3,3] = v
    return S

def to_skew_symetric(x):
    m = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return m

def inv_transform(T):
    rotM = T[:3,:3]
    T_inv = np.zeros((4,4))
    T_inv[3,3] = 1
    T_inv[:3,:3] = rotM.T
    T_inv[:3,3] = -np.matmul(rotM.T,T[:3,3])
    return T_inv

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    T50 = pose
    theta = np.zeros((5,8))
    for i in range(8):
        # theta 1
        p3_0 = T50[:3,3] - T50[:3,2]*dh_params[4][2]
        c1 = p3_0[0]/np.linalg.norm(p3_0[:2])
        s1 = p3_0[1]/np.linalg.norm(p3_0[:2])
        s1 *= (-1)**((i & 0x02) >> 1) # pattern is ++--
        theta[0,i] = np.arctan2(s1,c1)

        # theta 3
        T10 = np.array([[np.cos(theta[0,i]), 0, np.sin(theta[0,i]), 0],
                        [np.sin(theta[0,i]), 0, -np.cos(theta[0,i]), 0],
                        [0,1,0,dh_params[0][2]],
                        [0,0,0,1]])
        T01 = inv_transform(T10)

        p3_0 = np.append(p3_0,1)
        p3_1 = np.matmul(T01,p3_0)

        a2 = dh_params[1][0]
        a3 = dh_params[2][0]
        c3 = (p3_1[0]**2 + p3_1[1]**2 - a2**2 - a3**2)/(2*a2*a3)
        s3 = np.sqrt(1-c3**2)
        s3 *= (-1)**((i & 0x01)) # pattern is +-+-
        theta[2,i] = -np.arctan2(s3,c3)

        #theta 2
        theta[1,i] = np.arctan2(p3_1[1],p3_1[0]) - np.arctan2(a3*np.sin(theta[2,i]), a2+a3*np.cos(theta[2,i]))

        #theta 5
        T30 = np.zeros((4,4))
        c23 = np.cos(theta[1,i]+theta[2,i])
        s23 = np.sin(theta[1,i]+theta[2,i])
        T30[0,0] = c23*np.cos(theta[0,i])
        T30[0,1] = -s23*np.cos(theta[0,i])
        T30[0,2] = np.sin(theta[0,i])
        T30[0,3] = np.cos(theta[0,i]) * (a3*c23 + a2*np.cos(theta[1,i]))
        T30[1,0] = c23*np.sin(theta[0,i])
        T30[1,1] = -s23*np.sin(theta[0,i])
        T30[1,2] = -np.cos(theta[0,i])
        T30[1,3] = np.sin(theta[0,i]) * (a3*c23 + a2*np.cos(theta[1,i]))
        T30[2,0] = s23
        T30[2,1] = c23
        T30[2,3] = dh_params[0,2]+a3*s23+a2*np.sin(theta[1,i])
        T30[3,3] = 1

        T03 = inv_transform(T30)
        T53 = np.matmul(T03, T50)

        theta[4,i] = np.arctan2(T53[2,0], T53[2,1])
        theta[3,i] = np.arctan2(T53[0,2], -T53[1,2])


        theta[0,i] -= np.pi/2
        theta[1,i] -= np.arctan2(200,50)
        theta[2,i] += np.arctan2(200,50)
        theta[3,i] -= np.pi/2
        
    return theta