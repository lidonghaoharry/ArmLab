"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
import modern_robotics as mr


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


def FK_dh(dh_params, joint_angles, link, wrist_pos_ref=None, elbow_pos_ref=None):
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
    @param      wrist_pos_ref  Reference to nparray where [x,y,z] of the wrist is stored
    @param      elbow_pos_ref  Reference to nparray where [x,y,z] of the wrist is stored

    @return     a transformation matrix representing the pose of the desired link
    """
    T = np.eye(4)
    theta = np.copy(joint_angles)
    if len(joint_angles) == 5:
        theta[0] += np.pi/2
        theta[1] -= np.arctan2(200,50)
        theta[2] -= np.arctan2(200,50)
        theta[3] += np.pi/2
    elif len(joint_angles) == 6:
        theta[0] += np.pi/2
        theta[1] += np.arctan2(200,50)
        theta[2] += np.arctan2(50,200)

    for i in range(0,link):

        Ti = get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], theta[i])
        T = np.matmul(T,Ti)

        #modify the passed nparrays with the locations of the wrist and elbow
        if i==1 and elbow_pos_ref is not None:
            elbow_pos_ref[:3] = T[:3,3] + T[:3,1]*0.05 #offset in y to estimate farthest point
        elif i==2 and elbow_pos_ref is not None:
            wrist_pos_ref[:3] = T[:3,3]

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
    rot = R.from_dcm(T[:3,:3])
    # rot = R.from_matrix(T[:3,:3]) for python 3
    return rot.as_euler('ZYZ')


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi,theta,psi) where angles
                are zyz euler angles

    @param      T     transformation matrix

    @return     The pose from T as [x,y,z,phi,theta,psi].
    """
    pos = T[:3,3]
    ang = get_euler_angles_from_T(T)
    return np.hstack((pos,ang))


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
    '''
    Returns skew symetric representation of 3-vector x
    '''
    m = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return m

def inv_transform(T):
    '''
    @param T 4x4 transformation matrix

    @return T_inv The inverse of that transformation matrix
    '''
    rotM = T[:3,:3]
    T_inv = np.zeros((4,4))
    T_inv[3,3] = 1
    T_inv[:3,:3] = rotM.T
    T_inv[:3,3] = -np.matmul(rotM.T,T[:3,3])
    return T_inv

def IK_geometric(dh_params, T):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      T       The desired pose as a transformation matrix

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    T50 = T
    theta = np.zeros((5,4))
    for i in range(4):
        # theta 1
        p3_0 = T50[:3,3] - T50[:3,2]*dh_params[4][2]
        mag_p3_0 = np.linalg.norm(p3_0[:2])
        mag_p3_0 *= (-1.0)**((i & 0x02) >> 1)
        c1 = p3_0[0]/mag_p3_0
        s1 = p3_0[1]/mag_p3_0

        theta[0,i] = np.arctan2(s1,c1) # pattern is ++--

        # theta 3
        T10 = np.array([[c1, 0, s1, 0],
                        [s1, 0, -c1, 0],
                        [0,1,0,dh_params[0][2]],
                        [0,0,0,1]])
        T01 = inv_transform(T10)

        p3_0 = np.append(p3_0,1)
        p3_1 = np.matmul(T01,p3_0)

        a2 = dh_params[1][0]
        a3 = dh_params[2][0]
        c3 = (p3_1[0]**2 + p3_1[1]**2 - a2**2 - a3**2)/(2*a2*a3)
        #check if the desired position is possible for th arm to reach
        if (1-c3**2) > 0:
            s3 = np.sqrt(1-c3**2)
            s3 *= (-1)**((i & 0x01)) # pattern is +-+-
            theta[2,i] = -np.arctan2(s3,c3)
        else:
            print("IK Failed")
            raise Exception("Outside of dexterous workspace")

        #theta 2
        theta[1,i] = -(np.arctan2(p3_1[1],p3_1[0]) - np.arctan2(a3*np.sin(theta[2,i]), a2+a3*np.cos(theta[2,i])))

        #theta 5
        R30 = np.zeros((3,3))
        c2_3 = np.cos(theta[1,i]-theta[2,i])
        s2_3 = np.sin(theta[1,i]-theta[2,i])
        R30[0,0] = c2_3*c1
        R30[0,1] = s2_3*c1
        R30[0,2] = s1
        R30[1,0] = c2_3*s1
        R30[1,1] = s2_3*s1
        R30[1,2] = -c1
        R30[2,0] = -s2_3
        R30[2,1] = c2_3

        R53 = np.matmul(R30.T, T50[:3,:3])

        theta[4,i] = np.arctan2(R53[2,0], R53[2,1])
        theta[3,i] = np.arctan2(R53[0,2], -R53[1,2])
        
        # correct for differences between DH zero and motor zero positions
        theta[0,i] -= np.pi/2
        theta[1,i] += np.arctan2(200,50)
        theta[2,i] += np.arctan2(200,50)
        theta[3,i] -= np.pi/2

        #check whether the pose matches the desired
        T_soln = FK_dh(dh_params, theta[:,i], 5)
        tol = 1e-6
        if np.max(np.abs(T_soln - T50)) > tol:
            print("Impossible orientation -- solved for T50: ")
            print(T_soln)
            print("Wanted T50:")
            print(T50)
            raise Exception("Imposible Wrist Orientation")


        theta = ((theta + np.pi) % (2*np.pi)) - np.pi # constrain to [-pi,pi]
        
    return theta


def IK_from_top(dh_params, pos, theta=0):
    '''
    Picks up a block with the gripper pointing down

    @param dh_params DH table
    @param pos desired [x,y,z] position in mm
    @param theta rotation around the world z axis, defaults to zero

    @return theta a vector of angles for the joints
    
    '''
    rot = R.from_euler("ZYZ", [-theta, np.pi, np.pi/2])
    T = np.eye(4)
    T[:3,:3] = rot.as_dcm()
    T[:3,3] = pos/1000 # mm to m
    # print("IK top pos: " + str(T))
    theta = IK_geometric(dh_params, T)
    
    if theta[0,0] <= 3*np.pi/4 and theta[0,0] >= -3*np.pi/4:
        # print("FK: " + str(FK_dh(dh_params, theta[:,0], 5)))
        # print("theta: " + str(theta[:,0]))
        if theta[3,0] > 1.7 or theta[3,0] < -2.146: #joint limit
            print("joint limits" + str(theta[3,0]))
            raise Exception("Joint Limits J4")
        return theta[:,0]
    else:
        # print("FK: " + str(FK_dh(dh_params, theta[:,2], 5)))
        # print("theta: " + str(theta[:,2]))
        if theta[3,2] > 1.7 or theta[3,2] < -2.146: #joint limit
            print("joint limits" + str(theta[3,2]))
            raise Exception("Joint Limits J4")
        return theta[:,2]

def IK_from_side(dh_params, pos):
    '''
    Picks up a block with the gripper pointing to the side

    @param dh_params DH table
    @param pos desired [x,y,z] position
    @param theta rotation around the world z axis, defaults to zero

    @return theta a vector of angles for the joints
    
    '''
    alpha = np.arctan2(pos[1],pos[0])
    rot = R.from_euler("ZYZ", [alpha-np.pi, -np.pi/2, 0])
    T = np.eye(4)
    T[:3,:3] = rot.as_dcm()
    T[:3,3] = pos/1000 # mm to m
    # print("IK side pos: " + str(T))
    theta = IK_geometric(dh_params, T)

    if theta[0,0] <= 3*np.pi/4 and theta[0,0] >= -3*np.pi/4:
        # print("FK: " + str(FK_dh(dh_params, theta[:,0], 5)))
        # print("theta: " + str(theta[:,0]))
        if theta[3,0] > 1.7 or theta[3,0] < -2.146: #joint limit
            print("joint limits" + str(theta[3,0]))
            raise Exception("Joint Limits J4")
        return theta[:,0]
    else:
        # print("FK: " + str(FK_dh(dh_params, theta[:,2], 5)))
        # print("theta: " + str(theta[:,2]))
        if theta[3,2] > 1.7 or theta[3,2] < -2.146: #joint limit
            print("joint limits" + str(theta[3,2]))
            raise Exception("Joint Limits J4")
        return theta[:,2]



def IK_6dof(dh_params, pose):
    T60 = pose
    theta = np.zeros((6,8))
    for i in range(8):
        p4_0 = T60[:3,3] - T60[:3,2]*dh_params[5][2]
        mag_p4_0 = np.linalg.norm(p4_0[:2])
        mag_p4_0 *= (-1)**((i & 0x04) >> 2) # pattern is ++++----
        c1 = p4_0[0]/mag_p4_0
        s1 = p4_0[1]/mag_p4_0
        theta[0,i] = np.arctan2(s1,c1)
        
        # theta 3
        T10 = np.array([[c1, 0, s1, 0],
                        [s1, 0, -c1, 0],
                        [0,1,0,dh_params[0][2]],
                        [0,0,0,1]])
        T01 = inv_transform(T10)

        p4_0 = np.append(p4_0,1)
        p4_1 = np.matmul(T01,p4_0)

        a2 = dh_params[1][0]
        d4 = dh_params[3][2]
        c3 = (p4_1[0]**2 + p4_1[1]**2 - a2**2 - d4**2)/(2*a2*d4)
        if 1-c3**2>=0:
            s3 = np.sqrt(1-c3**2)
            s3 *= (-1)**((i & 0x02) >> 1) # pattern is ++--++--
            theta[2,i] = np.pi/2 - np.arctan2(s3,c3)
        else:
            raise Exception("Outside of workspace")

        #theta 2
        c3 = np.cos(theta[2,i])
        s3 = np.sin(theta[2,i])
        A = np.array([[d4*s3 + a2, d4*c3],
                      [-d4*c3, d4*s3 + a2]])
        vec = np.linalg.solve(A, p4_1[:2])
        theta[1,i] = np.arctan2(vec[1], vec[0])

        R30 = np.zeros((3,3))
        c23 = np.cos(theta[1,i]+theta[2,i])
        s23 = np.sin(theta[1,i]+theta[2,i])
        R30[0,0] = c23*c1
        R30[0,1] = s1
        R30[0,2] = s23*c1
        R30[1,0] = c23*s1
        R30[1,1] = -c1
        R30[1,2] = s23*s1
        R30[2,0] = s23
        R30[2,2] = -c23

        R63 = np.matmul(R30.T, T60[:3,:3])
        c5 = R63[2,2]
        s5 = np.sqrt(1-c5**2)
        s5 *= (-1)**((i & 0x01)) # pattern is +-+-+-
        theta[4,i] = np.arctan2(s5, c5)
        if c5 == 1:
            print("WARNING: Wrist in singularity")
            # choosing to leave theta 4 = 0
            theta[5,i] = np.arctan2(R63[1,0], R63[1,1])
        else: 
            theta[3,i] = np.arctan2(R63[1,2]/s5, R63[0,2]/s5)
            theta[5,i] = np.arctan2(R63[2,1]/s5, -R63[2,0]/s5)

        theta[0,i] -= np.pi/2
        theta[1,i] -= np.arctan2(200,50)
        theta[2,i] -= np.arctan2(50,200)

        theta = ((theta + np.pi) % (2*np.pi)) - np.pi # constrain to [-pi,pi]

    return theta


def get_grav(theta, s_lst):
    '''
    Return the torque each joint needs to overcome gravity

    @param theta the joint angles of the arm
    @param s_lst screw vectors
    '''
    M01 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.10391],
                    [0, 0, 0, 1]])
    M12 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0.01195],
                    [0, 0, 1, 0.13943],
                    [0, 0, 0, 1]])
    M23 = np.array([[0, 1, 0, 0],
                    [-1, 0, 0, 0.1528],
                    [0, 0, 1, 0.0657],
                    [0, 0, 0, 1]])
    M34 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0.12761],
                    [0, 0, 1, 0.01058],
                    [0, 0, 0, 1]])
    M45 = np.array([[1, 0, 0, -0.01058],
                    [0, 1, 0, 0.08864],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    M56 = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0.04315],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])                
    Mlist = np.array([M01,M12,M23,M34,M45,M56])
    m1 = 0.257774
    m2 = 0.297782
    m3 = 0.258863
    m4 = 0.084957
    m5 = 0.139576
    G1 = np.diag([0.0,0.0,0.0,m1,m1,m1])
    G2 = np.diag([0.0,0.0,0.0,m2,m2,m2])
    G3 = np.diag([0.0,0.0,0.0,m3,m3,m3])
    G4 = np.diag([0.0,0.0,0.0,m4,m4,m4])
    G5 = np.diag([0.0,0.0,0.0,m5,m5,m5])
    Glist = np.array([G1,G2,G3,G4,G5])
    g = np.array([0, 0, -9.8])
    g_forces = mr.GravityForces(theta, g, Mlist, Glist, s_lst.T)
    return g_forces