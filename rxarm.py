"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import *
from sensor_msgs.msg import JointState
import rospy
import kinematics
import cv2

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixRobot):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True # True if open
        self.has_block = False
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = np.array([[0, -np.pi/2, 0.10391, 0],
                      [.20573, np.pi, 0, 0],
                      [.200, 0, 0, 0],
                      [0, np.pi/2, 0, 0],
                      [0, 0, .17415, 0]])

        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        # POX params
        self.M_matrix = []
        self.S_list = np.array([[0, 0, 1, 0, 0, 0],
                                [-1, 0, 0, 0, -0.10391, 0],
                                [1, 0, 0, 0, 0.30391, -0.05],
                                [1, 0, 0, 0, 0.30391, -0.25],
                                [0, 1, 0, -0.30391, 0, 0]])

        # max speed (arbitrarily set atm) radians/sec
        self.max_speed = 0.75
        
        self.gearbox_k = np.array([0.0, 0.0194, 0.0819, 0.0, 0.0])

        # end effector pose 
        self.ee_pose = [0.0 for i in range(6)]

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        rospy.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.set_gripper_pressure(1.0)
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.open_gripper()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def open_gripper(self, delay=1):
        self.gripper_state = True 
        return super().open_gripper(delay)

    def close_gripper(self, delay=1):
        self.gripper_state = False
        return super().close_gripper(delay)

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """

         #To Do
        self.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)


    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_off(self.joint_names)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_on(self.joint_names)

    def enable_torque_gripper(self):
        self.torque_joints_on('gripper')

    def disable_torque_gripper(self):
        self.torque_joints_off('gripper')

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi, theta, psi]
        """
        T = kinematics.FK_dh(self.dh_params, self.get_positions(), self.num_joints)
        # print(T)
        self.ee_pose = kinematics.get_pose_from_T(T).tolist()
        return self.ee_pose

    def get_ee_T(self):
        return kinematics.FK_dh(self.dh_params, self.get_positions(), self.num_joints)

    def collect_deflect_data(self):
        theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        n = 8
        th_data = np.zeros((n+4, 5))
        pose_data = np.zeros((4,4,n+4))
        for i in range(n):
            self.set_g_corrected_positions(theta)
            # self.set_positions(theta)
            time.sleep(3)
            pose_data[:,:,i] = self.get_ee_T()
            th_data[i,:] = theta
            theta[1] += 0.1
            theta[2] += 0.1
        theta = np.array([0.0, 0.1, -0.1, -1.0, 0.0])
        for i in range(n,n+4):
            # self.set_positions(theta)
            self.set_g_corrected_positions(theta)
            time.sleep(3)
            pose_data[:,:,i] = self.get_ee_T()
            th_data[i,:] = theta
            theta[1] += 0.1
            theta[2] += 0.1
        np.save('theta_data3', th_data)
        np.save('pose_data3', pose_data)

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params

    def pick_from_top(self, pos_w, pos_c, block_info, theta=0):
        block = self.which_block(pos_c, block_info)
        # print("which block return: " + str(block))

        # set gripper angle to the angle of the block
        if block == -1:
            block_angle = 0
            id = -1
        else:
            block_angle = block[1][3]
            id = block[0]
            
        if self.has_block:
            pos_w[2] += 20
        else:
            pos_w[2] -= 25

        approach_point = np.array([pos_w[0], pos_w[1], pos_w[2] + 70])
        joint_angles_approach = kinematics.IK_from_top(self.dh_params, approach_point, block_angle*np.pi/180)
        self.set_g_corrected_positions(joint_angles_approach)
        time.sleep(2)

        joint_angles = kinematics.IK_from_top(self.dh_params, pos_w, block_angle*np.pi/180)
        self.set_g_corrected_positions(joint_angles)
        time.sleep(2)

        if self.has_block:
            self.open_gripper()
            self.has_block = False
        else:
            self.close_gripper()
            self.has_block = True

        time.sleep(2)
        self.set_g_corrected_positions(joint_angles_approach)

        return id 
    
    def pick_from_side(self, pos):
        if self.has_block:
            pos[2] += 30
        else:
            pos[2] += 30

        approach_point = np.array([pos[0], pos[1], pos[2] + 70])
        joint_angles_approach = kinematics.IK_from_side(self.dh_params, approach_point)
        self.set_g_corrected_positions(joint_angles_approach)
        time.sleep(2)

        joint_angles = kinematics.IK_from_side(self.dh_params, pos)
        self.set_g_corrected_positions(joint_angles)
        time.sleep(2)

        if self.has_block:
            self.open_gripper()
            self.has_block = False
        else:
            self.close_gripper()
            self.has_block = True

        time.sleep(2)
        self.set_positions(joint_angles_approach)

    def pick_block(self, pos_w, pos_c, block_info, theta=0):
        try:
            self.pick_from_top(pos_w, pos_c, block_info, theta)
        except:
            self.pick_from_side(pos_w)

    def which_block(self, point, block_info):
        ret = -1
        # print("point: " + str(point))

        for block in block_info:
            # contour = block[3]
            id = block[0]
            box = block[2]
            # print("contour: " + str(contour))
            # print("color: " + str(block[2]))
            # print("box: " + str(box))

            if cv2.pointPolygonTest(box, (point[0], point[1]), False) == 1:
                ret = (id, block)

        return ret
    
    def set_g_corrected_positions(self, joint_angles):
        g_forces = kinematics.get_grav(joint_angles, self.S_list)
        corrections = self.gearbox_k * g_forces
        self.set_positions(joint_angles + corrections)



class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        rospy.Subscriber('/rx200/joint_states', JointState, self.callback)
        rospy.sleep(0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:

            rospy.spin()


if __name__ == '__main__':
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.close_gripper()
        rxarm.go_to_home_pose()
        rxarm.open_gripper()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
