"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
from collections import OrderedDict

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.recorded_positions = []

        # dictionary mapping locations to the number of blocks stacked there
        self.r_slot_dict = OrderedDict()
        self.l_slot_dict = OrderedDict()

        self.REG_BLOCK_HEIGHT = 40
        self.SMALL_BLOCK_HEIGHT = 20

        self.regrip_pos = np.array([0.0, 175.0, 0.0, 1])


    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "teach":
            self.teach()
        
        if self.next_state == "replay":
            self.replay_waypoints()
        
        if self.next_state == "event1":
            self.event1()

        if self.next_state == "event2":
            self.event2()

        if self.next_state == "event3":
            self.event3()

        if self.next_state == "event4":
            self.event4()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def event1(self):
        self.l_slot_dict[(-125, -75)] = 0
        self.l_slot_dict[(-175, -75)] = 0
        self.l_slot_dict[(-225, -75)] = 0
        self.l_slot_dict[(-275, -75)] = 0
        self.l_slot_dict[(-325, -75)] = 0

        self.r_slot_dict[(125, -75)] = 0
        self.r_slot_dict[(175, -75)] = 0
        self.r_slot_dict[(225, -75)] = 0
        self.r_slot_dict[(275, -75)] = 0
        self.r_slot_dict[(325, -75)] = 0

        # iterate while blocks exist in pos half plane 
        p_blocks = self.camera.positive_blocks()
        while len(p_blocks) > 0:
            id = p_blocks.keys()[0]
            block = p_blocks[id]
            size = block[6]

            # convert to world coords
            b_pos_w = block[1]

            # move to pick up block
            theta = block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            # place block in correct slot
            if size == 'l':
                coord, d = self.min_slot(self.r_slot_dict, self.camera.get_height_img(), p_blocks)
                self.r_slot_dict[coord] += self.REG_BLOCK_HEIGHT
            else:
                coord, d  = self.min_slot(self.l_slot_dict, self.camera.get_height_img(), p_blocks)
                self.l_slot_dict[coord] += self.SMALL_BLOCK_HEIGHT

            coord = np.array([coord[0], coord[1], d], dtype=np.float)
            self.rxarm.pick_block(coord, theta=90, size=size)
        
            # get new set of blocks
            p_blocks = self.camera.positive_blocks()

        self.next_state = "idle"

    def event2(self):
        self.l_slot_dict[(-150, 50)] = 0
        self.l_slot_dict[(-250, 50)] = 0
        self.l_slot_dict[(-300, 50)] = 0
        self.l_slot_dict[(-350, 50)] = 0

        self.r_slot_dict[(325, -75)] = 0
        self.r_slot_dict[(250, -75)] = 0
        self.r_slot_dict[(175, -75)] = 0

        # unstack blocks 
        self.unstack_all()

        p_blocks = self.camera.positive_blocks()
        while len(p_blocks) > 0:
            block = self.get_large_block(p_blocks)
            if block is None:
                id = p_blocks.keys()[0]
                block = p_blocks[id]
            size = block[6]

            # convert to world coords
            b_pos_w = block[1]

            # move to pick up block
            theta = block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            # place block in correct slot
            coord = self.min_slot(self.r_slot_dict, self.camera.get_height_img(), self.camera.block_info)
            # if size == 'l':
            #     self.r_slot_dict[coord] += self.REG_BLOCK_HEIGHT
            # else:
            #     self.r_slot_dict[coord] += self.SMALL_BLOCK_HEIGHT

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([200.0, -75.0, 200.0]), theta=90)

            # coord = np.array([coord[0], coord[1], d], dtype=np.float)
            self.rxarm.pick_block(coord, theta=90, size=size)

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([200.0, -75.0, 200.0]), theta=90)

            # get new set of blocks
            p_blocks = self.camera.positive_blocks()

        self.next_state = "idle"

    def event3(self):
        '''
        line up small and large blocks by color
        '''
        l_drop_pos = np.array([-350.0,-75.0, 0.0])
        r_drop_pos = np.array([400.0,-75.0, 0.0])

        #move everything to positive half plane
        self.move_all_positive()
        
        #unstack everything into the positive half-plane
        self.unstack_all(rand_pos=True)

        #move the arm out off the way so all blocks are visible
        self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))
        
        #sort the large blocks
        p_blocks = self.camera.positive_blocks()
        l_block = self.find_next_block(p_blocks, 'l')
        while l_block is not None:
            # print('p_blocks: ', p_blocks)
            size = l_block[6]
            b_pos_w = l_block[1]
            theta = l_block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            #move the arm to neutral position
            self.rxarm.move_to_pos(np.array([0.0, 200.0, 200.0]))

            drop_pos_c = self.camera.to_camera_coords(np.append(r_drop_pos, 1))
            if self.rxarm.which_block(drop_pos_c, self.camera.block_info) != -1:
                r_drop_pos[0] -= 40
            
            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([200.0, -75.0, 200.0]), theta=90)
            
            self.rxarm.pick_block(r_drop_pos.copy(), theta=90, size=size, x_offset = -30) #approach from the side a bit


            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([200.0, -75.0, 200.0]), theta=90)

            # print("drop position: " + str(r_drop_pos))
            #get the next large block by color
            p_blocks = self.camera.positive_blocks()
            l_block = self.find_next_block(p_blocks, 'l')

        #sort the small blocks
        p_blocks = self.camera.positive_blocks()
        s_block = self.find_next_block(p_blocks, 's')
        while s_block is not None:
            size = s_block[6]
            b_pos_w = s_block[1]
            theta = s_block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            #move the arm to neutral position
            self.rxarm.move_to_pos(np.array([0.0, 200.0, 200.0]))

            drop_pos_c = self.camera.to_camera_coords(np.append(l_drop_pos, 1))
            if self.rxarm.which_block(drop_pos_c, self.camera.block_info) != -1:
                l_drop_pos[0] += 35

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([-200.0, -75.0, 200.0]), theta=90)

            self.rxarm.pick_block(l_drop_pos.copy(), theta=90, size=size, x_offset = 30) #approach from the side a bit

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([-200.0, -75.0, 200.0]), theta=90)

            # print("drop position " + str(l_drop_pos))
            #get the next small block by color
            p_blocks = self.camera.positive_blocks()
            s_block = self.find_next_block(p_blocks, 's')

        self.next_state = 'idle'

    def event4(self):
        l_drop_pos = np.array([-200.0,-75.0, 0.0])
        r_drop_pos = np.array([200.0,-75.0, 0.0])

        #move everything to positive half plane
        self.move_all_positive()
        
        #unstack everything into the positive half-plane
        self.unstack_all(rand_pos=True)

        #move the arm out off the way so all blocks are visible
        self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))

        # move blocks out of regrip pos 
        self.clear_regrip_pos()
        
        #sort the large blocks
        p_blocks = self.camera.positive_blocks()
        l_block = self.find_next_block(p_blocks, 'l')
        while l_block is not None:
            # print('p_blocks: ', p_blocks)
            size = l_block[6]
            b_pos_w = l_block[1]
            theta = l_block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            # move the arm to neutral position
            self.rxarm.move_to_pos(np.array([0.0, 200.0, 200.0]))

            # regrip block 
            self.regrip(size)

            drop_pos_c = self.camera.to_camera_coords(np.append(r_drop_pos, 1))
            block = self.rxarm.which_block(drop_pos_c, self.camera.block_info)
            if block != -1:
                id, block = block 
                center = block[1]
                depth = block[7]

                # convert to world 
                center = [center[0], center[1], 1]
                pos_w = self.camera.to_world_coords(depth, center)
                r_drop_pos = pos_w[:3]
            
            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([100.0, -75.0, 300.0]), theta=90)
            
            self.rxarm.pick_block(r_drop_pos.copy(), theta=90, size=size, z_offset=30)

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([100.0, -75.0, 300.0]), theta=90)

            # print("drop position: " + str(r_drop_pos))
            #get the next large block by color
            p_blocks = self.camera.positive_blocks()
            l_block = self.find_next_block(p_blocks, 'l')

        #sort the small blocks
        p_blocks = self.camera.positive_blocks()
        s_block = self.find_next_block(p_blocks, 's')
        while s_block is not None:
            size = s_block[6]
            b_pos_w = s_block[1]
            theta = s_block[3]
            self.rxarm.pick_block(np.array(b_pos_w[:3]), theta=theta, size=size)

            #move the arm to neutral position
            self.rxarm.move_to_pos(np.array([0.0, 200.0, 200.0]))

            # regrip block 
            self.regrip(size)

            drop_pos_c = self.camera.to_camera_coords(np.append(l_drop_pos, 1))
            block = self.rxarm.which_block(drop_pos_c, self.camera.block_info)
            if block != -1:
                id, block = block 
                center = block[1]
                depth = block[7]

                # convert to world 
                center = [center[0], center[1], 1]
                pos_w = self.camera.to_world_coords(depth, center)
                l_drop_pos = pos_w[:3]

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([-100.0, -75.0, 200.0]), theta=90)

            fudge_factors = np.array([7.0, 0.0, 3.0])
            self.rxarm.pick_block(l_drop_pos.copy() + fudge_factors, theta=90, size=size) #approach from the side a bit

            # go to approach point for stacks
            self.rxarm.move_to_pos(np.array([-100.0, -75.0, 200.0]), theta=90)

            # print("drop position " + str(l_drop_pos))
            #get the next small block by color
            p_blocks = self.camera.positive_blocks()
            s_block = self.find_next_block(p_blocks, 's')

        self.next_state = 'idle'

    def regrip(self, size):
        # drop block at regrip position 
        self.rxarm.pick_block(self.regrip_pos[:3].copy(), theta=0, size=size)

        # move the arm out off the way so all blocks are visible
        self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))

        # get block in regrip pos
        self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))
        p = self.camera.to_camera_coords(self.regrip_pos)
        block = self.rxarm.which_block(p, self.camera.block_info)

        # pick up block in regrip position 
        if block != -1:
            id, block = block 
            center = block[1]
            theta = block[3]
            size = block[6]
            depth = block[7]

            # convert to world 
            center = [center[0], center[1], 1]
            pos_w = self.camera.to_world_coords(depth, center)

            # pick up block 
            self.rxarm.pick_block(np.array(pos_w[:3]), theta=theta, size=size)

    def clear_regrip_pos(self):
        # move the arm out off the way so all blocks are visible
        self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))
        p = self.camera.to_camera_coords(self.regrip_pos)

        block = self.rxarm.which_block(p, self.camera.block_info)
        print(block)

        # while there exists a block in the regrip position 
        while block != -1:
            id, block = block 
            center = block[1]
            theta = block[3]
            size = block[6]
            depth = block[7]

            # convert to world 
            center = [center[0], center[1], 1]
            pos_w = self.camera.to_world_coords(depth, center)

            # pick up block 
            self.rxarm.pick_block(np.array(pos_w[:3]), theta=theta, size=size)

            # move to random pos 
            pos_w = self.get_random_drop_pos(self.camera.block_info)

            # drop block
            self.rxarm.pick_block(np.array(pos_w), theta=0, size=size)

            # move the arm out off the way so all blocks are visible
            self.rxarm.move_to_pos(np.array([-100.0, 0.0, 200.0]))

            # get block in regrip pos
            block = self.rxarm.which_block(p, self.camera.block_info)
            print(block)

    def unstack_all(self, rand_pos=False):
        p_blocks = self.camera.positive_blocks()
        s_block = self.get_stacked_block(p_blocks)

        while s_block is not None:
            # convert to world coords
            s_pos_w = s_block[1]
            size = s_block[6]

            # pick up block
            theta = s_block[3]
            self.rxarm.pick_block(np.array(s_pos_w[:3]), theta=theta, size=size)

            if rand_pos:
                coord = self.get_random_drop_pos(p_blocks)
                self.rxarm.pick_block(coord, theta=0, size=size)
            else:
                coord = self.min_slot(self.l_slot_dict, self.camera.get_height_img(), self.camera.block_info)

                # if size == 'l':
                #     self.l_slot_dict[coord] += self.REG_BLOCK_HEIGHT
                # else:
                #     self.l_slot_dict[coord] += self.SMALL_BLOCK_HEIGHT

                # coord = np.array([coord[0], coord[1], d], dtype=np.float)
                self.rxarm.pick_block(coord, theta=-90, size=size)

            # get new block to unstack
            p_blocks = self.camera.positive_blocks()
            s_block = self.get_stacked_block(p_blocks)

    def move_all_positive(self):
        '''
        move all blocks into the positive half plane
        '''
        n_blocks = self.camera.negative_blocks(y_cutoff=25)
        while len(n_blocks) > 0:
            id = n_blocks.keys()[0]
            block = n_blocks[id]
            pos_w = block[1]
            size = block[6]
            theta = block[3]

            #pick up the block
            self.rxarm.pick_block(np.array(pos_w[:3]), theta=theta, size=size)

            #find a spot to put it
            pos = self.get_random_drop_pos(self.camera.block_info)
            self.rxarm.pick_block(pos, theta=0, size=size)
            
            n_blocks = self.camera.negative_blocks(y_cutoff=50)

    def get_random_drop_pos(self, blocks):
        '''
        finds an open drop location near the arm 

        returns last tried if it didn't work
        '''
        for _ in range(300):
            x = np.random.randint(-240, 240)
            y = np.random.randint(150, 330)
            point_w = np.array([x,y,0,1], dtype=np.float)
            point_c = self.camera.to_camera_coords(point_w)
            # print("point_c: " + str(point_c))
            if self.rxarm.which_block(point_c, blocks, thresh=75) == -1:
                return point_w[:3]
        print("no valid location found")
        return point_w[:3]
    
    def find_next_block(self, blocks, size):
        '''
        returns the next block by color of a particular size
        '''
        color_ids = {'red':0, 'orange':1, 'yellow':2, 'green':3, 'blue':4, 'violet':5}
        ret = None
        min_color_id = 10000000
        for id in blocks:
            block = blocks[id]

            if block[6] == size:
                block_color = block[4]
                color_id = color_ids[block_color]
                if color_id < min_color_id:
                    print("color:" + str(color_id))
                    min_color_id = color_id
                    ret = block 

        return ret

    def get_stacked_block(self, blocks):
        ret = None 
        for id in blocks:
            block = blocks[id]

            if block[1][2] > self.REG_BLOCK_HEIGHT:
                # print(block)
                ret = block 

        return ret

    def get_large_block(self, blocks):
        ret = None 
        for id in blocks:
            block = blocks[id]

            if block[6] == 'l':
                # print(block)
                ret = block 

        return ret
                
    def min_slot(self, s_dict, height_img, blocks):
        #move the arm to neutral position
        self.rxarm.move_to_pos(np.array([0.0, 200.0, 200.0]))
        max = 0
        point = None
        for coord in s_dict:
            w_c = [coord[0], coord[1], 0, 1]
            cam_coord = self.camera.to_camera_coords(w_c)
            block = self.rxarm.which_block(cam_coord, blocks)

            print("which block return: " + str(block))
            if block == -1:
                num = 100000
                p = np.array([coord[0], coord[1], 0], dtype=np.float)  
            else:
                id, block = block
                num = block[7]
                p = block[1]
                p = self.camera.to_world_coords(num, np.append(np.array(p),1))
                p = p[:3]

            # print(height_img[int(cam_coord[1]) - 10: int(cam_coord[1]) + 10, int(cam_coord[0]) - 10: int(cam_coord[0]) + 10].shape)
            print("min slot height: " + str(num))
            if num > max:
                max = num
                point = p
                
        return point
            

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"

        for i,waypoint in enumerate(self.waypoints):
            self.rxarm.set_positions(waypoint)
            rospy.sleep(5)

        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.camera.auto_calibrate()
        self.status_message = "Calibration - Completed Calibration"

    def replay_waypoints(self):
        self.current_state = "replay"

        #FOR TESTING
        # self.recorded_positions = np.load("recorded_pos.npy", allow_pickle=True)

        n = len(self.recorded_positions)
        th_data = np.zeros((n,5))
        pose_data = np.zeros((4,4,n))
        for i,waypoint in enumerate(self.recorded_positions):
            arm_pose, gripper_state = waypoint

            self.rxarm.set_move_time(arm_pose)

            # move arm 
            # self.rxarm.set_positions(arm_pose)
            self.rxarm.set_g_corrected_positions(arm_pose)
            rospy.sleep(self.rxarm.moving_time + self.rxarm.wait_time)

            # open/close gripper
            if gripper_state == True:
                self.rxarm.open_gripper()
                rospy.sleep(self.rxarm.wait_time)
            else:
                self.rxarm.close_gripper()
                rospy.sleep(self.rxarm.wait_time)
            #recording data
            rospy.sleep(1)
            pose_data[:,:,i] = self.rxarm.get_ee_T()
            th_data[i,:] = arm_pose
        np.save('corr_theta_data7', th_data)
        np.save('corr_pose_data7', pose_data)
            
        self.next_state = "idle"

    def kinematics_grid(self):
        z = 50
        x = np.linspace(-400, -100, 4)
        y = np.linspace(-50, 350, 5)
        lgridx, lgridy = np.meshgrid(x,y)
        lpairs = np.vstack([lgridx.ravel(), lgridy.ravel()])

        x = 0
        y = np.linspace(150, 350, 3)
        cgridy, cgridx = np.meshgrid(y, x)
        cpairs = np.vstack([cgridx.ravel(), cgridy.ravel()])

        x = np.linspace(400, 100, 4)
        y = np.linspace(-50, 350, 5)
        rgridy, rgridx = np.meshgrid(y, x)
        rpairs = np.vstack([rgridx.ravel(), rgridy.ravel()])
        points = np.hstack((lpairs, cpairs, rpairs))

        n = points.shape[1]
        expected_pos = np.zeros((n,3))
        achieved_pos = np.zeros((n,3))
        for i in range(points.shape[1]):
            pos = np.array([points[0,i], points[1,i], z])
            self.rxarm.move_to_pos(pos) #uses g compensation
            # self.rxarm.move_to_pos(pos, use_g_correction = False) # no g compensation
            rospy.sleep(1)
            expected_pos[i,:] = pos
            achieved_pos[i,:] = self.rxarm.get_ee_T()[:3,3]
        np.save('kinematics_grid_expected',expected_pos)
        np.save('kinematics_grid_actual', achieved_pos)


    def teach(self):
        '''teach and repeat state'''
        if self.current_state != "teach":
            self.recorded_positions = []
        self.current_state = "teach"
        self.next_state = "teach"

        # disable torque for everything besides the gripper 
        self.rxarm.disable_torque()
        # self.rxarm.enable_torque_gripper()

        self.status_message = "Teach Mode"

    def stop_teaching(self):
        np.save('recorded_pos', self.recorded_positions)
        self.next_state = "idle"

    def record_position(self):
        if self.current_state == "teach":
            self.recorded_positions.append((self.rxarm.get_positions(), self.rxarm.gripper_state))
            print(self.recorded_positions)
            self.status_message = "recorded position: " + str(len(self.recorded_positions))



    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)