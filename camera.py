"""!
Class to represent the camera.
"""

import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
import random

class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        # for checkpoint 1 part 2
        self.rough_Hinv = np.linalg.inv(np.array([[-1, 0, 0, 5], [0, 1, 0, -175], [0, 0, -1, 970], [0, 0, 0, 1]]))
        self.auto_Hinv = None

        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = np.array([[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]]).T
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        # self.block_info = []
        self.block_info = {}
        self.ids = []

        # image for template matching
        self.template_img = cv2.imread("template.png", 0)

        # workspace boundary for drawing 
        self.top_left = None 
        self.bottom_right = None
        self.arm_top_left = None 
        self.arm_top_right = None
        self.ee_pose = [0.0 for i in range(6)]
        self.wrist_pos = np.zeros(3)
        self.elbow_pos = np.zeros(3)
        self.base = np.array([670, 250])

        # RGB colors
        self.colors = list((
            {'id': 'red', 'color': (255, 0, 0)},
            {'id': 'orange', 'color': (255,69,0)},
            {'id': 'yellow', 'color': (204,204,0)},
            {'id': 'green', 'color': (0, 255, 0)},
            {'id': 'blue', 'color': (0,0,255)},
            {'id': 'violet', 'color': (148,0,211)},
            {'id': 'pink', 'color': (255,20,147)}))

        # self.colors = list((
        #             {'id': 'red', 'color': [[[255, 0, 0]]]},
        #             {'id': 'orange', 'color': [[[255,165,0]]]},
        #             {'id': 'yellow', 'color': [[[204,204,0]]]},
        #             {'id': 'green', 'color': [[[0, 255, 0]]]},
        #             {'id': 'blue', 'color': [[[0,0,255]]]},
        #             {'id': 'violet', 'color': [[[148,0,211]]]},
        #             {'id': 'pink', 'color': [[[255,20,147]]]},
        #             {'id': 'black', 'color': [[[255,255,255]]]}))

        # HSV color
        # self.colors = list((
        #     {'id': 'red', 'color': (0, 100, 100)},
        #     {'id': 'orange', 'color': (39, 100, 100)},
        #     {'id': 'yellow', 'color': (60, 100, 80)},
        #     {'id': 'green', 'color': (120, 100, 100)},
        #     {'id': 'blue', 'color': (240, 100, 100)},
        #     {'id': 'violet', 'color': (282, 100, 82.7)})
        # )
        # HSV color
        # self.colors = list((
        #     {'id': 'red', 'color': (0, 100, 100)},
        #     {'id': 'red', 'color': (170, 100, 100)},
        #     {'id': 'orange', 'color': (20, 100, 100)},
        #     {'id': 'yellow', 'color': (30, 100, 80)},
        #     {'id': 'green', 'color': (60, 100, 100)},
        #     {'id': 'blue', 'color': (120, 100, 100)},
        #     {'id': 'violet', 'color': (140, 100, 82.7)})
        #     {'id': 'violet', 'color': (150, 255, 255)})
        # )
            

        self.font = cv2.FONT_HERSHEY_SIMPLEX


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detect_workspace_boundary(self):
        """
        Use template matching to automatically detect workspace boundary
        """
        w, h = self.template_img.shape[::-1]

        video_gray = cv2.cvtColor(self.VideoFrame, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(video_gray, self.template_img, cv2.TM_CCOEFF_NORMED)

        threshold = 0.9
        rows, cols = np.where( res >= threshold)

        # calc workspace bounding box
        top_left = (cols.min() - w/2 + 50, rows.min() - h/2 + 50)
        bottom_right = (cols.max() + w - 25, rows.max() + h - 25)

        # calc location of arm
        middle_top = (np.abs(top_left[0] - bottom_right[0]) // 2, top_left[1])
        arm_top_left = (middle_top[0] + top_left[0] - 75, middle_top[1])
        arm_bottom_right = (middle_top[0] + top_left[0] + 75, top_left[1] + 225)

        return top_left, bottom_right, arm_top_left, arm_bottom_right
        
    
    def retrieve_area_color(self, data, contour, labels):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(data, mask=mask)[:3]
        min_dist = (np.inf, None)
        # print("mean: " + str(mean))
        for label in labels:
            # d = np.linalg.norm(cv2.cvtColor(np.uint8(label["color"]), cv2.COLOR_BGR2HSV)[0, 0] - np.array(mean))
            # print("HSV: " + str(cv2.cvtColor(np.uint8(label["color"]), cv2.COLOR_BGR2HSV)[0, 0]) + " d: " + str(d))

            d = np.linalg.norm(label["color"] - np.array(mean))
            # print("d: " + str(d))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        return min_dist[1] 

    def add_arm_to_mask(self, mask):
        '''
        add the arm to a mask based on FK values 
        '''
        w_ee = [self.ee_pose[0]*1000, self.ee_pose[1]*1000, self.ee_pose[2]*1000, 1]
        w_ee = np.array(w_ee)

        # print(w_ee)
        #find ee, wrist and elbow locations in camera coords
        c_ee = self.to_camera_coords(w_ee)
        w_wrist = self.wrist_pos*1000
        w_elbow = self.elbow_pos*1000
        c_wrist = self.to_camera_coords(np.append(w_wrist, 1))
        c_elbow = self.to_camera_coords(np.append(w_elbow, 1))

        #create and draw the box
        a = -90 + 180.0/np.pi * np.arctan2(self.base[1] - c_ee[1], self.base[0] - c_ee[0])
        h = np.linalg.norm(np.array(self.base) - c_ee[:2]) + 150
        w = 150.0
        c = np.mean([np.array(self.base), c_ee[:2]], axis=0)

        box = cv2.boxPoints(((c[0], c[1]), (w, h), a))
        box = np.int0(box)
        cv2.drawContours(self.VideoFrame, [box], 0, (255, 0, 0), 2)
        cv2.fillPoly(mask, [box], 0)

        #check if the wrist and elbow are outside the created mask
        # print("elbow camera pos")
        # print(c_elbow)
        # print("ee mask value")
        # print(mask[int(c_ee[0]), int(c_ee[1])])
        if mask[int(c_wrist[1]), int(c_wrist[0])] > 0 or mask[int(c_elbow[1]), int(c_elbow[0])] > 0:
            #if it is outside make a new box that covers it
            # print("wrist/elbow not masked")
            a = -90 + 180.0/np.pi * np.arctan2(c_elbow[1] - c_wrist[1], c_elbow[0] - c_wrist[0])
            h = np.linalg.norm(c_elbow[:2] - c_wrist[:2]) + 150
            w = 150.0
            c = np.mean([c_elbow[:2], c_wrist[:2]], axis=0)

            box = cv2.boxPoints(((c[0], c[1]), (w, h), a))
            box = np.int0(box)
            cv2.drawContours(self.VideoFrame, [box], 0, (0, 0, 255), 2)
            cv2.fillPoly(mask, [box], 0)

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        lower = 500

        if self.auto_Hinv is None:
            upper = self.rough_Hinv[2, 3]
        else:
            upper = self.auto_Hinv[2, 3]

        upper -= 18

        if self.top_left is None or self.bottom_right is None:
            # ideally this should automatically get the workspace boundary
            self.top_left, self.bottom_right, self.arm_top_left, self.arm_top_right \
                = self.detect_workspace_boundary()

        # draw workspace boundary
        cv2.rectangle(self.VideoFrame, self.top_left, self.bottom_right, (255, 0, 0), 2)
        cv2.rectangle(self.VideoFrame, self.arm_top_left, self.arm_top_right, (255, 0, 0), 2)

        # mask out arm base & outside board
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        cv2.rectangle(mask, self.top_left, self.bottom_right, 255, cv2.FILLED)
        cv2.rectangle(mask, self.arm_top_left, self.arm_top_right, 0, cv2.FILLED)

        # mask out rest of arm based on ee pose 
        self.add_arm_to_mask(mask)
        
        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)
      
        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find contours of the top of stacks 
        final_contours = []
        d_offset = 10
        for contour in contours:
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)

            if M['m00'] != 0:
                mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                thresh = cv2.bitwise_and(self.DepthFrameRaw, self.DepthFrameRaw, mask=mask)
                
                depth = np.min(thresh[np.nonzero(thresh)])

                thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, depth - d_offset, depth + d_offset), mask)
                _, c, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                final_contours.extend(c)

         # convert video frame from rgb to hsv
        added_ids = []
        for contour in final_contours:
            # color = self.retrieve_area_color(hsvImg, contour, self.colors)
            color = self.retrieve_area_color(self.VideoFrame, contour, self.colors)
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(self.VideoFrame, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
                
                # print(color, int(theta), cx, cy)

                # draw actual contour
                cv2.drawContours(self.VideoFrame, [contour], -1, (0,255,255), thickness=1)

                # draw bounding box around blocks
                rect = cv2.minAreaRect(contour)
                area = rect[1][0]* rect[1][1]
                # print(area)
                if area > 300:
                    if area > 1000:
                        size = 'l'
                        cv2.putText(self.VideoFrame, size, (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
                    else: 
                        size = 's'
                        cv2.putText(self.VideoFrame, 's', (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
                    
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(self.VideoFrame, [box], 0, (0, 255, 0), 2)

                    # track block info 
                    center = rect[0]
                    id = self.to_add(center)
                    if id is None:
                        id = self.generate_id()
                        depth = self.DepthFrameRaw[int(center[1]), int(center[0])]
                        self.block_info[id] = (id, center, box, theta, color, contour, size, depth)
           
                    # display id
                    # cv2.putText(self.VideoFrame, str(id), (cx+30, cy+30), self.font, 0.5, (255,255,255), thickness=2)

                    # track ids that have been detected on the current timestep
                    added_ids.append(id)

                    # self.block_info.append((1, center, box, theta, color, contour))

        # delete block info not detected on current timestep
        self.delete_blocks(added_ids)

        # print("number of blocks: " + str(len(self.block_info)))

    def positive_blocks(self):
        p_blocks = {}
        for id in self.block_info:
            block = self.block_info[id]
            center = block[1]
            depth = block[7]

            # convert to world to check half plane 
            center = [center[0], center[1], 1]
            pos_w = self.to_world_coords(depth, center)

            if pos_w[1] > 0:
                p_blocks[id] = (id, pos_w, block[2], block[3], block[4], block[5], block[6], block[7])

        return p_blocks

    def negative_blocks(self, y_cutoff=0):
        blocks = {}
        for id in self.block_info:
            block = self.block_info[id]
            center = block[1]
            depth = block[7]

            # convert to world to check half plane 
            center = [center[0], center[1], 1]
            pos_w = self.to_world_coords(depth, center)

            if pos_w[1] < y_cutoff:
                blocks[id] = (id, pos_w, block[2], block[3], block[4], block[5], block[6], block[7])

        return blocks

    def to_add(self, center, thresh=5):
        min = 10000000
        min_id = None 

        center = np.array(center)
        for id in self.block_info:
            block = self.block_info[id]
            bc = np.array(block[1])

            # s = np.stack((center, bc))
            # print(np.array(center))
            # print(np.array(bc))
            # print("norm: " + str(np.linalg.norm(center - bc)))

            # track id of closest block
            dist = np.linalg.norm(center - bc)
            if dist < min:
                min = dist 
                min_id = id

        if min > thresh:
            return None
                
        return min_id

    def delete_blocks(self, added_ids):
        to_delete = []
        for id in self.block_info:
            if id not in added_ids:
                to_delete.append(id)
        
        for id in to_delete:
            del self.block_info[id]

    def get_height_img(self):
        img = self.auto_Hinv[2, 3] - self.DepthFrameRaw
        print("H inv 2, 3: " + str(self.auto_Hinv[2, 3]))
        return img

    def generate_id(self):
        id = random.randint(0, 100000)
        while id in self.ids:
            id = random.randint(0, 100000)
        
        self.ids.append(id)
        return id

    def to_world_coords(self, z, uv_cam):
        X_c = z * np.matmul(np.linalg.inv(self.intrinsic_matrix), uv_cam)
        X_c = np.append(X_c, 1)
        if self.auto_Hinv is None:
            X_w = np.matmul(self.rough_Hinv, X_c)
        else:
            X_w = np.matmul(self.auto_Hinv, X_c)

        return X_w

    def to_camera_coords(self, w_c):
        if self.auto_Hinv is None:
            x_c = np.matmul(np.linalg.inv(self.rough_Hinv), w_c)
        else:
            x_c = np.matmul(np.linalg.inv(self.auto_Hinv), w_c)

        c_c = np.matmul(self.intrinsic_matrix, x_c[:3])
        c_c = c_c/c_c[2] 

        return c_c

    def auto_calibrate(self):
        self.tag_detections = self.tag_detections.T * 1000
        projection = np.hstack((np.eye(3), np.zeros([3,1])))
        
        h_detections = np.vstack((self.tag_detections, np.ones((1,4))))
        tag_pixel_detections = np.matmul(np.matmul(self.intrinsic_matrix, projection), h_detections)
        tag_pixel_detections = tag_pixel_detections/self.tag_detections[2,:]


        for i in range(tag_pixel_detections.shape[1]):
            #correct tag_detections depth
            d = self.DepthFrameRaw[int(tag_pixel_detections[1,i]), int(tag_pixel_detections[0,i])]
            self.tag_detections[2,i] = d

        # Kabsch algorithm wikipedia.org/wiki/Kabsh_algorithm
        locations_centroid = np.mean(self.tag_locations, axis=1).reshape(-1,1)
        detections_centroid = np.mean(self.tag_detections, axis=1).reshape(-1,1)

        H = np.matmul((self.tag_detections - detections_centroid), np.transpose(self.tag_locations - locations_centroid))

        [U, _, V_t] = np.linalg.svd(H)

        R = np.matmul(V_t,U.T)
        # print(R)
        if np.linalg.det(R) < 0:
            V_t.T[:,2] *= -1
            R = np.matmul(V_t, U.T)
        print(R)

        # t = detections_centroid - np.matmul(R, locations_centroid)
        t = locations_centroid - np.matmul(R, detections_centroid)
        self.auto_Hinv = np.vstack((np.hstack((R,t.reshape((3,1)))), [0,0,0,1]))

        # print(self.tag_detections, detections_centroid)
        # print(self.tag_locations, locations_centroid)

        # print(self.auto_Hinv)
        # print(self.rough_Hinv)
        


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            # cv_image = np.zeros((720, 1280, 3)).astype(np.uint8)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        det_map = {}        
        for detection in data.detections:
            id = detection.id[0]
            p = detection.pose.pose.pose.position

            # add to map for easy loc assignment
            det_map[id] = [p.x, p.y, p.z]

        d = np.zeros((4, 3))
        for id in det_map:
            # print(det_map[id])
            d[id - 1] = np.array(det_map[id])

        self.camera.tag_detections = d


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()

        # update block detections from depth frame 
        self.camera.detectBlocksInDepthImage()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
