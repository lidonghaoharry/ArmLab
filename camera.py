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
        self.block_info = []

        # image for template matching
        self.template_img = cv2.imread("template.png", 0)

        # workspace boundary for drawing 
        self.top_left = None 
        self.bottom_right = None
        self.arm_top_left = None 
        self.arm_top_right = None

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
        arm_bottom_right = (middle_top[0] + top_left[0] + 75, top_left[1] + 325)

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

        # mask out arm & outside board
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        cv2.rectangle(mask, self.top_left, self.bottom_right, 255, cv2.FILLED)
        cv2.rectangle(mask, self.arm_top_left, self.arm_top_right, 0, cv2.FILLED)
        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)
      
        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


         # convert video frame from rgb to hsv
        # hsvImg = cv2.cvtColor(self.VideoFrame, cv2.COLOR_BGR2HSV)
        # self.block_info = [] # reset block info every timestep
        new_blocks = []
        for contour in contours:
            # color = self.retrieve_area_color(hsvImg, contour, self.colors)
            color = self.retrieve_area_color(self.VideoFrame, contour, self.colors)

            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(self.VideoFrame, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
                # cv2.putText(self.VideoFrame, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
                # print(color, int(theta), cx, cy)

                # draw actual contour
                cv2.drawContours(self.VideoFrame, [contour], -1, (0,255,255), thickness=1)

                # draw bounding box around blocks
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.VideoFrame, [box], 0, (0, 255, 0), 2)

                # # track block info 
                # center = rect[0]
                # if self.to_add(center, thresh=20):
                #     new_blocks.append((center, box, theta, color, contour))


    def to_add(self):
        pass


    def to_world_coords(self, z, uv_cam):
        X_c = z * np.matmul(np.linalg.inv(self.intrinsic_matrix), uv_cam)
        X_c = np.append(X_c, 1)
        if self.auto_Hinv is None:
            X_w = np.matmul(self.rough_Hinv, X_c)
        else:
            X_w = np.matmul(self.auto_Hinv, X_c)

        return X_w

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
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
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
