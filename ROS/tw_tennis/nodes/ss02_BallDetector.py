#!/usr/bin/env python

# -*- coding:utf-8 -*-

"""
Author: Terrance Williams
Date: 07 July 2023
Description: Creates a ROS Subscriber to transfer images to the /image_hub topic

Credit: Addison Sears-Collins
https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/
https://wiki.ros.org/actionlib_tutorials/Tutorials/
"""

from __future__ import print_function
import cv2 as cv
import numpy as np
# ROS-specifc imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from numpy_msgs.msg import ROSNumpyList
from numpy_msgs.rosnp_helpers import construct_rosnumpy
from tw_tennis.msg import ProcessDetectionAction, ProcessDetectionGoal, MoveRobotAction, MoveRobotGoal
from actionlib import SimpleActionClient


class BallDetector:
    # ================
    # Class Variables
    # ================
    # Image Processing Parameters
    
    img_params = rospy.get_param("img_params")

    HUE_LOWER = img_params["HUE_LOWER"]
    HUE_UPPER = img_params["HUE_UPPER"]
    SATURATION_LOWER = img_params["SATURATION_LOWER"]
    SATURATION_UPPER = img_params["SATURATION_UPPER"]
    BRIGHTNESS_LOWER = img_params["BRIGHTNESS_LOWER"]
    BRIGHTNESS_UPPER = img_params["BRIGHTNESS_UPPER"]
    NUM_BLURS = img_params["NUM_BLURS"]
    RADIUS_MIN = img_params["RADIUS_MIN"]
    RADIUS_MAX = img_params["RADIUS_MAX"]
    PARAM_1 = img_params["PARAM_1"]
    PARAM_2 = img_params["PARAM_2"]
    # ALPHA = img_params["ALPHA"]


    # Detection-Specific Infrastructure
    center_container = []
    count = 0
    # Number of consecutive detections needed to be a "true" detection. 
    # Make this a ROS Parameter?
    DETECTION_THRESH = rospy.get_param("DETECTION_THRESH") 
    TENNIS_THRESH = DETECTION_THRESH # this way, get_param isn't continuously called
    image_topic = rospy.get_param("ss01_ss02_topic")
    pd_topic = rospy.get_param("ss02_ss03_topic")
    move_topic = rospy.get_param("ss02_ss04_topic")

    def __init__(self):
        """
        Instantiates the node;
        Sets up subscriber
        """
        # ROS Setup
        rospy.init_node("ss02_BallDetector", anonymous=False)
        self.name = rospy.get_name()
        # Subscriber Setup
        rospy.loginfo("{}: Beginning listener.".format(self.name))
        sub = rospy.Subscriber(BallDetector.image_topic, Image, self.callback)
        rospy.loginfo("{}: Online.".format(self.name))

    # =================
    # Private Functions
    # =================
    def _reset_detections(self):
        # Reset circle container
        # Reset count
        del BallDetector.center_container[:]
        BallDetector.count = 0
        assert (len(BallDetector.center_container) == 0) and (BallDetector.count == 0)
    # =================
    # Public Functions
    # =================
    def callback(self, msg):
        """
        Essentially the "main" function. Everything is coordinated here.
        The logic is bulleted below.

        Parameter(s):
        msg: Image

        Outputs:
        None
        """
        center_container = BallDetector.center_container
        last_count = BallDetector.count

        # ROS <--> OpenCV bridge
        br = CvBridge()
        img = br.imgmsg_to_cv2(msg)

        """
        Logic Flow:
        - As circle_centers are detected consecutively, fill container
        - If detection missed, empty container (restart filling)
        - If consec_thresh hit, convert all arrays in container to ROSNumpy msgs
        - Then, convert container into ROSNumpyList msg
        - Send message as a SimpleActionClient (block until SimpleActionServer result)
        - Send returned angle (if successful) to SS04
        - Begin next cycle (may need to add intermediate step for when robot moves?)
        """

        # Conduct Image Processing
        masked = self.img_mask(img)

        circles = self.houghCircles(masked, debug=False)
        if circles is not None:
            BallDetector.count += 1
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                # circle center
                cv.circle(img, center, 1, (0, 0, 255), 3)
                # circle outline
                radius = j[2]
                cv.circle(img, center, radius, (255, 0, 255), 3)

        # Check the chain of detections.        
        if last_count == BallDetector.count:
            # Detection Chain Broken; reset global variables
            # last_count = BallDetector.count
            self._reset_detections()
           
        else:
            # Chain continued; update variables
            # NOTE: At this point, the detected circle centers are Numpy arrays!
            center_container.append(circles)
            assert len(BallDetector.center_container) == BallDetector.count
            rospy.loginfo("Consec. Detection Count: {}"
                    .format(BallDetector.count))

        # Display images	
        # The blurred image from houghCircles is also displayed
        cv.imshow("Circles", img)
        cv.moveWindow("Circles", 0, int(img.shape[0] + 10))
        cv.waitKey(1)

        # Detection Threshold Check
        if BallDetector.count >= BallDetector.TENNIS_THRESH:
            ''' 
            DO EVERYTHING ELSE HERE
            '''
            rospy.loginfo("{}: Ball Detected! Sending to Processing Division.".format(rospy.get_name()))
            # Convert arrays to messages
            arrays = [construct_rosnumpy(arr) for arr in center_container]
            detections = ROSNumpyList(arrays)

            # Perform action client role for SS03
            msg = ProcessDetectionGoal()
            msg.img_shape = img.shape
            msg.detections = detections
            # print(msg.img_shape)
            # print(msg.detections)
            result = self.pd_client(msg)
            
            if result is None:
                rospy.loginfo('{}: Processing FAIL.'.format(self.name))
                rospy.signal_shutdown("SS02: Processing Error from SS03.")
            elif result == 360:
                # Angle 360 means no ball was found. Try again.
                rospy.loginfo('{}: No ball found. Resetting...'.format(self.name))
                self._reset_detections()
            else:    
                rospy.loginfo(('{}: Processing was successful. '.
                               format(self.name) +
                               'Sending result to SS04 for motion control.')
                               )
            
                angle = result
                move_successful = self.move_robot_client(angle)
                if not move_successful:
                    rospy.signal_shutdown('{}: Unsuccessful Move Operation'.
                                          format(self.name))
                else:
                    rospy.loginfo('{}: Resetting for next detection.'
                                  .format(self.name))
                    self._reset_detections()
                    rospy.sleep(5)  

    """
    Image Processing Functions
    """
    def img_mask(self, image):
        # input raw image
        # outputs array of masked images (red, yellow, green, blue, original)

        # Constants
        HUE_LOWER = BallDetector.HUE_LOWER
        HUE_UPPER = BallDetector.HUE_UPPER
        SATURATION_LOWER = BallDetector.SATURATION_LOWER
        SATURATION_UPPER = BallDetector.SATURATION_UPPER
        BRIGHTNESS_LOWER = BallDetector.BRIGHTNESS_LOWER
        BRIGHTNESS_UPPER = BallDetector.BRIGHTNESS_UPPER



        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Generate HSV Threshold (yellow to light blue)
        color_thresh = np.array([[HUE_LOWER, SATURATION_LOWER, BRIGHTNESS_LOWER],
            [HUE_UPPER, SATURATION_UPPER, BRIGHTNESS_UPPER]])

        # Generate Mask
        color_mask = cv.inRange(img_hsv, color_thresh[0], color_thresh[1])

        img_masked = cv.bitwise_and(image, image, mask=color_mask)

        return img_masked
        

    def houghCircles(self, image, debug=False):
        PARAM_1 = BallDetector.PARAM_1
        PARAM_2 = BallDetector.PARAM_2
        RADIUS_MIN = BallDetector.RADIUS_MIN
        RADIUS_MAX = BallDetector.RADIUS_MAX
        NUM_BLURS = BallDetector.NUM_BLURS


        # Gray image
        imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img = imgray.copy()
        for i in range(NUM_BLURS):
            img = cv.medianBlur(img, ksize=5)
            img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
        # Show blurred image for debugging purposes
        if debug:    
            cv.imshow("Blurred", img)
        # Hough Params
        rows, cols= img.shape[0:2]
        DIST = min(rows / 6,  cols / 6)
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1,
                minDist=DIST,
                param1=PARAM_1,
                param2=PARAM_2,
                minRadius=RADIUS_MIN,
                maxRadius=RADIUS_MAX)

        return circles
    
    """
    Communication Functions

    pd_client -> SS02 is a SAC for SS03
    move_robot_client -> SS02 is a SAC for SS04
    """
    def pd_client(self, goal):
        """
        The client portion of the Action between Subsystem 2 (this node) 
        and Subsystem 3 (processing node).

        Parameter(s):
        goal: ProcessDetectionGoal

        Output(s):
        bool
        """

        # QUESTION: Is this instantiating a client each callback?
        client = SimpleActionClient(BallDetector.pd_topic, ProcessDetectionAction)

        # Wait for Server
        client.wait_for_server()

        # Generate goal and send to SimpleActionServer
        client.send_goal(goal)

        # Wait for server to finish performing action (Blocking)
        client.wait_for_result()
        
        # Return result
        result = client.get_result()
        return result.deg_angle
    
    def move_robot_client(self, angle):
        """
        The client portion of the Action between Subsystem 2 (this node) 
        and Subsystem 4 (robot controller node).

        Parameter(s):
        angle: float

        Output(s):
        bool
        """
        rospy.loginfo('{}: Establishing Connection to SS04.'.format(self.name))
        client = SimpleActionClient(BallDetector.move_topic, MoveRobotAction)


        # Wait for Server
        client.wait_for_server()
        rospy.loginfo('{}: Connected to SS04'.format(self.name))
        # Generate goal and send to SimpleActionServer
        goal = MoveRobotGoal(angle)
        client.send_goal(goal)

        # Wait for server to finish performing action (Blocking)
        client.wait_for_result()
        
        # Return result
        result = client.get_result()
        return result.move_successful

    def stop(self):
        rospy.loginfo("{}: Stopping.".format(self.name))
        cv.destroyAllWindows()
        

if __name__ ==   '__main__':
    ss02 = BallDetector()
    rospy.on_shutdown(ss02.stop)
    rospy.spin()

