#!/usr/bin/env python

# coding: utf-8

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
from tw_tennis.msg import ProcessDetectionAction, ProcessDetectionGoal
from actionlib import SimpleActionClient


# Define Globals
SATURATION_LOWER = 28
SATURATION_MAX = 255  # Max 'S' value in HSV
BRIGHTNESS_LOWER = 108
BRIGHTNESS_MAX = 255  # Max 'V' Value in HSV
HUE_LOWER = 20
HUE_UPPER = 70
NUM_BLURS = 3
RADIUS_MIN = 10
RADIUS_MAX = 80
PARAM_1 = 100
PARAM_2 = 32

TENNIS_THRESH = 50 # Number of consecutive detections needed to be a "true" detection.
center_container = []
count = 0


def img_mask(image):
# input raw image
# outputs array of masked images (red, yellow, green, blue, original)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Generate HSV Threshold (yellow to light blue)
    color_thresh = np.array([[HUE_LOWER, SATURATION_LOWER, BRIGHTNESS_LOWER],
        [HUE_UPPER, SATURATION_MAX, BRIGHTNESS_MAX]])

    # Generate Mask
    color_mask = cv.inRange(img_hsv, color_thresh[0], color_thresh[1])

    img_masked = cv.bitwise_and(image, image, mask=color_mask)

    return img_masked
    

def houghCircles(image):
	
    # Gray image
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = imgray.copy()
    for i in range(NUM_BLURS):
        img = cv.medianBlur(img, ksize=5)
        img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)    
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



def pd_client(goal):
    """
    The client portion of the Action between Subsystem 2 (this node) 
    and Subsystem 3 (processing node).

    Parameter(s):
    goal: ProcessDetectionGoal

    Output(s):
    bool
    """
    client = SimpleActionClient('ProcessDetection', ProcessDetectionAction)

    # Wait for Server
    client.wait_for_server()

    # Generate goal and send to SimpleActionServer
    client.send_goal(goal)

    # Wait for server to finish performing action (Blocking)
    client.wait_for_result()
    
    # Return result
    result = client.get_result()
    return result.was_successful



def callback(msg):
    """
    Essentially the "main" function. Everything is coordinated here.
    The logic is bulleted below.

    Parameter(s):
    msg: Image

    Outputs:
    None
    """
    global center_container
    global count
    last_count = count
    # ROS <--> OpenCV bridge
    br = CvBridge()

    # ball_found = False

    # get message data
    #rospy.loginfo("Receiving image...")
    img = br.imgmsg_to_cv2(msg)

    """
    Logic Flow: Create center container
    - As circle_centers are detected consecutively, fill container
    - If detection missed, empty container (restart filling)
    - If consec_thresh hit, convert all arrays in container to ROSNumpy msgs
    - Then, convert container into ROSNumpyList msg
    - Send message as a SimpleActionClient (block until SimpleActionServer result)
    - Begin next cycle (may need to add intermediate step for when robot moves?)
    """


    # Color filter the image
    masked = img_mask(img)
    # Apply Hough
    circles = houghCircles(masked)
    if circles is not None:
        count += 1
        circles = np.uint16(np.around(circles))
        for j in circles[0, :]:
            center = (j[0], j[1])
            # circle center
            cv.circle(img, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = j[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)
    if last_count == count:
        # Detection Chain Broken; reset variables
        count = 0
        last_count = count
        del center_container[:]
    else:
        # Chain continued; update variables
        # NOTE: At this point, the detected circle centers are Numpy arrays!
        last_count = count
        center_container.append(circles)
        rospy.loginfo("Consec. Detection Count: {}".format(count))

    # Display images	
    # The blurred image from houghCircles is also displayed
    cv.imshow("Test", img)
    cv.waitKey(1)

    # Detection Threshold Check
    if count >= TENNIS_THRESH:
        # DO EVERYTHING ELSE HERE
        rospy.loginfo("{}: Ball Detected! Sending to Processing Division.".format(rospy.get_name()))
        # Convert arrays to messages
        arrays = [construct_rosnumpy(arr) for arr in center_container]
        detections = ROSNumpyList(arrays)
        # Perform action client role
        msg = ProcessDetectionGoal()
        msg.img_shape = img.shape
        msg.detections = detections
        # print(msg.img_shape)
        # print(msg.detections)
        result = pd_client(msg)
        rospy.loginfo('{}: Processing was successful: {}'.format(rospy.get_name(), result))
        if result==True:
            # Reset Center container and counters
            count = 0
            last_count = count
            del center_container[:]
        else:
            rospy.signal_shutdown("SS02: Processing Error from SS03.")

def img_sub():
    """
    Instantiates the node;
    Sets up subscriber
    Begin Blocking.
    """
    # ROS Setup
    rospy.init_node("ss02_BallDetector", anonymous=True)
    name = rospy.get_name()
    rospy.loginfo("{}: Online.".format(name))
    # Subscriber Setup
    TOPIC = "image_hub"
    sub = rospy.Subscriber(TOPIC, Image, callback)
    rospy.loginfo("{}: Beginning listener.".format(name))

    # Don't do anything until the exit
    rospy.spin()

    cv.destroyAllWindows()


if __name__ ==   '__main__':
    img_sub()

