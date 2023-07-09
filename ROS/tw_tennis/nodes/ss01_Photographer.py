#!/usr/bin/env python

# -*- coding:utf-8 -*-

"""
Author: Terrance Williams
Date: 13 June 2023
Description: Creates a ROS Publisher to transfer images to the /image_hub topic

Credit: Addison Sears-Collins
https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/
"""

import cv2 as cv
import sys
# ROS-specific imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def publish_msg():
    PUB_RATE = 10 # Hz (same as FPS in this case?)
    # ROS setup
    rospy.init_node("ss01_Photographer", anonymous=False) # Only one camera
    
    image_topic = rospy.get_param("ss01_ss02_topic")
    pub = rospy.Publisher(image_topic, Image, queue_size=1)  # adjust the queue_size
    rate = rospy.Rate(PUB_RATE)
    rospy.loginfo("{}: Online.".format(rospy.get_name()))
    
    # Create ROS <--> OpenCV Bridge
    br = CvBridge()

    # OpenCV Image Capture
    cap = cv.VideoCapture(0)  # capture JetHexa camera

    if not cap.isOpened():
        ros.loginfo("Could not open camera. Exiting...")
        cap.release()
        sys.exit()

    # Capture images and send
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            #rospy.loginfo("Sending Image")
            # scale image down (16:9 ratio width:height)
            height, width, _ = frame.shape
            height, width = int(height/2), int(width/2)
            #height = 720
            #width = int(16*(height/9))
            down_frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
            # send image
            msg = br.cv2_to_imgmsg(down_frame)
            pub.publish(msg)

            # Display Live image feed (because SS02's feed gets blocked)
            cv.imshow("SS01 Live Feed", down_frame)
            cv.moveWindow("SS01 Live Feed", 0, 0)
            cv.waitKey(1)
        rate.sleep()
    # Clean up for exit
    cap.release()
    cv.destroyAllWindows()


if __name__ ==   '__main__':
    try:
        publish_msg()
    except rospy.ROSInterruptException:
        pass


