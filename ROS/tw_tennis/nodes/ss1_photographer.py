#!/usr/bin/env python
# coding: utf-8

"""
Title: Photographer Node
Author: Terrance Williams
Date: 13 June 2023
Description: Creates a ROS Publisher to transfer images to the /image_hub topic

Credit: Addison Sears-Collins
https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/

NOTE: This is a Python 2 script
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import sys

def publish_msg():
	PUB_RATE = 10 # Hz (same as FPS in this case?)
	# ROS setup
	pub = rospy.Publisher("image_hub", Image, queue_size=1)  # adjust the queue_size
	rospy.init_node("JH_camera", anonymous=False) # Only one camera
	rate = rospy.Rate(PUB_RATE)

	# Create ROS <--> OpenCV Bridge
	br = CvBridge()

	# OpenCV Image Capture
	cap = cv.VideoCapture(0)  # capture JetHexa camera

	if not cap.isOpened():
		rospy.signal_shudown("Could not open camera." )

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

		rate.sleep()


if __name__ ==   '__main__':
	try:
		publish_msg()
	except rospy.ROSInterruptException:
		pass
