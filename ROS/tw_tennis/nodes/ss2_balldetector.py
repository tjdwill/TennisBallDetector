#!/usr/bin/env python
# coding: utf-8

"""
Title: Tennis Ball Detector Node
Author: Terrance Williams
Date: 13 June 2023
Description: Creates a ROS Subscriber to transfer images to the /image_hub topic

Credit: Addison Sears-Collins
https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/

NOTE: This is a Python 2 script
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np


# Define Globals
SATURATION_LOWER = 50
SATURATION_MAX = 255  # Max 'S' value in HSV
BRIGHTNESS_LOWER = 20
BRIGHTNESS_MAX = 255  # Max 'V' Value in HSV
YELLOW_LOWER = 22
GREEN_UPPER = 85
TENNIS_THRESH = 10 # Number of consecutive detections needed to be a "true" detection.
count = 0

def img_mask(image):
# input raw image
# outputs array of masked images (red, yellow, green, blue, original)
	img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	# Generate HSV Threshold (yellow to light blue)
	color_thresh = np.array([[YELLOW_LOWER, SATURATION_LOWER, BRIGHTNESS_LOWER],
							  [GREEN_UPPER, SATURATION_MAX, BRIGHTNESS_MAX]])

	# Generate Mask
	color_mask = cv.inRange(img_hsv, color_thresh[0], color_thresh[1])

	img_masked = cv.bitwise_and(image, image, mask=color_mask)

	return img_masked
	

def houghCircles(image):
	
# Gray image
	imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
	img = cv.medianBlur(imgray, ksize=5)
		    
	# Hough Params
	rows, cols= img.shape[0:2]
	DIST = min(rows / 8, cols/8)

	CIRCLE_RADIUS_MIN = 1
	CIRCLE_RADIUS_MAX = 120
	PARAM_1 = 50
	PARAM_2 = 25
	circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1,
		                      minDist=DIST, 
		                      param1=PARAM_1, param2=PARAM_2,
		                      minRadius=CIRCLE_RADIUS_MIN,
		                      maxRadius=CIRCLE_RADIUS_MAX)

	return circles


def callback(msg):
	global count
	last_count = count
	# ROS <--> OpenCV bridge
	br = CvBridge()

	# ball_found = False

	# get message data
	#rospy.loginfo("Receiving image...")
	img = br.imgmsg_to_cv2(msg)

	# Color filter the image
	masked = img_mask(img)
	# Apply Hough
    '''https://docs.opencv.org/4.6.0/d4/d70/tutorial_hough_circle.html'''
	circles = houghCircles(masked)
    # Draw circles; Track consec. detections
	if circles is not None:
		count += 1
		circles = np.uint16(np.around(circles))
		for j in circles[0, :]:
			center = (j[0],j[1])
			# circle center
			cv.circle(img, center, 1, (0, 0, 255), 3)
			# circle outline
			radius = j[2]
			cv.circle(img, center, radius, (255, 0, 255), 3)
    
    # Thresholding
	if last_count == count:
		count = 0
		last_count = count
	else:
		last_count = count
	print "Detection Count: {}".format(count)
	if count >= TENNIS_THRESH:
		print("Ball Detected!")
        ''' PUT BOOLEAN PUBLISH COMMAND HERE IF DESIRED'''
	# Display images
	cv.imshow("Mask", masked)	
	cv.imshow("Test", img)
	cv.waitKey(1)


def img_sub():

	# ROS Setup
	rospy.init_node("ball_detector", anonymous=True)
	name = rospy.get_name()
	TOPIC = "image_hub"
	sub = rospy.Subscriber(TOPIC, Image, callback)
	rospy.loginfo("{}: Beginning listener.".format(name))
    
	# Don't do anything until the exit
	rospy.spin()
	cv.destroyAllWindows()


if __name__ ==   '__main__':
	'''TO-DO: Add conditional argument "inside":bool '''
    img_sub()
