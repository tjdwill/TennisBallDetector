#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:11:52 2023
@title: OpenCV Camera Tuner
@author: Tj
@description: Facilitates the use of a given camera for color thresholding,
circle detection, and (possibly) contrast balancing.

@inspiration:
    https://github.com/SaifRehman/Real-Time-RGB-Color-Filtering-with-Python
    https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
"""
from sys import exit
import numpy as np
import cv2 as cv
import rospy

# ==========
# Variables
# ==========
CAMERA = 0

# Contrast Params (alpha is gain, beta is brightness)
# Use new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
ALPHA = 1  # ranges from 1-3
BETA = 0  # Ranges from 0-100 (may be redundant with the use of HSV)
alpha_slider_max = 300
max_blurs = 100

# ==========
# Functions
# ==========


def img_mask(image) -> np.ndarray:
    # input raw image
    # outputs array of masked images (red, yellow, green, blue, original)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Generate HSV Threshold (yellow to light blue)
    color_thresh = np.array([[HUE_LOWER, SATURATION_LOWER, BRIGHTNESS_LOWER],
                             [HUE_UPPER, SATURATION_UPPER, BRIGHTNESS_UPPER]])
    # Generate Mask
    color_mask = cv.inRange(img_hsv, color_thresh[0], color_thresh[1])

    img_masked = cv.bitwise_and(image, image, mask=color_mask)

    return img_masked


def Hough(image: np.ndarray) -> np.ndarray:

    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = imgray.copy()
    for i in range(NUM_BLURS):
        img = cv.medianBlur(img, ksize=5)
        img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    cv.imshow("Blurred", img)
    # cv.waitKey(1)
    rows, cols = img.shape[0:2]
    cv.moveWindow("Blurred", cols+5, 0)
    DIST = min(rows / 6, cols/6)
    
    # print(f'DIST:{DIST}\nRADIUS_MAX: {RADIUS_MAX}')
    # print(PARAM_1, PARAM_2, RADIUS_MIN)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1,
                              minDist=DIST,
                              param1=PARAM_1, param2=PARAM_2,
                              minRadius=RADIUS_MIN,
                              maxRadius=RADIUS_MAX)
    return circles


def track_callback(val):
    global HUE_LOWER
    global HUE_UPPER
    global SATURATION_LOWER
    global SATURATION_UPPER
    global BRIGHTNESS_LOWER
    global BRIGHTNESS_UPPER
    global RADIUS_MIN
    global RADIUS_MAX
    global PARAM_1
    global PARAM_2
    global ALPHA
    global NUM_BLURS

    # Update Variables
    HUE_LOWER = cv.getTrackbarPos('Hmin', windowName)
    HUE_UPPER = cv.getTrackbarPos('Hmax', windowName)
    SATURATION_LOWER = cv.getTrackbarPos('Smin', windowName)
    SATURATION_UPPER = cv.getTrackbarPos('Smax', windowName)
    BRIGHTNESS_LOWER = cv.getTrackbarPos('Vmin', windowName)
    BRIGHTNESS_UPPER = cv.getTrackbarPos('Vmax', windowName)
    RADIUS_MIN = cv.getTrackbarPos('Rmin', windowName)
    RADIUS_MAX = cv.getTrackbarPos('Rmax', windowName)
    PARAM_1 = cv.getTrackbarPos('Param 1', windowName)
    PARAM_2 = cv.getTrackbarPos('Param 2', windowName)
    ALPHA = cv.getTrackbarPos("Contrast", windowName)/100
    NUM_BLURS = cv.getTrackbarPos('Num Blurs', windowName)
    



if __name__ == '__main__':
    # Setup ROS Node
    
    rospy.init_node("cameraTuner", anonymous=False)
    # Get Parameters from RPS
    img_params = rospy.get_param("img_params")
    rospy.sleep(1)
    
    print(img_params)
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
    ALPHA = img_params["ALPHA"]
    '''
    print(HUE_LOWER, HUE_UPPER, SATURATION_LOWER, SATURATION_UPPER, 
            BRIGHTNESS_LOWER, BRIGHTNESS_UPPER, NUM_BLURS,
            RADIUS_MIN, RADIUS_MAX, PARAM_1, PARAM_2, ALPHA)
    '''   
    # Construct Trackbars
    # I set the default value of each trackbar to the value on the ROS
    # Parameter Server. I do it by dictionary reference because
    # for some reason each trackbar initialization calls
    # track_callback, overwriting the global variables. 
    # This way, the program initializes correctly.
    param_config = cv.namedWindow("Parameter Configuration")
    windowName = "Parameter Configuration"
    
    cv.createTrackbar('Hmin', windowName, img_params["HUE_LOWER"], 180, track_callback)
    cv.createTrackbar('Hmax', windowName, img_params['HUE_UPPER'], 180, track_callback)
    cv.createTrackbar('Smin', windowName, img_params['SATURATION_LOWER'], 255, track_callback)
    cv.createTrackbar('Smax', windowName, img_params['SATURATION_UPPER'], 255, track_callback)
    cv.createTrackbar('Vmin', windowName, img_params['BRIGHTNESS_LOWER'], 255, track_callback)
    cv.createTrackbar('Vmax', windowName, img_params['BRIGHTNESS_UPPER'], 255, track_callback)
    cv.createTrackbar('Rmin', windowName, img_params['RADIUS_MIN'], 200, track_callback)
    cv.createTrackbar('Rmax', windowName, img_params['RADIUS_MAX'], 200, track_callback)
    cv.createTrackbar('Param 1', windowName, img_params['PARAM_1'], 250, track_callback)
    cv.createTrackbar('Param 2', windowName, img_params['PARAM_2'], 100, track_callback)
    cv.createTrackbar('Contrast', windowName, int(100 * img_params["ALPHA"]),
                      alpha_slider_max, track_callback)
    cv.createTrackbar('Num Blurs', windowName, img_params['NUM_BLURS'], max_blurs, track_callback)
    
    # Get camera image
    cap = cv.VideoCapture(CAMERA)
    if not cap.isOpened():
        print("Camera Error. Exiting.")
        exit()

    # Get images
    while True:
        ret, frame = cap.read()
        frame = frame[:, ::-1]  # Flip horizontally
        if not ret:
            print("Streaming Error. Exiting")
            break
        else:
            # cv.imshow("Original", frame)
            # cv.moveWindow("Original", 0, 0)
            # run contrast
            image = cv.convertScaleAbs(frame.copy(), alpha=ALPHA, beta=BETA)
            masked = img_mask(image)

            #print(f'Average HSV: {np.mean(masked.reshape(-1, 3), axis=0).round()}') #noqa

            circles = Hough(masked)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    center = (j[0], j[1])
                    # Draw circle center
                    cv.circle(image, center, 1, (0, 0, 255), 3)
                    radius = j[2]
                    # Draw circle outline
                    cv.circle(image, center, radius, (255, 0, 255), 3)
            row, col, _ = frame.shape
            
            # Display:
            cv.imshow("Circles", image)
            # cv.imshow("Masked", masked)
            cv.moveWindow("Circles", 0, 0)
            cv.moveWindow(windowName, col + 1, row)
            key = cv.waitKey(1)
            if key == 27 or key == ord('q'):
                break
    
    # Update ROS Parameter Server
    img_params["HUE_LOWER"] = HUE_LOWER
    img_params["HUE_UPPER"] = HUE_UPPER
    img_params["SATURATION_LOWER"] = SATURATION_LOWER
    img_params["SATURATION_UPPER"] = SATURATION_UPPER
    img_params["BRIGHTNESS_LOWER"] = BRIGHTNESS_LOWER
    img_params["BRIGHTNESS_UPPER"] = BRIGHTNESS_UPPER
    img_params["NUM_BLURS"] = NUM_BLURS
    img_params["RADIUS_MIN"] = RADIUS_MIN
    img_params["RADIUS_MAX"] = RADIUS_MAX
    img_params["PARAM_1"] = PARAM_1
    img_params["PARAM_2"] = PARAM_2
    img_params["ALPHA"] = ALPHA

    rospy.set_param("img_params", img_params)

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
