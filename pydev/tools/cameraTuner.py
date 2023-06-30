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

# ==========
# Variables
# ==========
CAMERA = 0

# 'H' value in HSV (ranges 0, 179)
HUE_LOWER, HUE_MAX = 0, 179 
# 'S' value in HSV [0, 255]
SATURATION_LOWER, SATURATION_MAX = 0, 255
# 'V' Value in HSV [0, 255]
BRIGHTNESS_LOWER, BRIGHTNESS_MAX = 0, 255
# Hough Circle Params (define minDist later)
CIRCLE_RADIUS_MIN = 10
CIRCLE_RADIUS_MAX = 70
PARAM_1 = 105
PARAM_2 = 35
# Contrast Params (alpha is gain, beta is brightness)
# Use new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
ALPHA = 1 # ranges from 1-3
BETA = 0 # Ranges from 0-100 (may be redundant with the use of HSV)
alpha_slider_max = 300
max_blurs = 100
BLUR_IT = 2

# ==========
# Functions
# ==========

def img_mask(image)->np.ndarray:
# input raw image
# outputs array of masked images (red, yellow, green, blue, original)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Generate HSV Threshold (yellow to light blue)
    color_thresh = np.array([[HUE_LOWER, SATURATION_LOWER, BRIGHTNESS_LOWER],
    						  [HUE_MAX, SATURATION_MAX, BRIGHTNESS_MAX]])
    
    # Generate Mask
    color_mask = cv.inRange(img_hsv, color_thresh[0], color_thresh[1])
    
    img_masked = cv.bitwise_and(image, image, mask=color_mask)

    return img_masked


def Hough(image:np.ndarray)->np.ndarray:
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = imgray.copy()
    for i in range(BLUR_IT):
        img = cv.medianBlur(img, ksize=5)
        img = cv.GaussianBlur(img, ksize=(5,5), sigmaX=0)

    cv.imshow("Blurred", img)
    #cv.waitKey(1)
    rows, cols= img.shape[0:2]
    cv.moveWindow("Blurred", cols+5, 0)
    DIST = min(rows / 6, cols/6)
    
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1,
                              minDist=DIST, 
                              param1=PARAM_1, param2=PARAM_2,
                              minRadius=CIRCLE_RADIUS_MIN,
                              maxRadius=CIRCLE_RADIUS_MAX)
    return circles

def track_callback(val):
    global HUE_LOWER
    global HUE_UPPER
    global SATURATION_LOWER
    global SATURATION_MAX
    global BRIGHTNESS_LOWER
    global BRIGHTNESS_MAX
    global CIRCLE_RADIUS_MIN
    global CIRCLE_RADIUS_MAX
    global PARAM_1
    global PARAM_2
    global ALPHA
    global BLUR_IT
    
    # Update Variables
    HUE_LOWER = cv.getTrackbarPos('Hmin', windowName)
    HUE_UPPER = cv.getTrackbarPos('Hmax', windowName)
    SATURATION_LOWER = cv.getTrackbarPos('Smin', windowName)
    SATURATION_MAX = cv.getTrackbarPos('Smax', windowName)
    BRIGHTNESS_LOWER = cv.getTrackbarPos('Vmin', windowName)
    BRIGHTNESS_MAX = cv.getTrackbarPos('Vmax', windowName)
    CIRCLE_RADIUS_MIN = cv.getTrackbarPos('Rmin', windowName)
    CIRCLE_RADIUS_MAX = cv.getTrackbarPos('Rmax', windowName)
    PARAM_1 = cv.getTrackbarPos('Param 1', windowName)
    PARAM_2 = cv.getTrackbarPos('Param 2', windowName)
    ALPHA = cv.getTrackbarPos("Contrast", windowName)/100
    BLUR_IT = cv.getTrackbarPos('Num Blurs', windowName)
    
if __name__ == '__main__':
    
    # Construct Trackbars
    param_config = cv.namedWindow("Parameter Configuration")
    windowName = "Parameter Configuration"
    
    cv.createTrackbar('Hmin', windowName, 20, 180, track_callback)
    cv.createTrackbar('Hmax', windowName, 70, 180, track_callback)
    cv.createTrackbar('Smin', windowName, 16, 255, track_callback)
    cv.createTrackbar('Smax', windowName, 255, 255, track_callback)
    cv.createTrackbar('Vmin', windowName, 108, 255, track_callback)
    cv.createTrackbar('Vmax', windowName, 255, 255, track_callback)
    cv.createTrackbar('Rmin', windowName, 10, 200, track_callback)
    cv.createTrackbar('Rmax', windowName, 80, 200, track_callback)
    cv.createTrackbar('Param 1', windowName, 100, 250, track_callback)
    cv.createTrackbar('Param 2', windowName, 32, 100, track_callback)
    cv.createTrackbar('Contrast', windowName, 85, alpha_slider_max, track_callback)
    cv.createTrackbar('Num Blurs', windowName, 0, max_blurs, track_callback)
    
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
            #cv.imshow("Original", frame)
            #cv.moveWindow("Original", 0, 0)
            # run contrast 
            image = cv.convertScaleAbs(frame.copy(), alpha=ALPHA, beta=BETA)
            masked = img_mask(image)
            
            #print(f'Average HSV: {np.mean(masked.reshape(-1, 3), axis=0).round()}')
            
            circles = Hough(masked)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    center = (j[0], j[1])
                    cv.circle(image, center, 1, (0, 0, 255), 3)  # circle center
                    radius = j[2]
                    cv.circle(image, center, radius, (255, 0, 255), 3)  # circle outline
            row, col, _ = frame.shape
            # Display:
            cv.imshow("Circles", image)
            #cv.imshow("Masked", masked)
            cv.moveWindow("Circles", 0, 0)
            #cv.moveWindow("Masked", col + 1, 0)
            key = cv.waitKey(1)
            if key == 27 or key == ord('q'):
                break   
    # Cleanup      
    cap.release()
    cv.destroyAllWindows()
