# Pre-program stuff

import sys
def add_module(modules):
    for module in modules:
        if module not in sys.path:
            sys.path.append(module)
tjhelpers = 'C:/Users/Tj/Documents/Programming/Python_Scripts/tjCV/tjCVHelpers/'
proj2help = r'C:\Users\Tj\Documents\Graduate_School\UWF\Engineering_M.S\Summer_2023\CAP6665-Computer_Vision\Projects\Project_2\Code\scripts\tools'.replace('\\','/')
modules = [tjhelpers, proj2help]
add_module(modules)

import os 
import numpy as np
import cv2 as cv
from cv_helpers import KMeans, Image
import proj2helpfuncs as phf

#%%
# ==================
# Define Parameters
# ==================

day: bool = False  # Change the parameters depending on lighting.

# Image Parameters
if day:
    HUE_LOWER, HUE_MAX = 20, 70 
    SATURATION_LOWER, SATURATION_MAX = 16, 255
    BRIGHTNESS_LOWER, BRIGHTNESS_MAX = 108, 255
    ALPHA, BETA = 0.85, 0
    
    CIRCLE_RADIUS_MIN = 10
    CIRCLE_RADIUS_MAX = 80
    PARAM_1 = 100
    PARAM_2 = 32
    
    BLUR_IT = 3 # How many times we blur the image.
else:
    HUE_LOWER, HUE_MAX = 20, 70 
    SATURATION_LOWER, SATURATION_MAX = 16, 255
    BRIGHTNESS_LOWER, BRIGHTNESS_MAX = 108, 255
    ALPHA, BETA = 1.0, 0
    
    CIRCLE_RADIUS_MIN = 10
    CIRCLE_RADIUS_MAX = 80
    PARAM_1 = 100
    PARAM_2 = 32
    
    BLUR_IT = 3 # How many times we blur the image.

# Detection Parameters
DETECT_THRESH = 75
MIN_POINT_COUNT = int(0.8*DETECT_THRESH)


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
    # Blur image
    for _ in range(BLUR_IT):
        img = cv.medianBlur(img, ksize=5)
        img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
            
    rows, cols = img.shape[0:2]
    cv.imshow("Blurred", img)
    cv.moveWindow("Blurred", cols, 0)
    cv.waitKey(1)
    
    DIST = min(rows / 6, cols/6)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1,
                              minDist=DIST, 
                              param1=PARAM_1, param2=PARAM_2,
                              minRadius=CIRCLE_RADIUS_MIN,
                              maxRadius=CIRCLE_RADIUS_MAX)
    return circles


def record_circles():
    ...
    center_container = [] # Container for circle centers
    consec_detections = 0
    last_count = consec_detections
    THRESH_REACHED = False
    
    cap = cv.VideoCapture(0)
    # get images
    if not cap.isOpened():
        print("Could not open camera")
        cap.release()
        sys.exit()
            
    while not THRESH_REACHED:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't receive from video camera.")
            break
        if ret is True:
            # Flip horizontally
            frame = frame[:,::-1]
            #frame_cpy = cv.GaussianBlur(frame.copy() , ksize=(5,5), sigmaX=0)
            frame_cpy = frame.copy()
            # Processing
            masked = img_mask(frame_cpy)
            
           
            # Circle Detection
            circles = Hough(masked)
            if circles is not None:
                consec_detections += 1
                circles = np.uint16(np.around(circles))
                center_container.append(circles)
                #print(f'Container:\n{center_container}')
                for j in circles[0, :]:
                    center = (j[0],j[1])
                    # circle center
                    #print(frame.shape)
                    cv.circle(frame_cpy, center, 1, (0, 0, 255), 3)
                    # circle outline
                    radius = j[2]
                    cv.circle(frame_cpy, center, radius, (255, 0, 255), 3)
            # Display
            cv.imshow("Original", frame_cpy)
            #cv.imshow("Masked", masked)
            cv.moveWindow("Original", 0,0)
            key = cv.waitKey(1)
            if key == 27 or key==ord('q'):
                break   
            # Thresholding
            if last_count == consec_detections:
                consec_detections = 0
                last_count = consec_detections
                center_container.clear()
            else:
                last_count = consec_detections
                print("Detection Count: {}".format(consec_detections))
            if consec_detections >= DETECT_THRESH:
                print("Ball Detected!\n")
                THRESH_REACHED = True
                
    cap.release()
    #cv.destroyAllWindows()
    return center_container, frame

if __name__ == '__main__':
    
    centers, frame = record_circles()
    img_dimensions = frame.shape
    # Data Pre-Processing
    flat, means, k = phf.process_centers(centers)    

    # Run k-Means
    km = KMeans(flat, k, means)
    clusters, centroids, _ = km.cluster()
    centroids = np.int16(centroids)
    
    # Cluster Plot config
    y_max, x_max = img_dimensions[0] - 1, img_dimensions[1] - 1
    grnd_pxl = np.array(phf.get_ground_pixel(img_dimensions))
        
    # Update Plot
    km._axes2D.scatter(grnd_pxl[1], grnd_pxl[0], color='k',
                       label='Ground Pixel', edgecolor='k', zorder=100)
    
    
    #===================
    # Cluster Filtering
    #===================   
    candidates = phf.filter_clusters(clusters, centroids, grnd_pxl, MIN_POINT_COUNT)
    
    #%%
    # ========
    # Scoring
    # ========
    scoreWin = phf.score(candidates)
    voteWin = phf.vote(candidates)
    
    angle = phf.calc_pixel_angle(voteWin, grnd_pxl)
    print(f'\nAngle between Tennis Ball and GPXL {angle}')
    
    # ============
    # Update Plot
    # ============
    km._axes2D.plot((voteWin[1], grnd_pxl[1]), (voteWin[0], grnd_pxl[0]), linestyle='--', color='k')
    km._axes2D.plot((grnd_pxl[1], grnd_pxl[1]), (0, grnd_pxl[0]), linestyle='--', color='k')
    km._axes2D.grid(visible=True, axis='both')
    km._axes2D.invert_yaxis()
    km._axes2D.set_xlim(0, x_max)
    km._axes2D.set_ylim(y_max, 0) 
    km._axes2D.legend()
    
    # ==========================
    # Save frame and rotate img
    # ==========================
    os.chdir(r'C:\Users\Tj\Documents\Programming\Python_Scripts\tjCV\tjCVHelpers\output'.replace('\\', '/'))
    filename = './angle_test.png'
    cv.imwrite(filename, frame)
    img = Image(filename)
    Image.view_img(img.transform(angle=(-angle), get_data=True, rotation_center=grnd_pxl[::-1], display=False))
    
