# Pre-program stuff

import sys
def add_module(module):
    if module not in sys.path:
        sys.path.append(module)
tjhelpers = 'C:/Users/Tj/Documents/Programming/Python_Scripts/tjCV/tjCVHelpers/'       
add_module(tjhelpers)

import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from cv_helpers import KMeans, Image

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

def get_ground_pixel(image_shape:np.ndarray)->tuple:
    '''
    Calculate the bottom-center pixel of an image.
    '''
    rows, cols = image_shape[0:2]
    return (rows-1, int(cols/2 + (cols % 2)/2))


def calc_pixel_angle(circle_center:np.ndarray, 
                     ground_pixel:np.ndarray)->float:
    
    # Pass in the discovered centroid of chosen cluster of
    # detected centers from the Hough Circle detection. The point is in order
    # [y, x] (row, col) as desired. 
    # Angle positive is in CCW direction from vertical located along horizontal
    # center.
    a,b = ground_pixel # This is the bottom, center pixel of an image.
    i,j = circle_center
    
    # Calculate vector
    v = np.array([i-a, j-b])
    print(f'Vector v: {v}')
    # Angle Calculation: Due to unusual positioning of picture coordinates, 
    # theta = arctan(j-b/i-a) where 
    if v[0] == 0:
        if v[1] == 0:
            raise ValueError(f"The two points are identical\n"\
                             f'Vector v: {v}')
        elif v[1] > 0:
            theta = -90
        else:
            theta = 90    
        return theta
    else:
        theta = np.degrees(np.arctan(v[1]/v[0]))
        return theta
    
    
def test_calc_pixel_angle()->bool:
    ''' Test cases for the calc_pixel_angles function'''
    ground_pixel = (4,2)
    print("Beginning Pixel Angle Calculation Test.")
    assert calc_pixel_angle((4,4), ground_pixel) == -90
    assert calc_pixel_angle((4,0), ground_pixel) == 90
    assert calc_pixel_angle((3,2), ground_pixel) == 0
    assert calc_pixel_angle((0,2), ground_pixel) == 0
    assert calc_pixel_angle((2,0), ground_pixel) == 45
    assert calc_pixel_angle((2,4), ground_pixel) == -45
    try:
        assert calc_pixel_angle((4,2), ground_pixel) 
    except ValueError:
        print("Same point check passed.")
    print("All tests passed!")
    
    return True


def process_centers(center_container:list)->tuple:
    '''
    Gets the list of circle centers and reorganizes them to desired format.
    Also determines which subcontainer has the most entries. This is the 
    container that will be used as the initial means for k-Means segmentation.
    
    The "length" of this subcontainer is the number of segments, k.
    
    Parameters
    ----------
    center_container : list
        A list containing all appended circle_center arrays from OpenCV's
        HoughCircles function. It is in format [x, y, r], where:
            x = horizontal index (along columns)
            y = vertical index (along rows)
            r = estimated circle radius
    Returns
    -------
    indices:list
        The reformated circle centers container. Now ordered as 
        [y, x]. The radii are discarded as they are unneeded.
        
    means:np.ndarray
        The array with the most entries; chosen to help mitigate effects of 
        outliers in the clustering because the outlier(s) would (hopefully) 
        serve as cluster centroids; meaning they are less likely to be included
        in the *actual* desired cluster.
    
    segments:int
        The number of segments to use for k-Means clustering. 
    '''
    # Reformat circle centers indices container. 
    # Step 1:
    #   Reshape; collapses a given array of indices from (1, n, 3) to 
    #   (n, 3).
    # Step 2:
    #   Remove third column of each array (this is the est. radii column)

    indices = [arr.reshape(-1, 3)[:,0:2] for arr in center_container]
    
    # Determine means and segments
    means, index, max_length = [], None, 0
    for i in range(len(indices)):
        length = indices[i].shape[0]
        if length > max_length:
            max_length = length
            index = i
    else:
        means = indices[index]
        segments = max_length
    flattened_indices = [point for arr in indices for point in arr]
    return (flattened_indices, means, segments) 



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


def filter_clusters(clusters: dict,
                    centroids: np.ndarray, 
                    ground_pixel: tuple) -> dict:
    '''
    

    Parameters
    ----------
    clusters : dict
        Clusters from KMeans.cluster function.
        Comes in [x,y] order.
        
    centroids : np.ndarray
        Centroids from KMeans.cluster function. 
        Comes in [x,y] order.
        
    ground_pixel : tuple
        Pass in the output from the get_ground_pixel function.
        This is in [y, x] order.
    Returns
    -------
    comparison_dict
        A Dictionary with the following information:
            key: cluster number
            value: tuple
            Value Contents:
                cluster_centroid ([y, x] order): np.ndarray
                    Center of the cluster
                cluster_point_count: int
                    Number of points in the cluster
                cluster_point_density: float
                    Point concentration (pts/sqr. pixel)
                cluster_GPD: float
                    Distance from cluster centroid to ground pixel.
    '''
    # Data setup
    grnd_pxl = np.array(ground_pixel)
    comparison_data = {}

    
    #=================
    # Begin Filtering
    #=================
    for key in clusters:
        
        # Get given cluster and centroid
        cluster = clusters[key]
        centroid = centroids[key]
        
        # Stage 1: Minimum point count
        clust_point_count = len(cluster)
        if clust_point_count >= MIN_POINT_COUNT:
            ...
            # Stage 2: Density Calculation
            distances = np.array([np.linalg.norm(cluster[i] - centroid)
                         for i in range(clust_point_count)])
            distances = np.sort(distances)
            
            vel = distances[1:]-distances[0:-1]
            accel = vel[1:] - vel[:-1]
            if max(accel) > 5:
                index = np.nonzero(np.equal(accel, max(accel)))[0]
                index_num = index[0]
                r_max = distances[index_num]
            else:
                r_max = max(distances)
            area = np.pi*(r_max**2)
            clust_point_density = clust_point_count / area # this is an np.flat64
            # ADD CIRCLE TO PLOT
            km._axes2D.add_patch(plt.Circle(centroid,
                                            radius=r_max, 
                                            fill=False, 
                                            linestyle='--'))
            

            # Stage 3: Ground Distance
            clust_centroid = centroid[::-1] # [col, row] -> [row, col]
            clust_GPD = np.linalg.norm(clust_centroid[0] - grnd_pxl[0])
            
            # Gather data
            comparison_data.update( 
                {key:(clust_centroid, 
                 clust_point_count, clust_point_density, clust_GPD)}
                )
        else:
            continue
    else:
        # Wrap-up
        print("Data filtering complete.")
        return comparison_data
    
    
def vote(cluster_candidates: dict) -> np.ndarray:
    """
    Produces best guess on the true tennis ball cluster

    Parameters
    ----------
    cluster_candidates : dict
        Output from the filter_clusters function.

    Returns
    -------
    winning_centroid: np.ndarray
        The centroid belonging to the winning cluster (in [y, x] form).

    """
    
    # Recall the dict values have order 
    # (clust_centroid, clust_point_count, clust_point_density, clust_GPD)
    # Index: 0, 1, 2, 3
    
    DENSITY_THRESH = 1.75
    num_candidates = len(cluster_candidates)
    
    # Simple Case Checks
    if num_candidates == 0:
        print("Vote: No winner.")
        return np.array([])
    elif num_candidates == 1:
        key = [*cluster_candidates][0]
        winner = cluster_candidates[key]
        print(f"Vote: Winner is obvious; Cluster {key}")
        return winner[0]
    
    # Begin winner analysis
    # Two stages: Density then Ground Pixel Distance
    first_place_density, fpindex = 0, None
    second_place_density, spindex = 0, None
    count = 0
    
    for key in cluster_candidates:
        # Stage 1
        candidate = cluster_candidates[key]
        density = candidate[2]
        if count == 0:
            first_place_density, fpindex = density, key
            second_place_density, spindex = density, key
        else:
            if density > first_place_density:
                # update metric trackers
                second_place_density, spindex = first_place_density, fpindex
                first_place_density, fpindex = density, key
            elif count == 1:
                # Get the second place contender; Doing this will result in the 
                # Function voting properly in the case of only two candidates
                # with identical densities, but the first candidate considered
                # has a higher GPD. We want it to lose in that case. Otherwise,
                # it would both 1st and 2nd place would be the same candidate.
                second_place_density, spindex = density, key
        count += 1
    else:
        # Compare densities
        if first_place_density/second_place_density >= DENSITY_THRESH:
            # We have a winner
            winner =  cluster_candidates[fpindex][0]
            print(f"Vote: Winner Decided by Density! Cluster {fpindex}")
        else:
            # Stage 2: distance comp
            GPD1 = cluster_candidates[fpindex][3]
            GPD2 = cluster_candidates[spindex][3]
            
            if GPD1 <= GPD2:
                winner = cluster_candidates[fpindex][0]
                print(f"Vote: Winner Decided by GPD! Cluster {fpindex}")

            else:
                winner =  cluster_candidates[spindex][0]
                print(f"Vote: Winner Decided by GPD! Cluster {spindex}")

        return winner
        
def score(cluster_candidates: dict) -> np.ndarray:
    """
    Produces best guess on the true tennis ball cluster

    Parameters
    ----------
    cluster_candidates : dict
        Output from the filter_clusters function.

    Returns
    -------
    winning_centroid: np.ndarray
        The centroid belonging to the winning cluster (in [y, x] form).

    """
    
    # Calculate scores for each candidate cluster.
    # Recall the dict values have order 
    # (clust_centroid, clust_point_count, clust_point_density, clust_GPD)
    # Index: 0, 1, 2, 3
    num_candidates = len(cluster_candidates)
    if num_candidates == 0:
        print("Score: No winner.")
        return np.array([])
    elif num_candidates == 1:
        key = [*cluster_candidates][0]
        winner = cluster_candidates[key]
        print(f"Score: Winner is obvious; Cluster {key}")
        return winner[0]
    
    first_place = None 
    best_score = -1E1000

    for key in cluster_candidates:
        candidate = cluster_candidates[key]
        point_count = candidate[1]
        density = candidate[2]
        GPD = candidate[3]
        score = 0.80*(1000*density) + 0.07*(point_count) - 0.13*(GPD)
        print(f"Cluster {key}'s score: {score}")
        #print(f'Score greater than current best? {score > best_score}')
        if score > best_score:
            best_score = score
            first_place = key
    else:
        print(f"\nScore: Winner is Cluster {first_place} with score {best_score}")
        # return centroid
        winner = cluster_candidates[first_place][0]
        return winner

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
    flat, means, k = process_centers(centers)    

    # Run k-Means
    km = KMeans(flat, k, means)
    clusters, centroids, _ = km.cluster()
    centroids = np.int16(centroids)
    
    # Cluster Plot config
    y_max, x_max = img_dimensions[0] - 1, img_dimensions[1] - 1
    grnd_pxl = np.array(get_ground_pixel(img_dimensions))
        
    # Update Plot
    km._axes2D.scatter(grnd_pxl[1], grnd_pxl[0], color='k',
                       label='Ground Pixel', edgecolor='k', zorder=100)
    
    
    #===================
    # Cluster Filtering
    #===================   
    candidates = filter_clusters(clusters, centroids, grnd_pxl)
    
    #%%
    # ========
    # Scoring
    # ========
    scoreWin = score(candidates)
    voteWin = vote(candidates)
    
    angle = calc_pixel_angle(voteWin, grnd_pxl)
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
    
    
    #%% Debugging Portion: Plotting Distance Acceleration
    # How quickly does the change in distance change per point?
    # Can I detect outliers this way?
    # Honestly, this may be better done interactively via console.
    def inspect_acceleration(clusters, centroids, key):
        ...
        """I'm just putting this here as an example so I won't lose the code"""
        from numpy.linalg import norm
        # Calculate Differences
        clust0dist = [norm(clusters[key][i] - centroids[key])
                  for i in range(len(clusters[key]))]
        clust0dist = np.array(clust0dist)
        
        # Sort the distances 
        sorted_0dist = np.sort(clust0dist)
        
        # Get vel and accel
        sorted_0vel = sorted_0dist [1:] - sorted_0dist[:-1]
        sorted_0accel = sorted_0vel [1:] - sorted_0vel[:-1]
        
        # Find index of max accel
        index = np.nonzero(np.equal(sorted_0accel, max(sorted_0accel)))[0]
        index_num = index[0]
        
        try:
            print(f'Max Acceleration: {sorted_0accel[index_num -1 :index_num+2]}')
            print(f'Velocities around point w/ max accel:\n {sorted_0vel[index_num-2:index_num + 5]}\n')
            print(f'Distances around point w/ max accel:\n {sorted_0dist[index_num:index_num + 5]}\n')
        except IndexError:
            print(f'Print up to it instead: Dist: {sorted_0dist[index_num:index_num + 3]}')
            print(f'Velocities around point w/ max accel:\n {sorted_0vel[index_num-1:index_num + 1]}\n')
            print(f'Max Acceleration: {sorted_0accel[index_num]}')
            
        # Plot
        clust0x = np.arange(sorted_0dist.shape[0])
        fig, ax = plt.subplots()
        fig.suptitle(f"Cluster {key} Analysis")
        ax.clear()
        ax.grid()
        ax.plot(clust0x, sorted_0dist, label=f"Cluster {key} Distance", zorder=3)
        ax.plot(clust0x[1:], sorted_0vel, color='g', label=f'Cluster {key} Velocity', zorder=3)
        ax.plot(clust0x[2:], sorted_0accel, color='r', label=f'Cluster {key} Acceleration', zorder=3)
        ax.set(xlabel="point", ylabel='Value')
        ax.legend()
        plt.show()
        return sorted_0dist, sorted_0vel, sorted_0accel, index
    

    
    