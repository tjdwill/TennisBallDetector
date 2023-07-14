#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:42:47 2023

@author: Terrance Williams
"""
import numpy as np
import matplotlib.pyplot as plt


def get_ground_pixel(image_shape: tuple) -> tuple:
    '''
    Calculate the bottom-center pixel of an image.
    '''
    rows, cols = image_shape[0:2]
    return (rows - 1, int(cols/2 + (cols % 2)/2))


def calc_pixel_angle(circle_center: np.ndarray, ground_pixel: np.ndarray):
    
    # Check for an empty list or Numpy array.
    # This means the voting mechanism could not decide on a winner. 
    # Return 360 as an angle because that is not an expected angle value. 
    if len(circle_center) == 0:
        return 360

    # Pass in the discovered centroid of chosen cluster of
    # detected centers from the Hough Circle detection. The point is in order
    # [y, x] (row, col) as desired.
    # Angle positive is in CCW direction from vertical located along horizontal
    # center.
    a, b = ground_pixel  # This is the bottom, center pixel of an image.
    i, j = circle_center

    # Calculate vector
    v = np.array([i-a, j-b])
    v_norm = np.linalg.norm(v)
    print(f'Vector v: {v}\n')
    print(f'Vector length: {v_norm}')
    # Angle Calculation: Due to unusual positioning of picture coordinates,
    # theta = arctan(j-b/i-a)
    if v[0] == 0:
        if v[1] == 0:
            raise ValueError(f"The two points are identical\n"
                             f'Vector v: {v}')
        elif v[1] > 0:
            theta = -90
        else:
            theta = 90
        return theta
    else:
        theta = np.degrees(np.arctan(v[1]/v[0]))
        theta_norm = theta / (v_norm/10)
        print(f'Angles: {theta}, {theta_norm}')
        return theta


def process_centers(center_container: list):
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
        [x, y]. The radii are discarded as they are unneeded.

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
    # Step 3:
    #   Cast each array as a signed integer to prevent wonky distance
    #   calculation results.
    indices = [np.int16(arr.reshape(-1, 3)[:, 0:2])
               for arr in center_container]

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


def filter_clusters(clusters: dict,
                    centroids: np.ndarray,
                    ground_pixel: tuple,
                    COUNT_THRESH: int) -> dict:
    '''
    Remove clusters that are likely not valid detections.

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

    COUNT_THRESH: int
        Minimum points in a cluster to be a qualifying candidate.

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
    ACCEL_MAX = 5
    MIN_POINT_COUNT = COUNT_THRESH

    # ================
    # Begin Filtering
    # ================
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
            if max(accel) > ACCEL_MAX:
                index = np.nonzero(np.equal(accel, max(accel)))[0]
                index_num = index[0]
                r_max = distances[index_num]
            else:
                r_max = max(distances)
            area = np.pi * (r_max**2)
            # Data is now np.float64
            clust_point_density = clust_point_count / area
            '''
            # ADD CIRCLE TO PLOT
            km._axes2D.add_patch(plt.Circle(centroid,
                                            radius=r_max,
                                            fill=False,
                                            linestyle='--'))
            '''

            # Stage 3: Ground Distance
            clust_centroid = centroid[::-1]  # [col, row] -> [row, col]
            clust_GPD = np.linalg.norm(clust_centroid[0] - grnd_pxl[0])

            # Gather data
            comparison_data.update(
                {key: (clust_centroid,
                 clust_point_count, clust_point_density, clust_GPD)}
                )
        else:
            continue
    else:
        # Wrap-up
        print("FILTER_CLUSTERS: Data filtering complete.\n")
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
    # NOTE: DEBUG
    print('Vote DEBUGGING')
    for key in cluster_candidates:
        print(cluster_candidates[key])

    # Recall the dict values have order
    # (clust_centroid, clust_point_count, clust_point_density, clust_GPD)
    # Matching Index: 0, 1, 2, 3

    DENSITY_THRESH = 1.5
    num_candidates = len(cluster_candidates)

    # Simple Case Checks
    if num_candidates == 0:
        print("VOTE: No winner.")
        return np.array([])
    elif num_candidates == 1:
        key = [*cluster_candidates][0]
        winner = cluster_candidates[key]
        print(f"VOTE: Winner is obvious; Cluster {key}")
        return winner[0]

    # Begin winner analysis
    # Two stages: Density then Ground Pixel Distance
    first_place_density, fpindex = 0, None
    second_place_density, spindex = 0, None
    count = 0

    for key in cluster_candidates:
        # Stage 1
        print(f'VOTE: Cluster {key}')
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
            elif density > second_place_density:
                second_place_density, spindex = density, key
            elif count == 1 and (fpindex == spindex):
                # Get the second place contender; Doing this will result in the
                # Function voting properly in the case of only two candidates
                # with identical densities, but the first candidate considered
                # has a higher GPD. We want it to lose in that case. Otherwise,
                # both 1st and 2nd place would be the same candidate.
                second_place_density, spindex = density, key
        count += 1
    else:
        # Compare densities
        if first_place_density/second_place_density >= DENSITY_THRESH:
            # We have a winner
            winner = cluster_candidates[fpindex][0]
            print(f"VOTE: Winner Decided by Density! Cluster {fpindex}")
        else:
            # Stage 2: distance comp
            GPD1 = cluster_candidates[fpindex][3]
            GPD2 = cluster_candidates[spindex][3]

            if GPD1 <= GPD2:
                winner = cluster_candidates[fpindex][0]
                print(f"VOTE: Winner Decided by GPD! Cluster {fpindex}")

            else:
                winner = cluster_candidates[spindex][0]
                print(f"VOTE: Upset! Winner Decided by GPD! Cluster {spindex}")

        return winner

'''
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
    # Matching Index: 0, 1, 2, 3
    num_candidates = len(cluster_candidates)
    if num_candidates == 0:
        print("SCORE: No winner.")
        return np.array([])
    elif num_candidates == 1:
        key = [*cluster_candidates][0]
        winner = cluster_candidates[key]
        print(f"SCORE: Winner is obvious; Cluster {key}")
        return winner[0]

    first_place = None
    best_score = -1E1000

    for key in cluster_candidates:
        candidate = cluster_candidates[key]
        point_count = candidate[1]
        density = candidate[2]
        GPD = candidate[3]
        score = 0.80*(1000*density) + 0.07*(point_count) - 0.13*(GPD)
        print(f"SCORE: Cluster {key}'s score: {score}")
        # print(f'SCORE greater than current best? {score > best_score}')
        if score > best_score:
            best_score = score
            first_place = key
    else:
        print(f"SCORE: Winner is Cluster {first_place} with {best_score}")
        # return centroid
        winner = cluster_candidates[first_place][0]
        return winner
'''
# %% Debugging Portion: Plotting Distance Acceleration
# How quickly does the change in distance change per point?
# Can I detect outliers this way?
# Honestly, this may be better done interactively via console.


def inspect_acceleration(clusters, centroids, key):
    """
    Plots the intra-cluster centroid distances, how those distances
    change from point to point (intra-cluster velocty), and
    how those changes change (intra-cluster acceleration)
    """
    # Calculate Differences
    clustdist = [np.linalg.norm(clusters[key][i] - centroids[key])
                 for i in range(len(clusters[key]))]
    clustdist = np.array(clustdist)

    # Sort the distances
    sorted_dist = np.sort(clustdist)

    # Get vel and accel
    sorted_vel = sorted_dist[1:] - sorted_dist[:-1]
    sorted_accel = sorted_vel[1:] - sorted_vel[:-1]

    # Find index of max accel
    index = np.nonzero(np.equal(sorted_accel, max(sorted_accel)))[0]
    index_num = index[0]

    try:
        print(f'Max Acceleration: {sorted_accel[index_num -1:index_num+2]}')
        print('Velocities around point w/ max accel:\n'
              f'{sorted_vel[index_num-2:index_num + 5]}\n')
        print('Distances around point w/ max accel:\n'
              f'{sorted_dist[index_num:index_num + 5]}\n')
    except IndexError:
        print(f'Dist: {sorted_dist[index_num:index_num + 3]}')
        print(f'Velocities around point w/ max accel:\n'
              f'{sorted_vel[index_num-1:index_num + 1]}\n')
        print(f'Max Acceleration: {sorted_accel[index_num]}')

    # Plot
    clust_x = np.arange(sorted_dist.shape[0])
    fig, ax = plt.subplots()
    fig.suptitle(f"Cluster {key} Analysis")
    ax.clear()
    ax.grid()
    ax.plot(clust_x, sorted_dist, label=f"Cluster {key} Distance", zorder=3)
    ax.plot(clust_x[1:], sorted_vel, color='g',
            label=f'Cluster {key} Velocity', zorder=3)
    ax.plot(clust_x[2:], sorted_accel, color='r',
            label=f'Cluster {key} Acceleration', zorder=3)
    ax.set(xlabel="point", ylabel='Value')
    ax.legend()
    plt.show()
    return sorted_dist, sorted_vel, sorted_accel, index
