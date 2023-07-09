#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# ROS-specific imports
import rospy
from numpy_msgs.rosnp_helpers import open_rosnumpy
from tw_tennis.kmeans import KMeans
import tw_tennis.processing_helpers as ph
from tw_tennis.msg import ProcessDetectionAction, ProcessDetectionGoal, ProcessDetectionResult
from actionlib import SimpleActionServer


"""
Program Flow:

- Initialize SimpleActionServer (SAS)
- Wait for Goal from client
- Get goal; unpack data
    - Calculate Ground Pixel
    - Process data (use process_centers function)
    - Perform KMeans clustering
    - filter_clusters
    - vote
    - Calculate Angle
"""

class DataProcessor:
    _result = ProcessDetectionResult()
    pd_topic = rospy.get_param("ss02_ss03_topic")

    def __init__(self):
        # ROS Node
        rospy.init_node("ss03_DetectionProcessor", anonymous=False)
        self.name = rospy.get_name()
        self.COUNT_THRESH = int(0.8 * rospy.get_param("DETECTION_THRESH"))

        # Action Server things
        rospy.loginfo('{}: Booting up...'.format(self.name))
        self._as = SimpleActionServer(DataProcessor.pd_topic,
                ProcessDetectionAction,
                execute_cb=self.processing_callback,
                auto_start=False)
        self._as.start()
        rospy.sleep(1)
        rospy.loginfo('{}: Online.'.format(self.name))
 
    def processing_callback(self, goal: ProcessDetectionGoal):
        """
        Callback for ss02 client's action goal.
        """
        # Get relevant data from message
        img_shape, detections = goal.img_shape, goal.detections.rosnp_list
        detections = [open_rosnumpy(arr) for arr in detections]

        # Check for correct structure
        try:
            assert type(detections) == list
            assert type(detections[0]) == np.ndarray
        except AssertionError:
            rospy.loginfo('{}: Received data is in wrong format.'.format(self.name))
            self._result.deg_angle = None
            self._as.set_aborted()

        # Calculate ground_pixel [y, x]
        ground_pixel = ph.get_ground_pixel(img_shape)

        # Pre-process the data; remove radii estimations and revise array shape.
        (centers, initial_means, k) = ph.process_centers(detections)

        # Perform k-means clustering; display plot if debugging.
        km =  KMeans(data=centers, segments=k,
                initial_means=initial_means, threshold=0.01)
        display = True
        clusters, centroids, _ = km.cluster(display=display)

        # Filter clusters to get candidates
        # Output Centroids will be in [y, x] form
        candidates = ph.filter_clusters(clusters=clusters,
                centroids=centroids, ground_pixel=ground_pixel,
                COUNT_THRESH=self.COUNT_THRESH)
        
        # Choose the winning cluster.
        winning_coordinates = ph.vote(candidates)
        
        # Close plot to preserve memory across detections
        if display:
            rospy.sleep(15)
        km._closeplot(close_all=True)

        # Calculate angle
        angle = ph.calc_pixel_angle(circle_center=winning_coordinates,
                ground_pixel=ground_pixel)
        
        print(f'{self.name}: Angle: {angle}')

        # Report to SimpleActionClient
        self._result.deg_angle = angle
        self._as.set_succeeded(self._result)
        

if __name__ =='__main__':
    server = DataProcessor()
    rospy.spin()

