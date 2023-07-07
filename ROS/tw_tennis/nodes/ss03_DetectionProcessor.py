#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# ROS-specific imports
import rospy
from numpy_msgs.rosnp_helpers import open_rosnumpy
from tw_tennis.kmeans import KMeans
import tw_tennis.processing_helpers as ph
from tw_tennis.msg import ProcessDetectionAction, ProcessDetectionResult
from tw_tennis.srv import MoveRobot
from actionlib import SimpleActionServer

# Change this to work via ROS Parameter Server later.
COUNT_THRESH = int(0.8 * 50)  # 80% of detection threshold 
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
    def __init__(self):
        # ROS Node
        rospy.init_node("ss03_DetectionProcessor", anonymous=False)
        self.name = rospy.get_name()
        rospy.loginfo('{}: Online.'.format(self.name))
        # Action Server things
        self._as = SimpleActionServer("ProcessDetection",
                ProcessDetectionAction,
                execute_cb=self.processing_callback,
                auto_start=False)
        self._as.start()

    def processing_callback(self, goal: ProcessDetectionAction):
        # Get relevant data from message
        success = False
        img_shape, detections = goal.img_shape, goal.detections.rosnp_list
        detections = [open_rosnumpy(arr) for arr in detections]
        try:
            assert type(detections) == list
            assert type(detections[0]) == np.ndarray
        except AssertionError:
            rospy.loginfo('{}: Received data is in wrong format.'.format(self.name))
            self._result.was_successful = success
            self._as.set_aborted()
        # Calculate ground_pixel [y, x]
        ground_pixel = ph.get_ground_pixel(img_shape)

        # Begin to process the data
        (centers, initial_means, k) = ph.process_centers(detections)

        km =  KMeans(data=centers, segments=k,
                initial_means=initial_means, threshold=0.1)
        display = True
        clusters, centroids, _ = km.cluster(display=display)
        # Filter clusters to get candidates
        # Output Centroids will be in [y, x] form
        candidates = ph.filter_clusters(clusters=clusters,
                centroids=centroids, ground_pixel=ground_pixel,
                COUNT_THRESH=COUNT_THRESH)
        # Choose the winner
        winning_coordinates = ph.vote(candidates)
        
        # Close plot
        if display:
            rospy.sleep(15)
        km._closeplot(close_all=True)

        # Calculate angle
        angle = ph.calc_pixel_angle(circle_center=winning_coordinates,
                ground_pixel=ground_pixel)

        # Report to SimpleActionClient
        success = True
        self._result.was_successful = success
        self._as.set_succeeded(self._result)
        
        print(f'{self.name}: Angle: {angle}')


if __name__ =='__main__':
    server = DataProcessor()
    rospy.spin()
