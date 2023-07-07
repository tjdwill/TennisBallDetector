#!/usr/bin/env python3

import argparse
import sys
from typing import Tuple
import numpy as np
import rospy
from jethexa_controller import jethexa

class RotationNode:

    def __init__(self, node) -> None:
        rospy.init_node('jethexa_rotator', anonymous=True)
        self.jethexa = jethexa.JetHexa(self)
        self.jethexa.set_build_in_pose('DEFAULT_POSE_M', 1)
        self.rotation_speed = 15  # degrees per second
        rospy.sleep(2)

    def transform(self, rotation_angle: Tuple[float]):
        angle = float(rotation_angle)
        duration = round(abs(angle))/self.rotation_speed
        rospy.loginfo('Beginning Rotation.')

        self.jethexa.transform_pose_2((0, 0, 0),
                                      'xyz',
                                      (0, 0, angle),
                                      duration=duration,
                                      degrees=True)
        
        rospy.sleep(duration + 1)
    
    def reset(self):
        self.jethexa.set_build_in_pose('DEFAULT_POSE_M', 1)

if __name__ == "__main__":
    # ros will pass in some of its own parameters when starting the py file,
    # myargv will remove these and return the original parameters

    args = rospy.myargv(argv=sys.argv) 
    parser = argparse.ArgumentParser(
        description='Program to practice controlling JetHexa.')
    parser.add_argument('-a','--angle', help='Pass in a degreed angle.',
                        dest='angle', required=True)
    cmls = parser.parse_args(args[1:])
    angle = cmls.angle

    node = RotationNode() # create related files
    rospy.loginfo("start")
    rospy.on_shutdown(node.reset)
    rospy.sleep(3)
    node.transform(angle)
    