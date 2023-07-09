#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
# ROS-specific imports
import rospy
from actionlib import SimpleActionServer
from tw_tennis.msg import  MoveRobotAction, MoveRobotResult, MoveRobotGoal
from jethexa_controller import client

"""
What subsystem04 needs to do:
- Receive service requests from ss03 and send positive response.
- Convert angle to integer
- Move the Robot
    - Instantiate the client
    - Send the move command
"""

class RobotController:
    # Action infromation as class variables.
    _result = MoveRobotResult()
    try:
        move_robot_topic = rospy.get_param("ss02_ss04_topic")
    except KeyError:
        print("Something went wrong. Check topic namespace.")
        raise

    def __init__(self):
        # Start ROS Node
        rospy.init_node("ss04_RobotController", anonymous=False)
        self.name = rospy.get_name()
        rospy.loginfo('{}: Initializing "MoveRobot" SimpleActionServer'.format(self.name))
        # Initialize robot client
        self.jethexa_client = client.Client(self)
        rospy.sleep(3)
        # Begin SimpleActionServer
        self.server = SimpleActionServer(RobotController.move_robot_topic,
                                         MoveRobotAction,
                                         execute_cb=self.move_robot_action, 
                                         auto_start=False)
        self.server.start()
        rospy.sleep(2)
        rospy.loginfo('{}: Online.'.format(self.name))

    def move_robot_action(self, goal: MoveRobotGoal):
        """
        Handles the MoveRobot request to move the JetHexa.

        Received angles are truncated because JetHexa has at best a 1 degree 
        resolution for this application.

        Rotation is at one degree per complete movement cycle (6 legs move). Each cycle
        takes one second.
        We must calculate the number of steps based on the angle.
        """
        angle = goal.deg_angle
        # move (+/-) 1 degree per mvmt cycle (expressed in rad)
        if angle < 0:
            rotation_rate = -(np.pi / 180) 
        else:
            rotation_rate = (np.pi / 180) 
        time_per_cycle = 1
        # Number of movement cycles [unit check]: 
        # [degrees] / [rad/cycle * degrees/rad] -> [degrees] * [cycle/degrees]
        # [cycles] as desired.
        cycles = int(angle / (rotation_rate * (180 / np.pi))
                     )
        try:
            self.jethexa_client.traveling(
                    gait=1, # ripple gait 
                    stride=0.0, # stride 0mm; move in-place
                    height=10.0, # gait 10mm
                    direction=0, # move in the direction of 180 that is move backward
                    rotation=rotation_rate, 
                    time=time_per_cycle, # the time taken for each mvmt cycle (sec)
                    steps=cycles, # 0 means infinite; Don't use 0 for this application.
                    interrupt=True,
                    relative_height=False)
            # Sleep until robot moves (3 seconds longer than time the complete motion takes.)
            # time_taken = [s/cycle]*[cycle] -> s
            time_taken = round(time_per_cycle * cycles)
            rospy.sleep(int(2*time_taken))
        except Exception as e:
            rospy.loginfo("{}: Could not move robot.\nError: {}", self.name, e)
            self._result.move_successful = False
            self.server.set_aborted()
            rospy.signal_shutdown("Robot Movement Error")
        
        # Return response
        rospy.loginfo('{}: Successful move.'.format(self.name))
        self._result.move_successful = True
        self.server.set_succeeded(self._result)
        
    def stop(self):
        self.jethexa_client.traveling(gait=0)

if __name__ == '__main__':
    controller = RobotController()
    rospy.on_shutdown(controller.stop)
    rospy.spin()
