# ROS Scripts

In this subdirectory, I upload the scripts for the ROS portion of the project. 
The project is "ROS-ified" by segmenting the program into nodes. 
For example, one node may capture images via the robot camera, and publish those photos to a topic. 
Another node would then subscribe to this topic, receive the images, and run the relevant processing to detect tennis balls.

<p align="center">
  <img src="https://github.com/tjdwill/TennisBallDetector/assets/118497355/633b7d1a-528f-466d-83f0-fdb9fde4fef5" />
</p>
