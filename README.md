# TennisBallDetector
A project for CAP6665 Computer Vision course, the idea is to use OpenCV and ROS to implement a tennis ball detection algorithm to the JetHexa robot platform.


Videos regarding this project are found at the following links:

- [Tennis Ball Detection](https://www.youtube.com/watch?v=LeLQ7bqcsco)
- [Facing Behavior Implementation](https://www.youtube.com/watch?v=au4tLTOkUE0)

## Retrospective (26 January 2024)

Looking back at this project, I consider it a success, especially in terms of adapting on-the-fly in order to solve the problem of angular convergence.
My solution to scale the angle down by a scaling factor in order to have to robot converge the centering position was effective in achieving the desired behavior.
A similar but perhaps more "sure-fire" approach would have been to segment the image and assign a concrete number of rotation steps at a pre-defined angle to each one. This is the method used in 
Ahmad, Moazami, and Zargarzadeh, 2019.
```
S. Ahmad, S. Moazami and H. Zargarzadeh, "Autonomous Color Based Object Tracking of a Hexapod with Efficient Intuitive Characteristics," 2019 IEEE International Symposium on Measurement and Control in Robotics (ISMCR), Houston, TX, USA, 2019, pp. D3-4-1-D3-4-7, doi: 10.1109/ISMCR47492.2019.8955728. keywords: {Legged locomotion;Cameras;Neck;Image color analysis;Target tracking;hexapod;object detection;vision tracking;efficient neck feature},
```

However, given what I know today, perhaps the most direct solution would have been to use the JetHexa's depth camera to calculate transforms between coordinate frames and calculate the rotation angle directly in terms of the body frame.
This is the method I am currently using in the spiritual successor to this project, the [UAV Detector](https://github.com/tjdwill/UAVDetector) (which also happens to be my Master's Thesis).
For platforms that are solely equipped with RGB cameras, however, the incremental rotation method is sufficient.

I am grateful to have worked on the Tennis Ball Detector Project. I learned a lot about project management, program design and architecture, and gained experience diving into a codebase to understand a defined system.
