# DJI-Tello-Face-Follower
A project made in python that enables a DJI Tello Drone to follow a face
It uses a simple resnet10 model to identify faces in the video stream from the drone and follow the face with the highest confidence
by checking the position of the bounding box. 

Run the ```telloai.py``` file (keep the .py file just outside the resnet10 folder) with following arguments : 

1. ```-s <flag, if set, will save video to output.avi with XVID encoding>```
2. ```-c <float confidence value, default is 0.5>```
3. ```-p <prototxt file, default is given>```
4. ```-m <model file, default is given>```

**Demo :**
<video src="demo.mp4" width="400" height="300" controls preload></video>
