# Simple Object Segmentation
ROS Node for segmenting unknown objects using simple geometrical heuristics. 
The node takes point cloud data from RGB--D sensor or stereo camera as input.

## Requirements
- [ROS](http://wiki.ros.org/indigo) >= indigo
- [PCL](https://github.com/PointCloudLibrary/pcl) >= 1.7
- [OpenCV](https://github.com/opencv/opencv)

## Compilation

```bash
git clone https://github.com/iKrishneel/simple_object_segmentation.git
catkin build simple_object_segmentation
source $HOME/.bashrc
```

## Running
to run fully automatic segmentation
```bash
roslaunch simple_object_segmentation simple_object_segmentation.launch
```
### options:
[**DEFAULT**] setting is none, whereby whole scene is segmented 
* [jsk_recognition](https://github.com/jsk-ros-pkg/jsk_recognition) *is required for visualization*
```bash
roslaunch simple_object_segmentation simple_object_segmentation.launch user_input:=none
```
user marks a point on the image_view2 and the region the point is segmented automatically 
* [image_view2](https://github.com/jsk-ros-pkg/jsk_common) *is required*
```bash
roslaunch simple_object_segmentation simple_object_segmentation.launch user_input:=point
```
user marks the object region using a 2D bounding box and the program segments the region of interest. 
* [image_view2](https://github.com/jsk-ros-pkg/jsk_common) *for marking the region on the image*
```bash
roslaunch simple_object_segmentation simple_object_segmentation.launch user_input:=rect
```

## Sample Results
<img src="data/sample.png" width="80%" height="80%"/>