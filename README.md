# Simple Object Segmentation
ROS Node for segmenting unknown objects using simple geometrical heuristics. 

## Requirements
- [ROS](http://wiki.ros.org/indigo) >= indigo
- [PCL](https://github.com/PointCloudLibrary/pcl) >= 1.7
- [OpenCV](https://github.com/opencv/opencv)
- [jsk_recognition](https://github.com/jsk-ros-pkg/jsk_recognition)
- [image_view2](https://github.com/jsk-ros-pkg/jsk_common)

## Compilation

```bash
git clone https://github.com/iKrishneel/simple_object_segmentation.git
cd simple_object_segmentation
catkin bt
```

## Running
```bash
roslaunch simple_object_segmentation simple_object_segmentation.launch
```

## Network
<img src="data/network.png" height="50%" />