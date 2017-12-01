// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#pragma once
#ifndef _SIMPLE_OBJECT_SEGMENTATION_HPP_
#define _SIMPLE_OBJECT_SEGMENTATION_HPP_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <simple_object_segmentation/supervoxel_segmentation.hpp>

class SimpleObjectSegmentation: public SupervoxelSegmentation {

 private:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;
    typedef pcl::PointCloud<PointNormalT> PointCloudNormal;
   
    typedef std::map<uint32_t, int> UInt32Map;

    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, geometry_msgs::PolygonStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> sub_rect_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, geometry_msgs::PointStamped> SyncPolicy2;
    // message_filters::Subscriber<sensor_msgs::PointCloud2> sub_points_;
    message_filters::Subscriber<geometry_msgs::PointStamped> sub_screen_pt_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy2> >sync2_;
   
    pcl::IntegralImageNormalEstimation<PointT, NormalT> ne_;
    int num_threads_;

    std_msgs::Header header_;
    std::string user_in_type_;
   
    cv::Size input_size_;
    cv::Rect_<int> rect_;

    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
    int neigbor_size_;
   
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
   
 public:
    SimpleObjectSegmentation();
    void callback(const sensor_msgs::PointCloud2::ConstPtr &);
    void callbackRect(const sensor_msgs::PointCloud2::ConstPtr &,
                         const geometry_msgs::PolygonStamped::ConstPtr &);
    void callbackPoint(const sensor_msgs::PointCloud2::ConstPtr &,
                       const geometry_msgs::PointStamped::ConstPtr &);
    void getNormals(PointNormal::Ptr, const PointCloud::Ptr);
    void segment(const PointCloud::Ptr);
    float convexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                            Eigen::Vector4f, Eigen::Vector4f,
                            const float = -0.01f, bool = false);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   const float = -0.01f);
    void segmentRecursiveCC(UInt32Map &, int &, const AdjacentList,
                            const SupervoxelMap, const uint32_t);
    void fastSeedRegionGrowing(PointCloudNormal::Ptr, const PointCloud::Ptr,
                               const PointNormal::Ptr);
    void seedCorrespondingRegion(std::vector<int> &, const PointCloud::Ptr,
                                 const PointNormal::Ptr, const int);
    template<class T>
    void getPointNeigbour(std::vector<int> &, const PointCloud::Ptr,
                          const PointT, const T = 16, bool = true);
};



#endif /* _SIMPLE_OBJECT_SEGMENTATION_HPP_ */
