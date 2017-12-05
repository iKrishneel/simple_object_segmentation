// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <simple_object_segmentation/supervoxel_segmentation.hpp>

SupervoxelSegmentation::SupervoxelSegmentation() {
    srv_ = boost::shared_ptr<dynamic_reconfigure::Server<Config> >(
       new dynamic_reconfigure::Server<Config>);
    dynamic_reconfigure::Server<Config>::CallbackType f =
       boost::bind(
          &SupervoxelSegmentation::configCallback, this, _1, _2);
    srv_->setCallback(f);

}

void SupervoxelSegmentation::supervoxelSegmentation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &supervoxel_clusters,
    std::multimap<uint32_t, uint32_t> &supervoxel_adjacency) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: Supervoxel input cloud empty...\n Incorrect Seed");
       return;
    }
    pcl::SupervoxelClustering<PointT> super(voxel_resolution_,
                                            seed_resolution_,
                                            use_transform_);
    super.setInputCloud(cloud);
    super.setColorImportance(color_importance_);
    super.setSpatialImportance(spatial_importance_);
    super.setNormalImportance(normal_importance_);
    supervoxel_clusters.clear();
    super.extract(supervoxel_clusters);
    super.getSupervoxelAdjacency(supervoxel_adjacency);
}

void SupervoxelSegmentation::supervoxelSegmentation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &supervoxel_clusters,
    pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList &adjacency_list) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: Supervoxel input cloud empty...");
       return;
    }
    pcl::SupervoxelClustering<PointT> super(voxel_resolution_,
                                            seed_resolution_,
                                            use_transform_);
    super.setInputCloud(cloud);
    super.setColorImportance(color_importance_);
    super.setSpatialImportance(spatial_importance_);
    super.setNormalImportance(normal_importance_);
    supervoxel_clusters.clear();
    super.extract(supervoxel_clusters);
    super.getSupervoxelAdjacencyList(adjacency_list);
}

void SupervoxelSegmentation::publishSupervoxel(
    const std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters,
    sensor_msgs::PointCloud2 &ros_cloud,
    jsk_recognition_msgs::ClusterPointIndices &ros_indices,
    const std_msgs::Header &header) {
    pcl::PointCloud<PointT>::Ptr output (new pcl::PointCloud<PointT>);
    std::vector<pcl::PointIndices> all_indices;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr >::const_iterator
           it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end();
         ++it) {
      pcl::Supervoxel<PointT>::Ptr super_voxel = it->second;
      pcl::PointCloud<PointT>::Ptr super_voxel_cloud = super_voxel->voxels_;
      pcl::PointIndices indices;
      for (size_t i = 0; i < super_voxel_cloud->size(); i++) {
        indices.indices.push_back(i + output->points.size());
      }
      all_indices.push_back(indices);
      *output = *output + *super_voxel_cloud;
    }
    ros_indices.cluster_indices.clear();
    ros_indices.cluster_indices = this->convertToROSPointIndices(
       all_indices, header);
    ros_cloud.data.clear();
    pcl::toROSMsg(*output, ros_cloud);
    ros_indices.header = header;
    ros_cloud.header = header;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

void SupervoxelSegmentation::sortSupervoxelsByCentroid(
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &supervoxel_clusters,
    std::multimap<uint32_t, uint32_t> & adjacency_list) {
    if (supervoxel_clusters.empty()) {
       ROS_ERROR("[::sortSupervoxelsByCentroid]: EMPTY INPUT");
       return;
    }
    std::vector<float> pt_height;
    std::vector<uint32_t> indices;
    int icount = 0;
    for (auto it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       pt_height.push_back(it->second->centroid_.y);
       indices.push_back(it->first);
    }

    SupervoxelMap sorted_sv_cluster;
    std::multimap<uint32_t, uint32_t>  sorted_al;
    for (auto i: sort_indexes(pt_height)) {
       sorted_sv_cluster[indices[i]] =
          supervoxel_clusters.at(indices[i]);
       
    }
    supervoxel_clusters.clear();
    supervoxel_clusters = sorted_sv_cluster;
}

std::vector<pcl_msgs::PointIndices>
SupervoxelSegmentation::convertToROSPointIndices(
    const std::vector<pcl::PointIndices> cluster_indices,
    const std_msgs::Header& header) {
    std::vector<pcl_msgs::PointIndices> ret;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg;
       ros_msg.header = header;
       ros_msg.indices = cluster_indices[i].indices;
       ret.push_back(ros_msg);
    }
    return ret;
}

void SupervoxelSegmentation::configCallback(
    Config &config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    this->color_importance_ = config.color_importance;
    this->spatial_importance_ = config.spatial_importance;
    this->normal_importance_ = config.normal_importance;
    this->voxel_resolution_ = config.voxel_resolution;
    this->seed_resolution_ = config.seed_resolution;
    this->use_transform_ = config.use_transform;
    this->coplanar_threshold_ = config.coplanar_threshold;
    this->distance_threshold_ = config.distance_threshold;
    this->angle_threshold_ = config.angle_threshold;
}
