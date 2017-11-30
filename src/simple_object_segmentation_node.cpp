
#include <simple_object_segmentation/simple_object_segmentation.hpp>

SimpleObjectSegmentation::SimpleObjectSegmentation() :
    num_threads_(2) {
    this->onInit();
}

void SimpleObjectSegmentation::onInit() {
    this->subscribe();

    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cloud", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>("/indices", 1);

}

void SimpleObjectSegmentation::subscribe() {
    this->sub_cloud_ = this->pnh_.subscribe(
       "points", 1, &SimpleObjectSegmentation::callback, this);
    
}

void SimpleObjectSegmentation::unsubscribe() {
    this->sub_cloud_.shutdown();
   
}

void SimpleObjectSegmentation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }
    this->header_ = cloud_msg->header;

    assert((cloud->width != 1 || cloud->height != 1)  &&
           "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    // PointNormal::Ptr normals(new PointNormal);
    // this->getNormals(normals, cloud);

    this->segment(cloud);
}

void SimpleObjectSegmentation::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    ne_.setNormalEstimationMethod(ne_.AVERAGE_3D_GRADIENT);
    ne_.setMaxDepthChangeFactor(0.02f);
    ne_.setNormalSmoothingSize(10.0f);
    ne_.setInputCloud(cloud);
    ne_.compute(*normals);
}


void SimpleObjectSegmentation::segment(
    const PointCloud::Ptr in_cloud) {
    SupervoxelMap supervoxel_clusters;
    AdjacentList adjacency_list;
    this->supervoxelSegmentation(in_cloud, supervoxel_clusters, adjacency_list);
    UInt32Map voxel_labels;
    for (auto it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       voxel_labels[it->first] = -1;
    }
    
    int label = 0;
    for (auto itr = adjacency_list.begin(); itr != adjacency_list.end();) {
       int32_t vindex = itr->first;
       if (voxel_labels[vindex] == -1) {
          if (!supervoxel_clusters.at(vindex)->voxels_->empty()) {
             voxel_labels[vindex] = label;
             this->segmentRecursiveCC(voxel_labels, label, adjacency_list,
                                      supervoxel_clusters, vindex);
             label++;
          }
       }
       itr = adjacency_list.upper_bound(vindex);
    }

    SupervoxelMap sv_clustered;
    AdjacentList update_adjlist;

    //! initalization
    for (int i = 0; i < label + 1; i++) {
       pcl::Supervoxel<PointT>::Ptr tmp_sv(new pcl::Supervoxel<PointT>);
       sv_clustered[i] = tmp_sv;
    }

    pcl::Supervoxel<PointT>::Ptr tmp_sv(new pcl::Supervoxel<PointT>);
    for (auto it = voxel_labels.begin(); it != voxel_labels.end(); it++) {
       if (it->second != -1) {
          *(sv_clustered[it->second]->voxels_) +=
             *supervoxel_clusters.at(it->first)->voxels_;
          *(sv_clustered[it->second]->normals_) +=
             *supervoxel_clusters.at(it->first)->normals_;

          auto v_label = it->second;
          for (auto it2 = adjacency_list.equal_range(it->first).first;
               it2 != adjacency_list.equal_range(it->first).second; ++it2) {
             auto n_label = voxel_labels[it2->second];
             if (n_label != v_label) {
                update_adjlist.insert(std::make_pair(it->second, n_label));
             }
          }
       }
    }

    ROS_INFO("DONE PROCESSING");

    bool is_visualize = true;
    if (is_visualize) {
       sensor_msgs::PointCloud2 ros_voxels;
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       this->publishSupervoxel(sv_clustered,
                               ros_voxels, ros_indices,
                               this->header_);
       this->pub_cloud_.publish(ros_voxels);
       this->pub_indices_.publish(ros_indices);
    }
}

void SimpleObjectSegmentation::segmentRecursiveCC(
    UInt32Map &voxel_labels, int &label, const AdjacentList adjacency_list,
    const SupervoxelMap supervoxel_clusters, const uint32_t vindex) {
    for (auto it = adjacency_list.equal_range(vindex).first;
         it != adjacency_list.equal_range(vindex).second; ++it) {
      uint32_t n_vindex = it->second;
      auto it2 = voxel_labels.find(n_vindex);
      if (vindex != n_vindex && it2->second == -1) {
         Eigen::Vector4f sp = supervoxel_clusters.at(
            vindex)->centroid_.getVector4fMap();
         Eigen::Vector4f sn = supervoxel_clusters.at(
            vindex)->normal_.getNormalVector4fMap();
         Eigen::Vector4f np = supervoxel_clusters.at(
            n_vindex)->centroid_.getVector4fMap();
         Eigen::Vector4f nn = supervoxel_clusters.at(
            n_vindex)->normal_.getNormalVector4fMap();
         auto cc = convexityCriteria(sp, sn, np, nn, -0.0025f, true);
         if (cc == 1.0f) {
            voxel_labels[n_vindex] = label;
            segmentRecursiveCC(voxel_labels, label, adjacency_list,
                               supervoxel_clusters, n_vindex);
         }
      }
    }
}

float SimpleObjectSegmentation::convexityCriteria(
    Eigen::Vector4f seed_point, Eigen::Vector4f seed_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float thresh, bool hard_label) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point).dot(n_normal);
    seed2pt_relation = (seed_point - n_centroid).dot(seed_normal);
    float angle = std::acos((seed_point.dot(n_centroid)) / (
                               seed_point.norm() * n_centroid.norm()));

    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       float w = std::exp(-angle / (2.0f * M_PI));
       if (hard_label) {
          return 1.0f;
       } else {
          return 0.75f;
       }
    } else {
       float w = std::exp(-angle / (M_PI/6.0f));
       if (hard_label) {
          return 0.0f;
       } else {
          return 1.50f;
       }
    }
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "simple_object_segmentation");
    SimpleObjectSegmentation sos;
    ros::spin();
    return 0;
}

