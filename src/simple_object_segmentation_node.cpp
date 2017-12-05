// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <simple_object_segmentation/simple_object_segmentation.hpp>

SimpleObjectSegmentation::SimpleObjectSegmentation() :
    num_threads_(8), neigbor_size_(26), cc_thresh_(-0.01f) {
    std::string type;
    this->pnh_.param<std::string>("user_input", this->user_in_type_, "none");
    this->pnh_.param<float>("cc_thresh", this->cc_thresh_, -0.01);
    
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
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
    if (this->user_in_type_ == "rect") {
       ROS_INFO("\033[34mRUN TYPE RECT\033[0m");
       this->sub_point_.subscribe(this->pnh_, "points", 1);
       this->sub_rect_.subscribe(this->pnh_, "rect", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       this->sync_->connectInput(this->sub_point_, this->sub_rect_);
       this->sync_->registerCallback(
          boost::bind(&SimpleObjectSegmentation::callbackRect, this, _1, _2));
    } else if (this->user_in_type_ == "point") {
       ROS_INFO("\033[34mRUN TYPE POINT\033[0m");
       this->sub_point_.subscribe(this->pnh_, "points", 1);
       this->sub_screen_pt_.subscribe(this->pnh_, "screen_pt", 1);
       this->sync2_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy2> >(100);
       this->sync2_->connectInput(this->sub_point_, this->sub_screen_pt_);
       this->sync2_->registerCallback(
          boost::bind(&SimpleObjectSegmentation::callbackPoint, this, _1, _2));
    } else {
       ROS_INFO("\033[34mRUN TYPE AUTO\033[0m");
       this->sub_cloud_ = this->pnh_.subscribe(
          "points", 1, &SimpleObjectSegmentation::callback, this);
    }
}

void SimpleObjectSegmentation::unsubscribe() {
    if (this->user_in_type_ == "rect") {
       this->sub_point_.unsubscribe();
       this->sub_rect_.unsubscribe();
    } else {
       this->sub_cloud_.shutdown();
    }
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
    this->segment(cloud);
}

void SimpleObjectSegmentation::callbackRect(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PolygonStamped::ConstPtr &rect_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty() || rect_msg->polygon.points.size() == 0) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }
    assert((cloud->width != 1 || cloud->height != 1)  &&
          "\033[31m UNORGANIZED INPUT CLOUD \033[0m");
    
    this->input_size_ = cv::Size(cloud->width, cloud->height);

    int x = rect_msg->polygon.points[0].x;
    int y = rect_msg->polygon.points[0].y;
    int width = rect_msg->polygon.points[1].x - x;
    int height = rect_msg->polygon.points[1].y - y;

    x -= width/2;
    y -= height/2;
    width *= 2;
    height *= 2;

    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;
    width -= x + width > cloud->width ? (x + width) - cloud->width : 0;
    height -= y + height > cloud->height ? (y + height) - cloud->height : 0;
    this->rect_ = cv::Rect_<int>(x, y, width, height);

    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    PointCloudNormal::Ptr src_pts(new PointCloudNormal);
    this->fastSeedRegionGrowing(src_pts, cloud, normals);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*src_pts, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void SimpleObjectSegmentation::callbackPoint(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PointStamped::ConstPtr &screen_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }

    assert((cloud->width != 1 || cloud->height != 1)  &&
          "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    int seed_index = screen_msg->point.x + (cloud->width * screen_msg->point.y);
    this->seed_point_ = cloud->points[seed_index];
    if (isnan(seed_point_.x) || isnan(seed_point_.x) || isnan(seed_point_.x)) {
       ROS_ERROR("SELETED POINT IS NAN");
       return;
    }

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, nan_indices);

    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < cloud->size(); i++) {
       double d = pcl::distances::l2(cloud->points[i].getVector4fMap(),
                                     seed_point_.getVector4fMap());
       if (d < dist) {
          dist = d;
          idx = i;
       }
    }
    
    PointNormal::Ptr normals(new PointNormal);
    this->getNormals<float>(cloud, normals, 0.03f, false);
    
    seed_index = idx;
    this->seed_point_ = cloud->points[seed_index];
    this->seed_normal_ = normals->points[seed_index];

    // TODO(^_^): change to pointnormal type
    PointCloud::Ptr in_cloud(new PointCloud);

    /*
      PointNormal::Ptr in_normal(new PointNormal);
    // double dist = DBL_MAX;
    int index = -1;
    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       NormalT nt = normals->points[i];
       if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z) &&
           !std::isnan(nt.normal_x) && !std::isnan(nt.normal_y) &&
           !std::isnan(nt.normal_z)) {
          in_cloud->push_back(pt);
          in_normal->push_back(nt);
          
          double d = pcl::distances::l2(pt.getVector4fMap(),
                                       seed_point_.getVector4fMap());
          if (d < dist) {
             dist = d;
             index = static_cast<int>(in_cloud->size()) - 1;
          }
       }
    }
    seed_index = index;
    */
    
    if (cloud->size() != normals->size()) {
       ROS_ERROR("CLOUD AND NORMALS SIZES ARE DIFF");
       return;
    }
    
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);
    
    if (labels.size() < seed_index) {
       ROS_WARN("INCORRECT SIZES");
       return;
    }
    labels[seed_index] = 1;
    
    this->kdtree_->setInputCloud(cloud);
    this->seedCorrespondingRegion(labels, cloud, normals, seed_index);

    cloud->clear();
    for (int i = 0; i < labels.size(); i++) {
       if (labels[i] != -1) {
          in_cloud->push_back(cloud->points[i]);
       }
    }
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*in_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void SimpleObjectSegmentation::seedCorrespondingRegion(
    std::vector<int> &labels, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals, const int parent_index) {
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, cloud,
                                cloud->points[parent_index],
                                this->neigbor_size_);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());

    
    std::vector<int> merge_list(neigb_lenght);
    merge_list[0] = -1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)        \
    shared(merge_list, labels)
#endif
    for (int i = 1; i < neigbor_indices.size(); i++) {
       int index = neigbor_indices[i];
       if (index != parent_index && labels[index] == -1) {
          Eigen::Vector4f parent_pt = cloud->points[
             parent_index].getVector4fMap();
          Eigen::Vector4f parent_norm = normals->points[
             parent_index].getNormalVector4fMap();
          Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
          Eigen::Vector4f child_norm = normals->points[
             index].getNormalVector4fMap();
          if (this->seedVoxelConvexityCriteria(
                 parent_norm, child_pt, child_norm,
                 this->cc_thresh_) == 1) {
             merge_list[i] = index;
             labels[index] = 1;
          } else {
             merge_list[i] = -1;
          }
       } else {
          merge_list[i] = -1;
       }
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) schedule(guided, 1)
#endif
    for (int i = 0; i < merge_list.size(); i++) {
       int index = merge_list[i];
       if (index != -1) {
          seedCorrespondingRegion(labels, cloud, normals, index);
       }
    }
}

template<class T>
void SimpleObjectSegmentation::getPointNeigbour(
    std::vector<int> &neigbor_indices, const PointCloud::Ptr cloud,
    const PointT seed_point_, const T K, bool is_knn) {
    if (cloud->empty() || isnan(seed_point_.x) ||
        isnan(seed_point_.y) || isnan(seed_point_.z)) {
       ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN GET NEIGBOUR");
       return;
    }
    neigbor_indices.clear();
    std::vector<float> point_squared_distance;
    if (is_knn) {
       int search_out = kdtree_->nearestKSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    } else {
       int search_out = kdtree_->radiusSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    }
}

void SimpleObjectSegmentation::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    this->ne_.setNormalEstimationMethod(this->ne_.AVERAGE_3D_GRADIENT);
    this->ne_.setMaxDepthChangeFactor(0.02f);
    this->ne_.setNormalSmoothingSize(10.0f);
    this->ne_.setInputCloud(cloud);
    this->ne_.compute(*normals);
}

template<class T>
void SimpleObjectSegmentation::getNormals(
    const PointCloud::Ptr cloud, PointNormal::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(this->num_threads_);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
       ne.setKSearch(k);
    } else {
       ne.setRadiusSearch(k);
    }
    ne.compute(*normals);
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
         auto cc = convexityCriteria(sp, sn, np, nn, this->cc_thresh_, true);
         if (cc == 1.0f) {
            voxel_labels[n_vindex] = label;
            segmentRecursiveCC(voxel_labels, label, adjacency_list,
                               supervoxel_clusters, n_vindex);
         }
      }
    }
}

float SimpleObjectSegmentation::convexityCriteria(
    Eigen::Vector4f seed_point_, Eigen::Vector4f seed_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float thresh, bool hard_label) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point_).dot(n_normal);
    seed2pt_relation = (seed_point_ - n_centroid).dot(seed_normal);
    float angle = std::acos((seed_point_.dot(n_centroid)) / (
                               seed_point_.norm() * n_centroid.norm()));

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

void SimpleObjectSegmentation::fastSeedRegionGrowing(
    PointCloudNormal::Ptr src_points, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       return;
    }
    int seed_index = (rect_.x + rect_.width/2)  +
       (rect_.y + rect_.height/2) * input_size_.width;
    Eigen::Vector4f seed_point_ = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);
    const int window_size = 3;
    const int wsize = window_size * window_size;
    const int lenght = std::floor(window_size/2);
    std::vector<int> processing_list;
    for (int j = -lenght; j <= lenght; j++) {
       for (int i = -lenght; i <= lenght; i++) {
          int index = (seed_index + (j * input_size_.width)) + i;
          if (index >= 0 && index < cloud->size()) {
             processing_list.push_back(index);
          }
       }
    }

    std::vector<int> temp_list;
    while (true) {
       if (processing_list.empty()) {
          break;
       }
       temp_list.clear();
       for (int i = 0; i < processing_list.size(); i++) {
          int idx = processing_list[i];
          if (labels[idx] == -1) {
             Eigen::Vector4f c = cloud->points[idx].getVector4fMap();
             Eigen::Vector4f n = normals->points[idx].getNormalVector4fMap();
             if (this->seedVoxelConvexityCriteria(
                    seed_point_, seed_normal, c, n, this->cc_thresh_) == 1) {
                labels[idx] = 1;
                for (int j = -lenght; j <= lenght; j++) {
                   for (int k = -lenght; k <= lenght; k++) {
                      int index = (idx + (j * input_size_.width)) + k;
                      if (index >= 0 && index < cloud->size()) {
                         temp_list.push_back(index);
                      }
                   }
                }
             }
          }
       }
       processing_list.clear();
       processing_list.insert(processing_list.end(), temp_list.begin(),
                              temp_list.end());
    }
    src_points->clear();
    for (int i = 0; i < labels.size(); i+=5) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
}

int SimpleObjectSegmentation::seedVoxelConvexityCriteria(
    Eigen::Vector4f seed_point_, Eigen::Vector4f seed_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal, const float thresh) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point_).dot(n_normal);
    seed2pt_relation = (seed_point_ - n_centroid).dot(seed_normal);
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       return 1;
    } else {
       return -1;
    }
}

int SimpleObjectSegmentation::seedVoxelConvexityCriteria(
    Eigen::Vector4f c_normal, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float thresh) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid -
                        this->seed_point_.getVector4fMap()).dot(n_normal);
    seed2pt_relation = (this->seed_point_.getVector4fMap() - n_centroid).dot(
       this->seed_normal_.getNormalVector4fMap());
    float norm_similarity = (M_PI - std::acos(
                                c_normal.dot(n_normal) /
                                (c_normal.norm() * n_normal.norm()))) / M_PI;

    if (seed2pt_relation > thresh &&
        pt2seed_relation > thresh && norm_similarity > 0.50f) {
       return 1;
    } else {
       return -1;

    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "simple_object_segmentation");
    SimpleObjectSegmentation sos;
    ros::spin();
    return 0;
}
