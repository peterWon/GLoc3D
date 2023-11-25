
#include <iostream>

#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "torch/script.h"
#include "torch/torch.h"


#include "3d/submap_3d.h"

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
using namespace cartographer::mapping;
using namespace nanoflann;
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;

/////////////////////////////codes for libtorch-based loop detection//////////////////////////////////
// Caution: this class assumes that the input scans are approximately gravity aligned 
struct OccupancyGrid{
  Eigen::Vector3f ox_oy_res;
  cv::Mat occupancy;
};

class RpyPCLoopDetector{
public:
  using PointType = pcl::PointXYZI;
  RpyPCLoopDetector();
  ~RpyPCLoopDetector(){}
  
  void load_model(const std::string& model_file){
    module_ = torch::jit::load(model_file);
    
    torch::DeviceType device_type_;
    device_type_ = torch::kCUDA;
    // device_type = torch::kCPU;
    torch::Device device_(device_type_);
    module_.to(device_);
    module_.eval();
    // torch::set_num_threads(12);
  }

  void add_keyframe(const pcl::PointCloud<PointType>::Ptr db_pc);
  
  // for slam, using the last frame as query
  bool detect(size_t& q_idx, size_t& loop_idx);

  // TODO(wz): remove from this class.
  bool match(const size_t q_idx,  const size_t db_idx, 
      Eigen::Vector3f& xy_yaw, double& estimated_scale);
  
  // for global localization
  void detect(const pcl::PointCloud<PointType>::Ptr q_pc,
              OccupancyGrid& q_grid, 
              std::vector<size_t>& loop_indices,
              std::vector<float>& out_dists_sqr);
  bool match(const OccupancyGrid& q_grid, 
             const size_t db_idx, 
             Eigen::Vector3f& xy_yaw, double& estimated_scale);

  const int NUM_EXCLUDE_RECENT = 30;
private:
  bool match(const cv::Mat& src1,  const cv::Mat& src2, 
      const Eigen::Vector3f& oxy_res_1, const Eigen::Vector3f& oxy_res_2,
      Eigen::Vector3f& xy_yaw, double& estimated_scale,
      bool visualize = false);
  void detect(const size_t q_idx, std::vector<size_t>& ret_indexes, std::vector<float>& out_dists_sqr);
  cv::Mat crop_pad_occupancy(const cv::Mat& src, size_t width, size_t height);

  cartographer::sensor::RangeData point_cloud_to_range_data(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud);

  cv::Mat get_projected_grid(pcl::PointCloud<PointType>::Ptr q_pc, Eigen::Vector3f& xy_res);

  std::vector<float> get_place_feature(
      torch::jit::script::Module& module, 
      pcl::PointCloud<PointType>::Ptr q_pc,
      cv::Mat& occupancy_grid,
      Eigen::Vector3f& xy_res);
private:
  const size_t k_dim_ = 512;
  const size_t top_k_ = 20;
  const size_t num_exclude_recent_ = 30;
  const size_t tree_making_period_ = 30;
  size_t tree_making_period_counter_ = 0;
  // currently only check the top-1, can relax to top-5 if the computing resource allows.
  const float loop_metric_dist_th_ = 0.8;
  
  torch::jit::script::Module module_; 

  KeyMat db_features_;
  KeyMat db_features_to_search_;
  std::unique_ptr<InvKeyTree> kdtree_;
  

  std::vector<OccupancyGrid> db_grids_;

  RangeDataInserter3D range_data_inserter_;
  const float high_resolution_max_range_ = 100.;
  const float high_resolution_ = 0.2;
  const float low_resolution_ = 0.5;
  cartographer::transform::Rigid3d identity_transform_;
};