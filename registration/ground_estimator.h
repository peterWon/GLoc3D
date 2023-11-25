
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <glog/logging.h>

using PointType = pcl::PointXYZI;
class GroundEstimator{
public:
  GroundEstimator(){}
  Eigen::Matrix4f EsitmateGroundAndTransform(
      pcl::PointCloud<PointType>::Ptr cloud_in,
      pcl::PointCloud<PointType>::Ptr cloud_out);
private:
  Eigen::VectorXf EstimateGround(
    pcl::PointCloud<PointType>::Ptr ground_points);

  pcl::PointCloud<PointType>::Ptr FilterGroundByNormals(
        pcl::PointCloud<PointType>::Ptr cloud,
        pcl::search::KdTree<PointType>::Ptr tree,
        pcl::NormalEstimationOMP<PointType, pcl::Normal>& n);

  Eigen::Matrix4f TransformPointsToGround(const Eigen::VectorXf& coeff,
      pcl::PointCloud<PointType>::Ptr cloud_in,
      pcl::PointCloud<PointType>::Ptr cloud_out);
  
  bool visualize_ = false;
};