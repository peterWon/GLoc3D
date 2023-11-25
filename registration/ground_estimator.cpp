#include "ground_estimator.h"
#include "3d/transform.h"

namespace{
  // 实现argsort功能
template<typename T> std::vector<int> argsort(const std::vector<T>& array){
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] > array[pos2]);});

	return array_index;
}

}
Eigen::VectorXf GroundEstimator::EstimateGround(
    pcl::PointCloud<PointType>::Ptr ground_points){
  pcl::SampleConsensusModelPlane<PointType>::Ptr model_plane(
    new pcl::SampleConsensusModelPlane<PointType>(ground_points));
  pcl::RandomSampleConsensus<PointType> ransac(model_plane);
  
  // 设置距离阈值，与平面距离小于0.03的点作为内点
  ransac.setDistanceThreshold(0.1);	
  ransac.computeModel();

  pcl::PointCloud<PointType>::Ptr cloud_plane(
      new pcl::PointCloud<PointType>());	
  std::vector<int> inliers;
  ransac.getInliers(inliers);
  pcl::copyPointCloud<PointType>(*ground_points, inliers, *cloud_plane);
  
  /// 输出模型参数Ax+By+Cz+D=0
  Eigen::VectorXf coefficient;
  ransac.getModelCoefficients(coefficient);
  
  if(visualize_){
    pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("ground_fit"));

    viewer->addPointCloud<PointType>(ground_points, "cloud_ori");
    viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud_ori");
    viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_ori");

    viewer->addPointCloud<PointType>(cloud_plane, "ground");
    viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "ground");
    viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "ground");

    while (!viewer->wasStopped()){
      viewer->spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
  }
  return coefficient;
}

pcl::PointCloud<PointType>::Ptr GroundEstimator::FilterGroundByNormals(
      pcl::PointCloud<PointType>::Ptr cloud,
      pcl::search::KdTree<PointType>::Ptr tree,
      pcl::NormalEstimationOMP<PointType, pcl::Normal>& n){

  pcl::PointCloud<pcl::Normal> normals;
  
  const size_t pt_num = cloud->size();
  int bin_flags[pt_num];
  
  n.setInputCloud(cloud);
  n.setSearchMethod(tree);

  //点云法向计算时，需要所搜的近邻点大小
  n.setKSearch(10);
  //搜索半径
  //n.setRadiusSearch(0.03);
  n.compute(normals);

  //每10度为一个bin
  int degree_bins[18]={0};
  const float rad2deg = 180. / M_PI;
  
  CHECK(normals.size()==cloud->size());
  for(size_t i = 0; i<normals.size(); ++i){
    const pcl::Normal& np = normals.points[i];
    const PointType& gp = cloud->points[i];
    float nx = np.normal_x;
    float ny = np.normal_y;
    float nz = np.normal_z;
    float curvature = np.curvature;
    
    float xy = sqrt(nx*nx + ny*ny);
    float theta = (atan2(nz, xy) + M_PI_2) * rad2deg;
    
    int idx = int(floor(theta / 10));
    assert(idx>=0 && idx<=17);//why CHECK fails?
    bin_flags[i] = idx;
    degree_bins[idx] = degree_bins[idx] + 1;
  }

  std::vector<int> bins_vec(18);
  std::copy(degree_bins, degree_bins + 18, bins_vec.begin());

  std::vector<int> index = argsort(bins_vec);

  pcl::PointCloud<pcl::Normal>::Ptr ground_normals(
      new pcl::PointCloud<pcl::Normal>());
  pcl::PointCloud<PointType>::Ptr ground_points(
      new pcl::PointCloud<PointType>());
  
  int ground_bin = -1;
  for(int idx : index){
    if(idx > 4 && idx < 13) continue;//TODO(wz): hard parameters
    ground_bin = idx;
    break;
  }

  if(ground_bin == -1){
    LOG(WARNING)<<"No valid ground found!";
    return nullptr;
  }
  // LOG(INFO)<<"ground_bin: "<<ground_bin;

  for(int i = 0; i < pt_num; ++i){
    int idx = bin_flags[i];
    
    if(idx == ground_bin){
      ground_normals->push_back(normals.points[i]);
      ground_points->push_back(cloud->points[i]);
    }
  }
  
  
  if(visualize_){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("Normal viewer"));
  
    viewer->setBackgroundColor(0.3, 0.3, 0.3);
    viewer->addText("faxian", 10, 10, "text");
    pcl::visualization::PointCloudColorHandlerCustom<
        PointType> single_color(ground_points, 0, 225, 0);
    viewer->addCoordinateSystem(0.1);
    viewer->addPointCloud<PointType>(
        ground_points, single_color, "sample cloud");
  
    //添加需要显示的点云法向。cloud为原始点云模型，normal为法向信息，20表示需要显示法向的点云间隔，
    //即每20个点显示一次法向，0.02表示法向长度。
    viewer->addPointCloudNormals<PointType, pcl::Normal>(
        ground_points, ground_normals, 1, 0.5, "normals");
    //设置点云大小
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    while (!viewer->wasStopped()){
      viewer->spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    } 
  }

  return ground_points;
}

Eigen::Matrix4f GroundEstimator::TransformPointsToGround(const Eigen::VectorXf& coeff,
    pcl::PointCloud<PointType>::Ptr cloud_in,
    pcl::PointCloud<PointType>::Ptr cloud_out){
  CHECK(cloud_in);
  CHECK(cloud_out);
  Eigen::Vector3f z_l(0., 0., 1.);

  Eigen::Vector3f gn_l(coeff[0], coeff[1], coeff[2]);
  // The ground must be lower than the lidar (nearly upward installed)
  float d = abs(coeff[3]) / gn_l.norm();
  if(coeff[2] < 0){
    gn_l *= -1.; // make sure the normal vector is upward
  }
  
  // 从LiDAR系（z轴在LiDAR下的向量gn_l）到地面系(z_l) 
  gn_l.normalize();
  Eigen::Quaternionf q_l2g = Eigen::Quaternionf::FromTwoVectors(gn_l, z_l);
  q_l2g.normalize();
  Eigen::Vector3f ypr = q_l2g.toRotationMatrix().eulerAngles(2, 1, 0);
  Eigen::Quaternionf q_l2g_noyaw = 
    cartographer::transform::RollPitchYaw(ypr[2], ypr[1], 0).cast<float>();
  Eigen::Matrix4f T_l2g = Eigen::Matrix4f::Identity();
  T_l2g.topLeftCorner<3, 3>() = q_l2g_noyaw.toRotationMatrix();
  
  T_l2g.block<3, 1>(0, 3) << 0, 0, d;
  pcl::transformPointCloud(*cloud_in, *cloud_out, T_l2g);
  return T_l2g;
}

Eigen::Matrix4f GroundEstimator::EsitmateGroundAndTransform(
    pcl::PointCloud<PointType>::Ptr cloud_in,
    pcl::PointCloud<PointType>::Ptr cloud_out){
  
  // 保留15米范围内的点计算法向量
  pcl::PointCloud<PointType>::Ptr q_ground(
      new pcl::PointCloud<PointType>);
  for(const PointType& pt: cloud_in->points){
    if(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < 400.){//kGroundIntensity
      PointType pt_tmp;
      pt_tmp.x = pt.x;
      pt_tmp.y = pt.y;
      pt_tmp.z = pt.z;
      q_ground->points.push_back(pt_tmp);
    }
  }
  
  // 利用法向量筛选地面点
  pcl::search::KdTree<PointType>::Ptr tree(
    new pcl::search::KdTree<PointType>());
  pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_estimator;
  normal_estimator.setNumberOfThreads(10);//设置openMP的线程数
  pcl::PointCloud<PointType>::Ptr q_ground_pts = FilterGroundByNormals(
      q_ground, tree, normal_estimator);
  
  if(q_ground_pts == nullptr){
    return Eigen::Matrix4f::Identity();
  }

  // 地面拟合: ax + by + cz + d = 0, q_coeff=[a,b,c,d]
  Eigen::VectorXf q_coeff = EstimateGround(q_ground_pts);
  
  // 转换点云到地面坐标系
  return TransformPointsToGround(q_coeff, cloud_in, cloud_out);
}