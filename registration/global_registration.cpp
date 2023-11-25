#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <glog/logging.h>
#include <iostream>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "3d/submap_3d.h"
#include "3d/transform.h"
#include "2d/probability_grid.h"
#include "2d/fast_correlative_scan_matcher_2d.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

// #include "Scancontext/Scancontext.h"
// #include "ground_segment/efficient_online_segmentation.h"

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cartographer::mapping;

typedef pcl::PointXYZ PointType;

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(
    std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

template <typename T, typename A>
int arg_min(std::vector<T, A> const& vec) {
  return static_cast<int>(
    std::distance(vec.begin(), std::min_element(vec.begin(), vec.end())));
}

vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if("" == str) return res;

	char * strs = new char[str.length() + 1];
	strcpy(strs, str.c_str()); 
 
	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());
 
	char *p = strtok(strs, d);
	while(p) {
		string s = p;
		res.push_back(s);
		p = strtok(NULL, d);
	}

	return res;
}

void ReadValset(const string& filename, vector<string>& db_files, 
                vector<string>& q_files, vector<vector<int>>& pos_idx){
  ifstream ifs(filename);
  
  db_files = {};
  q_files={};
  pos_idx={};
  string line = "";
  vector<string> substrs = {};
  int db_num = 0;
  int q_num = 0;
  if(!ifs.is_open()){
    std::cout<<"failed to open file "<<filename<<"\n";
    return; 
  }
  getline(ifs, line);
  substrs = split(line, " ");
  db_num = atoi(substrs[0].c_str());
  q_num = atoi(substrs[1].c_str());
  // the scan path of db
  for(int i = 0; i < db_num; ++i){
    getline(ifs, line);
    db_files.push_back(line);
  }
  // the scan path of query
  for(int i = 0; i < q_num; ++i){
    getline(ifs, line);
    q_files.push_back(line);
  }
  // the positive samples of each query
  for(int i = 0; i < q_num; ++i){
    if(!getline(ifs, line)){
      break;
    }
    if(line.empty()) break;
    substrs = split(line, ":");
    if(substrs.size()==1){
      pos_idx.push_back({});
      continue;
    }
    int q_idx = atoi(substrs[0].c_str());
    
    string pos_idx_str = substrs[1];
    if(pos_idx_str.empty()){
      pos_idx.push_back({});
      continue;
    }
    substrs = split(pos_idx_str, " ");
    vector<int> pos_idx_tmp = {};
    for(int k = 0; k < substrs.size(); ++k){
      pos_idx_tmp.push_back(atoi(substrs[k].c_str()));
    }
    pos_idx.push_back(pos_idx_tmp);
  }
  ifs.close();  
  LOG(INFO)<<"db_num and db_files: "<<db_num<<", "<<db_files.size();
  LOG(INFO)<<"q_num and q_files: "<<q_num<<", "<<q_files.size();
  LOG(INFO)<<"q_num and q_pos_index: "<<q_num<<", "<<pos_idx.size();
}

void ReadValsetPose(const string& filename, std::vector<Eigen::Matrix4f>& poses){
  ifstream ifs(filename);
  
  poses = {};
  
  string line = "";
  vector<string> substrs = {};
  if(!ifs.is_open()){
    LOG(ERROR)<<"failed to open file "<<filename;
    return; 
  }

  while(getline(ifs, line)){
    substrs = split(line, " ");
    CHECK(substrs.size()==7);
    
    // w,x,y,z
    float qw = atof(substrs[3].c_str());
    float qx = atof(substrs[0].c_str());
    float qy = atof(substrs[1].c_str());
    float qz = atof(substrs[2].c_str());
    float x = atof(substrs[4].c_str());
    float y = atof(substrs[5].c_str());
    float z = atof(substrs[6].c_str());
    Eigen::Quaternionf q(qw, qx, qy, qz);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.topLeftCorner(3, 3) = q.toRotationMatrix();
    pose.topRightCorner(3, 1) << x,y,z;
    poses.emplace_back(pose);
  }
  ifs.close();  
  LOG(INFO)<<"Read poses with size: "<<poses.size();
}

// The raw data format of NCLT dataset.
pcl::PointCloud<PointType> read_lidar_data_nclt(
    const std::string lidar_data_path){
  std::ifstream lidar_data_file(
      lidar_data_path, std::ifstream::in | std::ifstream::binary);
 
  
  float scaling = 0.005;
  float offset_ = -100.0;
  pcl::PointCloud<PointType> laser_cloud;
  ushort x, y, z;
  uchar i, l;
  while(true){
    if(lidar_data_file.eof()) break;
    lidar_data_file.read(reinterpret_cast<char*>(&x), sizeof(ushort));
    lidar_data_file.read(reinterpret_cast<char*>(&y), sizeof(ushort));
    lidar_data_file.read(reinterpret_cast<char*>(&z), sizeof(ushort));
    lidar_data_file.read(reinterpret_cast<char*>(&i), sizeof(uchar));
    lidar_data_file.read(reinterpret_cast<char*>(&l), sizeof(uchar));
    
    PointType point;
    point.x = float(x) * scaling + offset_;
    point.y = float(y) * scaling + offset_;
    point.z = float(z) * scaling + offset_;
    laser_cloud.push_back(point);
  }
  lidar_data_file.close();
  
  return laser_cloud;
}

// Read the KITTI lidar data and our resaved NCLT lidar data in 'velodyne_xyzi'.
pcl::PointCloud<PointType> read_lidar_data(const std::string lidar_data_path){
  std::ifstream lidar_data_file(
      lidar_data_path, std::ifstream::in | std::ifstream::binary);
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(reinterpret_cast<char*>(
      &lidar_data_buffer[0]), num_elements*sizeof(float));

  pcl::PointCloud<PointType> laser_cloud;
  for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 4){
    PointType point;
    point.x = lidar_data_buffer[i];
    point.y = lidar_data_buffer[i + 1];
    point.z = lidar_data_buffer[i + 2];
    // point.intensity = 0;
    
    laser_cloud.push_back(point);
  }
  return laser_cloud;
}


void icp_match_3d(pcl::PointCloud<PointType>::Ptr input_cloud,
    pcl::PointCloud<PointType>::Ptr target_cloud,
    const Eigen::Matrix4f& initial_guess, Eigen::Matrix4f& pose){
  
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaximumIterations(30);
  icp.setInputSource(input_cloud);
  icp.setInputTarget(target_cloud);
  pcl::PointCloud<PointType>::Ptr cloud_icp(new pcl::PointCloud<PointType>());
  icp.align(*cloud_icp, initial_guess);
  pose = icp.getFinalTransformation();
}
  
void ndt_match_3d(pcl::PointCloud<PointType>::Ptr input_cloud,
    pcl::PointCloud<PointType>::Ptr target_cloud,
    const Eigen::Matrix4f& initial_guess, Eigen::Matrix4f& pose){
  pcl::PointCloud<PointType>::Ptr filtered_cloud(
      new pcl::PointCloud<PointType>);
  pcl::ApproximateVoxelGrid<PointType> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud(input_cloud);
  approximate_voxel_filter.filter(*filtered_cloud);
  // std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            // << " data points from room_scan2.pcd" << std::endl;

  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<PointType, PointType> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon(0.01);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize(0.1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution(0.5);

  // Setting max number of registration iterations.
  ndt.setMaximumIterations(35);

  // Setting point cloud to be aligned.
  ndt.setInputSource(filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget(target_cloud);

  // Set initial alignment estimate found using robot odometry.
  // Eigen::AngleAxisf init_rotation(0, Eigen::Vector3f::UnitZ ());
  // Eigen::Translation3f init_translation(0, 0, 0);
  // Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<PointType>::Ptr output_cloud(
      new pcl::PointCloud<PointType>);
  ndt.align(*output_cloud, initial_guess);

  LOG(INFO) << "Normal Distributions Transform has converged:" 
            << ndt.hasConverged () 
            << " score: " << ndt.getFitnessScore () << std::endl;
  pose = ndt.getFinalTransformation();

  // Transforming unfiltered, input cloud using found transform.
  pcl::transformPointCloud(
    *input_cloud, *output_cloud, ndt.getFinalTransformation());

  // Initializing point cloud visualizer
  pcl::visualization::PCLVisualizer::Ptr
  viewer_final(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor(0, 0, 0);

  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  target_color(target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<PointType>(
    target_cloud, target_color, "target cloud");
  viewer_final->setPointCloudRenderingProperties (
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

  // Coloring and visualizing transformed input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  output_color(output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<PointType>(
      output_cloud, output_color, "output cloud");
  viewer_final->setPointCloudRenderingProperties (
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output cloud");
  
  // Starting visualizer
  viewer_final->addCoordinateSystem(1.0, "global");
  viewer_final->initCameraParameters();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ()){
    viewer_final->spinOnce (100);
    // std::this_thread::sleep_for(0.1);
  }
}

void ndt_match_2d(pcl::PointCloud<PointType>::Ptr input_cloud,
    pcl::PointCloud<PointType>::Ptr target_cloud){
  
  for(auto& p: input_cloud->points){
    p.z = 0.;
  }
  for(auto& p: target_cloud->points){
    p.z = 0.;
  }
  pcl::PointCloud<PointType>::Ptr filtered_cloud(
      new pcl::PointCloud<PointType>);
  pcl::ApproximateVoxelGrid<PointType> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud(input_cloud);
  approximate_voxel_filter.filter(*filtered_cloud);
  std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            << " data points from room_scan2.pcd" << std::endl;
  
  
  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform2D<PointType, PointType> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setMaximumIterations (40);
  ndt.setGridCentre(Eigen::Vector2f(0, 0));
  ndt.setGridExtent(Eigen::Vector2f(20, 20));
  ndt.setGridStep(Eigen::Vector2f (20, 20));
  ndt.setOptimizationStepSize(Eigen::Vector3d(0.4, 0.4, 0.1));
  ndt.setTransformationEpsilon(1e-9);

  // Setting point cloud to be aligned.
  ndt.setInputSource(filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget(target_cloud);

  // // Set initial alignment estimate found using robot odometry.
  // Eigen::AngleAxisf init_rotation(0, Eigen::Vector3f::UnitZ ());
  // Eigen::Translation3f init_translation(0, 0, 0);
  // Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<PointType>::Ptr output_cloud(
      new pcl::PointCloud<PointType>);
  ndt.align(*output_cloud);

  LOG(INFO) << "Normal Distributions Transform has converged:" 
            << ndt.hasConverged () 
            << " score: " << ndt.getFitnessScore () << std::endl;

  // Transforming unfiltered, input cloud using found transform.
  pcl::transformPointCloud(
    *input_cloud, *output_cloud, ndt.getFinalTransformation());

  // Initializing point cloud visualizer
  pcl::visualization::PCLVisualizer::Ptr
  viewer_final(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor(0, 0, 0);

  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  target_color(target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<PointType>(
    target_cloud, target_color, "target cloud");
  viewer_final->setPointCloudRenderingProperties (
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

  // Coloring and visualizing transformed input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  output_color(output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<PointType>(
      output_cloud, output_color, "output cloud");
  viewer_final->setPointCloudRenderingProperties (
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output cloud");

  // Starting visualizer
  viewer_final->addCoordinateSystem(1.0, "global");
  viewer_final->initCameraParameters();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ()){
    viewer_final->spinOnce (100);
    // std::this_thread::sleep_for(0.1);
  }
}


// void project_2d(pcl::PointCloud<PointType>::Ptr input_cloud){
//   float side_range = 40.;
//   float fwd_range = 40.;

//   for(PointType pt: input_cloud->points){
    
//   }
// }
cartographer::sensor::RangeData point_cloud_to_range_data(
  const pcl::PointCloud<PointType>::Ptr point_cloud){
  cartographer::sensor::RangeData result;
  result.origin << 0., 0., 0.;
  for(const PointType& pt: point_cloud->points){
    if(sqrt(pt.x*pt.x+pt.y*pt.y+pt.z*pt.z) > 100.){
      result.misses.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
    }else{
      result.returns.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
    }
  }
  return result;
}

void test_contours(const cv::Mat& thres_gray){
  vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(thres_gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
 
	//绘制轮廓
	cv::Mat drawing = Mat::zeros(thres_gray.size(), CV_8UC3);
	int count1 = 0;
	int count2 = 0;
 
	//-------------------------------------------------------------------------------
	for (size_t i = 0; i < contours.size(); i++)
	{
		count1 += 1;
		Scalar color = Scalar(0, 0, 255);
		drawContours(drawing, contours, (int)i, color, 1, LINE_AA, hierarchy, 0);
	}
	cv::imshow("xxx", drawing);
	cv::waitKey();
}


std::vector<std::vector<cv::Point>> fillContour(
  const std::vector<std::vector<cv::Point>> & _contours){
    // sort as x descent y descent.
    std::vector<std::vector<cv::Point>> contours(_contours);
    for(size_t i = 0; i<contours.size(); ++i){
      std::vector<cv::Point> sub(contours[i]);
      std::sort(sub.begin(), sub.end(), [&](cv::Point & A, cv::Point & B) {
          if (A.x == B.x)
              return A.y < B.y;
          else
              return A.x < B.x;
      });

      contours[i] = sub;
    }
 
    // restore as pairs with same ys.
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours_pair;
    for (size_t i = 0; i < contours.size(); ++i){
        std::vector<std::pair<cv::Point, cv::Point>> pairs;
 
        for (size_t j = 0; j < contours[i].size(); ++j){
            // j==0
            if (pairs.size() == 0){
                pairs.push_back({ contours[i][j],contours[i][j] });
                continue;
            }
 
            // j>0
            if (contours[i][j].x != pairs[pairs.size() - 1].first.x){
                pairs.push_back({ contours[i][j],contours[i][j] });
                continue;
            }
 
            if (contours[i][j].x == pairs[pairs.size() - 1].first.x){
                if (contours[i][j].y > pairs[pairs.size() - 1].second.y)
                    pairs[pairs.size() - 1].second = contours[i][j];
                continue;
            }
        }
 
        contours_pair.push_back(pairs);
    }
 
    // fill contour coordinates.
    std::vector<std::vector< cv::Point>> fill_con;
    for (auto pair_set : contours_pair){
        std::vector<cv::Point> pointSet;
        for (auto aPair : pair_set)
        {
            if (aPair.first == aPair.second)
            {
                pointSet.push_back(aPair.first);
         
            }
            else
            {
                for (int i = aPair.first.y; i <= aPair.second.y; ++i)
                {
                    pointSet.push_back(cv::Point(aPair.first.x, i));
                }
            }
 
        }
        fill_con.push_back(pointSet);
    }
 
    return fill_con;
}

std::vector<cv::Point> get_points_in_contour(
    const std::vector<cv::Point>& contour){
  cv::Rect rc = cv::boundingRect(contour);
  std::vector<cv::Point> result = {};
  for(int c = 0; c < rc.width; c++){
    for(int r = 0; r < rc.height; r++){
      if(pointPolygonTest(contour, cv::Point(rc.x+c, rc.y+r), true)>0){
        result.push_back(cv::Point(rc.x+c, rc.y+r));
      }
    }
  }
  return result;
}

cartographer::sensor::PointCloud GetVirtualPointCloud(
  const cartographer::mapping::Grid2D& grid){
  cartographer::sensor::PointCloud output={};
  Eigen::Vector3f pt;
  int cell_x = grid.limits().cell_limits().num_x_cells;
  int cell_y = grid.limits().cell_limits().num_y_cells;
  double resolution = grid.limits().resolution();
  // LOG(INFO)<<cell_x<<","<<cell_y;
  // LOG(INFO)<<grid.ox()<<","<<grid.oy()<<","<<resolution;
  for(int i = 0; i < cell_x; ++i){
    for(int j = 0; j < cell_y; ++j){
      if(!grid.limits().Contains(Eigen::Array2i(i, j))) continue;
      if(grid.GetCorrespondenceCost(Eigen::Array2i(i, j)) < 0.11){//0.12
        pt << grid.ox() + i * resolution, grid.oy() + j * resolution, 0.f;
        output.emplace_back(pt);
      }
    }
  }
  return output;
}

cartographer::sensor::PointCloud get_scan_from_contours(
    const cv::Mat& src, double ox, double oy, double resolution){
  cv::Mat img;
  
  threshold(src, img, 100, 255, CV_THRESH_BINARY);
  auto ele = getStructuringElement(MORPH_RECT, Size(3, 3));
  erode(img, img, ele);
  cv::Mat img_copy;
  img.copyTo(img_copy);
  cv::Mat img_draw = Mat::zeros(img.size(), CV_8UC3);
  vector<vector<Point> > contours, contours_filter;
	vector<Vec4i> hierarchy;
	findContours(img_copy, contours, hierarchy, RETR_TREE,CHAIN_APPROX_SIMPLE);

  for (size_t i = 0; i < contours.size(); i++){
    float area = contourArea(contours[i]);
    if(area > 100 && area < img.cols * img.rows / 4.){
      contours_filter.push_back(contours[i]);
    }
	}
  // int i = 0;
  cartographer::sensor::PointCloud pts={};
  // std::vector<cv::Point> pts = {};
  for(const auto& contour: contours_filter){
    // drawContours(img_draw, contours_filter, i++, Scalar(0,0,255));
    cv::Rect rc = cv::boundingRect(contour);
    for(int c = 0; c < rc.width; c++){
      for(int r = 0; r < rc.height; r++){
        double d = pointPolygonTest(contour, cv::Point(rc.x+c, rc.y+r), true);
        if(d > 0){
          // pts.push_back(cv::Point(rc.x+c, rc.y+r));
          Eigen::Vector3f pt;
          pt<<(rc.x+c) * resolution + ox, (rc.y+r) * resolution + oy, 0.;
          pts.push_back(pt);
        }
      }
    }
  }
  return pts;
  // cv::imshow("contour", img_draw);
  // cv::waitKey(0);
}


bool detect_and_match(
    const cv::Mat& src1, 
    const cv::Mat& src2, 
    const Eigen::Vector3f& oxy_res_1,
    const Eigen::Vector3f& oxy_res_2,
    Eigen::Vector3f& xy_yaw, 
    double& estimated_scale,
    bool visualize = false){
  cv::Mat img1, img2;
  threshold(src1, img1, 100, 255, CV_THRESH_BINARY_INV);
  threshold(src2, img2, 100, 255, CV_THRESH_BINARY_INV);
  // auto ele = getStructuringElement(MORPH_RECT, Size(3, 3));
  // erode(img1, img1, ele);
  // erode(img2, img2, ele);
  // cv::imshow("src1", src1);
  // cv::imshow("erode", img1);
  // cv::waitKey();
  // test_contours(img1);
  // cv::imshow("img1-before", img1);
  // cv::imshow("img2-before", img2);
  // cv::waitKey(0);
  
  /* cv::Mat img1_copy;
  img1.copyTo(img1_copy);
  cv::Mat img2_copy;
  img2.copyTo(img2_copy);
  cv::Mat img1_draw = Mat::zeros(img1.size(), CV_8UC3);
  cv::Mat img2_draw = Mat::zeros(img2.size(), CV_8UC3);
  vector<vector<Point> > contours1, contours1_filter, contours2, contours2_filter;
	vector<Vec4i> hierarchy1, hierarchy2;
	findContours(img1_copy, contours1, hierarchy1, RETR_TREE,CHAIN_APPROX_SIMPLE);
	findContours(img2_copy, contours2, hierarchy2, RETR_TREE,CHAIN_APPROX_SIMPLE);

  for (size_t i = 0; i < contours1.size(); i++){
    float area = contourArea(contours1[i]);
    if(area > 100 && area < img1.cols * img1.rows / 4.){
      contours1_filter.push_back(contours1[i]);
    }
	}
  
  for (size_t i = 0; i < contours2.size(); i++){
    float area = contourArea(contours2[i]);
    if(area > 100 && area < img2.cols * img2.rows / 4.){
      contours2_filter.push_back(contours2[i]);
    }
	}


  cv::Mat img1_filter = Mat(img1.size(), CV_8UC1, Scalar(255)); 
  cv::Mat img2_filter = Mat(img2.size(), CV_8UC1, Scalar(255)); 
  int i = 0;
  int j = 0;
  for(const auto& contour: contours1_filter){
    // drawContours(img1_draw, contours1_filter, i++, Scalar(0,0,255));
    auto pts = get_points_in_contour(contour);
    for(const auto& pt: pts){
      img1_filter.at<uchar>(pt.y, pt.x) = 0;
    }
  }
  for(const auto& contour: contours2_filter){
    // drawContours(img2_draw, contours2_filter, j++, Scalar(0,0,255));
    auto pts = get_points_in_contour(contour);
    for(const auto& pt: pts){
      img2_filter.at<uchar>(pt.y, pt.x) = 0;
    }
  }

  cv::imshow("img1", img1_filter);
  cv::imshow("img2", img2_filter);
  cv::waitKey(0); */

  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create(minHessian);
  // Ptr<SIFT> detector = SIFT::create();
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2);
  //-- Step 2: Matching descriptor vectors with a FLANN based matcher
  // Since SURF is a floating-point descriptor NORM_L2 is used
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
      DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<DMatch> > knn_matches;
  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.85;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++){
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  //-- Draw matches
  if(visualize){
    Mat img_matches;
    drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches, 
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), 
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", img_matches );
    imshow("src1", img1 );
    imshow("src2", img2 );
  }
  
  // drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches);
  //-- Show detected matches
  // imshow("Good Matches", img_matches );
  // waitKey();

  std::vector<cv::Point2f> from_pts={};
  std::vector<cv::Point2f> to_pts={};
  std::vector<cv::Point2f> from_pix={};
  std::vector<cv::Point2f> to_pix={};
  // std::cout << "good_matches.size: " << good_matches.size() << std::endl;
  if(good_matches.size() >= 5){
    cv::Point2f tl_from(oxy_res_1[0], oxy_res_1[1]);
    cv::Point2f tl_to(oxy_res_2[0], oxy_res_2[1]);
    for(int i = 0; i < good_matches.size(); ++i){
      auto gmt = good_matches[i];
      
      from_pts.push_back(
          tl_from + keypoints1.at(gmt.queryIdx).pt * oxy_res_1[2]);
      to_pts.push_back(
          tl_to + keypoints2.at(gmt.trainIdx).pt * oxy_res_2[2]);

      from_pix.push_back(keypoints1.at(gmt.queryIdx).pt);
      to_pix.push_back(keypoints2.at(gmt.trainIdx).pt);
    }
    std::vector<uchar> inliers;
    
    cv::Mat transform = cv::estimateAffinePartial2D(from_pts, to_pts, 
        inliers, cv::RANSAC, 3 * oxy_res_2[2]/*in meter*/, 3000);
    if(!transform.empty()){
      double scale = sqrt(transform.at<double>(0,0)*transform.at<double>(0,0) 
          + transform.at<double>(0,1)*transform.at<double>(0,1));
      
      if(abs(1-scale)<0.1){
        if(visualize){
          cv::Mat transform_pix = cv::estimateAffinePartial2D(
            from_pts, to_pts, inliers, cv::RANSAC, 3/*in pix*/, 3000);
          Mat affine_img1, img3;
          warpAffine(
              src1, affine_img1, transform_pix, Size(src1.cols, src1.rows));
          drawMatches(affine_img1, keypoints1, src2, keypoints2, {}, 
            img3, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), 
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
          
          imshow("Affine", img3);
          waitKey();
        }
        // float yaw = atan2(transform.at<double>(1, 0), transform.at<double>(1, 1));
        float yaw = asin(transform.at<double>(1, 0) / scale);
        xy_yaw << transform.at<double>(0, 2),
                  transform.at<double>(1, 2),//whether to divide scale?
                  yaw;
        estimated_scale = scale;
        return true;
      }
    }
  }
  
  return false;
}



void TestGridMatch(const cv::Mat& grid_a, const cv::Mat& grid_b,
    Eigen::Vector2i& translation, double& rot_angle){
  std::vector<Eigen::Vector2d> cells_b;
  for(int r = 0; r < grid_b.rows; ++r){
    for(int c = 0; c < grid_b.cols; ++c){
      if(grid_b.at<uchar>(r, c) > 200){
        cells_b.push_back(Eigen::Vector2d(c, r)); 
      }
    }
  }
  
  size_t rot_num = 90;
  double angle_step_size = 2. * M_PI / static_cast<double>(rot_num);
  std::vector<float> probs(rot_num*grid_a.cols*grid_a.rows, 0.);

  std::vector<std::vector<Eigen::Vector2i>> rot_cells_b={};
  rot_cells_b.resize(rot_num);
  for(int i = 0; i < rot_num; ++i){
    rot_cells_b[i].resize(cells_b.size());
  }
  for(int i = 0; i < rot_num; ++i){
    double theta = static_cast<double>(i) * angle_step_size; 
    cartographer::transform::Rigid2d rotation = 
        cartographer::transform::Rigid2d::Rotation(theta);
    for(int j = 0; j < cells_b.size(); ++j) {
      const auto& cell = cells_b[j];
      Eigen::Vector2d rc = rotation * cell;
      Eigen::Vector2i rc_i(floor(rc[0]), floor(rc[1]));
      rot_cells_b.at(i).at(j) = rc_i;
    }
  }
  // LOG(INFO)<<rot_cells_b.size();
  // LOG(INFO)<<rot_cells_b[0].size();
  // LOG(INFO)<<grid_a.rows<<","<<grid_a.cols;

  for(int i = 0; i < rot_num; ++i){
    for(int r = 0; r < grid_a.rows; ++r){
      for(int c = 0; c < grid_a.cols; ++c){
        int sum_num_irc = 0;
        for(const auto& cell: rot_cells_b[i]) {
          int shift_c = c + cell[0];
          int shift_r = r + cell[1];
          if( shift_c < 0 ||shift_r < 0 || 
              shift_c >= grid_a.cols-1 || shift_r >= grid_a.rows-1){
            continue;
          }
          if(grid_a.at<uchar>(shift_r, shift_c) > 200){
            sum_num_irc++;
          }
        }
        probs.at(i * grid_a.cols * grid_a.rows + r * grid_a.cols + c) = 
          static_cast<double>(sum_num_irc) / static_cast<double>(cells_b.size());
      }
    }
  }
  int max_id = arg_max(probs);
  int id_angle = max_id / (grid_a.cols * grid_a.rows);
  int id_row = (max_id % (grid_a.cols * grid_a.rows)) / grid_a.cols;
  int id_col = (max_id % (grid_a.cols * grid_a.rows)) % grid_a.cols;
  translation <<  id_col, id_row;
  rot_angle = angle_step_size * id_angle;
  // LOG(INFO)<<id_col<<","<<id_row<<","<<rot_angle*180./M_PI;
}

// void SetGroundSegmentParams(SegmentationParams& params){
//    params.kLidarRows = 64;
//    params.kLidarCols = 2169;
   
//    params.kNumSectors = 360;

//    params.kGroundYInterceptTolerance = 0.5;
//    params.kGroundPointLineDistThres = 0.1;
//    params.kWallLineMinBinNum = 3;
//    params.kWallPointLineDistThres = 0.1;

//    std::vector<float> ext_trans_vec = {0., 0., 1.73};
//    std::vector<float> ext_rot_vec = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
//    params.kExtrinsicTrans = Eigen::Map<const Eigen::Matrix<
//      float, -1, -1, Eigen::RowMajor>>(ext_trans_vec.data(), 3, 1);
//    params.kExtrinsicRot = Eigen::Map<const Eigen::Matrix<
//      float, -1, -1, Eigen::RowMajor>>(ext_rot_vec.data(), 3, 3);

//    const float factor = M_PI / 180.;
//    params.kLidarProjectionError = 0.5 / factor;
//    params.kLidarHorizRes = 0.166 * factor;
//    params.kLidarVertRes = 0.4 * factor;
//    params.kLidarVertFovMax = 2.0 * factor;
//    params.kLidarVertFovMin = -24.9 * factor;
//    params.kGroundSameLineTolerance = 2 * factor;
//    params.kGroundSlopeTolerance = 10 * factor;
//    params.kWallSameLineTolerance = 10 * factor;
//    params.kWallSlopeTolerance = 75 * factor;
// }

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

Eigen::VectorXf EstimateGround(
    pcl::PointCloud<PointType>::Ptr ground_points, bool visualize = false){
  pcl::SampleConsensusModelPlane<PointType>::Ptr model_plane(
     new pcl::SampleConsensusModelPlane<PointType>(ground_points));
	pcl::RandomSampleConsensus<PointType> ransac(model_plane);
  
  // 设置距离阈值，与平面距离小于0.03的点作为内点
	ransac.setDistanceThreshold(0.1);	
	ransac.computeModel();

	pcl::PointCloud<PointType>::Ptr cloud_plane(
      new pcl::PointCloud<PointType>());	
  vector<int> inliers;
	ransac.getInliers(inliers);
	pcl::copyPointCloud<PointType>(*ground_points, inliers, *cloud_plane);
  
  /// 输出模型参数Ax+By+Cz+D=0
	Eigen::VectorXf coefficient;
	ransac.getModelCoefficients(coefficient);
	
  if(visualize){
    pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("ground_fit"));

    viewer->addPointCloud<PointType>(ground_points, "cloud_ori");
    viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.5, 0.5, "cloud_ori");
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

pcl::PointCloud<PointType>::Ptr FilterGroundByNormals(
      pcl::PointCloud<PointType>::Ptr cloud,
      pcl::search::KdTree<PointType>::Ptr tree,
      pcl::NormalEstimationOMP<PointType, pcl::Normal>& n,
      bool visualize = false){

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
  
	
  if(visualize){
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

Eigen::Matrix4f TransformPointsToGround(const Eigen::VectorXf& coeff,
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

Eigen::Matrix4f EsitmateGroundAndTransform(
    pcl::PointCloud<PointType>::Ptr cloud_in,
    pcl::PointCloud<PointType>::Ptr cloud_out,
    bool visualize = false){
  
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
      q_ground, tree, normal_estimator, visualize);
  
  if(q_ground_pts == nullptr){
    return Eigen::Matrix4f::Identity();
  }

  // 地面拟合: ax + by + cz + d = 0, q_coeff=[a,b,c,d]
  Eigen::VectorXf q_coeff = EstimateGround(q_ground_pts, visualize);
  
  // 转换点云到地面坐标系
  return TransformPointsToGround(q_coeff, cloud_in, cloud_out);
}

Submap3D BuildSubmap(pcl::PointCloud<PointType>::Ptr cloud_in){
  RangeDataInserter3D range_data_inserter;
  const float high_resolution_max_range = 70.;
  const float high_resolution = 0.2;
  const float low_resolution = 0.5;
  auto identity_transform = cartographer::transform::Rigid3d::Identity();

  Submap3D submap(high_resolution, low_resolution, identity_transform);
  auto range_data = point_cloud_to_range_data(cloud_in);
  submap.InsertRangeData(
      range_data, range_data_inserter, high_resolution_max_range);
  return submap;
}

cv::Mat BuildSubmapAndProject(
    pcl::PointCloud<PointType>::Ptr cloud_in,
    double& ox, double& oy, double& resolution){
  scan_matching::FastCorrelativeScanMatcherOptions2D option;
  RangeDataInserter3D range_data_inserter;
  const float high_resolution_max_range = 70.;
  const float high_resolution = 0.2;
  const float low_resolution = 0.5;
  auto identity_transform = cartographer::transform::Rigid3d::Identity();

  Submap3D submap(high_resolution, low_resolution, identity_transform);
  auto range_data = point_cloud_to_range_data(cloud_in);
  submap.InsertRangeData(
      range_data, range_data_inserter, high_resolution_max_range);

  cv::Mat img_q = ProjectToCvMat(
      &submap.high_resolution_hybrid_grid(), identity_transform,
      ox, oy, resolution);

  ProbabilityGrid q_grid = ProjectToGrid(
      &submap.high_resolution_hybrid_grid(), identity_transform,
      ox, oy, resolution);
  scan_matching::FastCorrelativeScanMatcher2D scan_matcher(q_grid, option); 
  // 这里如果取降采样的，那么应该将所返回的分辨率对应降低
  const int k = 1;
  cv::Mat img_k_q = scan_matcher.precomputation_grid_stack_->Get(k).ToCvImage();
  // resolution *= (k+1);
  return img_q;
}

void view_bin(const std::string& filename){
  pcl::PointCloud<PointType>::Ptr scan(new pcl::PointCloud<PointType>());
  *scan = read_lidar_data(filename);
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
       new pcl::visualization::PCLVisualizer("viewer"));
	
  viewer->setBackgroundColor(0.3, 0.3, 0.3);
  viewer->addText("binfile", 10, 10, "text");
  pcl::visualization::PointCloudColorHandlerCustom<
      PointType> single_color(scan, 0, 225, 0);
  viewer->addCoordinateSystem(0.1);
  viewer->addPointCloud<PointType>(
      scan, single_color, "sample cloud");

  //设置点云大小
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  while (!viewer->wasStopped()){
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  } 
}

std::string imgpath_to_binpath(const std::string& image_path){
  std::string prob_img = "prob_img";
  size_t pos_0 = image_path.find(prob_img);
  size_t pos_1 = image_path.find_last_of('/');
  size_t pos_2 = image_path.find_last_of('.');
  
  CHECK(pos_0!=std::string::npos);
  CHECK(pos_1!=std::string::npos);
  CHECK(pos_2!=std::string::npos);

  std::string basedir = image_path.substr(0, pos_0);
  std::string basename = image_path.substr(pos_1+1, pos_2-pos_1 - 1);
  
  std::string scan_path = basedir + "velodyne_points/data/" + basename + ".bin";
  
  return scan_path;
}

void caculate_mean_std(
    const std::vector<double>& resultSet, double& mean, double& stdev){
  double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	mean =  sum / resultSet.size();
 
	double accum  = 0.0;
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum  += (d-mean)*(d-mean);
	});
 
	stdev = sqrt(accum/(resultSet.size()-1));
}
	

int main(int argc, char *argv[]){
  // Read evaluation filenames
  string valset_filename = argv[1];
  string pose_filename = argv[2];
  vector<string> db_files = {};
  vector<string> q_files = {};
  vector<vector<int>> gt_q_pos_idx = {};
  vector<Eigen::Matrix4f> poses_db_q;
  
  ReadValset(valset_filename, db_files, q_files, gt_q_pos_idx);
  ReadValsetPose(pose_filename, poses_db_q);

  CHECK(q_files.size()==gt_q_pos_idx.size());
  
  double res_q, ox_q, oy_q;
  double res_db, ox_db, oy_db;
  // When the query scan is captured from the same platform as the one in database
  // and the LiDAR is horizontally installed, we do not need to estimate the ground plane.
  // Otherwise, turn estimate_roll_pitch on, the algorithm will automatically 
  // estimate the relative transform to the ground.
  // Usually, we do not need to estimate roll and pitch for db.
  const bool estimate_roll_pitch_q = true;
  const bool estimate_roll_pitch_db = false;
  const bool enable_ergodic_roll_pitch = false;
  const bool use_icp = false;
  // 如果不估计database帧到地面的关系，需显式提供雷达到地面的外参数以便转换点云到地面
  // 以KITTI作测试
  Eigen::Matrix4f T_L2G = Eigen::Matrix4f::Identity();
  T_L2G << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1.73, 0, 0, 0, 1;
  
  // Perform pose estimation
  int all_tests = 0;
  int no_positive_num = 0;
  int succeed_tests = 0;
  std::vector<double> rot_err={};
  std::vector<double> pos_err={};

  for(int i = 0; i < q_files.size(); ++i){
    pcl::PointCloud<PointType>::Ptr q_pc_raw(
        new pcl::PointCloud<PointType>); 
    LOG(INFO)<<"Query: "<<q_files[i];
    *q_pc_raw = read_lidar_data_nclt(q_files[i]);

    // 转换点云到地面坐标系
    Eigen::Matrix4f Tq_l2g = Eigen::Matrix4f::Identity();
    pcl::PointCloud<PointType>::Ptr q_pc(new pcl::PointCloud<PointType>);
    
    const double rad2deg = 180. / M_PI;

    if(estimate_roll_pitch_q){
      Tq_l2g = EsitmateGroundAndTransform(q_pc_raw, q_pc, true);
      if(q_pc == nullptr){
        LOG(ERROR)<<"Failed to fit the ground plane from "<<q_files[i];
        q_pc = q_pc_raw;
      }
      // Eigen::Matrix3f Rq_l2g = Tq_l2g.topLeftCorner(3, 3);
      // Eigen::Vector3f ypr = Rq_l2g.eulerAngles(2, 1, 0);
      // LOG(INFO) << "roll: " << ypr[2] * rad2deg << " pitch: "
      //           << ypr[1] * rad2deg << " yaw: " << ypr[0] * rad2deg;
    }else{
      Tq_l2g = T_L2G;
      q_pc = q_pc_raw;
    }
    
    auto identity_transform = cartographer::transform::Rigid3d::Identity();
    // save candidate horizontally projected images of the query frame.
    std::vector<cv::Mat> imgs_q = {};
    std::vector<Eigen::Vector3f> imgs_oxy_res = {};
    // TODO(wz): transform submap instead of transforming the point cloud
    if(enable_ergodic_roll_pitch){
      Eigen::Quaternionf q_est(Tq_l2g.block<3,3>(0, 0));
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      // go through roll and pitch to improve recall
      for(float r = -3.; r < 4.; r += 1.){ //in degree
        for(float p = -3.; p < 4.; p += 1.){
          Eigen::Quaternionf q_rp = cartographer::transform::RollPitchYaw(  
              r / rad2deg, p / rad2deg,  0.).cast<float>();
          q_rp = q_rp * q_est;
          T.topLeftCorner(3, 3) = q_rp.toRotationMatrix();
          
          pcl::PointCloud<PointType>::Ptr pc_tmp(new pcl::PointCloud<PointType>());
          pcl::transformPointCloud(*q_pc_raw, *pc_tmp, T);
          
          Submap3D submap_q = BuildSubmap(pc_tmp);
          
          cv::Mat img_rp = ProjectToCvMat(
              &submap_q.high_resolution_hybrid_grid(), 
              identity_transform, ox_q, oy_q, res_q);

          imgs_q.emplace_back(img_rp);
          imgs_oxy_res.emplace_back(Eigen::Vector3f(ox_q, oy_q, res_q));
        }
      }
    }else{
      cv::Mat img_k_q = BuildSubmapAndProject(q_pc, ox_q, oy_q, res_q);
      imgs_q.emplace_back(img_k_q);
      imgs_oxy_res.emplace_back(Eigen::Vector3f(ox_q, oy_q, res_q));
    }
    CHECK(imgs_oxy_res.size() == imgs_q.size());
    
    
    // 逐一与数据库中的回环帧匹配
    for(int j: gt_q_pos_idx[i]){
      all_tests++;
      LOG(INFO)<<"Db: "<<db_files[i];
      pcl::PointCloud<PointType>::Ptr db_pc_raw(new pcl::PointCloud<PointType>); 
      *db_pc_raw = read_lidar_data_nclt(db_files[j]);

      Eigen::Matrix4f Tdb_l2g = Eigen::Matrix4f::Identity();
      
      // 转换db点云到地面坐标系
      // usually, we do not need to estimate roll and pitch for db.
      pcl::PointCloud<PointType>::Ptr db_pc(new pcl::PointCloud<PointType>());
      
      if(estimate_roll_pitch_db){
        Tdb_l2g = EsitmateGroundAndTransform(db_pc_raw, db_pc, true);
        if(db_pc == nullptr){
          LOG(WARNING)<<"Failed to fit the ground plane from "<<db_files[i];
          continue;
        }
      }else{
        Tdb_l2g = T_L2G;
        db_pc = db_pc_raw;
      } 
     
      cv::Mat img_k_db = BuildSubmapAndProject(db_pc, ox_db, oy_db, res_db);

      // 2D match between projected 2D scans
      // estimate the 3-DoF transform (x, y, yaw) in the ground plane.
      // How about the returned yaw angle in (-pi/2, pi/2), do we need to check yaw+pi?
      Eigen::Vector3f xy_yaw, oxy_res_db;
      oxy_res_db << ox_db, oy_db, res_db;
      
      std::vector<float> scales = {};
      std::vector<float> remapped_scales = {};
      std::vector<Eigen::Vector3f> xy_yaws = {};
      scales.resize(imgs_q.size(), 999.);
      remapped_scales.resize(imgs_q.size(), 999.);
      xy_yaws.resize(imgs_q.size());
      double estimated_scale = 999.;
      for(int k = 0; k < imgs_q.size(); ++k){
        const cv::Mat img_k_q = imgs_q.at(k);
        const Eigen::Vector3f oxy_res_q_k = imgs_oxy_res.at(k);

        bool matched = detect_and_match(img_k_q, img_k_db, 
            oxy_res_q_k, oxy_res_db, xy_yaw, estimated_scale, false);//q-to-db
        scales[k] = estimated_scale;
        remapped_scales[k] = abs(1. - estimated_scale);
        xy_yaws[k] = xy_yaw;
      }
      
      int min_idx = arg_min(remapped_scales);
      if(remapped_scales[min_idx] < 0.1){
        // estimate (roll, pitch, z) from their ground plane normals.
        // roll，pitch和dz由与地平面的关系直接可得
        Eigen::Matrix4f T_q2db_rpz = Tdb_l2g.inverse() * Tq_l2g;
        Eigen::Matrix3f R_q2db_rpz = T_q2db_rpz.topLeftCorner(3, 3);
        Eigen::Vector3f ypr_rpz = R_q2db_rpz.eulerAngles(2, 1, 0);
        
        // dx, dy及yaw需要叠加3-DoF的2维变换
        cartographer::transform::Rigid2f T_q2db_h(xy_yaw.head<2>(), xy_yaw[2]);
        cartographer::transform::Rigid3f T_q2db_h_3d 
            = cartographer::transform::Embed3D(T_q2db_h);
        Eigen::Matrix3f R_q2db_h_3d = T_q2db_h_3d.rotation().toRotationMatrix();
        Eigen::Vector3f t_q2db_h_3d = T_q2db_h_3d.translation();
        Eigen::Matrix4f T_qg_dbg = Eigen::Matrix4f::Identity();
        T_qg_dbg.topLeftCorner(3, 3) = R_q2db_h_3d;
        T_qg_dbg.block<3, 1>(0, 3) = t_q2db_h_3d;
        Eigen::Matrix4f T_q2db_yawxy = Tdb_l2g.inverse() * T_qg_dbg * Tq_l2g;

        Eigen::Matrix3f R_q2db_yawxy = T_q2db_yawxy.topLeftCorner(3,3);
        Eigen::Vector3f ypr_yawxy = R_q2db_yawxy.eulerAngles(2,1,0);

        float dx = T_q2db_yawxy(0, 3);
        float dy = T_q2db_yawxy(1, 3);
        float dz = T_q2db_rpz(2, 3);
        float roll = ypr_rpz[2];
        float pitch = ypr_rpz[1];
        float yaw = ypr_yawxy[0];
        
        // Ground truth
        Eigen::Matrix4f q2db = poses_db_q[j].inverse() * poses_db_q[i+db_files.size()];
        Eigen::Matrix3f gt_rot = q2db.topLeftCorner(3, 3);
        Eigen::Vector3f gt_ypr = gt_rot.eulerAngles(2, 1, 0);

        // Rotation matrix restored from roll, pitch, and yaw  
        Eigen::Quaterniond q_restored = cartographer::transform::RollPitchYaw(
            roll, pitch, yaw);
        Eigen::Matrix3f R_restored = Eigen::Matrix3d(q_restored).cast<float>();
        
        if(use_icp){
          Eigen::Matrix4f T_guess = Eigen::Matrix4f::Identity();
          Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();
          T_guess.block(0,0,3,3) = R_restored;
          T_guess.block(0,3,3,1) = Eigen::Vector3f(dx,dy,dz);
          icp_match_3d(q_pc_raw, db_pc_raw, T_guess, T_icp);
          R_restored = T_icp.topLeftCorner(3,3);
          dx = T_icp(0,3);
          dy = T_icp(1,3);
          dz = T_icp(2,3);
        }
        

        Eigen::Matrix3f err_R = gt_rot.transpose() * R_restored;

        float trace = err_R.trace();
        float offset_trace = 0.5 * (trace - 1);
        offset_trace = offset_trace < -0.999999 ? -0.999999 : offset_trace;
        offset_trace = offset_trace > 0.999999 ? 0.999999 : offset_trace;
        float err_rot = abs(std::acos(offset_trace));
        Eigen::Vector3f err_trans;
        err_trans << q2db(0, 3) - dx, q2db(1, 3) - dy, q2db(2, 3) - dz;
        float err_pos = err_trans.norm();
        // LOG(INFO) << j << "-RPZ: "<<roll*rad2deg<<", " <<pitch*rad2deg<<", "<<dz;
        // LOG(INFO) << j << "-Gt: "<<q2db(0, 3) <<", "<<q2db(1, 3) <<", " << gt_ypr[0] * rad2deg;
        // LOG(INFO) << j <<"-Est: "<<dx<<", "<<dy<<", "<<yaw*rad2deg;
       
        // LOG(INFO) << "XYZ-1: "<<xy_yaw.transpose();
        err_rot = err_rot * rad2deg;
        LOG(INFO)<<err_pos<<", "<<err_rot;

        if(abs(err_rot - 180.) < 5.) err_rot = abs(err_rot - 180.);
        if(err_pos < 1.0 && err_rot < 5.){
          succeed_tests++;
          
          rot_err.push_back(err_rot);
          pos_err.push_back(err_pos);
        } 
      }else{
        // cv::imwrite("./failed_pairs/"+std::to_string(i)+"-"+std::to_string(j)+"-q.jpg", img_k_q);
        // cv::imwrite("./failed_pairs/"+std::to_string(i)+"-"+std::to_string(j)+"-db.jpg", img_k_db);
      } 
    }
  }
  double mean_rot, std_rot, mean_pos, std_pos;
  caculate_mean_std(pos_err, mean_pos, std_pos);
  caculate_mean_std(rot_err, mean_rot, std_rot);

  // LOG(INFO)<<"dx-dy-dz: "<<dx_sum/succeed_tests<<", "<<dy_sum/succeed_tests<<", "<<dz_sum/succeed_tests;
  // LOG(INFO)<<"rpy: "<<roll_sum/succeed_tests<<", "<<pitch_sum/succeed_tests<<", "<<yaw_sum/succeed_tests;
  LOG(INFO)<<succeed_tests<<", "<<all_tests;
  LOG(INFO)<<"Success rate: "
      <<static_cast<float>(succeed_tests) / static_cast<float>(all_tests);
  LOG(INFO)<<"Rot error: "<<mean_rot<<", "<<std_rot;
  LOG(INFO)<<"Pos error: "<<mean_pos<<", "<<std_pos;
  return 0;
}