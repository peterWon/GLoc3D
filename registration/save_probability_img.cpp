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

#include <pcl/visualization/pcl_visualizer.h>

#include "3d/submap_3d.h"
#include "2d/probability_grid.h"
#include "2d/fast_correlative_scan_matcher_2d.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include<chrono>

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cartographer::mapping;

vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if("" == str) return res;
	//先将要切割的字符串从string类型转换为char*类型
	char * strs = new char[str.length() + 1] ; //不要忘了
	strcpy(strs, str.c_str()); 
 
	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());
 
	char *p = strtok(strs, d);
	while(p) {
		string s = p; //分割得到的字符串转换为string类型
		res.push_back(s); //存入结果数组
		p = strtok(NULL, d);
	}

	return res;
}

pcl::PointCloud<pcl::PointXYZI> read_lidar_data(
    const std::string lidar_data_path){
  std::ifstream lidar_data_file(
    lidar_data_path, std::ifstream::in | std::ifstream::binary);
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(
    reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));

  pcl::PointCloud<pcl::PointXYZI> laser_cloud;
  for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 4){
    pcl::PointXYZI point;
    point.x = lidar_data_buffer[i];
    point.y = lidar_data_buffer[i + 1];
    point.z = lidar_data_buffer[i + 2];
    point.intensity = 0;
    
    laser_cloud.push_back(point);
  }
  return laser_cloud;
}

pcl::PointCloud<pcl::PointXYZI> read_lidar_data_nuscenes(
    const std::string lidar_data_path){
  std::ifstream lidar_data_file(
    lidar_data_path, std::ifstream::in | std::ifstream::binary);
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(
    reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));

  pcl::PointCloud<pcl::PointXYZI> laser_cloud;
  for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 5){
    pcl::PointXYZI point;
    point.x = lidar_data_buffer[i];
    point.y = lidar_data_buffer[i + 1];
    point.z = lidar_data_buffer[i + 2];
    point.intensity = 0;
    
    laser_cloud.push_back(point);
  }
  return laser_cloud;
}

int scanFiles(vector<string> &fileList, string inputDirectory){
  inputDirectory = inputDirectory.append("/");

  DIR *p_dir;
  const char* str = inputDirectory.c_str();

  p_dir = opendir(str);   
  if( p_dir == NULL){
    cout<< "can't open :" << inputDirectory << endl;
  }

  struct dirent *p_dirent;

  while(p_dirent = readdir(p_dir)){
    string tmpFileName = p_dirent->d_name;
    if( tmpFileName == "." || tmpFileName == ".."){
      continue;
    }else{
      fileList.push_back(tmpFileName);
    }
  }
  closedir(p_dir);
  return fileList.size();
}

cartographer::sensor::RangeData point_cloud_to_range_data(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud){
  cartographer::sensor::RangeData result;
  result.origin << 0., 0., 0.;
  for(const pcl::PointXYZI& pt: point_cloud->points){
    if(sqrt(pt.x*pt.x+pt.y*pt.y+pt.z*pt.z) > 100.){
      result.misses.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
    }else{
      result.returns.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
    }
  }
  return result;
}


int main(int argc, char *argv[]){
  string seqdir = argv[1];
  // kitti raw
  string srcdir = seqdir + "/velodyne_points/data/";
  string resdir = seqdir + "/prob_img/";
  
  // NuScenes
  // string srcdir = seqdir + "/LIDAR_TOP/";
  // string resdir = seqdir + "/prob_img/";
  
  // NCLT
  // string srcdir = seqdir + "/velodyne_sync_xyzi/";
  // string resdir = seqdir + "/prob_img/";
  
  
  int status = mkdir(resdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  vector<string> bin_files = {};
  int fsize = scanFiles(bin_files, srcdir);
  LOG(INFO)<<"Total "<<fsize<<" binary files to process.";
  
  scan_matching::FastCorrelativeScanMatcherOptions2D option;
  RangeDataInserter3D range_data_inserter;
  
  LOG(INFO)<<"branch_and_bound_depth "<<option.branch_and_bound_depth();
  const float high_resolution_max_range = 100.;
  double ox, oy, resolution;
  double ox_db, oy_db, resolution_db;
  auto identity_transform = cartographer::transform::Rigid3d::Identity();

  double tsum = 0.;
  for(const string& name: bin_files){
    // const string& name = bin_files[i];
    pcl::PointCloud<pcl::PointXYZI>::Ptr q_pc(
      new pcl::PointCloud<pcl::PointXYZI>); 
    *q_pc = read_lidar_data_nuscenes(srcdir + "/" + name);
    // *q_pc = read_lidar_data(srcdir + "/" + name);
    auto start = std::chrono::steady_clock::now();
    Submap3D submap_q(0.2, 0.5, identity_transform);
    auto range_data = point_cloud_to_range_data(q_pc);
    submap_q.InsertRangeData(
      range_data, range_data_inserter, high_resolution_max_range);
    cv::Mat img_q = ProjectToCvMat(
      &submap_q.high_resolution_hybrid_grid(), identity_transform,
      ox, oy, resolution);
    auto end = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<microseconds>(end - start);
    // tsum+= (tt.count()/1000);
    size_t idx = name.find_last_of(".");
    cv::imwrite(resdir+"/"+name.substr(0, idx)+".jpg", img_q);
    LOG(INFO)<<"Saved "<<name.substr(0, idx)+".jpg";
  }
  LOG(INFO)<<"Done!";

  return 0;
}