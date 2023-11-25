#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>

using namespace std;

typedef pcl::PointXYZI PointType;
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
    point.intensity = lidar_data_buffer[i + 3];
    
    laser_cloud.push_back(point);
  }
  return laser_cloud;
}

void points_to_voxels(
    pcl::PointCloud<PointType>::Ptr points_xyz, 
    const std::vector<bool>& points_mask,
    const Eigen::Vector3f& grid_range_x, //min_max_res_x
    const Eigen::Vector3f& grid_range_y, 
    const Eigen::Vector3f& grid_range_z,
    std::vector<Eigen::Vector3f>& local_points_xyz,
    std::vector<Eigen::Vector3f>& point_centroids,
    std::vector<Eigen::Vector3f>& pts_to_voxel_centers,
    std::vector<size_t>& voxel_point_count,
    std::vector<size_t>& voxel_indices){
  const size_t batch_size = 1;
  const size_t num_points = points_xyz->size();
  
  point_centroids.resize(num_points);
  local_points_xyz.resize(num_points);
  voxel_point_count.resize(num_points);
  pts_to_voxel_centers.resize(num_points);
  voxel_indices.resize(num_points);
  // voxel大小
  float voxel_size_x = grid_range_x[2];
  float voxel_size_y = grid_range_y[2];
  float voxel_size_z = grid_range_z[2];
  std::vector<float> voxel_size = {voxel_size_x, voxel_size_y, voxel_size_z};

  // voxel数量
  std::vector<int> grid_size ={
    (grid_range_x[1]-grid_range_x[0]) / voxel_size_x,
    (grid_range_y[1]-grid_range_y[0]) / voxel_size_y,
    (grid_range_z[1]-grid_range_z[0]) / voxel_size_z};
  
  int num_voxels = grid_size[0] * grid_size[1] * grid_size[2];
  
  std::vector<float> grid_offset = 
    {grid_range_x[0], grid_range_y[0], grid_range_z[0]};

  std::vector<Eigen::Vector3f> shifted_points_xyz(num_points);
  std::vector<Eigen::Vector3f> voxel_xyz(num_points);
  std::vector<Eigen::Vector3f> voxel_centers(num_points);
  std::vector<Eigen::Vector3i> voxel_coords(num_points);
  
  std::vector<bool> voxel_paddings(num_points);

  std::vector<int> points_per_voxel(num_voxels, 0);
  std::vector<Eigen::Vector3f> voxel_xyz_sum(num_voxels, Eigen::Vector3f(0,0,0));
  
  for(size_t i = 0; i< num_points; ++i){
    const auto& pt = points_xyz->points.at(i);
    Eigen::Vector3f spt(pt.x - grid_offset[0], 
        pt.y - grid_offset[1], pt.z - grid_offset[2]);
    shifted_points_xyz[i] = spt;
    Eigen::Vector3f xyz = Eigen::Vector3f(
        spt[0] / voxel_size_x, spt[1] / voxel_size_y, spt[2] / voxel_size_z);
    voxel_xyz[i] = xyz;
    voxel_coords[i] = Eigen::Vector3i(xyz[0], xyz[1], xyz[2]);
    voxel_paddings[i] = points_mask[i] < 1 
                        || voxel_coords[i][0] >= grid_size[0]
                        || voxel_coords[i][1] >= grid_size[1]
                        || voxel_coords[i][2] >= grid_size[2]
                        || voxel_coords[i][0] < 0
                        || voxel_coords[i][1] < 0
                        || voxel_coords[i][2] < 0;
    /// w*h*z + w*y + x               
    voxel_indices[i] = grid_size[0] * grid_size[1] * voxel_coords[i][2]
               + grid_size[0] * voxel_coords[i][1] + voxel_coords[i][0];
    if(voxel_paddings[i]){
      voxel_coords[i] = Eigen::Vector3i(0, 0, 0);
      voxel_xyz[i] = Eigen::Vector3f(0., 0., 0.);
      voxel_indices[i] = 0;
    }
    
    voxel_centers[i]<<
      (0.5 + float(voxel_coords[i][0])) * voxel_size_x + grid_offset[0],
      (0.5 + float(voxel_coords[i][1])) * voxel_size_y + grid_offset[1],
      (0.5 + float(voxel_coords[i][2])) * voxel_size_z + grid_offset[2];
    
    pts_to_voxel_centers[i] = Eigen::Vector3f(pt.x, pt.y, pt.z) - voxel_centers[i];
    points_per_voxel[voxel_indices[i]]++;
    voxel_xyz_sum[voxel_indices[i]] += Eigen::Vector3f(pt.x, pt.y, pt.z);
  }
  
  std::vector<Eigen::Vector3f> voxel_centroids(num_voxels);
  
  
  for(size_t i = 0; i < num_voxels; ++i){
    voxel_centroids[i] =  voxel_xyz_sum[i] / points_per_voxel[i];
  }
  for(size_t  i = 0; i < num_points; ++i){
    voxel_point_count[i] = points_per_voxel[voxel_indices[i]];
    point_centroids[i] = voxel_centroids[voxel_indices[i]];

    const auto& pt_raw = points_xyz->points.at(i);
    local_points_xyz[i] << pt_raw.x - point_centroids[i][0],
                           pt_raw.y - point_centroids[i][1],
                           pt_raw.z - point_centroids[i][2];
  }
}

int main(){
  std::stringstream lidar_data_path;
  lidar_data_path << "../000000.bin";

  const size_t pts_num = 122480;
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr padded_cloud(new pcl::PointCloud<PointType>());
  *cloud = read_lidar_data(lidar_data_path.str());
  
  padded_cloud->points.resize(pts_num);
  std::vector<bool> points_mask(pts_num, true);
  if(cloud->size() < pts_num){
    std::copy(cloud->points.begin(), cloud->points.end(), padded_cloud->points.begin());
    for(int i = cloud->size(); i < pts_num; ++i){
      points_mask[i] = false;
    }
  }else{
    std::copy(cloud->points.begin(), cloud->points.begin()+pts_num, padded_cloud->points.begin());
  }
  
  const size_t dim = 16;
  const size_t input_size = pts_num * dim;
  float input[input_size];
  
  Eigen::Vector3f grid_range_x(-35., 35., 0.5);
  Eigen::Vector3f grid_range_y(-20., 20., 0.5); 
  Eigen::Vector3f grid_range_z(-10., 10., 20.);
  
  std::vector<Eigen::Vector3f> local_points_xyz = {};
  std::vector<Eigen::Vector3f> point_centroids = {};
  std::vector<Eigen::Vector3f> pts_to_voxel_centers = {};
  std::vector<size_t> voxel_point_count = {};
  std::vector<size_t> voxel_indices = {};
  points_to_voxels(padded_cloud, points_mask, 
        grid_range_x, grid_range_y, grid_range_z, 
        local_points_xyz, point_centroids, pts_to_voxel_centers, 
        voxel_point_count, voxel_indices);
  
  for(size_t i = 0; i < pts_num; ++i){
    // 0-3
    input[i * dim] =  padded_cloud->points[i].x;
    input[i * dim + 1] =  padded_cloud->points[i].y;
    input[i * dim + 2] =  padded_cloud->points[i].z;
    input[i * dim + 3] =  padded_cloud->points[i].intensity;
    // 4
    input[i * dim + 4] =  voxel_point_count[i];
    // 5-7
    input[i * dim + 5] =  local_points_xyz[i][0];
    input[i * dim + 6] =  local_points_xyz[i][1];
    input[i * dim + 7] =  local_points_xyz[i][2];
    // 8-10
    input[i * dim + 8] =  point_centroids[i][0];
    input[i * dim + 9] =  point_centroids[i][1];
    input[i * dim + 10] =  point_centroids[i][2];
    // 11-13
    input[i * dim + 11] =  pts_to_voxel_centers[i][0];
    input[i * dim + 12] =  pts_to_voxel_centers[i][1];
    input[i * dim + 13] =  pts_to_voxel_centers[i][2];
    // 14
    input[i * dim + 14] =  voxel_indices[i];
    // 15
    input[i * dim + 15] =  points_mask[i];
  }

  torch::DeviceType device_type;
  // device_type = torch::kCUDA;
  device_type = torch::kCPU;
  torch::Device device(device_type);
  std::cout<<"cuda support:"<< (torch::cuda::is_available()?"ture":"false")<<std::endl;

  torch::jit::script::Module module = torch::jit::load("../s2s_kitti.pt");
  std::cout<<"Load model succeed.\n";
  // module.to(torch::kCUDA);
  module.eval();

  // torch::Tensor tester  = torch::from_blob(input, {1, pts_num, dim}, torch::kFloat).to(device);
  torch::Tensor tester  = torch::from_blob(input, {1, pts_num, dim}, torch::kFloat);

  std::cout<<"Read blob data succeed.\n";
  double desc_gen_time = 0;
  for (int i=0; i<1000; i++){
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    torch::Tensor result = module.forward({tester}).toTensor();
    
    // result = result.to(torch::kCPU);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    desc_gen_time += (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
  }
  std::cout << "Processing time per frame = " << desc_gen_time/1000.0 << " sec" <<std::endl;
  return 0;
}
