/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "submap_3d.h"

#include <cmath>
#include <limits>

#include "carto_math.h"
#include "range_data.h"
#include "probability_values.h"
#include "glog/logging.h"


namespace cartographer {
namespace mapping {
namespace {
 
struct PixelData {
  int min_z = INT_MAX;
  int max_z = INT_MIN;
  int count = 0;
  float probability_sum = 0.f;
  float max_probability = 0.5f;
};

// Filters 'range_data', retaining only the returns that have no more than
// 'max_range' distance from the origin. Removes misses and reflectivity
// information.
sensor::RangeData FilterRangeDataByMaxRange(const sensor::RangeData& range_data,
                                            const float max_range) {
  sensor::RangeData result{range_data.origin, {}, {}};
  for (const Eigen::Vector3f& hit : range_data.returns) {
    if ((hit - range_data.origin).norm() <= max_range) {
      result.returns.push_back(hit);
    }
  }
  return result;
}

std::vector<PixelData> AccumulatePixelData(
    const int width, const int height, const Eigen::Array2i& min_index,
    const Eigen::Array2i& max_index,
    const std::vector<Eigen::Array4i>& voxel_indices_and_probabilities) {
  std::vector<PixelData> accumulated_pixel_data(width * height);
  for (const Eigen::Array4i& voxel_index_and_probability :
       voxel_indices_and_probabilities) {
    const Eigen::Array2i pixel_index = voxel_index_and_probability.head<2>();
    if ((pixel_index < min_index).any() || (pixel_index > max_index).any()) {
      // Out of bounds. This could happen because of floating point inaccuracy.
      continue;
    }
    const int x = max_index.x() - pixel_index[0];
    const int y = max_index.y() - pixel_index[1];
    PixelData& pixel = accumulated_pixel_data[x * width + y];
    ++pixel.count;
    pixel.min_z = std::min(pixel.min_z, voxel_index_and_probability[2]);
    pixel.max_z = std::max(pixel.max_z, voxel_index_and_probability[2]);
    const float probability =
        ValueToProbability(voxel_index_and_probability[3]);
    pixel.probability_sum += probability;
    pixel.max_probability = std::max(pixel.max_probability, probability);
  }
  return accumulated_pixel_data;
}

// The first three entries of each returned value are a cell_index and the
// last is the corresponding probability value. We batch them together like
// this to only have one vector and have better cache locality.
std::vector<Eigen::Array4i> ExtractVoxelData(
    const HybridGrid& hybrid_grid, const transform::Rigid3f& transform,
    Eigen::Array2i* min_index, Eigen::Array2i* max_index) {
  std::vector<Eigen::Array4i> voxel_indices_and_probabilities;
  const float resolution_inverse = 1.f / hybrid_grid.resolution();

  constexpr float kXrayObstructedCellProbabilityLimit = 0.501f;
  for (auto it = HybridGrid::Iterator(hybrid_grid); !it.Done(); it.Next()) {
    const uint16 probability_value = it.GetValue();
    const float probability = ValueToProbability(probability_value);
    if (probability < kXrayObstructedCellProbabilityLimit) {
      // We ignore non-obstructed cells.
      continue;
    }

    const Eigen::Vector3f cell_center_submap =
        hybrid_grid.GetCenterOfCell(it.GetCellIndex());
    const Eigen::Vector3f cell_center_global = transform * cell_center_submap;
    const Eigen::Array4i voxel_index_and_probability(
        common::RoundToInt(cell_center_global.x() * resolution_inverse),
        common::RoundToInt(cell_center_global.y() * resolution_inverse),
        common::RoundToInt(cell_center_global.z() * resolution_inverse),
        probability_value);

    voxel_indices_and_probabilities.push_back(voxel_index_and_probability);
    const Eigen::Array2i pixel_index = voxel_index_and_probability.head<2>();
    *min_index = min_index->cwiseMin(pixel_index);
    *max_index = max_index->cwiseMax(pixel_index);
  }
  return voxel_indices_and_probabilities;
}

// Builds texture data containing interleaved value and alpha for the
// visualization from 'accumulated_pixel_data'.
std::string ComputePixelValues(
    const std::vector<PixelData>& accumulated_pixel_data) {
  std::string cell_data;
  cell_data.reserve(2 * accumulated_pixel_data.size());
  constexpr float kMinZDifference = 3.f;
  constexpr float kFreeSpaceWeight = 0.15f;
  for (const PixelData& pixel : accumulated_pixel_data) {
    // TODO(whess): Take into account submap rotation.
    // TODO(whess): Document the approach and make it more independent from the
    // chosen resolution.
    const float z_difference = pixel.count > 0 ? pixel.max_z - pixel.min_z : 0;
    if (z_difference < kMinZDifference) {
      cell_data.push_back(0);  // value
      cell_data.push_back(0);  // alpha
      continue;
    }
    const float free_space = std::max(z_difference - pixel.count, 0.f);
    const float free_space_weight = kFreeSpaceWeight * free_space;
    const float total_weight = pixel.count + free_space_weight;
    const float free_space_probability = 1.f - pixel.max_probability;
    const float average_probability = ClampProbability(
        (pixel.probability_sum + free_space_probability * free_space_weight) /
        total_weight);
    const int delta = 128 - ProbabilityToLogOddsInteger(average_probability);
    const uint8 alpha = delta > 0 ? 0 : -delta;
    const uint8 value = delta > 0 ? delta : 0;
    cell_data.push_back(value);                         // value
    cell_data.push_back((value || alpha) ? alpha : 1);  // alpha
  }
  return cell_data;
}


}  // namespace


Submap3D::Submap3D(const float high_resolution, const float low_resolution,
                   const transform::Rigid3d& local_submap_pose)
    : Submap(local_submap_pose),
      high_resolution_hybrid_grid_(
          common::make_unique<HybridGrid>(high_resolution)),
      low_resolution_hybrid_grid_(
          common::make_unique<HybridGrid>(low_resolution)) {}


void Submap3D::InsertRangeData(const sensor::RangeData& range_data,
                               const RangeDataInserter3D& range_data_inserter,
                               const int high_resolution_max_range) {
  CHECK(!finished());
  // HybridGrid的原点就是submap的原点，submap以第一帧作为参考系
  // local_pose是第一帧到local frame的转换矩阵
  const sensor::RangeData transformed_range_data = sensor::TransformRangeData(
      range_data, local_pose().inverse().cast<float>());
  range_data_inserter.Insert(
      FilterRangeDataByMaxRange(transformed_range_data,
                                high_resolution_max_range),
      high_resolution_hybrid_grid_.get());
  range_data_inserter.Insert(transformed_range_data,
                             low_resolution_hybrid_grid_.get());
  set_num_range_data(num_range_data() + 1);
}

void Submap3D::Finish() {
  CHECK(!finished());
  set_finished(true);
}


/*****************************************************************************/
/* 
ProbabilityGrid To2DProbabilityGrid(const HybridGrid* hybrid_grid){
  const float resolution = hybrid_grid->resolution();

  // Compute a bounding box for the texture.
  Eigen::Array2i min_index(INT_MAX, INT_MAX);
  Eigen::Array2i max_index(INT_MIN, INT_MIN);
  const std::vector<Eigen::Array4i> voxel_indices_and_probabilities =
      ExtractVoxelData(*hybrid_grid, local_pose().cast<float>(),
                       &min_index, &max_index);
  const int width = max_index.x() - min_index.x() + 1;
  const int height = max_index.y() - min_index.y() + 1;

  // Initialize the 2D probability grid  with computed geometry characteristics.
  CellLimits cell_limits(width, height);//check this
  MapLimits map_limits(
    resolution, Eigen::Vector2d(
      max_index.x()*resolution, max_index.y()*resolution), cell_limits);
  ProbabilityGrid prob_grid = ProbabilityGrid(map_limits);
  
  // Update grid's probabilities
  std::vector<PixelData> accumulated_pixel_data(width * height);
  for (const Eigen::Array4i& voxel_index_and_probability :
       voxel_indices_and_probabilities) {
    const Eigen::Array2i pixel_index = voxel_index_and_probability.head<2>();
    if ((pixel_index < min_index).any() || (pixel_index > max_index).any()) {
      // Out of bounds. This could happen because of floating point inaccuracy.
      continue;
    }
    const int x = pixel_index[0] - min_index.x();
    const int y = pixel_index[1] - min_index.y();
    PixelData& pixel = accumulated_pixel_data[y * width + x];
    ++pixel.count;
    pixel.min_z = std::min(pixel.min_z, voxel_index_and_probability[2]);
    pixel.max_z = std::max(pixel.max_z, voxel_index_and_probability[2]);
    const float probability =
        ValueToProbability(voxel_index_and_probability[3]);
    pixel.probability_sum += probability;
    pixel.max_probability = std::max(pixel.max_probability, probability);
  }
  
  for(int k = 0; k < accumulated_pixel_data.size(); ++k){
    int ix = k % width;
    int iy = k / width;
    Eigen::Array2i cell_index;
    cell_index << ix, iy;
    prob_grid.SetProbability(
      cell_index, accumulated_pixel_data.at(k).probability_sum);
  }
  return prob_grid;
} */

cv::Mat ProjectToCvMat(const HybridGrid* hybrid_grid,
    const transform::Rigid3d& transform,        
    double& ox, double& oy, double& resolution){
  resolution = hybrid_grid->resolution();
  transform::Rigid3f gravity_aligned = transform::Rigid3d::Rotation(
      transform.rotation()).cast<float>();
  // 去除全局yaw角影响
  double yaw = transform::GetYaw(transform);
  auto inv_yaw_rot = transform::Embed3D(transform::Rigid2d::Rotation(-yaw));
  gravity_aligned = inv_yaw_rot.cast<float>() * gravity_aligned;

  // Compute a bounding box for the texture.
  Eigen::Array3i min_index(INT_MAX, INT_MAX, INT_MAX);
  Eigen::Array3i max_index(INT_MIN, INT_MIN, INT_MIN);
  
  // TODO(wz): redundant codes
  std::vector<Eigen::Array4i> voxel_indices_and_probabilities;
  const float resolution_inverse = 1.f / resolution;
  constexpr float kXrayObstructedCellProbabilityLimit = 0.501f;
  for (auto it = HybridGrid::Iterator(*hybrid_grid); 
      !it.Done(); it.Next()) {
    const uint16 probability_value = it.GetValue();
    const float probability = ValueToProbability(probability_value);
    if (probability < kXrayObstructedCellProbabilityLimit) {
      // We ignore non-obstructed cells.
      continue;
    }
    // transform to gravity aligned
    const Eigen::Vector3f cell_center_submap =
        hybrid_grid->GetCenterOfCell(it.GetCellIndex());
    const Eigen::Vector3f cell_center_aligned 
        = gravity_aligned * cell_center_submap;
    const Eigen::Array4i voxel_index_and_probability(
        common::RoundToInt(cell_center_aligned.x() * resolution_inverse),
        common::RoundToInt(cell_center_aligned.y() * resolution_inverse),
        common::RoundToInt(cell_center_aligned.z() * resolution_inverse),
        probability_value);

    voxel_indices_and_probabilities.push_back(voxel_index_and_probability);
    const Eigen::Array3i pixel_index = voxel_index_and_probability.head<3>();
    min_index = min_index.cwiseMin(pixel_index);
    max_index = max_index.cwiseMax(pixel_index);
  }
  //////////////////////////
  ox = min_index.x() * resolution;
  oy = min_index.y() * resolution;

  const int width = max_index.x() - min_index.x() + 1;
  const int height = max_index.y() - min_index.y() + 1;
  cv::Mat result = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
  
  // Update grid's probabilities
  std::vector<PixelData> accumulated_pixel_data(width * height);
  for (const Eigen::Array4i& voxel_index_and_probability :
       voxel_indices_and_probabilities) {
    const Eigen::Array2i pixel_index = voxel_index_and_probability.head<2>();
    if ((pixel_index < min_index.head<2>()).any() 
      || (pixel_index > max_index.head<2>()).any()) {
      // Out of bounds. This could happen because of floating point inaccuracy.
      continue;
    }
    // comforming to cv mat's convention
    const int x = pixel_index[0] - min_index.x();
    const int y = pixel_index[1] - min_index.y();
    PixelData& pixel = accumulated_pixel_data[y * width + x];
    ++pixel.count;
    pixel.min_z = std::min(pixel.min_z, voxel_index_and_probability[2]);
    pixel.max_z = std::max(pixel.max_z, voxel_index_and_probability[2]);
    const float probability =
        ValueToProbability(voxel_index_and_probability[3]);
    pixel.probability_sum += probability;
    pixel.max_probability = std::max(pixel.max_probability, probability);
  }

  for(int k = 0; k < accumulated_pixel_data.size(); ++k){
    int ix = k % width;
    int iy = k / width;
    auto probability = accumulated_pixel_data.at(k).probability_sum;
    int cell_value = common::RoundToInt((probability - kMinProbability) 
      * (255.f / (kMaxProbability - kMinProbability)));
    if(probability > kMaxProbability){
      cell_value = 0;
    }else{
      cell_value = 255;
    }
    result.at<uchar>(iy, ix) = cell_value;
  }
  return result; 
}

ProbabilityGrid ProjectToGrid(const HybridGrid* hybrid_grid,
    const transform::Rigid3d& transform,        
    double& ox, double& oy, double& resolution){
  resolution = hybrid_grid->resolution();
  
  transform::Rigid3f gravity_aligned = transform::Rigid3d::Rotation(
      transform.rotation()).cast<float>();

  // 去除全局yaw角影响
  double yaw = transform::GetYaw(transform);
  auto inv_yaw_rot = transform::Embed3D(transform::Rigid2d::Rotation(-yaw));
  gravity_aligned = inv_yaw_rot.cast<float>() * gravity_aligned;

  // LOG(INFO)<<gravity_aligned;

  // Compute a bounding box for the texture.
  Eigen::Array3i min_index(INT_MAX, INT_MAX, INT_MAX);
  Eigen::Array3i max_index(INT_MIN, INT_MIN, INT_MIN);
  
  // TODO(wz): redundant codes
  std::vector<Eigen::Array4i> voxel_indices_and_probabilities;
  const float resolution_inverse = 1.f / resolution;
  constexpr float kXrayObstructedCellProbabilityLimit = 0.501f;
  for (auto it = HybridGrid::Iterator(*hybrid_grid); 
      !it.Done(); it.Next()) {
    const uint16 probability_value = it.GetValue();
    const float probability = ValueToProbability(probability_value);
    if (probability < kXrayObstructedCellProbabilityLimit) {
      // We ignore non-obstructed cells.
      continue;
    }
    // transform to gravity aligned
    const Eigen::Vector3f cell_center_submap =
        hybrid_grid->GetCenterOfCell(it.GetCellIndex());
    const Eigen::Vector3f cell_center_aligned 
        = gravity_aligned * cell_center_submap;
    const Eigen::Array4i voxel_index_and_probability(
        common::RoundToInt(cell_center_aligned.x() * resolution_inverse),
        common::RoundToInt(cell_center_aligned.y() * resolution_inverse),
        common::RoundToInt(cell_center_aligned.z() * resolution_inverse),
        probability_value);

    voxel_indices_and_probabilities.push_back(voxel_index_and_probability);
    const Eigen::Array3i pixel_index = voxel_index_and_probability.head<3>();
    min_index = min_index.cwiseMin(pixel_index);
    max_index = max_index.cwiseMax(pixel_index);
  }
  //////////////////////////
  ox = min_index.x() * resolution;
  oy = min_index.y() * resolution;
  double max_x = max_index.x() * resolution;
  double max_y = max_index.y() * resolution;
  const int width = max_index.x() - min_index.x() + 1;
  const int height = max_index.y() - min_index.y() + 1;
  // cv::Mat result = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
  
  CellLimits climit(width, height);
  MapLimits mlimit(resolution, Eigen::Vector2d(max_x, max_y), climit);
  ProbabilityGrid prob_grid_2d(mlimit);
  prob_grid_2d.SetOrigin(ox, oy);

  // Update grid's probabilities
  std::vector<PixelData> accumulated_pixel_data(width * height);
  for (const Eigen::Array4i& voxel_index_and_probability :
       voxel_indices_and_probabilities) {
    const Eigen::Array2i pixel_index = voxel_index_and_probability.head<2>();
    if ((pixel_index < min_index.head<2>()).any() 
      || (pixel_index > max_index.head<2>()).any()) {
      // Out of bounds. This could happen because of floating point inaccuracy.
      continue;
    }
    // comforming to cv mat's convention
    const int x = pixel_index[0] - min_index.x();
    const int y = pixel_index[1] - min_index.y();
    PixelData& pixel = accumulated_pixel_data[y * width + x];
    ++pixel.count;
    pixel.min_z = std::min(pixel.min_z, voxel_index_and_probability[2]);
    pixel.max_z = std::max(pixel.max_z, voxel_index_and_probability[2]);
    const float probability =
        ValueToProbability(voxel_index_and_probability[3]);
    pixel.probability_sum += probability;
    pixel.max_probability = std::max(pixel.max_probability, probability);
  }

  for(int k = 0; k < accumulated_pixel_data.size(); ++k){
    int ix = k % width;
    int iy = k / width;
    auto probability = accumulated_pixel_data.at(k).probability_sum;
    int cell_value = common::RoundToInt((probability - kMinProbability) 
      * (255.f / (kMaxProbability - kMinProbability)));
    // result.at<uchar>(iy, ix) = cell_value;
    if(probability < kMaxProbability){
      prob_grid_2d.SetProbability(Eigen::Vector2i(ix, iy), 0);
    }else{
      prob_grid_2d.SetProbability(Eigen::Vector2i(ix, iy), probability);
    }
  }
  return prob_grid_2d; 
}

}  // namespace mapping
}  // namespace cartographer
