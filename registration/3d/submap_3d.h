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

#ifndef CARTOGRAPHER_MAPPING_3D_SUBMAP_3D_H_
#define CARTOGRAPHER_MAPPING_3D_SUBMAP_3D_H_

#include <memory>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "port.h"
#include "hybrid_grid.h"
#include "range_data_inserter_3d.h"
#include "id.h"
#include "submaps.h"
#include "range_data.h"
#include "rigid_transform.h"
#include "transform.h"

// #include "cartographer/mapping/2d/probability_grid.h"
#include <opencv2/opencv.hpp>
#include "2d/probability_grid.h"

namespace cartographer {
namespace mapping {


// Added by wz: For test 2d submap matching.
// ProbabilityGrid To2DProbabilityGrid(const HybridGrid* hybrid_grid);

// Project a HybridGrid to horizontal plane, return the grid in cv format 
// and the coordinate of the left-top pixel in the submap's frame.
cv::Mat ProjectToCvMat(const HybridGrid* hybrid_grid,
    const transform::Rigid3d& transform,
    double& ox, double& oy, double& resolution);

ProbabilityGrid ProjectToGrid(const HybridGrid* hybrid_grid,
    const transform::Rigid3d& transform,
    double& ox, double& oy, double& resolution);

class Submap3D : public Submap {
 public:
  Submap3D(float high_resolution, float low_resolution,
           const transform::Rigid3d& local_submap_pose);

  const HybridGrid& high_resolution_hybrid_grid() const {
    return *high_resolution_hybrid_grid_;
  }
  const HybridGrid& low_resolution_hybrid_grid() const {
    return *low_resolution_hybrid_grid_;
  }

  // Insert 'range_data' into this submap using 'range_data_inserter'. The
  // submap must not be finished yet.
  void InsertRangeData(const sensor::RangeData& range_data,
                       const RangeDataInserter3D& range_data_inserter,
                       int high_resolution_max_range);
  void Finish();
 private:
  std::unique_ptr<HybridGrid> high_resolution_hybrid_grid_;
  std::unique_ptr<HybridGrid> low_resolution_hybrid_grid_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_3D_SUBMAP_3D_H_
