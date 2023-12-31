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

#include "range_data_inserter_3d.h"

#include "Eigen/Core"
#include "probability_values.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace {

void InsertMissesIntoGrid(const std::vector<uint16>& miss_table,
                          const Eigen::Vector3f& origin,
                          const sensor::PointCloud& returns,
                          HybridGrid* hybrid_grid,
                          const int num_free_space_voxels) {
  const Eigen::Array3i origin_cell = hybrid_grid->GetCellIndex(origin);
  for (const Eigen::Vector3f& hit : returns) {
    const Eigen::Array3i hit_cell = hybrid_grid->GetCellIndex(hit);

    const Eigen::Array3i delta = hit_cell - origin_cell;
    const int num_samples = delta.cwiseAbs().maxCoeff();
    CHECK_LT(num_samples, 1 << 15);
    // 'num_samples' is the number of samples we equi-distantly place on the
    // line between 'origin' and 'hit'. (including a fractional part for sub-
    // voxels) It is chosen so that between two samples we change from one voxel
    // to the next on the fastest changing dimension.
    //
    // Only the last 'num_free_space_voxels' are updated for performance.
    for (int position = std::max(0, num_samples - num_free_space_voxels);
         position < num_samples; ++position) {
      const Eigen::Array3i miss_cell =
          origin_cell + delta * position / num_samples;
      hybrid_grid->ApplyLookupTable(miss_cell, miss_table);
    }
  }
}

}  // namespace


RangeDataInserter3D::RangeDataInserter3D()
    : hit_table_(
          ComputeLookupTableToApplyOdds(Odds(0.55))),
      miss_table_(
          ComputeLookupTableToApplyOdds(Odds(0.49))) {}

void RangeDataInserter3D::Insert(const sensor::RangeData& range_data,
                                 HybridGrid* hybrid_grid) const {
  CHECK_NOTNULL(hybrid_grid);

  for (const Eigen::Vector3f& hit : range_data.returns) {
    const Eigen::Array3i hit_cell = hybrid_grid->GetCellIndex(hit);
    hybrid_grid->ApplyLookupTable(hit_cell, hit_table_);
  }

  // By not starting a new update after hits are inserted, we give hits priority
  // (i.e. no hits will be ignored because of a miss in the same cell).
  InsertMissesIntoGrid(miss_table_, range_data.origin, range_data.returns,
                       hybrid_grid, 2);
  hybrid_grid->FinishUpdate();
}

}  // namespace mapping
}  // namespace cartographer
