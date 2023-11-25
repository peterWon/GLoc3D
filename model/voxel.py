import numpy as np
import torch
import torch_scatter


def pad_or_trim_to_np(x, shape, pad_val=0):
  shape = np.asarray(shape)
  pad = shape - np.minimum(np.shape(x), shape)
  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:shape[0], :shape[1]]


def raval_index(coords, dims):
    dims = torch.cat((dims, torch.ones(1, device=dims.device)), dim=0)[1:]
    dims = torch.flip(dims, dims=[0])
    dims = torch.cumprod(dims, dim=0) / dims[0]
    multiplier = torch.flip(dims, dims=[0])
    indices = torch.sum(coords * multiplier, dim=1)
    return indices

# points_mask是个啥形状？
def points_to_voxels(
  points_xyz,
  points_mask,
  grid_range_x,
  grid_range_y,
  grid_range_z
):
    batch_size, num_points, _ = points_xyz.shape
    
    # print(grid_range_x)
    # print(grid_range_y)
    # print(grid_range_z)
    # 3D Voxel的几何尺寸*
    voxel_size_x = grid_range_x[2]
    voxel_size_y = grid_range_y[2]
    voxel_size_z = grid_range_z[2]
    
    # 兴趣区内的3D Voxel的数量
    grid_size = np.asarray([
        (grid_range_x[1]-grid_range_x[0]) / voxel_size_x,
        (grid_range_y[1]-grid_range_y[0]) / voxel_size_y,
        (grid_range_z[1]-grid_range_z[0]) / voxel_size_z
    ]).astype('int32')
    voxel_size = np.asarray([voxel_size_x, voxel_size_y, voxel_size_z])
    voxel_size = torch.Tensor(voxel_size).to(points_xyz.device)
    num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
    grid_offset = torch.Tensor([grid_range_x[0], grid_range_y[0], grid_range_z[0]]).to(points_xyz.device)
    
    shifted_points_xyz = points_xyz - grid_offset
    voxel_xyz = shifted_points_xyz / voxel_size 
   
    # 表示在第几个voxel
    voxel_coords = voxel_xyz.int()
    
    grid_size = torch.from_numpy(grid_size).to(points_xyz.device)
    grid_size = grid_size.int()
    zeros = torch.zeros_like(grid_size)
    
    # |表示或
    voxel_paddings = ((points_mask < 1.0) |
                      torch.any((voxel_coords >= grid_size) |
                                (voxel_coords < zeros), dim=-1))

    # 大小与points_xyz一致，标识该point属于哪一个voxel                            
    voxel_indices = raval_index(
      torch.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
    voxel_indices = torch.reshape(voxel_indices, [batch_size, num_points])
    voxel_indices = torch.where(voxel_paddings,
                                torch.zeros_like(voxel_indices),
                                voxel_indices)
    voxel_centers = ((0.5 + voxel_coords.float()) * voxel_size + grid_offset)
    voxel_coords = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                               torch.zeros_like(voxel_coords),
                               voxel_coords)
    voxel_xyz = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                            torch.zeros_like(voxel_xyz),
                            voxel_xyz)
    voxel_paddings = voxel_paddings.float()

    voxel_indices = voxel_indices.long()
    points_per_voxel = torch_scatter.scatter_sum(
        torch.ones((batch_size, num_points), dtype=voxel_coords.dtype, device=voxel_coords.device) * (1-voxel_paddings),
        voxel_indices,
        dim=1,
        dim_size=num_voxels
    )
    
    # print(points_per_voxel.shape) (batch_size, 11200)
    
    voxel_point_count = torch.gather(points_per_voxel,
                                     dim=1,
                                     index=voxel_indices)
    # print(voxel_point_count.shape) (batch_size, point_num)

    voxel_centroids = torch_scatter.scatter_mean(
        points_xyz,
        voxel_indices,
        dim=1,
        dim_size=num_voxels)

    # 每个点对应的中心，shape与points_xyz一致
    point_centroids = torch.gather(voxel_centroids, dim=1, index=torch.unsqueeze(voxel_indices, dim=-1).repeat(1, 1, 3))
    local_points_xyz = points_xyz - point_centroids
    
    # print(num_points, local_points_xyz.shape, shifted_points_xyz.shape)
    # print(num_points, point_centroids.shape, points_xyz.shape)
    # print(grid_offset.shape, voxel_coords.shape, voxel_centers.shape)
    # print(voxel_indices.shape, voxel_paddings.shape)
    # print(num_voxels, grid_size )
    # print(voxel_point_count.shape, points_per_voxel.shape)
    result = {
        'local_points_xyz': local_points_xyz,
        'shifted_points_xyz': shifted_points_xyz,
        'point_centroids': point_centroids,
        'points_xyz': points_xyz,
        'grid_offset': grid_offset,
        'voxel_coords': voxel_coords,
        'voxel_centers': voxel_centers,
        'voxel_indices': voxel_indices,
        'voxel_paddings': voxel_paddings,
        'points_mask': 1 - voxel_paddings,
        'num_voxels': num_voxels,
        'grid_size': grid_size,
        'voxel_xyz': voxel_xyz,
        'voxel_size': voxel_size,
        'voxel_point_count': voxel_point_count,
        'points_per_voxel': points_per_voxel
    }


    return result
