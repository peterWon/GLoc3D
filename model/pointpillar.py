import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
import pytorch3d.transforms.rotation_conversions as rot_covert
from model.voxel import points_to_voxels

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class PoseLoss(nn.Module):
  def __init__(self, angle_scale_factor=1):
    super(PoseLoss, self).__init__()
    self.angle_scale_factor = angle_scale_factor

  def forward(self, pred, gt):
    '''
    pred: Nx6 [angleaxis, translation]
    gt: Nx6 [angleaxis, translation]
    '''
    N = gt.shape[0]
    pred_rot = rot_covert.axis_angle_to_matrix(pred[:,:3])
    gt_rot = rot_covert.axis_angle_to_matrix(gt[:,:3])
    pred_rot = pred_rot.to(torch.float32)
    gt_rot = gt_rot.to(torch.float32)
    gt = gt.to(torch.float32)

    lossr = 0
    losst = 0
    for i in range(N):
      p_rot = pred_rot[i]
      g_rot = gt_rot[i]
      dr = torch.mm(g_rot.t(), p_rot)
     
      dt = (pred[i, 3:] - gt[i, 3:]).unsqueeze(0).t()
      
      dt = torch.mm(g_rot.t(), dt)
      dr = rot_covert.matrix_to_axis_angle(dr)
      
      lossr += torch.sqrt(dr.pow(2).sum())
      losst += torch.sqrt(dt.pow(2).sum())
    
    # print(lossr.shape)
    # print(losst.shape)
    lossr /= N
    losst /= N
    print(lossr, losst)
    return self.angle_scale_factor * lossr + losst

class PoseLayer(nn.Module):
  def __init__(self):
    super(PoseLayer, self).__init__()
    
  def forward(self, encodingQ, encodingP):
    N, C, H, W = encodingQ.shape
    # print(encodingQ.shape)
    # print(encodingP.shape)
    input = torch.cat([encodingQ, encodingP], 1)
    # print(input.shape)
    # layers = []
    # layers.append(nn.Conv2d(C+C, 128, kernel_size=3, stride=1))
    # layers.append(nn.BatchNorm2d(128))
    # layers.append(nn.ReLU(inplace=True))
    # layers.append(nn.Linear(128, 6))
    # layers = nn.Sequential(*layers)
    # layers.to(input.device)
    # out = layers(input)
    # layers.append(nn.Conv2d(C+C, 128, kernel_size=3, stride=2))
    # layers.append(nn.BatchNorm2d(128))
    # layers.append(nn.ReLU(inplace=True))
    # layers.append(nn.Conv2d(128, 6 * N, kernel_size=3, stride=1))
    out = nn.Conv2d(C+C, 128, kernel_size=3, stride=2, device=input.device)(input)
    out = nn.BatchNorm2d(128, device=input.device)(out)
    out = nn.ReLU(inplace=True)(out)
    out = out.view(N, -1, 128)
    out = nn.Linear(128, 6, device=input.device)(out)
    # print(out.shape)
    out = out.mean(1)
    # print(out.shape)
    return out

class PillarBlock(nn.Module):
  def __init__(self, idims=64, dims=64, num_layers=1,
               stride=1):
    super(PillarBlock, self).__init__()
    layers = []
    self.idims = idims
    self.stride = stride
    for i in range(num_layers):
      layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                              padding=1, bias=False))
      layers.append(nn.BatchNorm2d(dims))
      layers.append(nn.ReLU(inplace=True))
      self.idims = dims
      self.stride = 1
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)


class PointNet(nn.Module):
  def __init__(self, idims=64, odims=64):
    super(PointNet, self).__init__()
    self.pointnet = nn.Sequential(
      nn.Conv1d(idims, odims, kernel_size=1, bias=False),
      nn.BatchNorm1d(odims),
      nn.ReLU(inplace=True)
    )

  def forward(self, points_feature, points_mask):
    batch_size, num_points, num_dims = points_feature.shape
    points_feature = points_feature.permute(0, 2, 1)
    mask = points_mask.view(batch_size, 1, num_points)
    return self.pointnet(points_feature) * mask


class PointPillar(nn.Module):
  def __init__(self, C, xbound, ybound, zbound, embedded_dim=16, direction_dim=37, cluster_mode = False, pose_mode = False, vlad_mode = False):
    super(PointPillar, self).__init__()
    self.xbound = xbound
    self.ybound = ybound
    self.zbound = zbound
    self.embedded_dim = embedded_dim
    self.cluster_mode = cluster_mode
    self.pose_mode = pose_mode
    self.vlad_mode = vlad_mode
    self.l2norm = None
    if cluster_mode:
      self.l2norm = L2Norm()
    self.pn = PointNet(14, 64)
    self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
    self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
    self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
    self.up1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      # nn.ReLU(inplace=True),
      # nn.Conv2d(128, C, 1),
    )

    self.conv_out_pose = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
    )
    
    self.layers = [self.pn, self.block1, self.block2, self.block3, self.up1, self.up2, self.up3]

  def forward(self, points, points_mask):
    # print(points.shape,points_mask.shape), ([4, 81920, 5]) ([4, 81920])
    points_xyz = points[:, :, :3] # xyz
    points_feature = points[:, :, 3:] # intensity, dt
    
    voxels = points_to_voxels(
      points_xyz, points_mask, self.xbound, self.ybound, self.zbound
    )
    # wz: 输入PointNet的每一个point的feature为15维的
    points_feature = torch.cat(
      [points, # 4
       torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
       voxels['local_points_xyz'], # 3
       voxels['point_centroids'], # 3
       points_xyz - voxels['voxel_centers'], # 3
      ], dim=-1
    )
    # print('before pointnet: ', points_feature.shape),torch.Size([4, 81920, 15])
    points_feature = self.pn(points_feature, voxels['points_mask'])
    # print('after pointnet: ', points_feature.shape),torch.Size([4, 64, 81920])

    # 取对应index的元素作平均
    voxel_feature = torch_scatter.scatter_mean(
      points_feature,
      torch.unsqueeze(voxels['voxel_indices'], dim=1),
      dim=2,
      dim_size=voxels['num_voxels'])
    batch_size = points.size(0)
    
    # print('1', voxel_feature.shape)
    voxel_feature = voxel_feature.view(
      batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
    # print('2', voxel_feature.shape) [(4, 64, 400, 200)]
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature2 = self.block2(voxel_feature1)
    voxel_feature3 = self.block3(voxel_feature2)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.up2(voxel_feature2)
    voxel_feature3 = self.up3(voxel_feature3)
    # print(voxel_feature1.shape)
    # print(voxel_feature2.shape)
    # print(voxel_feature3.shape)
    voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
    # print('3', voxel_feature.shape), [(4, 448, 400, 200)]

    # if self.cluster:

    # return self.conv_out(voxel_feature).transpose(3, 2), self.instance_conv_out(voxel_feature).transpose(3, 2), self.direction_conv_out(voxel_feature).transpose(3, 2)
    if self.cluster_mode:
      return self.l2norm(self.conv_out(voxel_feature).transpose(3, 2))
    elif self.pose_mode:
      return self.conv_out_pose(voxel_feature).transpose(3, 2)
    elif self.vlad_mode:
      return self.conv_out(voxel_feature).transpose(3, 2)
    else:
      return self.conv_out(voxel_feature).transpose(3, 2),\
             self.conv_out_pose(voxel_feature).transpose(3, 2)
    
  def fix_backbone_weights(self):
    for l in self.layers:
      for p in l.parameters():
        p.requires_grad = False
        
  def fix_pose_out_weights(self):
    for p in self.conv_out_pose.parameters():
      p.requires_grad = False

  def fix_vlad_out_weights(self):
    for p in self.conv_out.parameters():
      p.requires_grad = False

class PointPillarEncoder(nn.Module):
  def __init__(self, C, xbound, ybound, zbound):
    super(PointPillarEncoder, self).__init__()
    self.xbound = xbound
    self.ybound = ybound
    self.zbound = zbound
    self.pn = PointNet(15, 64)
    self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
    self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
    self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
    self.up1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, C, 1),
    )

  def forward(self, points, points_mask):
    points_xyz = points[:, :, :3]
    points_feature = points[:, :, 3:]
    voxels = points_to_voxels(
      points_xyz, points_mask, self.xbound, self.ybound, self.zbound
    )
    points_feature = torch.cat(
      [points, # 5
       torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
       voxels['local_points_xyz'], # 3
       voxels['point_centroids'], # 3
       points_xyz - voxels['voxel_centers'], # 3
      ], dim=-1
    )
    points_feature = self.pn(points_feature, voxels['points_mask'])
    voxel_feature = torch_scatter.scatter_mean(
      points_feature,
      torch.unsqueeze(voxels['voxel_indices'], dim=1),
      dim=2,
      dim_size=voxels['num_voxels'])
    batch_size = points.size(0)
    voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature2 = self.block2(voxel_feature1)
    voxel_feature3 = self.block3(voxel_feature2)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.up2(voxel_feature2)
    voxel_feature3 = self.up3(voxel_feature3)
    voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
    return self.conv_out(voxel_feature).transpose(3, 2)
