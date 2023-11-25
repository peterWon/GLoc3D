#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: Generate OT model for Libtorch

# modified from OverlapTransformer
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.s2s_merged import *

test_weights = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/vlad_pose_dataset/checkpoints/backup/model_best.pth.tar'
# # ============================================================================

checkpoint = torch.load(test_weights)

pool = netvlad_fc.NetVLAD(num_clusters=64, dim=128, vladv2=False)
encoder = PointPillarTest(
    10, #data_conf['num_channels'],
    [-35.0, 35.0, 0.5], #data_conf['xbound']
    [-20.0, 20.0, 0.5], #data_conf['ybound']
    [-10.0, 10.0, 20.0], #data_conf['zbound']
    embedded_dim=16,
    cluster_mode=False,
    pose_mode=False,
    vlad_mode=True)
# for KITTI
model = PointPillarVLAD()
model.add_module('encoder', encoder) 
model.add_module('pool', pool) 

model.load_state_dict(checkpoint['state_dict'],strict=False) 
# model.cuda()
model.eval()


# 64 for KITTI
# ([4, 81920, 4]) ([4, 81920])
points, points_mask = torch.rand(1, 122480, 4), torch.rand(1, 122480) 
points_xyz = points[:, :, :3] 
voxels = points_to_voxels(points_xyz, points_mask, \
    encoder.xbound, \
    encoder.ybound, \
    encoder.zbound)
input = torch.cat(
    [points, # 4
    torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
    voxels['local_points_xyz'], # 3
    voxels['point_centroids'], # 3
    points_xyz - voxels['voxel_centers'], # 3
    voxels['voxel_indices'].reshape(points.shape[0], -1, 1),
    points_mask.reshape(points.shape[0], -1, 1),
    ], dim=-1
)
# input = input.cuda()
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, input)
traced_script_module.save("./s2s_kitti.pt")