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
import torchvision.models as models
import model.netvlad_fc as netvlad_fc
import dataset.i2i_util as i2i_dataset


class VGGVLAD(nn.Module):
  def __init__(self):
    super(VGGVLAD, self).__init__()
    
  # add for tracing, should be called after \
  # add_module('encoder', encoder) & add_module('pool', pool)
  def forward(self, input):
    image_encoding = self.encoder(input)
    place_feature = self.pool(image_encoding)
    return place_feature

resume_ckpt = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/grid_to_grid/checkpoints/model_best.pth.tar'
# # ============================================================================

checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)


model = VGGVLAD() 
# vgg16 = models.vgg16(pretrained=False)
# layers = list(vgg16.features.children())[:-2]
# encoder = nn.Sequential(*layers)

encoder = models.vgg16(weights='IMAGENET1K_V1')
# encoder = models.vgg16(weights=VGG16_Weights.DEFAULT)
# capture only feature part and remove last relu and maxpool
layers = list(encoder.features.children())[:-2]
# if using pretrained then only train conv5_1, conv5_2, and conv5_3
for l in layers:             
  for p in l.parameters():
    p.requires_grad = False
encoder = nn.Sequential(*layers)

encoder_dim = 512
num_clusters = 64
pool = netvlad_fc.NetVLAD(
  num_clusters=num_clusters, dim=encoder_dim, vladv2=False, gating=False)

model.add_module('encoder', encoder) 
model.add_module('pool', pool) 

model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()
with torch.no_grad():
  # cope with what defined in dataset/i2i_util.py
  input = torch.ones(
    1, 3, i2i_dataset.INPUT_HEIGHT, i2i_dataset.INPUT_WIDTH,dtype=torch.float32) 
  input = input.cuda()
  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.

  traced_script_module = torch.jit.trace(model, input)
  traced_script_module.save("./i2i_vgg_vlad.pt")