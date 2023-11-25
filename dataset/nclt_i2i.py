import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import sys
from os.path import join, exists


import numpy as np
from random import randint, random
from collections import namedtuple
from scipy.io import loadmat
from scipy.io import savemat
from scipy import spatial

from sklearn.neighbors import NearestNeighbors
import h5py
import random
from model.voxel import pad_or_trim_to_np

import matplotlib.pyplot as plt
import scipy.spatial.transform as sst
import scipy.interpolate

import struct

import dataset.i2i_util as i2i_data

root_dir = '/home/wz/Data/sda1/data/NCLT/'
if not exists(root_dir):
  raise FileNotFoundError('root_dir is hardcoded, \
    please adjust to point to NCLT dataset')

struct_dir = join(root_dir, 'grid_to_grid/')
queries_dir = root_dir

data_interface = i2i_data.I2IDataInterface(
    'NCLT', root_dir, queries_dir, struct_dir)

def get_whole_training_set(onlyDB=False):
  return data_interface.get_whole_training_set(onlyDB)

def get_whole_val_set():
  return data_interface.get_whole_val_set()

def get_whole_test_set():
  return data_interface.get_whole_test_set()

def get_training_query_set(margin=0.1):
  return data_interface.get_training_query_set(margin)
  
def get_training_query_pose_set():
  return data_interface.get_training_query_pose_set()
  
def get_val_query_set():
  return data_interface.get_val_query_set()

def generate_struct_files(dataset_type, skip_frames = 5, dist_threshold = 20):
  def read_rtk(vel_ts, rtk_filename):
    gps = np.loadtxt(rtk_filename, delimiter = ",")
    
    interp = scipy.interpolate.interp1d(
      gps[:, 0], gps[:, 3:6], kind='nearest', axis=0, bounds_error=False)
    gps_interp = interp(vel_ts)
    
    num_sats = gps[:, 2]
    lat = gps_interp[:, 0]
    lng = gps_interp[:, 1]
    alt = gps_interp[:, 2]

    lat0 = lat[0]
    lng0 = lng[0]

    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000 # approx. radius of earth (m)
    x = r * np.cos(lat0) * np.sin(dLng)
    y = r * np.sin(dLat)
    
    return np.stack([x, y]).T.tolist()
  
  def read_gt(t_lidar, gt_filename):
    gt = np.loadtxt(gt_filename, delimiter = ",")
    interp = scipy.interpolate.interp1d(
      gt[:, 0], gt[:, 1:], kind='nearest', axis=0, bounds_error=False)
    pose_gt = interp(t_lidar)
    
    timestamps = gt[:, 0]
    rot = sst.Rotation.from_euler('zyx', pose_gt[:, 0:3], degrees=False)
    pos = pose_gt[:, 3:6]
    T = []
    for r, t in zip(rot, pos):
      T_tmp = np.eye(4, 4)
      T_tmp[:3, :3] = r.as_matrix()
      T_tmp[:3, 3] = t
      T.append(T_tmp)
    return T
 
  # for trainset
  # sequences = ["2012-01-08", "2012-03-25", "2012-08-04", "2012-12-01"] 
  sequences = ["2012-01-08"] 
  
  # for valset
  if dataset_type == 'val':
    sequences = ["2013-04-05"]

  mdict = {}
  whichset = dataset_type
  db_lidar_all = []
  db_pose_all = []
  db_utm_all = []
  db_lidar = []
  db_pose = []
  db_utm = []
  
  q_pose = []
  q_lidar = []
  q_utm = []
  for seq in sequences:
    seq_dir = root_dir+ '/' + seq + '/'
    vel_dir = seq_dir + 'velodyne_sync_xyzi'
    gps_filename = seq_dir + 'sensor' + '/' + 'gps_rtk.csv'
    gt_filename = seq_dir + 'groundtruth_' + seq + '.csv'
    
    vel_fnames = os.listdir(vel_dir)
    
    vel_fnames_filter = [vel_fnames[i] for i in range(0, len(vel_fnames), skip_frames)]
    
    vel_ts = [float(tn[:-4]) for tn in vel_fnames_filter]
    
    T = read_gt(vel_ts, gt_filename)
    db_pose_all.extend(T)
    
    db_utm_all.extend(read_rtk(vel_ts, gps_filename))
    
    db_lidar_all.extend(
      [seq + '/' + 'prob_img' + '/' + fname[:-4]+'.jpg' for fname in vel_fnames_filter])
      
  assert len(db_utm_all) == len(db_lidar_all)

  # randomly select query frames from db
  tmp_num_db = len(db_lidar_all)
  tmp_num_q = int(tmp_num_db * 0.2)
  q_index = np.random.choice(range(tmp_num_db),tmp_num_q,replace=False,p=None)
  for idx in q_index:
    if np.isnan(db_utm_all[idx]).any(): continue
    if np.isinf(db_utm_all[idx]).any(): continue
    q_utm.append(db_utm_all[idx])
    q_lidar.append(db_lidar_all[idx])
    q_pose.append(db_pose_all[idx])
  
  for idx in range(tmp_num_db):
    if idx not in q_index:
      if np.isnan(db_utm_all[idx]).any(): continue
      if np.isinf(db_utm_all[idx]).any(): continue
      db_lidar.append(db_lidar_all[idx])
      db_utm.append(db_utm_all[idx])
      db_pose.append(db_pose_all[idx])
  
  num_db = len(db_utm)
  num_q = len(q_utm)
  pos_dist_thr = dist_threshold
  pos_sq_dist_thr = pos_dist_thr * pos_dist_thr
  non_triv_pos_dist_sq_thr = 100
  
  mdict = {'dbStruct': [whichset, db_lidar, db_utm, db_pose, \
                      q_lidar, q_utm, q_pose, num_db, num_q, \
                      pos_dist_thr, pos_sq_dist_thr, non_triv_pos_dist_sq_thr]}
  
  savemat(os.path.join(root_dir, struct_dir, \
    'i2i_pose_{}_{}.mat'.format(dataset_type, str(skip_frames))), mdict)
  
def write_valset_to_txt(index_file, pose_file, sample_level='easy'):
  '''
    save the val set to txt file for C++ evaluation 
    of LiDAR-iris, ScanContext, and M2DP
  '''
  val_set = get_whole_val_set()
  with open(index_file, 'w') as ofile:
    ofile.writelines(
        str(val_set.dbStruct.numDb) + ' ' + str(val_set.dbStruct.numQ) + "\n")
    for pc_path in val_set.scans:
      pc_path = pc_path.replace('prob_img', 'velodyne_sync_xyzi')
      pc_path = pc_path[:-4]+'.bin'
      ofile.writelines(join(root_dir, pc_path) + "\n")
    
    positives, distances = val_set.getPositivesWithDistances()
    for qIdx, pos in enumerate(positives):
      ofile.write(str(qIdx)+':')
      for db_idx, dist in zip(pos, distances[qIdx]): 
        if sample_level=='easy':
          if dist > 5.: continue
        elif sample_level=='medium':
          if dist < 5. or dist > 10: continue
        elif sample_level=='hard':
          if dist < 10. or dist > 15: continue
        else:
          raise NotImplementedError
        
        ofile.write(str(db_idx)+" ")
      ofile.write("\n")
      
  # write query pose and database pose
  # caution: currently, the query and db should belong to the same sequence.
  with open(pose_file, 'w') as ofile:
    for pose in val_set.dbStruct.dbPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #z,y,z,w
      ofile.write(
        str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(
        str(pose[0,3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    for pose in val_set.dbStruct.qPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #z,y,z,w
      ofile.write(
        str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(
        str(pose[0, 3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    
  print('Save file to {} and {} succeed!'.format(index_file, pose_file))