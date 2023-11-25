import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import sys
from os.path import join, exists

from PIL import Image
import numpy as np
from random import randint, random
from collections import namedtuple
from scipy.io import loadmat
from scipy.io import savemat
from scipy import spatial

from sklearn.neighbors import NearestNeighbors
import h5py
import random

import matplotlib.pyplot as plt
import scipy.spatial.transform as sst
import pykitti
import cv2

from model.voxel import pad_or_trim_to_np
import dataset.i2i_util as i2i_data

def Log(msg):
  print(msg+' ,File: "'+__file__+'", Line '+str(sys._getframe().f_lineno)+' , in '+sys._getframe().f_code.co_name)

root_dir = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/'
if not exists(root_dir):
  raise FileNotFoundError('root_dir is hardcoded, \
    please adjust to point to KittiLidar dataset')

odometry_dir = "/home/wz/Data/sdb1/wz/Data/kitti-odometry/dataset/"
struct_dir = join(root_dir, 'grid_to_grid/')
queries_dir = root_dir

data_interface = i2i_data.I2IDataInterface(
    'KITTI', root_dir, queries_dir, struct_dir)

# 2011_09_26_drive_0067 is missing
# 07 and 08 have an overlap, 09 and 10 have an overlap
odom_raw_map = {'00': ('2011_10_03_drive_0027', "000000", "004540"),
                '01': ('2011_10_03_drive_0042', "000000", "001100"),
                '02': ('2011_10_03_drive_0034', "000000", "004660"),
                # '03': ('2011_09_26_drive_0067', "000000", "000800"),
                '04': ('2011_09_30_drive_0016', "000000", "000270"),
                '05': ('2011_09_30_drive_0018', "000000", "002760"),
                '06': ('2011_09_30_drive_0020', "000000", "001100"),
                '07': ('2011_09_30_drive_0027', "000000", "001100"),
                '08': ('2011_09_30_drive_0028', "001100", "005170"),
                '09': ('2011_09_30_drive_0033', "000000", "001590"),
                '10': ('2011_09_30_drive_0034', "000000", "001200")}

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
      pc_path = pc_path.replace('prob_img', 'velodyne_points/data')
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

def generate_struct_files(dataset_type, skip_frames = 5, dist_threshold = 20):
  # for trainset
  sequences = ["00","01","02","04","05","06","07","10"] 
  
  # for valset
  if dataset_type == 'val':
    sequences = ["08", "09"]

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
    seq_full_name = odom_raw_map[seq][0]
    date, drive = seq_full_name.split('_drive_')
    start = int(odom_raw_map[seq][1])
    end = int(odom_raw_map[seq][2])
    
    frames_odometry = range(0, end - start + 1, skip_frames)
    frames_raw = range(start, end + 1, skip_frames)
    raw = pykitti.raw(root_dir, date, drive, frames=frames_raw)
    odometry = pykitti.odometry(odometry_dir, seq, frames=frames_odometry)
    
    assert len(odometry.velo_files) == len(raw.velo_files)
    
    # using pose info from raw
    # T_velo_imu = raw.calib.T_velo_imu
    # T_imu_velo = np.linalg.inv(T_velo_imu)

    # using pose info from odometry
    T_c0_v = odometry.calib.T_cam0_velo
    T_v_c0 = np.linalg.inv(T_c0_v)
    
    # transform to velodyne frame
    db_pose_all.extend([(T_v_c0.dot(T)).dot(T_c0_v) for T in odometry.poses])
    # db_pose_all.extend([(T_velo_imu.dot(T)).dot(T_imu_velo) for T in odometry.poses])
    db_utm_all.extend([gps.T_w_imu[:2, 3] for gps in raw.oxts])
    
    # comform to probability image
    for path in raw.velo_files:
      rel_path = path.split(root_dir)[1]
      rel_path = rel_path.replace('velodyne_points/data', 'prob_img')
      db_lidar_all.append(rel_path[:-4]+'.jpg')  
      
  assert len(db_utm_all) == len(db_lidar_all)

  # randomly select query frames from db
  tmp_num_db = len(db_lidar_all)
  tmp_num_q = int(tmp_num_db * 0.2)
  q_index = np.random.choice(range(tmp_num_db),tmp_num_q,replace=False,p=None)
  for idx in q_index:
    q_utm.append(db_utm_all[idx])
    q_lidar.append(db_lidar_all[idx])
    q_pose.append(db_pose_all[idx])
  
  for idx in range(tmp_num_db):
    if idx not in q_index:
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
  
  savemat(os.path.join(struct_dir, 'i2i_pose_{}_{}.mat'.format(
      dataset_type, str(skip_frames))), mdict)