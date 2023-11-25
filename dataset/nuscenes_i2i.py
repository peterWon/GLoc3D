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

import matplotlib.pyplot as plt
import scipy.spatial.transform as sst
import scipy.interpolate


from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

import dataset.i2i_util as i2i_data

def Log(msg):
  print(msg+' ,File: "'+__file__+'", Line '+str(sys._getframe().f_lineno)+' , in '+sys._getframe().f_code.co_name)

root_dir = '/home/wz/Data/sda1/data/NuScenes-full/extracted/'
if not exists(root_dir):
  raise FileNotFoundError('root_dir is hardcoded, \
    please adjust to point to NuScenes dataset')

struct_dir = join(root_dir, 'grid_to_grid/')
queries_dir = root_dir

data_interface = i2i_data.I2IDataInterface(
    'NUSCENES', root_dir, queries_dir, struct_dir)

def get_whole_training_set(onlyDB=False):
  return data_interface.get_whole_training_set(onlyDB)

def get_whole_val_set():
  return data_interface.get_whole_val_set()

def get_whole_test_set():
  return data_interface.get_whole_test_set()

def get_whole_val_qp_set(sample_level, transform):
  return data_interface.get_whole_val_qp_pair_set(sample_level, transform)

def get_training_query_set(margin=0.1):
  return data_interface.get_training_query_set(margin)
  
def get_training_query_pose_set():
  return data_interface.get_training_query_pose_set()
  
def get_val_query_set():
  return data_interface.get_val_query_set()

# ---------------------------------------------------------------------------- #
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset

import os
import numpy as np
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points
  
class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, is_train):
        super(HDMapNetDataset, self).__init__()
        self.is_train = is_train
        
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.scenes = self.get_scenes(version, is_train)
        
        print(self.scenes)
        
        # 用了前4个blob
        # self.subset_scenes = []
        # with open(os.path.join(root_dir, 'subset_scenes.txt')) as ff:
        #   self.subset_scenes = ff.readlines()[0][:-1].split(', ')
        # filtered_scenes = [scene for scene in self.scenes if scene in self.subset_scenes]
        # self.scenes = filtered_scenes
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        # split = {
        #     'v1.0-trainval': {True: 'train', False: 'val'},
        #     'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        # }[version][is_train]

        
        scenes = [sc['name'] for sc in self.nusc.scene if self.nusc.get('log',sc['log_token'])['location']=='singapore-onenorth']
        return scenes

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_lidar(self, rec):
        # lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        # lidar_data = lidar_data.transpose(1, 0)
        # num_points = lidar_data.shape[0]
        
        # return lidar_data
        sample_data_token = rec['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        return current_sd_rec['filename']

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        # car_trans = ego_pose['translation']
        # pos_rotation = Quaternion(ego_pose['rotation'])
        # yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        # return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)
        return ego_pose

    def __getitem__(self, idx):
        rec = self.samples[idx]
        lidar_filename = self.get_lidar(rec)
        ego_pose = self.get_ego_pose(rec)

        return lidar_filename, ego_pose

  
def generate_struct_files(dataset_type, skip_frames = 5, dist_threshold = 20):
  dataset = HDMapNetDataset(version='v1.0-trainval', dataroot=root_dir, \
      is_train=dataset_type=='train')

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
  
  # print(dataset.__len__())
  # print(len(dataset.scenes))
  for i in range(dataset.__len__()):
    lidar_filename, ego_pose = dataset.__getitem__(i)
    T_tmp = np.eye(4, 4)
    T_tmp[:3, :3] = sst.Rotation.from_quat(ego_pose['rotation']).as_matrix()
    t = np.array(ego_pose['translation'])    
    T_tmp[:3, 3] = t
    db_pose_all.append(T_tmp)
    db_utm_all.append(np.array([T_tmp[0, 3], T_tmp[1, 3]]))
    
    # db_lidar_all.append(lidar_filename)
    # print(db_utm_all[-1], lidar_filename)
    basename = os.path.basename(lidar_filename)
    basedir = os.path.dirname(lidar_filename)
    basedir = basedir.replace('LIDAR_TOP', 'prob_img')
    db_lidar_all.append(os.path.join(basedir, basename[:-4]+'.jpg'))
        
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
      ofile.writelines(join(root_dir, pc_path) + "\n")
    
    positives, distances = val_set.getPositivesWithDistances()
    q_p_pairs = []
    for qIdx, pos in enumerate(positives):
      # ofile.write(str(qIdx)+':')
      for db_idx, dist in zip(pos, distances[qIdx]): 
        if sample_level=='easy':
          if dist > 5.: continue
        elif sample_level=='medium':
          if dist < 5. or dist > 10: continue
        elif sample_level=='hard':
          if dist < 10. or dist > 15: continue
        else:
          raise RuntimeError
        q_p_pairs.append((qIdx, db_idx))
        
      n_rows = len(q_p_pairs)
      q_p_pairs = random.sample(q_p_pairs, min(100, n_rows)) # filter test samples
      # save in older format. 
      last_qid = -1
      for i in range(len(q_p_pairs)):
        current_qid = q_p_pairs[i][0]
        db_idx = q_p_pairs[i][1]
        if current_qid != last_qid:
          last_qid = current_qid
          if i != 0: ofile.write("\n")
          ofile.write(str(last_qid)+':')
        
        ofile.write(str(db_idx)+" ")
      
  # write query pose and database pose
  # caution: currently, the query and db should belong to the same sequence.
  with open(pose_file, 'w') as ofile:
    for pose in val_set.dbStruct.dbPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #x,y,z,w
      ofile.write(
        str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(
        str(pose[0,3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    for pose in val_set.dbStruct.qPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #x,y,z,w
      ofile.write(
        str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(
        str(pose[0, 3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    
  print('Save file to {} and {} succeed!'.format(index_file, pose_file))
  
if __name__ == '__main__':
  # write_valset_to_txt(os.path.join(struct_dir, 'i2i_val_5_easy.txt'),
  #   os.path.join(struct_dir, 'i2i_val_5_poses_easy.txt'), 
  #   sample_level='easy')
  generate_struct_files('val', 5, 20)
  
  # val_dataset = get_whole_val_set()
  # plt.scatter(val_dataset.dbStruct.utmDb[:,0], val_dataset.dbStruct.utmDb[:,1], color='deepskyblue',linewidths=1)
  # # plt.scatter(val_dataset.dbStruct.utmQ[:,0], val_dataset.dbStruct.utmQ[:,1], color='red',linewidths=0.5)
  # plt.show()