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
import pykitti

def Log(msg):
  print(msg+' ,File: "'+__file__+'", Line '+str(sys._getframe().f_lineno)+' , in '+sys._getframe().f_code.co_name)

root_dir = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/'
if not exists(root_dir):
  raise FileNotFoundError('root_dir is hardcoded, \
    please adjust to point to KittiLidar dataset')

odometry_dir = "/home/wz/Data/sdb1/wz/Data/kitti-odometry/dataset/"
struct_dir = join(root_dir, 'vlad_pose_dataset/')
queries_dir = root_dir

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
                

def get_odom_sequence_poses():
  sequence_poses = {}
  for seq, raw in odom_raw_map.items():
      pose_i = []
      pose_file = join(odometry_dir, 'poses', seq + ".txt")
      calib_file = join(odometry_dir, 'sequences', seq, "calib.txt")
      
      Tr = None
      Tr_inv = None
      with open(calib_file, 'r') as cf:
        _, Tr_line = cf.readlines()[-1][:-1].split(": ")
        Tr_items = Tr_line.split(' ')
        Tr_items = list(Tr_items) + ['0', '0', '0', '1']
        Tr = np.array(Tr_items, dtype = np.float32).reshape(4, 4)
        Tr_inv = np.linalg.inv(Tr)

      with open(pose_file, 'r') as pf:
        for line in pf.readlines():
          pp = line[:-1].split(' ')
          pp = list(pp) + ['0', '0', '0', '1']
          pose_cam = np.array(pp, dtype = np.float32).reshape(4, 4)
          pose_tmp = np.matmul(Tr_inv, pose_cam) 
          pose_lidar = np.matmul(pose_tmp, Tr) 
          
          pose_i.append(pose_lidar)
      
      sequence_poses[raw[0]] = pose_i
  return sequence_poses

def get_whole_training_set(onlyDB=False):
  structFile = join(struct_dir, 'kitti_s2s_pose_train_5.mat')
  return WholeDatasetFromStruct(
    structFile, input_transform = None, onlyDB = onlyDB)

def get_whole_val_set():
  structFile = join(struct_dir, 'kitti_s2s_pose_val_5.mat')
  return WholeDatasetFromStruct(structFile, input_transform = None)

def get_whole_test_set():
  structFile = join(struct_dir, 'kitti_s2s_pose_val_5.mat')
  return WholeDatasetFromStruct(
      structFile, input_transform = None)

def get_training_query_set(margin=0.1):
  structFile = join(struct_dir, 'kitti_s2s_pose_train_5.mat')
  return QueryDatasetFromStruct(
      structFile, input_transform = None, margin=margin)

def get_training_query_pose_set():
  structFile = join(struct_dir, 'kitti_s2s_pose_train_5.mat')
  return QueryDatasetFromStruct(
      structFile, input_transform = None)

def get_val_query_set():
  structFile = join(struct_dir, 'kitti_s2s_pose_val_5.mat')
  return QueryDatasetFromStruct(
      structFile, input_transform = None)

def write_valset_to_txt(index_file, pose_file):
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
    for qIdx, pos in enumerate(val_set.getPositives()):
      ofile.write(str(qIdx)+':')
      for db_idx in pos: 
        ofile.write(str(db_idx)+" ")
      ofile.write("\n")
      
  # write query pose and database pose
  # caution: currently, the query and db should belong to the same sequence.
  with open(pose_file, 'w') as ofile:
    for pose in val_set.dbStruct.dbPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #z,y,z,w
      ofile.write(str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(str(pose[0,3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    for pose in val_set.dbStruct.qPose:
      q = sst.Rotation.from_matrix(pose[:3,:3]).as_quat() #z,y,z,w
      ofile.write(str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ')
      ofile.write(str(pose[0, 3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')
    
  print('Save file to {} and {} succeed!'.format(index_file, pose_file))

def write_valset_to_superglue_txt(file_path):
  '''
    save the val set to txt file for superglue evaluation
  '''
  val_set = get_whole_val_set()
  with open(file_path, 'w') as ofile:
    # ofile.writelines(str(val_set.dbStruct.numDb)+' '+str(val_set.dbStruct.numQ)+"\n")
    # for pc_path in val_set.scans:
    #   ofile.writelines(join(root_dir, pc_path) + "\n")
    num_db = val_set.dbStruct.numDb
    for qIdx, pos in enumerate(val_set.getPositives()):
      for db_idx in pos: 
        name0_bin = val_set.dbStruct.qLidar[qIdx]
        name1_bin = val_set.dbStruct.dbLidar[db_idx]
        name0 = name0_bin.replace('velodyne_points/data', 'grid')
        name1 = name1_bin.replace('velodyne_points/data', 'grid')
        name0 = name0.replace('bin', 'jpg')
        name1 = name1.replace('bin', 'jpg')
        ofile.write( name0 + ' ' + name1 + "\n")
  print('Save file to {} succeed!'.format(file_path))

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbLidar', 'utmDb', 'dbPose', 'qLidar', 'utmQ', 'qPose', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
  mat = loadmat(path)

  matStruct = mat['dbStruct'][0]
  
  dataset = 'kittilidarpose'
  
  whichSet = matStruct[0].item()

  dbLidar = [f for f in matStruct[1]] 
  utmDb = [xy for xy in matStruct[2]]
  dbPose = [T for T in matStruct[3]]  
  utmDb = np.array(utmDb)
  dbPose = np.array(dbPose)

  qLidar = [f for f in matStruct[4]]
  utmQ = [xy for xy in matStruct[5]] 
  qPose = [T for T in matStruct[6]]
  utmQ = np.array(utmQ)
  qPose = np.array(qPose)
  
  numDb = matStruct[7].item() #Db的数量
  numQ = matStruct[8].item()  #query的数量

  posDistThr = matStruct[9].item()
  posDistSqThr = matStruct[10].item()
  nonTrivPosDistSqThr = matStruct[11].item()

  print(whichSet, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)

  return dbStruct(whichSet, dataset, dbLidar, utmDb, dbPose, qLidar, 
          utmQ, qPose, numDb, numQ, posDistThr, 
          posDistSqThr, nonTrivPosDistSqThr)
    
class WholeDatasetFromStruct(data.Dataset):
  def __init__(self, structFile, input_transform=None, onlyDB=False):
    super().__init__()

    self.input_transform = input_transform

    self.dbStruct = parse_dbStruct(structFile)
    self.scans = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbLidar]

    if not onlyDB:
      self.scans += [join(queries_dir, qIm) for qIm in self.dbStruct.qLidar]

    self.whichSet = self.dbStruct.whichSet
    self.dataset = self.dbStruct.dataset
      
    self.positives = None
    self.distances = None

  def __getitem__(self, index):
    lidar_data = np.fromfile(
      self.scans[index], dtype=np.float32).reshape((-1, 4))
    num_points = lidar_data.shape[0]
    lidar_data = pad_or_trim_to_np(
      lidar_data, [122480, 4]).astype('float32')
    lidar_mask = np.ones(122480).astype('float32')
    lidar_mask[num_points:] *= 0.0
    return lidar_data, lidar_mask, index

  def __len__(self):
    return len(self.scans)

  def getPositives(self):
    # positives for evaluation are those within trivial threshold range
    # fit NN to find them, search by radius
    print('self.dbStruct.posDistThr ', self.dbStruct.posDistThr)
    if  self.positives is None:
      knn = NearestNeighbors(n_jobs=1)
      knn.fit(self.dbStruct.utmDb)
      # tree = spatial.KDTree(self.dbStruct.utmDb) # may be faster
      # self.positives = tree.query_ball_point(
      #   x=self.dbStruct.utmQ, 
      #   r=[self.dbStruct.posDistThr]*self.dbStruct.numQ)

      self.distances, self.positives = knn.radius_neighbors(
          self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)
      
        # TODO(wz): what if there are empty loops in the database
        # self.positives = np.where(np.array([len(x) for x in self.positives])>0)[0]
      empty_num = 0
      for x in self.positives:
        if len(x) == 0:
          empty_num+=1 
      print('query frames with no positives: {}'.format(empty_num))
    return self.positives


        
def collate_fn(batch):
  """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
  
  Args:
      data: list of tuple (query, positive, negatives). 
          - query: torch tensor of shape (3, h, w).
          - positive: torch tensor of shape (3, h, w).
          - negative: torch tensor of shape (n, 3, h, w).
  Returns:
      query: torch tensor of shape (batch_size, 3, h, w).
      positive: torch tensor of shape (batch_size, 3, h, w).
      negatives: torch tensor of shape (batch_size, n, 3, h, w).
  """

  batch = list(filter (lambda x:x is not None, batch))
  if len(batch) == 0: return None, None, None, None, None, None, None, None
  
  q_data, q_mask, pos_data, pos_mask, neg_data, neg_mask, indices = zip(*batch)

  q_data = data.dataloader.default_collate(q_data)
  q_mask = data.dataloader.default_collate(q_mask)
  pos_data = data.dataloader.default_collate(pos_data)
  pos_mask = data.dataloader.default_collate(pos_mask)
  
  negCounts = data.dataloader.default_collate([x.shape[0] for x in neg_data])
  neg_data = torch.cat(neg_data, 0)
  neg_mask = torch.cat(neg_mask, 0)
  
  import itertools
  indices = list(itertools.chain(*indices))

  return q_data, q_mask, pos_data, pos_mask, \
         neg_data, neg_mask, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
  def __init__(self, structFile, nNegSample=1000, nNeg=10, \
      margin=0.1, input_transform=None):
    super().__init__()

    self.input_transform = None
    self.margin = margin

    self.dbStruct = parse_dbStruct(structFile)
    self.whichSet = self.dbStruct.whichSet
    self.dataset = self.dbStruct.dataset
    self.nNegSample = nNegSample # number of negatives to randomly sample
    self.nNeg = nNeg # number of negatives used for training

    # potential positives are those within nontrivial threshold range
    #fit NN to find them, search by radius
    knn = NearestNeighbors(n_jobs=1)
    knn.fit(self.dbStruct.utmDb)
    
    # TODO use sqeuclidean as metric?
    self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
            radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
            return_distance=False))
    
    # 将每个查询帧的回环帧ID进行排序
    # radius returns unsorted, sort once now so we dont have to later
    for i, posi in enumerate(self.nontrivial_positives):
      self.nontrivial_positives[i] = np.sort(posi)

    # its possible some queries don't have any non trivial potential positives
    # lets filter those out+
    # 删掉在数据库中没有查询到足够的回环帧的查询帧
    self.queries = np.where(
      np.array([len(x) for x in self.nontrivial_positives])>0)[0]

    # potential negatives are those outside of posDistThr range
    potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
            radius=self.dbStruct.posDistThr, 
            return_distance=False)

    self.potential_negatives = []
    for pos in potential_positives:
      self.potential_negatives.append(np.setdiff1d(
        np.arange(self.dbStruct.numDb), pos, assume_unique=True))

    self.cache = None # filepath of HDF5 containing feature vectors for scans

    self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]
    

  def __getitem__(self, index):
    index = self.queries[index] # re-map index to match dataset
    with h5py.File(self.cache, mode='r') as h5: 
      h5feat = h5.get("features")
      
      # h5feat中前面存的是数据库的特征，后面才是查询库的特征
      qOffset = self.dbStruct.numDb
      qFeat = h5feat[index+qOffset]

      # 获取数据库中候选回环帧的特征
      posFeat = h5feat[self.nontrivial_positives[index].tolist()]

      knn = NearestNeighbors(n_jobs=1) # TODO replace with faiss?
      knn.fit(posFeat)

      # 以当前查询帧在后续回环帧中找特征最接近的回环帧及其对应索引
      dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)
      dPos = dPos.item()
      posIndex = self.nontrivial_positives[index][posNN[0]].item()

      # 从负样本中随机采样nNegSample个
      negSample = np.random.choice(
        self.potential_negatives[index], self.nNegSample)
      negSample = np.unique(np.concatenate(
        [self.negCache[index], negSample])).astype(np.int32)
      
      negFeat = h5feat[negSample.tolist()]
      knn.fit(negFeat)

      # to quote netvlad paper code: 10x is hacky but fine
      dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10) 
      dNeg = dNeg.reshape(-1)
      negNN = negNN.reshape(-1)

      # try to find negatives that are within margin, if there aren't any return none
      # 最近的负样本要离正样本足够远（margin）
      violatingNeg = dNeg < dPos + self.margin**0.5
      
      # 如果没有违反规则的负样本，则该batch对tripleloss没有作用
      if np.sum(violatingNeg) < 1:
        #if none are violating then skip this query
        return None

      negNN = negNN[violatingNeg][:self.nNeg]
      negIndices = negSample[negNN].astype(np.int32)

      self.negCache[index] = negIndices
    
    query = np.fromfile(join(queries_dir, \
      self.dbStruct.qLidar[index]), dtype=np.float32).reshape((-1, 4))
    positive = np.fromfile(join(root_dir, \
      self.dbStruct.dbLidar[posIndex]), dtype=np.float32).reshape((-1, 4))
    
    np_q = query.shape[0]
    q_data = pad_or_trim_to_np(query, [122480, 4]).astype('float32')
    q_mask = np.ones(122480).astype('float32')
    q_mask[np_q:] *= 0.0

    np_p = positive.shape[0]
    p_data = pad_or_trim_to_np(positive, [122480, 4]).astype('float32')
    p_mask = np.ones(122480).astype('float32')
    p_mask[np_p:] *= 0.0
    

    negatives = []
    negatives_mask = []
    for negIndex in negIndices:
      negative = np.fromfile(join(root_dir, \
        self.dbStruct.dbLidar[negIndex]), dtype=np.float32).reshape((-1, 4))
      np_n = negative.shape[0]
      n_data = pad_or_trim_to_np(negative, [122480, 4]).astype('float32')
      n_mask = np.ones(122480).astype('float32')
      n_mask[np_n:] *= 0.0
      n_data = torch.from_numpy(n_data)
      n_mask = torch.from_numpy(n_mask)
      negatives.append(n_data)
      negatives_mask.append(n_mask)
    
    negatives = torch.stack(negatives, 0)
    negatives_mask = torch.stack(negatives_mask, 0)
    
    return q_data, q_mask, p_data, p_mask, \
      negatives, negatives_mask, [index, posIndex]+negIndices.tolist()

  def __len__(self):
    return len(self.queries)

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
    db_lidar_all.extend([path.split(root_dir)[1] for path in raw.velo_files])
      
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
  
  savemat(os.path.join(root_dir, \
    'vlad_pose_dataset', 'kitti_s2s_pose_{}_{}.mat'.format(
      dataset_type, str(skip_frames))), mdict)

def eval_sequence_overlap():
  def overlap(box1, box2):
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
      return False
    else:
      return True
  base_raw_dir = '/home/wz/dev-sdb/wz/Data/kitti-raw/raw/2011_09_30/'
  odom_raw_map = {'00': '2011_10_03_drive_0027',
                '01': '2011_10_03_drive_0042',
                '02': '2011_10_03_drive_0034',
                # '03': '2011_09_26_drive_0067',
                '04': '2011_09_30_drive_0016',
                '05': '2011_09_30_drive_0018',
                '06': '2011_09_30_drive_0020',
                '07': '2011_09_30_drive_0027',
                # '08': '2011_09_30_drive_0028',//////////////////
                '09': '2011_09_30_drive_0033',
                '10': '2011_09_30_drive_0034'}
  
  sequences = ["00", "01","02","04","05","06","07","08","09","10"]

  intersected = False
  boxes = []
  for seq in sequences:
    gt_oxts_dir = os.path.join(
        base_raw_dir, odom_raw_map[seq]+'_sync', 'oxts', 'data')
    locations = []
    for name in os.listdir(os.path.join(
        base_raw_dir, odom_raw_map[seq]+'_sync', 'velodyne_points/data')):
      gt_oxts_file = os.path.join(gt_oxts_dir, name[:-4]+'.txt')
      
      with open(gt_oxts_file, 'r') as oxts:
        ss = oxts.readlines()[0].split(' ')  
        lat = float(ss[0]) * 180. / np.pi
        lon = float(ss[1]) * 180. / np.pi
        alt = float(ss[2])
      X, Y, Z = blh_to_enu(lat, lon, alt)
      locations.append(np.array([X, Y]))
    locations = np.array(locations)
    
    min_x, min_y = np.min(locations, axis=0)
    max_x, max_y = np.max(locations, axis=0)
    
    current_box = [min_x, min_y, max_x, max_y]
    boxes.append(current_box) 
    
  for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
      if(overlap(boxes[i], boxes[j])):
        print(sequences[i], " intersects ", sequences[j])
  # 07  intersects  08
  # 09  intersects  10


def view_dataset_split_trajectory(val_dataset):
  plt.scatter(val_dataset.dbStruct.utmDb[:,0], val_dataset.dbStruct.utmDb[:,1], color='deepskyblue',linewidths=1)
  plt.scatter(val_dataset.dbStruct.utmQ[:,0], val_dataset.dbStruct.utmQ[:,1], color='red',linewidths=0.5)
      
  plt.show()
