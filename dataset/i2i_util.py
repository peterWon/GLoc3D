import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.nn import functional as F

import math
import os
import sys
from os.path import join

import numpy as np
from collections import namedtuple
from scipy.io import loadmat


from sklearn.neighbors import NearestNeighbors
import h5py

import matplotlib.pyplot as plt
import cv2

# VGG
INPUT_WIDTH = 768
INPUT_HEIGHT = 768

def ToScaledTensor(img):
  # to cope with OpenCV in C++
  img = np.float32(img) * (1. / 255.)
  img = torch.Tensor(img)
  img = img.permute((2,0,1))
  return img

def input_transform():
  return transforms.Compose([
      transforms.ToTensor(),
      # transforms.Normalize(mean=[0.987481, 0.987481, 0.987481],
                          # std=[0.099150725, 0.099150725, 0.099150725]),
      # transforms.CenterCrop(size=(INPUT_WIDTH, INPUT_HEIGHT)),#0.2m分辨率，76.8m
  ])

def normalize():
  return transforms.Compose([
      transforms.Normalize(mean=[0.987481, 0.987481, 0.987481],
                          std=[0.099150725, 0.099150725, 0.099150725]),
  ])

def unnormalize():
  mean = torch.tensor([0.987481, 0.987481, 0.987481], dtype=torch.float32)
  std = torch.tensor(
    [0.099150725, 0.099150725, 0.099150725], dtype=torch.float32)
  return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

def pad_and_crop(img):
  h, w, c = img.shape
  res = np.ones(shape=(INPUT_HEIGHT, INPUT_WIDTH, c), dtype=img.dtype) * 255
  cw = w if w <= INPUT_WIDTH else INPUT_WIDTH
  ch = h if h <= INPUT_HEIGHT else INPUT_HEIGHT
  i_top = int(math.floor((h - ch) / 2.))
  i_left = int(math.floor((w - cw) / 2.))
  o_top = int(math.floor((INPUT_HEIGHT - ch) / 2.))
  o_left = int(math.floor((INPUT_WIDTH - cw) / 2.))
  
  # print(i_top, i_left, h, w)
  # print(o_top, o_left, ch, cw)
  res[o_top:o_top+ch,o_left:o_left+cw,:] = img[i_top:i_top+ch, i_left:i_left+cw,:]

  return res

def padwith255(img):
  # torchvision.transforms's CenterCrop automatively pad the image with 0, which
  # is not what we want for the probability grid.
  w, h, c = img.shape
  if w >= INPUT_WIDTH and h >= INPUT_HEIGHT:
    return img
  if w < INPUT_WIDTH and h < INPUT_HEIGHT:
    grid = np.ones(shape=(INPUT_WIDTH, INPUT_HEIGHT, c), dtype=img.dtype) * 255
    dw = int((INPUT_WIDTH - w) / 2)
    dh = int((INPUT_HEIGHT - h) / 2)
    grid[dw:dw+w, dh:dh+h, :] = img
    return grid
  if w < INPUT_WIDTH and h >= INPUT_HEIGHT:
    grid = np.ones(shape=(INPUT_WIDTH, h, c), dtype=img.dtype) * 255
    dw = int((INPUT_WIDTH - w) / 2)
    grid[dw:dw+w, :, :] = img
    return grid
  if w >= INPUT_WIDTH and h < INPUT_HEIGHT:
    grid = np.ones(shape=(w, INPUT_HEIGHT, c), dtype=img.dtype) * 255
    dh = int((INPUT_HEIGHT - h) / 2)
    grid[:, dh:dh+h, :] = img
    return grid
  return img

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbLidar', 'utmDb', 'dbPose', 'qLidar', 'utmQ', 'qPose', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
  mat = loadmat(path)

  matStruct = mat['dbStruct'][0]
  
  dataset = 'i2i'
  
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
  def __init__(self, name, root_dir, queries_dir, structFile, \
              input_transform=None, onlyDB=False):
    super().__init__()
    
    self.name = name #['KITTI','NCLT','NUSCENES']
    self.root_dir = root_dir
    self.queries_dir = queries_dir
    self.input_transform = input_transform
  
    self.dbStruct = parse_dbStruct(structFile)
    self.scans = [join(self.root_dir, dbIm) for dbIm in self.dbStruct.dbLidar]

    if not onlyDB:
      self.scans += [join(self.queries_dir, qIm) for qIm in self.dbStruct.qLidar]

    self.whichSet = self.dbStruct.whichSet #'i2i'
    self.dataset = self.dbStruct.dataset
      
    self.positives = None
    self.distances = None

  def __getitem__(self, index):
    img = cv2.imread(self.scans[index])   
    img = pad_and_crop(img)
    
    if self.input_transform:
        # img = self.input_transform(img)
        img = ToScaledTensor(img)
    return img, index

  def __len__(self):
    return len(self.scans)
  
  # add for test
  def getPositivesWithDistances(self):
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
    return self.positives, self.distances
  
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

        
class QueryDatasetFromStruct(data.Dataset):
  def __init__(self, name, root_dir, queries_dir, structFile, \
      nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
    super().__init__()
    
    self.name = name
    self.root_dir = root_dir
    self.queries_dir = queries_dir
    
    self.input_transform = input_transform
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
      
      # 如果没有违反规则的负样本（困难负样本），则该batch对tripleloss没有作用
      if np.sum(violatingNeg) < 1:
        #if none are violating then skip this query
        return None

      negNN = negNN[violatingNeg][:self.nNeg]
      negIndices = negSample[negNN].astype(np.int32)

      self.negCache[index] = negIndices
    
    query = cv2.imread(join(self.queries_dir, self.dbStruct.qLidar[index]))
    positive = cv2.imread(join(self.root_dir, self.dbStruct.dbLidar[posIndex]))
    
    query = pad_and_crop(query)
    positive = pad_and_crop(positive)
    
    if self.input_transform:
        # query = self.input_transform(query)
        # positive = self.input_transform(positive)
        query = ToScaledTensor(query)
        positive = ToScaledTensor(positive)

    negatives = []
    for negIndex in negIndices:
      negative = cv2.imread(join(self.root_dir, self.dbStruct.dbLidar[negIndex]))
      negative = pad_and_crop(negative)
    
      if self.input_transform:
          # negative = self.input_transform(negative)
          negative = ToScaledTensor(negative)
      negatives.append(negative)

    negatives = torch.stack(negatives, 0)
    
    return query, positive, negatives, [index, posIndex]+negIndices.tolist()

  def __len__(self):
    return len(self.queries)


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
  if len(batch) == 0: return None, None, None, None, None

  query, positive, negatives, indices = zip(*batch)

  query = data.dataloader.default_collate(query)
  positive = data.dataloader.default_collate(positive)
  negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
  negatives = torch.cat(negatives, 0)
  import itertools
  indices = list(itertools.chain(*indices))
  return query, positive, negatives, negCounts, indices


class I2IDataInterface():
  def __init__(self, name, root_dir, queries_dir, struct_dir, skip_frame=5):
    self.name = name
    self.root_dir = root_dir
    self.queries_dir = queries_dir
    self.struct_dir = struct_dir
    self.skip_frame = skip_frame
    print('using dataset with skip frame: ', self.skip_frame)
    self.val_filename = 'i2i_pose_val_'+str(self.skip_frame)+'.mat'
    self.train_filename = 'i2i_pose_train_'+str(self.skip_frame)+'.mat'
    
  def get_whole_training_set(self, onlyDB=False):
    structFile = join(self.struct_dir, self.train_filename)
    return WholeDatasetFromStruct(name=self.name,root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform(), onlyDB = onlyDB)

  def get_whole_val_set(self):
    structFile = join(self.struct_dir, self.val_filename)
    return WholeDatasetFromStruct(
      name=self.name, root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform())

  def get_whole_test_set(self):
    structFile = join(self.struct_dir, self.val_filename)
    return WholeDatasetFromStruct(
      name=self.name, root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform())

  def get_training_query_set(self, margin=0.1):
    structFile = join(self.struct_dir, self.train_filename)
    return QueryDatasetFromStruct(
      name=self.name, root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform(), margin=margin)
    
  def get_training_query_pose_set(self):
    structFile = join(self.struct_dir, self.train_filename)
    return QueryDatasetFromStruct(
      name=self.name, root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform())

  def get_val_query_set(self):
    structFile = join(self.struct_dir, self.val_filename)
    return QueryDatasetFromStruct(
      name=self.name, root_dir=self.root_dir, \
      queries_dir=self.queries_dir, structFile=structFile, \
      input_transform = input_transform())

def view_dataset_split_trajectory(val_dataset):
  plt.scatter(val_dataset.dbStruct.utmDb[:,0], val_dataset.dbStruct.utmDb[:,1],\
      color='deepskyblue',linewidths=1)
  plt.scatter(val_dataset.dbStruct.utmQ[:,0], val_dataset.dbStruct.utmQ[:,1], \
      color='red',linewidths=0.5)
      
  plt.show()

  
def estimate_transform(src1, src2): 
  surf = cv2.xfeatures2d.SURF_create(400)
  W, H = src1.shape
  center = np.array([W / 2., H / 2.])
  # saved probability grids are in 0~1?
  _, img1 = cv2.threshold(src1, 0.5, 255, cv2.THRESH_BINARY) 
  _, img2 = cv2.threshold(src2, 0.5, 255, cv2.THRESH_BINARY)
  
  img1 = np.uint8(img1)
  img2 = np.uint8(img2)
  
  # find the keypoints and descriptors with SURF
  kp1, des1 = surf.detectAndCompute(img1, None)
  kp2, des2 = surf.detectAndCompute(img2, None)
  
  # create Matcher
  matcher = cv2.FlannBasedMatcher()
  
  # Match descriptors.
  matches = matcher.knnMatch(des1, des2, k=2)
  
  good_matches = []
  pts1 = []
  pts2 = []
  ratio_thresh = 0.7
  
  for i, (m, n) in enumerate(matches):
    if m.distance < ratio_thresh * n.distance:
      good_matches.append(matches[i])
      pts1.append(kp1[m.queryIdx].pt - center) # comply with pytorch convention
      pts2.append(kp2[m.trainIdx].pt - center)
  
  pts1 = np.array(pts1, dtype=np.float32)       
  pts2 = np.array(pts2, dtype=np.float32)       
  if len(good_matches) > 5:
    transform, _ = cv2.estimateAffinePartial2D(pts1, pts2)
    
    if transform.size != 0:
      scale = np.sqrt(transform[0,0]*transform[0,0] 
          + transform[0,1]*transform[0,1])
      if abs(1-scale) < 0.1:
        # _, img1_th = cv2.threshold(src1, 0.5, 255, cv2.THRESH_BINARY_INV)
        # _, img2_th = cv2.threshold(src2, 0.5, 255, cv2.THRESH_BINARY_INV)
        
        # img1_th_trans = cv2.warpAffine(img1_th, transform, img1_th.shape)
        
        # h, w = img1_th.shape
        # num_db = 0
        # num_q = 0
        # num_overlap = 0
        # for r in range(h):
        #   for c in range(w):
        #     if img1_th_trans[r, c] == 255:
        #       num_q = num_q + 1
        #       if img2_th[r, c] == 255:
        #         num_overlap = num_overlap + 1
        # print('prob cv: ', num_overlap / num_q)
        
        # img_12 = cv2.drawMatches(img_t, pts1, img2, pts2)
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches1to2=good_matches,outImg=None)
        # cv2.imshow('match1', img1_th_trans)
        # cv2.imshow('match2', img2_th)
        # cv2.waitKey(0)
        
        
        return transform[0,2], transform[1,2], math.asin(transform[1,0] / scale), transform
  
  return []

def test_match():
  # root_dir = '/home/wz/Data/sda1/data/NCLT/'
  root_dir = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/'
  device = torch.device("cuda" if True else "cpu")
  
  struct_dir = join(root_dir, 'grid_to_grid/')
  queries_dir = root_dir
  cache_dir = join(struct_dir, 'cache')
  val = I2IDataInterface(root_dir, queries_dir, struct_dir)
  ds = val.get_training_query_set()
  ds.cache = os.path.join(cache_dir, 'train_feat_cache.hdf5')
  for i in range(ds.__len__()):
    query_raw, positive_raw, _, _ = ds.__getitem__(i)
    C, W, H = query_raw.shape
    
    query_img = query_raw[0,:,:].numpy()
    positive_img = positive_raw[0,:,:].numpy()
    res = estimate_transform(query_img, positive_img)
    
    if len(res) == 0:
      cv2.imshow('q', query_img)
      cv2.imshow('db', positive_img)
      cv2.waitKey(0)
      continue
    ox, oy, yaw, transform = res
    
    
    query_raw = query_raw[0,...]
    positive_raw = positive_raw[0,...]
    
    query_raw = query_raw.unsqueeze(0)
    positive_raw = positive_raw.unsqueeze(0)
    
    hit = torch.ones(query_raw.shape)
    miss = torch.zeros(query_raw.shape)
    query_prob = torch.where(query_raw < 0.1, hit, miss)
    positive_prob = torch.where(positive_raw < 0.1, hit, miss)
    
    # transform_cv = np.array([[math.cos(yaw), -math.sin(yaw), ox],
    #     [math.sin(yaw), math.cos(yaw), oy]],dtype=np.float)
    
    # q_img = query[0,:,:].numpy()
    # # q_trans = cv2.warpAffine(q_img, transform_cv, q_img.shape)
    # q_img_T = np.zeros_like(q_img, dtype = np.uint8)
    # R = np.array([[math.cos(yaw), -math.sin(yaw)],
    #     [math.sin(yaw), math.cos(yaw)]],dtype=np.float)
    # t = np.array([ox, oy]).reshape((2,1))
    # for x in range(W):
    #   for y in range(H):
    #     nx = x - W/2
    #     ny = y - H/2
    #     nxx, nyy = np.matmul(R, np.array([nx,ny]).reshape((2,1)))+t
    #     nxx = int(nxx)
    #     nyy = int(nyy)
    #     if nxx < -W/2 or nyy < -H/2 or nxx > W/2-1 or nyy > H/2-1: continue
    #     if q_img[y, x] > 0.5:
    #       q_img_T[int(nyy+H/2), int(nxx+W/2)] = 255
          
    # print(ox,oy)
    # transform[0,2]=0.
    # transform[1,2]=0.
    
    # caution this
    yaw *= -1.
    ox_scale = -ox / (W/2)
    oy_scale = -oy / (H/2)
    
    prob_qi = query_prob.to(device)
    prob_dbi = positive_prob.to(device)
    
    theta = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), ox_scale],
        [math.sin(yaw), math.cos(yaw), oy_scale]
    ], dtype=torch.float)
    
    
    grid = F.affine_grid(
      theta.unsqueeze(0), prob_qi.unsqueeze(0).size()).to(device)
    output = F.grid_sample(prob_qi.unsqueeze(0), grid)
    new_img_torch = output[0]
    res = new_img_torch.cpu().numpy()
    
    # get overlap probability
    bq = new_img_torch > 0.5
    bdb = prob_dbi > 0.5
    overlap_grid = torch.where(bq & bdb, torch.ones_like(bq), torch.zeros_like(bq))
    overlap = overlap_grid.sum() / bq.sum()
    print(overlap_grid.shape)
    
    # overlap_img = overlap_grid.cpu().numpy()[0]
    # overlap_img = overlap_img.astype('float32')
    # overlap_img *= 255.
    # overlap_img = overlap_img.astype('uint8')
    # print(overlap_img.shape)
    # plt.imshow(query[0,:,:].numpy())
    # plt.imshow(res[0,:,:])
    # plt.show()
    # cv2.imshow('cv', q_img_T)
    cv2.imshow('torch-q', res[0,:,:])
    cv2.imshow('torch-db', positive_prob[0,...].numpy())
    # cv2.imshow('overlap', overlap_img)
    cv2.waitKey(0)
    
    # plt.imshow(new_img_torch.cpu().numpy().transpose(1,2,0))
    # plt.show()
    
    # yaw_ts = torch.as_tensor(yaw)
    # syi = torch.sin(yaw_ts)
    # cyi = torch.cos(yaw_ts)
    
    # R = torch.Tensor([[cyi, -syi], [syi, cyi]]).to(device)
    # t = torch.Tensor([ox, oy]).to(device)
    
    
    # xy_q = torch.argwhere(prob_qi < 0.5)
    
    # xy_db = torch.matmul(xy_q.float(), R.T) + t
  
    # xy_db = xy_db.long()
    
    # W, H = prob_qi.shape
    # img_size = np.array([W, H])
    # img_size = torch.from_numpy(img_size).to(device)
    # img_size = img_size.int()
    # zeros = torch.zeros_like(img_size).to(device)
    # # print('xy_db: ', xy_db)
    # flags = torch.all((xy_db < img_size) & (xy_db > zeros), dim=-1)
    
    # indices = torch.nonzero(flags, as_tuple=True)[0] 
    # xy_db = torch.index_select(xy_db, 0, indices)
    
    # # print('xy_db-1: ', xy_db)
    
    # prob_idx = xy_db[:, 0] * W + xy_db[:, 1]
    
    # # print('prob_idx: ', prob_idx)
    
    # prob = torch.take(prob_dbi, prob_idx).sum() / prob_idx.shape[0]
    # print('prob_torch: ', prob)
  

if __name__ == '__main__':
  root_dir = '/home/wz/Data/sda1/data/NCLT/'
  root_dir = '/home/wz/Data/sdb1/wz/Data/kitti-raw/raw/'
  device = torch.device("cuda" if True else "cpu")
  
  struct_dir = join(root_dir, 'grid_to_grid/')
  queries_dir = root_dir
  cache_dir = join(struct_dir, 'cache')
  val = I2IDataInterface(root_dir, queries_dir, struct_dir)
  ds = val.get_training_query_set()
  ds.cache = os.path.join(cache_dir, 'train_feat_cache.hdf5')
  for i in range(ds.__len__()):
    query_raw, positive_raw, _, _ = ds.__getitem__(i)
    C, W, H = query_raw.shape
    
    query_img = query_raw[0,:,:].numpy()
    positive_img = positive_raw[0,:,:].numpy()
    print('query_raw.shape: ', query_raw.shape)
