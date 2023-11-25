from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
import os
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
import model.netvlad_fc as netvlad_fc
import model.netvlad as netvlad
import dataset.i2i_util as i2i_data

class VGGVLAD(nn.Module):
  def __init__(self):
    super(VGGVLAD, self).__init__()
    
  # add for tracing, should be called after \
  # add_module('encoder', encoder) & add_module('pool', pool)
  def forward(self, input):
    image_encoding = self.encoder(input)
    place_feature = self.pool(image_encoding)
    return place_feature


parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster','save_pt'])
parser.add_argument('--batchSize', type=int, default=2, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='grid_to_grid/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='grid_to_grid/runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='grid_to_grid/checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='grid_to_grid/cache', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='kitti', 
        help='Dataset to use', choices=['nuscenes', 'kitti', 'nclt'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet', 'mobilenet', 'resnet18'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad_fc', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg', 'netvlad_fc'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')

def train(epoch):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", 
                    [len(whole_train_set), pool_size], 
                    dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    # image_encoding = model.encoder(input)
                    # vlad_encoding = model.pool(image_encoding) 
                    vlad_encoding = model(input) 
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])
        print(len(indices), subIter)

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=i2i_data.collate_fn, pin_memory=cuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cachez

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def save_libtorch_model(filename):
  model.eval()
  with torch.no_grad():
    # cope with what defined in dataset/i2i_util.py
    input = torch.rand(1, 3, 768, 768, dtype=torch.float32) 
    input = input.cuda()
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.

    traced_script_module = torch.jit.trace(model, input)
    traced_script_module.save(filename)

def failure_case(eval_set, epoch=0, write_tboard=False):
    import cv2
    # TODO what if features dont fit in memory? 
    # torch.set_printoptions(precision=16)
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
    
    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            
            # image_encoding = model.encoder(input)
            # vlad_encoding = model.pool(image_encoding) 
            vlad_encoding = model(input) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
  
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 
    
    
    # for each query get those within threshold distance
    gt = eval_set.getPositives() 
    
    K = 5
    for qIx, pred in enumerate(predictions):
      in_top_k = np.any(np.in1d(pred[:K], gt[qIx]))
      q_idx_in_db = qIx + eval_set.dbStruct.numDb
      q_img_path = eval_set.scans[q_idx_in_db]
       
      if not in_top_k:
        # false positive
        # qimg = cv2.imread(q_img_path)
        # print("fp-q: ", q_img_path)
        # cv2.imwrite('failure_cases/'+'false-positive-'+str(qIx)+'.jpg', qimg)
        # for pred_idx in pred[:K]:
        #   pred_img_path = eval_set.scans[pred_idx]
        #   pred_img = cv2.imread(pred_img_path)
        #   print("fp-gt: ", pred_img_path)
        #   cv2.imwrite('failure_cases/'+'false-positive-'+str(qIx)+'-to-'+str(pred_idx)+'.jpg', pred_img)
        
        # false negative
        qimg = cv2.imread(q_img_path)
        cv2.imwrite('failure_cases/'+'false-negative-'+str(qIx)+'.jpg', qimg)
        print("fp-q: ", q_img_path)
        for gt_idx in gt[qIx]:
          missed_gt_path = eval_set.scans[gt_idx]
          print("fp-gt: ", missed_gt_path)
          missed_gt_img = cv2.imread(missed_gt_path)
          cv2.imwrite('failure_cases/'+'false-negative-'+str(qIx)+'-to-'+str(gt_idx)+'.jpg', missed_gt_img)
                    

  
def test(eval_set, epoch=0, write_tboard=False):
    # save_libtorch_model('i2i_vgg_vlad.pt')
    # torch.set_printoptions(precision=12)
    # print(eval_set.scans[0])
    # xxx = eval_set.__getitem__(0)[0]
    # # print(xxx.shape)
    # print(xxx[1, 360:370, 360:370])
    # return
  
    # TODO what if features dont fit in memory? 
    # torch.set_printoptions(precision=16)
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
    
    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            
            # image_encoding = model.encoder(input)
            # vlad_encoding = model.pool(image_encoding) 
            vlad_encoding = model(input) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 
    
    
    # for each query get those within threshold distance
    gt = eval_set.getPositives() 
    
    correct_at_n = np.zeros(len(n_values))
    
    # since we skipped frames when sampling, sometimes the query has not enough postive to caculate precision.
    false_positive_n = np.zeros(len(n_values))
    
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
      for i,n in enumerate(n_values):
        # if in top N then also in top NN, where NN > N
        if np.any(np.in1d(pred[:n], gt[qIx])):
          correct_at_n[i:] += 1
          break

    
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls

def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)
    
    print(len(cluster_set), nIm)
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def check_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == "__main__":
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)
    
    print(opt)

    if opt.dataset.lower() == 'kitti':
        import dataset.kitti_i2i as dataset
    elif opt.dataset.lower() == 'nclt':
        import dataset.nclt_i2i as dataset
    elif opt.dataset.lower() == 'nuscenes':
        import dataset.nuscenes_i2i as dataset
    else:
        raise Exception('Unknown dataset')
    
    # set directories here
    opt.dataPath = os.path.join(dataset.root_dir, 'grid_to_grid')
    opt.runsPath = os.path.join(opt.dataPath, 'runs')
    opt.savePath = os.path.join(opt.dataPath, 'checkpoints')
    opt.cachePath = os.path.join(opt.dataPath, 'cache')
    check_directory(opt.runsPath)
    check_directory(opt.savePath)
    check_directory(opt.cachePath)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')
    
    # pretrained = not opt.fromscratch
    pretrained = True
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        # encoder = models.vgg16(pretrained=pretrained)
        encoder = models.vgg16(weights='IMAGENET1K_V1')
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:             
                for p in l.parameters():
                    p.requires_grad = False
    elif opt.arch.lower() == 'mobilenet':
        encoder_dim = 320
        encoder = models.mobilenet_v2(weights='IMAGENET1K_V1')
        # capture only feature part and remove last relu and maxpool
        # print(list(encoder.features.children()))
        
        layers = list(encoder.features.children())[:-1]

        if pretrained:
            for l in layers[:-2]:             
                for p in l.parameters():
                    p.requires_grad = False
    elif opt.arch.lower() == 'resnet18':
        encoder_dim = 512
        encoder = models.resnet18(weights='IMAGENET1K_V1')
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.children())[:-2]  # delete the last fc layer and avgpool.

        if pretrained:
            for l in layers[:-2]:             
                for p in l.parameters():
                    p.requires_grad = False
      
    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    # model = nn.Module() 
    model = VGGVLAD()
    model.add_module('encoder', encoder)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs

            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'netvlad_fc':
          net_vlad = netvlad_fc.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
          if not opt.resume: 
              if opt.mode.lower() == 'train':
                  initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
              else:
                  initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

              if not exists(initcache):
                  raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

              with h5py.File(initcache, mode='r') as h5: 
                  clsts = h5.get("centroids")[...]
                  traindescs = h5.get("descriptors")[...]
                  net_vlad.init_params(clsts, traindescs) 
                  del clsts, traindescs

          model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        if opt.resume:
            print('===> Running evaluation step')
            epoch = 1
            recalls = test(whole_test_set, epoch, write_tboard=False)
            # failure_case(whole_test_set, epoch, write_tboard=False)
        else:
            raise RuntimeError('The resume path must be provided in test mode!')
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'save_pt':
        libtorch_filename = "./i2i_vgg_vlad.pt"
        print('===> Writing model to libtorch: {}'.format(libtorch_filename))
        save_libtorch_model(filename=libtorch_filename)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume and not os.path.exists(opt.savePath):
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
