import os
import dataset.kitti_s2s as kitti
import dataset.kitti_i2i as kitti_i2i
import dataset.nclt_s2s as nclt
import dataset.nclt_i2i as nclt_i2i
import dataset.nuscenes_s2s as nuscenes
import dataset.nuscenes_i2i as nuscenes_i2i
import dataset.i2i_util as i2i_utils

if __name__ == "__main__":
  
  # KITTI
  kitti_i2i.generate_struct_files(dataset_type = 'train', skip_frames=5)
  # valset = kitti_i2i.get_whole_val_set()
  # valset = kitti_i2i.get_whole_training_set()
  # i2i_utils.view_dataset_split_trajectory(valset)
  
  # NCLT
  # nclt_i2i.generate_struct_files(dataset_type = 'val', skip_frames=5)
  # valset = nclt_i2i.get_whole_val_set()
  # valset.getPositives()
  # valset = nclt_i2i.get_whole_training_set()
  # nclt_i2i.write_valset_to_txt(
  #   os.path.join(nclt_i2i.struct_dir, 'i2i_val_5_medium.txt'),
  #   os.path.join(nclt_i2i.struct_dir, 'i2i_val_5_poses_medium.txt'), 
  #   sample_level='medium')
  
  # NuScenes
  # nuscenes_i2i.generate_struct_files(dataset_type = 'val', skip_frames=5)
  # valset = nuscenes.get_whole_val_set()
  # valset.getPositives()
  # nuscenes_i2i.write_valset_to_txt(
  #   os.path.join(nuscenes.struct_dir, 'i2i_val_5_easy.txt'),
  #   os.path.join(nuscenes.struct_dir, 'i2i_val_5_poses_easy.txt'), 
  #   sample_level='easy')