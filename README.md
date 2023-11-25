# GlobalLoc3D

Implementation of our paper "Global Localization in Large-scale Point Clouds via Roll-pitch-yaw Invariant Place Recognition and Low-overlap Global Registration" (https://ieeexplore.ieee.org/document/10275135) in PyTorch.

# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)


# Usage

- Data preparation

   - Get into the folder of 'registration'
     -  ``` mkdir build && cd build && make -j8```

   - Download KITTI Odometry and KITTI raw

   - Set 'root_dir' in 'kitti_i2i.py' to your own directory that saves KITTI raw
   - Set 'odometry_dir' in 'kitti_i2i.py' to your own directory that saves KITTI odometry
   - Generate index files using 'gen_index_files.py'
   - Run ```./registration/build/save_probability_img YOUR_KITTI_RAW_DIR  ```. You can find two folders saving processed images on the same directory of your root directory of KITTI raw.

- Training

   - Check arguments in main.py and run:
     -  ```python main.py --mode=cluster --dataset=kitti --pooling=netvlad_fc ```
     - ```python main.py --mode=train --dataset=kitti --pooling=netvlad_fc```

- Test

   - Place recognition
     - ```python main.py --mode=test --dataset=kitti --pooling=netvlad_fc --resume=YOUR_TRAINING_DIR --ckpt=best```
   - Global localization
     - ```python main.py --mode=save_pt --dataset=kitti --pooling=netvlad_fc --resume=YOUR_TRAINING_DIR --ckpt=best``` 
     - ```registration/build/global_localization VALSET_FILENAME GT_POSE_FILENAME MODEL_FILENAME```

---

### Acknowledgements

- The authors of [Pytorch-NetVLAD](https://github.com/Nanne/pytorch-NetVlad)
- The authors of [Cartographer](https://github.com/cartographer-project/cartographer)

### license

```Following licenses of the above acknowledged repositories. ``` 
