
#include "loop_detector.h"

/////////////////////////////codes for libtorch-based loop detection//////////////////////////////////
  RpyPCLoopDetector::RpyPCLoopDetector(){
    identity_transform_=cartographer::transform::Rigid3d::Identity();
  }
  
  
  void RpyPCLoopDetector::add_keyframe(const pcl::PointCloud<PointType>::Ptr db_pc){
    // LOG(INFO)<<"added keyframe begins.";
    OccupancyGrid og;
    std::vector<float> feat = get_place_feature(
        module_, db_pc, og.occupancy, og.ox_oy_res);
    db_features_.push_back(feat);
    
    db_grids_.push_back(og);
    // ox_oy_res_.push_back(xy_res);
    // LOG(INFO)<<"added keyframe "<<db_features_.size();
  }

  void RpyPCLoopDetector::detect(
      const pcl::PointCloud<PointType>::Ptr q_pc, 
      OccupancyGrid& q_grid,
      std::vector<size_t>& ret_indexes, std::vector<float>& out_dists_sqr){
    
    if(db_features_.size() <= num_exclude_recent_ + top_k_) {
      std::cout << "Not enough keyframes in database."<<std::endl;
      return;
    }
    

    // just construct the search tree once in global localization.
    if(!kdtree_) { 
      kdtree_.reset(); 
      kdtree_ = std::make_unique<InvKeyTree>(k_dim_, db_features_, 10);
    }
    
    std::vector<float> feat = get_place_feature(
        module_, q_pc, q_grid.occupancy, q_grid.ox_oy_res);

    ret_indexes.resize(top_k_);
    out_dists_sqr.resize(top_k_);
    // std::cout<<"query begins..."<<std::endl;
    kdtree_->query(&feat[0], top_k_, &ret_indexes[0], &out_dists_sqr[0]);
  }

  bool RpyPCLoopDetector::detect(size_t& q_idx, size_t& loop_idx){
    size_t cur_idx = db_features_.size()-1;
    std::vector<size_t> ret_indexes={}; 
    std::vector<float> out_dists_sqr={};
    this->detect(cur_idx, ret_indexes, out_dists_sqr);
    if(ret_indexes.empty()) return false;
    if(out_dists_sqr[0] < loop_metric_dist_th_){
      q_idx = cur_idx;
      loop_idx = ret_indexes[0];
      return true;
    }
    return false;
  }

  void RpyPCLoopDetector::detect(const size_t q_idx,
        std::vector<size_t>& ret_indexes, std::vector<float>& out_dists_sqr){
    if(db_features_.size() <= num_exclude_recent_ + top_k_) return;
    // following the strategy used in scan-context
    if(tree_making_period_counter_ % tree_making_period_ == 0) {
      db_features_to_search_.clear();
      db_features_to_search_.assign(db_features_.begin(), db_features_.end() - num_exclude_recent_);

      kdtree_.reset(); 
      kdtree_ = std::make_unique<InvKeyTree>(k_dim_ /* dim */, db_features_to_search_, 10 /* max leaf */ );
    }
    tree_making_period_counter_ = tree_making_period_counter_ + 1;
    
    std::vector<float> feat = db_features_[q_idx];
    ret_indexes.resize(top_k_);
    out_dists_sqr.resize(top_k_);
    // std::cout<<"query begins..."<<std::endl;
    kdtree_->query(&feat[0], top_k_, &ret_indexes[0], &out_dists_sqr[0]);
    // std::cout<<"query ends..."<<std::endl;
  }

  cv::Mat RpyPCLoopDetector::crop_pad_occupancy(const cv::Mat& src, size_t width, size_t height){
    cv::Mat dst = cv::Mat::ones(height, width, CV_8UC3) * 255;
    int cw = src.cols; 
    int ch = src.rows;
    cw = cw >= width ? width: cw; 
    ch = ch >= height ? height: ch; 
    cv::Mat color_img = src.clone();
    if(color_img.channels() == 1){
      cv::cvtColor(color_img, color_img, CV_GRAY2BGR);
    }
    
    cv::Rect roi_src = cv::Rect(
      int(floor((src.cols - cw) / 2.)), 
      int(floor((src.rows - ch) / 2.)),
      cw, ch);
    cv::Rect roi_dst = cv::Rect(
      int(floor((dst.cols - cw) / 2.)), 
      int(floor((dst.rows - ch) / 2.)),
      cw, ch);
    
    cv::Mat crop = color_img(roi_src);
    crop.copyTo(dst(roi_dst));
    return dst;
  }

  cartographer::sensor::RangeData RpyPCLoopDetector::point_cloud_to_range_data(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud){
    cartographer::sensor::RangeData result;
    result.origin << 0., 0., 0.;
    for(const pcl::PointXYZI& pt: point_cloud->points){
      if(sqrt(pt.x*pt.x+pt.y*pt.y+pt.z*pt.z) > 100.){
        result.misses.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
      }else{
        result.returns.emplace_back(Eigen::Vector3f(pt.x,pt.y,pt.z));
      }
    }
    return result;
  }

  cv::Mat RpyPCLoopDetector::get_projected_grid(
      pcl::PointCloud<PointType>::Ptr q_pc, Eigen::Vector3f& xy_res){
    double ox, oy, resolution;
    
    Submap3D submap_q(high_resolution_, low_resolution_, identity_transform_);
    auto range_data = point_cloud_to_range_data(q_pc);
    submap_q.InsertRangeData(
      range_data, range_data_inserter_, high_resolution_max_range_);
    cv::Mat img_q = ProjectToCvMat(
      &submap_q.high_resolution_hybrid_grid(), identity_transform_,
      ox, oy, resolution);
    xy_res << ox, oy, resolution;
    return img_q;
  }

  std::vector<float> RpyPCLoopDetector::get_place_feature(
      torch::jit::script::Module& module, 
      pcl::PointCloud<PointType>::Ptr q_pc,
      cv::Mat& occupancy_grid,
      Eigen::Vector3f& xy_res){
     
    cv::Mat img_q = get_projected_grid(q_pc, xy_res);
    img_q.copyTo(occupancy_grid);

    const size_t width = 768;
    const size_t height = 768;

    cv::Mat input_img = crop_pad_occupancy(img_q, width, height);
    
    input_img.convertTo(input_img, CV_32FC3, 1.0 / 255.0);
    torch::Tensor img_tensor = torch::from_blob(input_img.data,
        {1, input_img.rows, input_img.cols, 3}, torch::dtype(torch::kFloat));
        
    img_tensor = img_tensor.permute({0, 3, 1, 2});    
    // img_tensor = img_tensor.to(torch::kCPU);
    img_tensor = img_tensor.to(torch::kCUDA);

    double desc_gen_time = 0;
    
    torch::NoGradGuard no_grad;
    torch::Tensor result = module.forward({img_tensor}).toTensor();
    result = result.to(torch::kCPU);

    std::vector<float> ret;
    ret.resize(result.size(1));
    auto res_a = result.accessor<float,2>();
    for(int i = 0; i < result.size(1); ++i){
      ret[i] = res_a[0][i];
    }
    return ret;
  }

  bool RpyPCLoopDetector::match(const size_t q_idx,  const size_t db_idx, 
      Eigen::Vector3f& xy_yaw, double& estimated_scale){
    const cv::Mat& qimg = db_grids_[q_idx].occupancy;
    const cv::Mat& dbimg = db_grids_[db_idx].occupancy;
    const Eigen::Vector3f& oxy_res_q = db_grids_[q_idx].ox_oy_res;
    const Eigen::Vector3f& oxy_res_db = db_grids_[db_idx].ox_oy_res;
    return this->match(qimg, dbimg, oxy_res_q, oxy_res_db, xy_yaw, estimated_scale);
  }

  bool RpyPCLoopDetector::match(const OccupancyGrid& q_grid,
      const size_t db_idx, Eigen::Vector3f& xy_yaw, double& estimated_scale){
    const cv::Mat& qimg = q_grid.occupancy;
    const cv::Mat& dbimg = db_grids_[db_idx].occupancy;
    const Eigen::Vector3f& oxy_res_q = q_grid.ox_oy_res;
    const Eigen::Vector3f& oxy_res_db = db_grids_[db_idx].ox_oy_res;
    return this->match(qimg, dbimg, oxy_res_q, oxy_res_db, xy_yaw, estimated_scale);
  }
  
  bool RpyPCLoopDetector::match(const cv::Mat& src1,  const cv::Mat& src2, 
      const Eigen::Vector3f& oxy_res_1, const Eigen::Vector3f& oxy_res_2,
      Eigen::Vector3f& xy_yaw, double& estimated_scale,
      bool visualize){
    cv::Mat img1, img2;
    cv::threshold(src1, img1, 100, 255, CV_THRESH_BINARY_INV);
    cv::threshold(src2, img2, 100, 255, CV_THRESH_BINARY_INV);
    
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.85;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
        good_matches.push_back(knn_matches[i][0]);
      }
    }
    
    //-- Draw matches
    if(visualize){
      cv::Mat img_matches;
      cv::drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches, 
          cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), 
          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      cv::imshow("Good Matches", img_matches );
      cv::imshow("src1", img1 );
      cv::imshow("src2", img2 );
    }
    
   
    std::vector<cv::Point2f> from_pts={};
    std::vector<cv::Point2f> to_pts={};
    std::vector<cv::Point2f> from_pix={};
    std::vector<cv::Point2f> to_pix={};
    
    if(good_matches.size() >= 5){
      cv::Point2f tl_from(oxy_res_1[0], oxy_res_1[1]);
      cv::Point2f tl_to(oxy_res_2[0], oxy_res_2[1]);
      for(int i = 0; i < good_matches.size(); ++i){
        auto gmt = good_matches[i];
        
        from_pts.push_back(
            tl_from + keypoints1.at(gmt.queryIdx).pt * oxy_res_1[2]);
        to_pts.push_back(
            tl_to + keypoints2.at(gmt.trainIdx).pt * oxy_res_2[2]);

        from_pix.push_back(keypoints1.at(gmt.queryIdx).pt);
        to_pix.push_back(keypoints2.at(gmt.trainIdx).pt);
      }
      std::vector<uchar> inliers;
      
      cv::Mat transform = cv::estimateAffinePartial2D(from_pts, to_pts, 
          inliers, cv::RANSAC, 3 * oxy_res_2[2]/*in meter*/, 3000);
      if(!transform.empty()){
        double scale = sqrt(transform.at<double>(0,0)*transform.at<double>(0,0) 
            + transform.at<double>(0,1)*transform.at<double>(0,1));
        
        if(abs(1-scale)<0.1){
          if(visualize){
            cv::Mat transform_pix = cv::estimateAffinePartial2D(
              from_pts, to_pts, inliers, cv::RANSAC, 3/*in pix*/, 3000);
            cv::Mat affine_img1, img3;
            cv::warpAffine(
                src1, affine_img1, transform_pix, cv::Size(src1.cols, src1.rows));
            cv::drawMatches(affine_img1, keypoints1, src2, keypoints2, {}, 
              img3, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), 
              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            cv::imshow("Affine", img3);
            cv::waitKey();
          }
          // float yaw = atan2(transform.at<double>(1, 0), transform.at<double>(1, 1));
          float yaw = asin(transform.at<double>(1, 0) / scale);
          xy_yaw << transform.at<double>(0, 2),
                    transform.at<double>(1, 2),//whether to divide scale?
                    yaw;
          estimated_scale = scale;
          return true;
        }
      }
    }
    
    return false;
  }


