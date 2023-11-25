#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>

#include "3d/submap_3d.h"
#include "2d/probability_grid.h"
#include "2d/fast_correlative_scan_matcher_2d.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "3d/submap_3d.h"
#include "2d/probability_grid.h"

#include "loop_detector.h"
#include "ground_estimator.h"
#include "tic_toc.h"

namespace{
using namespace std;

vector<string> split(const string& str, const string& delim) {
  vector<string> res;
  if("" == str) return res;

  char * strs = new char[str.length() + 1];
  strcpy(strs, str.c_str()); 

  char * d = new char[delim.length() + 1];
  strcpy(d, delim.c_str());

  char *p = strtok(strs, d);
  while(p) {
    string s = p;
    res.push_back(s);
    p = strtok(NULL, d);
  }

  return res;
}

void ReadValset(const string& filename, vector<string>& db_files, 
                vector<string>& q_files, vector<vector<size_t>>& pos_idx){
  ifstream ifs(filename);
  
  db_files = {};
  q_files={};
  pos_idx={};
  string line = "";
  vector<string> substrs = {};
  int db_num = 0;
  int q_num = 0;
  if(!ifs.is_open()){
    std::cout<<"failed to open file "<<filename<<"\n";
    return; 
  }
  getline(ifs, line);
  substrs = split(line, " ");
  db_num = atoi(substrs[0].c_str());
  q_num = atoi(substrs[1].c_str());
  // the scan path of db
  for(int i = 0; i < db_num; ++i){
    getline(ifs, line);
    db_files.push_back(line);
  }
  // the scan path of query
  for(int i = 0; i < q_num; ++i){
    getline(ifs, line);
    q_files.push_back(line);
  }
  // the positive samples of each query
  for(int i = 0; i < q_num; ++i){
    if(!getline(ifs, line)){
      break;
    }
    if(line.empty()) break;
    substrs = split(line, ":");
    if(substrs.size()==1){
      pos_idx.push_back({});
      continue;
    }
    int q_idx = atoi(substrs[0].c_str());
    
    string pos_idx_str = substrs[1];
    if(pos_idx_str.empty()){
      pos_idx.push_back({});
      continue;
    }
    substrs = split(pos_idx_str, " ");
    vector<size_t> pos_idx_tmp = {};
    for(size_t k = 0; k < substrs.size(); ++k){
      pos_idx_tmp.push_back(atoi(substrs[k].c_str()));
    }
    pos_idx.push_back(pos_idx_tmp);
  }
  ifs.close();  
  LOG(INFO)<<"db_num and db_files: "<<db_num<<", "<<db_files.size();
  LOG(INFO)<<"q_num and q_files: "<<q_num<<", "<<q_files.size();
  LOG(INFO)<<"q_num and q_pos_index: "<<q_num<<", "<<pos_idx.size();
}

void ReadValsetPose(const string& filename, std::vector<Eigen::Matrix4f>& poses){
  ifstream ifs(filename);
  
  poses = {};
  
  string line = "";
  vector<string> substrs = {};
  if(!ifs.is_open()){
    LOG(ERROR)<<"failed to open file "<<filename;
    return; 
  }

  while(getline(ifs, line)){
    substrs = split(line, " ");
    CHECK(substrs.size()==7);
    
    // w,x,y,z
    float qw = atof(substrs[3].c_str());
    float qx = atof(substrs[0].c_str());
    float qy = atof(substrs[1].c_str());
    float qz = atof(substrs[2].c_str());
    float x = atof(substrs[4].c_str());
    float y = atof(substrs[5].c_str());
    float z = atof(substrs[6].c_str());
    Eigen::Quaternionf q(qw, qx, qy, qz);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.topLeftCorner(3, 3) = q.toRotationMatrix();
    pose.topRightCorner(3, 1) << x,y,z;
    poses.emplace_back(pose);
  }
  ifs.close();  
  LOG(INFO)<<"Read poses with size: "<<poses.size();
}


// Read the KITTI lidar data and our resaved NCLT lidar data in 'velodyne_xyzi'.
pcl::PointCloud<PointType> read_lidar_data(const std::string lidar_data_path){
  std::ifstream lidar_data_file(
      lidar_data_path, std::ifstream::in | std::ifstream::binary);
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(reinterpret_cast<char*>(
      &lidar_data_buffer[0]), num_elements*sizeof(float));

  pcl::PointCloud<PointType> laser_cloud;
  for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 4){
    PointType point;
    point.x = lidar_data_buffer[i];
    point.y = lidar_data_buffer[i + 1];
    point.z = lidar_data_buffer[i + 2];
    // point.intensity = 0;
    
    laser_cloud.push_back(point);
  }
  return laser_cloud;
};


void caculate_mean_std(
    const std::vector<double>& resultSet, double& mean, double& stdev){
  double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	mean =  sum / resultSet.size();
 
	double accum  = 0.0;
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d){
		accum  += (d-mean)*(d-mean);
	});
 
	stdev = sqrt(accum/(resultSet.size()-1));
}
	

}
////////////////////////////////////////////////////////////////////////////////

class GlocEvaluator{
public:
  GlocEvaluator(){}
  ~GlocEvaluator(){}
  
  void load_valset(const std::string& q_db_file, const std::string& pose_file);
  void construct_db(const std::string& model_file_path);
  

  void locate_all_query(){
    if(queried_idx_.empty()){
      detect_all_query();
    }
    located_q_poses_.clear();
    global_registraion_all();
  }

  void detect_all_query();

  void recognition_recalls(){
    std::vector<int> k_values = {1, 5, 10, 20};
    std::vector<float> k_recalls = {0., 0., 0., 0.};
    int valid_query_num = 0;
    for(int i = 0; i < q_files_.size(); ++i){
      if(gt_q_pos_idx_[i].empty()) continue;
      valid_query_num++; 
      if(queried_idx_[i].empty()){
        failed_detect_indices_.push_back(i);
        continue;
      }
      std::vector<size_t> candidates_qi = queried_idx_[i];
      bool detected = false;
      for(int k = 0; k < k_values.size(); ++k){
        int top_k = k_values[k];
        for(int j = 0; j < top_k; ++j){
          if(std::find(gt_q_pos_idx_[i].begin(), gt_q_pos_idx_[i].end(), 
                candidates_qi[j]) != gt_q_pos_idx_[i].end()){
            k_recalls[k] += 1;
            detected = true;
            break;
          }
        }
      }
      if(!detected){
        failed_detect_indices_.push_back(i);
      }
    }
    
    if(valid_query_num > 0){
      for(int i = 0; i<k_recalls.size(); ++i){
        auto& recall = k_recalls[i];
        recall /= valid_query_num;
        LOG(INFO)<<"Recall @ "<<k_values[i]<<": "<<recall;
      }
    }
    
    std::string failed_indice_filename = "failed_detect_indices.txt";
    std::ofstream ofs(failed_indice_filename, std::ios::out);
    if(!ofs) {
      LOG(ERROR)<<"Failed open "<<failed_indice_filename;
    }
    for(int idx: failed_detect_indices_){
      ofs << idx << " ";
    }
    ofs << "\n";
    ofs.close();
  }
  
  void registration_recalls(){
    int all_tests = located_q_poses_.size();
    int no_positive_num = 0;
    int succeed_tests = 0;
    std::vector<double> rot_err={};
    std::vector<double> pos_err={};
    float rad2deg = 180. / M_PI;
    
    const int num_db = db_files_.size();
    for(int i = 0; i < located_q_poses_.size(); ++i){
      size_t db_idx = located_q_poses_[i].first;
      if(db_idx >= db_files_.size()){
        failed_registration_indices_.push_back(i);
        continue;
      }
      const auto& pose_db = poses_db_q_[db_idx];
      const auto& pose_q = poses_db_q_[i+num_db];
      const auto& located_q_in_db = located_q_poses_[i].second;
      Eigen::Matrix4f q2db = pose_db.inverse() * pose_q;
      const Eigen::Matrix3f& gt_rot = q2db.topLeftCorner(3, 3);
      const Eigen::Matrix3f& R_restored = located_q_in_db.topLeftCorner(3,3);
      Eigen::Matrix3f err_R = gt_rot.transpose() * R_restored;

      float trace = err_R.trace();
      float offset_trace = 0.5 * (trace - 1);
      offset_trace = offset_trace < -0.999999 ? -0.999999 : offset_trace;
      offset_trace = offset_trace > 0.999999 ? 0.999999 : offset_trace;
      float err_rot = abs(std::acos(offset_trace));
      Eigen::Vector3f err_trans;
      const Eigen::Vector3f& dxdydz = located_q_in_db.block(0,3,3,1);
      err_trans << q2db(0, 3) - dxdydz[0], 
                   q2db(1, 3) - dxdydz[1], q2db(2, 3) - dxdydz[2];
      float err_pos = err_trans.norm();
      
      err_rot = err_rot * rad2deg;
      if(abs(err_rot - 180.) < 5.) err_rot = abs(err_rot - 180.);
      if(err_pos < 1.0 && err_rot < 5){
        succeed_tests++;

        rot_err.push_back(err_rot);
        pos_err.push_back(err_pos);
      } 
    }

    double mean_rot, std_rot, mean_pos, std_pos;
    caculate_mean_std(pos_err, mean_pos, std_pos);
    caculate_mean_std(rot_err, mean_rot, std_rot);

    LOG(INFO)<<succeed_tests<<", "<<all_tests;
    LOG(INFO)<<"Success rate: "
        <<static_cast<float>(succeed_tests) / static_cast<float>(all_tests);
    LOG(INFO)<<"Rot error: "<<mean_rot<<", "<<std_rot;
    LOG(INFO)<<"Pos error: "<<mean_pos<<", "<<std_pos;

    std::string failed_indice_filename = "failed_registration_indices.txt";
    std::ofstream ofs(failed_indice_filename, std::ios::out);
    if(!ofs) {
      LOG(ERROR)<<"Failed open "<<failed_indice_filename;
    }
    for(int idx: failed_registration_indices_){
      ofs << idx << " ";
    }
    ofs << "\n";
    ofs.close();
    LOG(INFO)<<"Average 2D match costs "<<time_sum_match_ / times_call_match_<<"ms.";
  }

  bool align_ground_ = false;
private:
  bool global_registraion(const size_t q_idx, 
    size_t& located_db_idx, Eigen::Matrix4f& pose_in_db);

  void global_registraion_all(){
    size_t db_idx;
    Eigen::Matrix4f pose_in_db;
    located_q_poses_.resize(q_files_.size());
    CHECK(queried_idx_.size() == q_files_.size())
        << "Call detect_all_query first.";
    for(size_t qidx = 0; qidx < q_files_.size(); qidx++){
      if(global_registraion(qidx, db_idx, pose_in_db)){
        located_q_poses_[qidx] = {db_idx, pose_in_db};
      }else{
        located_q_poses_[qidx] = 
          {db_files_.size()+1, Eigen::Matrix4f::Identity()};
      }
    }
  }

  // deprecated
  void load_kf_time_pose(const std::string& filename);
  void associate_time_name(
      const std::string& kitti_odom_time_file_db,
      const std::string& kitti_odom_time_file_q){
    /* std::string line;
    std::size_t line_num = 0;

    std::ifstream t_db_file(kitti_odom_time_file_db, std::ifstream::in);
    while (std::getline(t_db_file, line)){
      float timestamp = stof(line);
      std::stringstream vel_name;
      vel_name << std::setfill('0') << std::setw(6) << line_num << ".bin";
      times_name_db_[ros::Time().fromSec(timestamp).toNSec()] = vel_name.str();
      line_num++;
    }
    line_num = 0; 
    t_db_file.close();
    std::ifstream t_q_file(kitti_odom_time_file_q, std::ifstream::in);
    while (std::getline(t_q_file, line)){
      float timestamp = stof(line);
      std::stringstream vel_name;
      vel_name << std::setfill('0') << std::setw(6) << line_num << ".bin";
      times_name_q_[ros::Time().fromSec(timestamp).toNSec()] = vel_name.str();
      line_num++;
    }
    t_q_file.close(); */
  }
  
  // std::vector<ros::Time> kf_timestamps_;
  // std::vector<std::string> kf_filenames_;
  std::vector<Eigen::Matrix4f> kf_poses_;
  
  RpyPCLoopDetector loop_detector_;
  GroundEstimator ground_estimator_;
  std::vector<Eigen::Matrix4f> db_rpz_estimates_ = {};
  std::vector<Eigen::Matrix4f> q_rpz_estimates_ = {};
  
  
  const size_t top_k_ = 20;//

  // for evaluation, not used now
  std::map<unsigned long, std::string> times_name_db_ = {};
  std::map<unsigned long, std::string> times_name_q_ = {};

  std::vector<std::string> db_files_ = {};
  std::vector<std::string> q_files_ = {};
  std::vector<OccupancyGrid> q_grids_ = {};
  std::vector<std::vector<size_t>> gt_q_pos_idx_ = {};
  std::vector<std::vector<size_t>> queried_idx_ = {};
  
  std::vector<Eigen::Matrix4f> poses_db_q_ = {};
  std::vector<std::pair<size_t, Eigen::Matrix4f>> located_q_poses_ = {};//{idx_of_db, pose_in_db}

  std::vector<size_t> failed_detect_indices_ = {};
  std::vector<size_t> failed_registration_indices_ = {};

  double time_sum_match_ = 0.;
  double times_call_match_ = 0.;
}; 

void GlocEvaluator::construct_db(const std::string& model_file_path){
    loop_detector_.load_model(model_file_path);
    LOG(INFO) << "LOAD libtorch model from: " << model_file_path;
  
  int i = 0;
  double t_align = 0.;
  double t_detect = 0.;
  std::vector<double> detect_ts;
  for(const auto& filename: db_files_){
    i++;
    pcl::PointCloud<PointType>::Ptr kf(new pcl::PointCloud<PointType>());
    *kf = read_lidar_data(filename);
    if(align_ground_){
      pcl::PointCloud<PointType>::Ptr g_kf(new pcl::PointCloud<PointType>());
      TicToc ta;
      Eigen::Matrix4f rpz = 
          ground_estimator_.EsitmateGroundAndTransform(kf, g_kf);
      db_rpz_estimates_.push_back(rpz);
      t_align += ta.toc();
      TicToc tb;
      loop_detector_.add_keyframe(g_kf);
      if(i > 2) t_detect += tb.toc();
    }else{
      TicToc tb;
      loop_detector_.add_keyframe(kf);
      if(i > 2) t_detect += tb.toc();
    }
  }
  LOG(INFO)<<"time cost for align to ground: " << t_align / double(i) <<"ms.";
  LOG(INFO)<<"time cost for feature extraction: " << t_detect / double(i - 2)<<"ms.";
}

void GlocEvaluator::load_valset(const std::string& q_db_file, 
                                const std::string& pose_file){
  ReadValset(q_db_file, db_files_, q_files_, gt_q_pos_idx_);
  ReadValsetPose(pose_file, poses_db_q_);
}

void GlocEvaluator::load_kf_time_pose(const std::string& filename){
  std::ifstream ifs(filename, std::ios::in);
  if(!ifs.is_open()){
    LOG(ERROR)<<"Failed to open file: "<<filename;
    return;
  }
  std::string line; 
  while(getline(ifs, line)){
    std::vector<std::string> items = split(line, " ");
    if(items.empty()) break;
    CHECK(items.size() == 13)<<"Check the times kitti format.";
    long stamp = stol(items[0]);
    // kf_timestamps_.push_back(ros::Time().fromNSec(stamp));
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        pose.row(i)[j] = stof(items[i*3+j+1]);
      }
    }
    kf_poses_.push_back(pose);
  }
  ifs.close();
  LOG(INFO)<<"Read file finished!";
}

void GlocEvaluator::detect_all_query(){
  double t_sum = 0.;
  for(const std::string& q_filename: q_files_){
    pcl::PointCloud<PointType>::Ptr q_pc(new pcl::PointCloud<PointType>());
    *q_pc = read_lidar_data(q_filename);
    
    Eigen::Matrix4f q_rpz;
    std::vector<size_t> loop_indices={};
    std::vector<float> out_dists_sqr={};
    OccupancyGrid og;
    Eigen::Vector3f xy_yaw; 
    double estimated_scale;
    // align to ground and detect candidates from database
    if(align_ground_){
      pcl::PointCloud<PointType>::Ptr g_pc(new pcl::PointCloud<PointType>());
      q_rpz = ground_estimator_.EsitmateGroundAndTransform(q_pc, g_pc);
      loop_detector_.detect(g_pc, og, loop_indices, out_dists_sqr);
      q_rpz_estimates_.push_back(q_rpz);
    }else{
      TicToc t;
      loop_detector_.detect(q_pc, og, loop_indices, out_dists_sqr);
      t_sum += t.toc();
    }
    q_grids_.push_back(og);
    queried_idx_.push_back(loop_indices);
  }
  LOG(INFO)<<"Each query cost: "<< t_sum/q_files_.size()<<"ms.";
}

bool GlocEvaluator::global_registraion(
    const size_t q_idx, size_t& located_db_idx, Eigen::Matrix4f& pose_in_db){
  CHECK(q_idx < queried_idx_.size());
  std::vector<size_t>& loop_indices = queried_idx_[q_idx];
  
  Eigen::Vector3f xy_yaw;
  double estimated_scale = 0.;
  // perform 2D match
  for(size_t i = 0; i < std::min(top_k_, loop_indices.size()); i++){
    size_t db_idx = loop_indices[i];
    TicToc t;    
    bool matched = loop_detector_.match(q_grids_[q_idx], db_idx, xy_yaw, estimated_scale);
    time_sum_match_ += t.toc();
    times_call_match_ += 1.;
    if(matched){
      float e_roll, e_pitch, e_yaw, e_dx, e_dy, e_dz;
      if (align_ground_){
        Eigen::Matrix4f Tq_l2g = q_rpz_estimates_[q_idx];
        Eigen::Matrix4f Tdb_l2g = db_rpz_estimates_[db_idx];

        Eigen::Matrix4f T_q2db_rpz = Tdb_l2g.inverse() * Tq_l2g;
        Eigen::Matrix3f R_q2db_rpz = T_q2db_rpz.topLeftCorner(3, 3);
        Eigen::Vector3f ypr_rpz = R_q2db_rpz.eulerAngles(2, 1, 0);

        // dx, dy及yaw需要叠加3-DoF的2维变换
        cartographer::transform::Rigid2f T_q2db_h(xy_yaw.head<2>(), xy_yaw[2]);
        cartographer::transform::Rigid3f T_q2db_h_3d 
          = cartographer::transform::Embed3D(T_q2db_h);
        Eigen::Matrix3f R_q2db_h_3d = T_q2db_h_3d.rotation().toRotationMatrix();
        Eigen::Vector3f t_q2db_h_3d = T_q2db_h_3d.translation();
        Eigen::Matrix4f T_qg_dbg = Eigen::Matrix4f::Identity();
        T_qg_dbg.topLeftCorner(3, 3) = R_q2db_h_3d;
        T_qg_dbg.block<3, 1>(0, 3) = t_q2db_h_3d;
        Eigen::Matrix4f T_q2db_yawxy = Tdb_l2g.inverse() * T_qg_dbg * Tq_l2g;

        Eigen::Matrix3f R_q2db_yawxy = T_q2db_yawxy.topLeftCorner(3, 3);
        Eigen::Vector3f ypr_yawxy = R_q2db_yawxy.eulerAngles(2, 1, 0);

        e_dx = T_q2db_yawxy(0, 3);
        e_dy = T_q2db_yawxy(1, 3);
        e_dz = T_q2db_rpz(2, 3);
        e_roll = ypr_rpz[2];
        e_pitch = ypr_rpz[1];
        e_yaw = ypr_yawxy[0];
      }else{
        e_roll = 0.;
        e_pitch = 0.;
        e_dz = 0.;
        e_dx = xy_yaw[0];
        e_dy = xy_yaw[1];
        e_yaw = xy_yaw[2];
      }
      Eigen::Quaterniond q = cartographer::transform::RollPitchYaw(
            e_roll, e_pitch, e_yaw);
      
      located_db_idx = db_idx;
      pose_in_db.topLeftCorner(3,3) = q.toRotationMatrix().cast<float>();
      pose_in_db.block(0, 3, 3, 1) << e_dx, e_dy, e_dz;

      return true;
    }
  }
  return false;
}


int main(int argc, char *argv[]){
  // Read evaluation filenames
  string valset_filename = argv[1];
  string pose_filename = argv[2];
  string model_filename = argv[3];
  
  GlocEvaluator gloc;
  if(argc == 5){
    gloc.align_ground_ = true;
  }else{
    gloc.align_ground_ = false;
  }
  gloc.load_valset(valset_filename, pose_filename);
  gloc.construct_db(model_filename);

  gloc.locate_all_query();

  // Currently, the c++ libtorch results are not the same exactly as that from python.
  // These c++ codes only showcase the usage of SLAM re-localization or global localization.
  gloc.recognition_recalls();
  gloc.registration_recalls();
  
  return 0;
}
