// ============================================================================
// Name        : main.cpp
// Author      : L.Sang, with many parts taken from C. Sommer
// Date        : 09/2021
// License     : GNU General Public License
// Description : 3D reconstruction using RGB-D data
// ============================================================================

// standard includes
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
// library includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <CLI/CLI.hpp>
// class includes
#include "Timer.h"
#include "normals/NormalEstimator.h"
#include "ps_optimizer/OptimizerSettings.h"
#include "sdf_tracker/TrackingSettings.h"
#include "ConfigLoader.h"
#include "sdf_tracker/VolumetricGradSdf.h"
#include "sdf_tracker/RigidPointOptimizer.h"
#include "img_loader/img_loader.h"
#include "ps_optimizer/PsOptimizer.h"
#include "ps_optimizer/LedOptimizer.h"
#include "ps_optimizer/SharpDetector.h"
// own includes
#include "mat.h"

// Vec3f compute_centroid(const Mat3f &K, const cv::Mat &depth);
Vec3f compute_centroid(const Mat3f &K, const cv::Mat &depth, const Mat4f& Trans);
void sampleKeyFrame(std::vector<int>& key_frames, std::vector<std::string>& key_stamps, std::vector<std::shared_ptr<cv::Mat>>& key_images, std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>>& key_poses, int max_num);

/**
 * main function
 */
int main(int argc, char *argv[]) {

    Timer T;

    // Default input sequence in folder
    std::string configfile = "";
    
    // bool REDIRECT = false;
    bool light = false;
    bool albedo = false;
    bool distance = false;
    bool pose = false;

    CLI::App app{"Volumetric Tracking Example Code"};
    app.add_option("--config_file", configfile, "folder of input sequence");
    
    // parse input arguments
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    
    // load config json file
    std::cout << "load the config file from: " << configfile << std::endl;

    OptimizerSettings* opt_set_;
    TrackingSettings* trac_set_;
    ImageLoader* loader;
    json config;

    if(!LoadConfig(configfile, &trac_set_, &opt_set_, &loader, config)){
        std::cout << "fail to load the config file!" << std::endl;
        return 1;
    }


    // save config file to results for reference
    if(config.contains("--light")) light = config.at("--light");
    if(config.contains("--albedo")) albedo = config.at("--albedo");
    if(config.contains("--distance")) distance = config.at("--distance");
    if(config.contains("--pose")) pose = config.at("--pose");


    // Trunction distance for the tSDF
    const float truncation = trac_set_->truncation_factor_ * trac_set_->voxel_size_;
    float voxel_size = trac_set_->voxel_size_;
    std::string input = trac_set_->input_;
    std::string output = trac_set_->output_;


    // Load camera intrinsics
    if (!loader->load_intrinsics("intrinsics.txt")) {
        std::cerr << "No intrinsics file found in " << input << "!" << std::endl;
        return 1;
    }
    const Mat3f K = loader->K();
    std::cout << "K: " << std::endl << K << std::endl;

    // load one frame to determine the image size
    cv::Mat color, depth;
    if (!loader->load_next(color, depth)) {
        std::cerr << " -> Frame could not be loaded!" << std::endl;
        return 1;
    }

    if (color.rows!=depth.rows || color.cols != depth.cols){
        std::cerr << "-> depth image and color image sizes don't match."<< std::endl;
        return 1;
    }
    loader->reset_counter(); // reset the loader to start from the first frame again.

    // create normal estimator
    T.tic();
    cv::NormalEstimator<float>* NEst = new cv::NormalEstimator<float>(color.cols, color.rows, K, cv::Size(2*5+1, 2*5+1));
    T.toc("Init normal estimation");

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << voxel_size;
    std::string voxel_size_str = stream.str();

    // Prepare tSDF volume
    size_t gridW = 128, gridH = 128, gridD = 128;
    Vec3i grid_dim(gridW, gridH, gridD);
       
    Sdf* tSDF;
    RigidPointOptimizer* pOpt;
    Optimizer* vOpt;

    std::ofstream pose_file(trac_set_->output_ + "tracking_poses.txt");
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> poses;
    
    std::vector<int> keyframes; // key frame index
    keyframes.push_back(0);
    // vector of sampled poses and frames
    std::vector<std::string> key_stamps;
    std::vector<std::shared_ptr<cv::Mat>> key_images;
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> key_poses;
    key_poses.push_back(Mat4f::Identity());


    // Frames to be processed
    // cv::Mat color, depth;
    int dist_to_last_keyframe = 0; //help to void to far distance for key frame selection
    bool GT_pose = false; // do we have GT poses?
    if(!loader->load_pose(trac_set_->pose_file_, poses)){
        std::cout << "GT poses is not avalible!" << std::endl;
        poses.push_back(Mat4f::Identity());
    }
    else{
        std::cout << poses.size() << " GT poses are loaded." << std::endl;
        GT_pose = true;
    }


    // Proceed until first frame
    for (size_t i = 0; i < trac_set_->first_; ++i) {
        loader->load_next(color, depth);
    }
    
    // Actual scanning loop
    for (size_t i = trac_set_->first_; i <= trac_set_->last_; ++i) {
        std::cout << "Working on frame: " << i << std::endl;
        
        // Load data
        T.tic();
        if (!loader->load_next(color, depth)) {
            std::cerr << " -> Frame " << i << " could not be loaded!" << std::endl;
            T.toc("Load data");
            break;
        }
        T.toc("Load data");

        // Get initial volume pose from centroid of first depth map
        if (i == trac_set_->first_) {
        
            // Initial pose for volume by computing centroid of first depth/vertex map
            Vec3f centroid = compute_centroid(K, depth, poses[0]);
            // std::cout << centroid << std::endl;

            // create SDF data
            T.tic();
            tSDF = new VolumetricGradSdf(grid_dim, voxel_size, centroid, truncation);
            T.toc("Create Sdf");
            tSDF->set_zmin(trac_set_->zmin_);
            tSDF->set_zmax(trac_set_->zmax_);
            
            T.tic();
			pOpt = new RigidPointOptimizer(tSDF);
			T.toc("Create RigidOptimizer");

            T.tic();
            switch (opt_set_->model) {
                case ModelType::SH1 :
                case ModelType::SH2 :
                    vOpt = new PsOptimizer(static_cast<VolumetricGradSdf*>(tSDF), voxel_size, K, output, opt_set_);
                    break;
                case ModelType::LED :
                    vOpt = new LedOptimizer(static_cast<VolumetricGradSdf*>(tSDF), voxel_size, K, output, opt_set_);
                    break;

            }
            T.toc("Create PhotometricOptimizer");

            // Initialize tSDF
            T.tic();
            tSDF->update(color, depth, K, SE3(poses[0]), NEst);
            T.toc("Integrate depth data into Sdf");
			// Initialize optimizer
			
            key_stamps.push_back(loader->rgb_timestamp());
            cv::Mat new_color;
            color.copyTo(new_color);
            key_images.push_back(std::make_shared<cv::Mat>(new_color));
        }
        else if(GT_pose){
            T.tic();
            tSDF->increase_counter();
            tSDF->update(color, depth, K, SE3(poses[i]), NEst);
            T.toc("Integrate depth data into Sdf");
            // select key frame if the frame is sharp enough or dist to last key frame is too large
                if (sharpDetector(color, trac_set_->sharpness_threshold_) || dist_to_last_keyframe > 5){
                    dist_to_last_keyframe = 0;
                    keyframes.push_back(i-trac_set_->first_);
                    key_stamps.push_back(loader->rgb_timestamp());
                    key_poses.push_back(poses[i]);
                    cv::Mat new_color;
                    color.copyTo(new_color);
                    key_images.push_back(std::make_shared<cv::Mat>(new_color));

                }
                else{
                    ++dist_to_last_keyframe;
                }
        }
        else{                  
            T.tic();
            tSDF->increase_counter();
            bool conv = pOpt->optimize(depth, K);
            T.toc("Point optimization");
            if(conv){
                T.tic();
                tSDF->update(color, depth, K, pOpt->pose(), NEst);
                T.toc("Integrate depth data into Sdf");
                // select key frame if the frame is sharp enough or dist to last key frame is too large
                if (sharpDetector(color, trac_set_->sharpness_threshold_) || dist_to_last_keyframe > 5){
                    dist_to_last_keyframe = 0;
                    keyframes.push_back(i-trac_set_->first_);
                    key_stamps.push_back(loader->rgb_timestamp());
                    key_poses.push_back(pOpt->pose().matrix());
                    cv::Mat new_color;
                    color.copyTo(new_color);
                    key_images.push_back(std::make_shared<cv::Mat>(new_color));

                }
                else{
                    ++dist_to_last_keyframe;
                }
            }

		}
		// write timestamp + pose in tx ty tz qx qy qz qw format
		Mat4f p;
        if(GT_pose) p = poses[i];
        else p = (pOpt->pose()).matrix();
		std::cout << "Current pose:" << std::endl
		          << p << std::endl;

		Vec3f t(p.topRightCorner(3,1));
        Mat3f R = p.topLeftCorner(3,3);
		Eigen::Quaternion<float> q(R);
   
		pose_file << loader->depth_timestamp() << " "
		          << t[0] << " " << t[1] << " " << t[2] << " "
		          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

    }
    
    pose_file.close();
   
    // extract mesh and write to file
    T.tic();
    if (!tSDF->extract_mesh(output + "init_mesh.ply")) {
        std::string filename = output + "init_mesh.ply";
        std::cerr << "Could not save mesh to " << filename << "!" << std::endl;
    }
    T.toc("Save mesh to disk");
    
    // extract point cloud and write to file
    T.tic();
    if (!tSDF->extract_pc(output + "init_pointcloud.ply")) { 
        std::string filename = output + "init_pointcloud.ply";
        std::cerr << "Could not save point cloud to " << filename << "!" << std::endl;
    }
    T.toc("Save point cloud to disk");
    
    // extract point cloud and write to file
    T.tic();
    if (!tSDF->saveSDF(output + "init_sdf.sdf")) { 
        std::string filename = output + "init_sdf.sdf";
        std::cerr << "Could not save sdf to " << filename << "!" << std::endl;
    }
    T.toc("Save point cloud to disk");
    // tSDF->check_vis_map();

    //=================================================== finished tracking =============================================================
    std::cout << " selected key frame: " << std::endl; 
    for(size_t i = 0; i < key_images.size(); i++){
        std::cout << keyframes[i] << " ";
    }
    std::cout << std::endl;
    if(keyframes.size()>40){
        sampleKeyFrame(keyframes, key_stamps, key_images, key_poses, 40);
    }
    std::cout << " selected key frame after sampling: " << std::endl; 
    for(size_t i = 0; i < key_images.size(); i++){
        std::cout << keyframes[i] << " ";
    }
    std::cout << std::endl;



    vOpt->setImages(key_images);
    vOpt->setKeyframes(keyframes);
    vOpt->setKeytimestamps(key_stamps);
    vOpt->setPoses(key_poses);

   
    vOpt->init();
    vOpt->alternatingOptimize(light, albedo, distance, pose);

    // run python script

    // tidy up
    delete tSDF;
    delete pOpt;
    delete loader;
    delete vOpt;

    cv::destroyAllWindows();

    return 0;
}

// centroid computation to initialize volume at first frame
Vec3f compute_centroid(const Mat3f &K, const cv::Mat &depth, const Mat4f& Trans) {

    Vec3f centroid(0., 0., 0.);
    int counter = 0;
    Mat3f R = Trans.topLeftCorner(3,3);
    Vec3f t = Trans.topRightCorner(3,1);

    int w = depth.cols;
    int h = depth.rows;
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);
    float fx_inv = 1.f / fx;
    float fy_inv = 1.f / fy;
    const float* depth_ptr = (const float*)depth.data;
    
    for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
        float z = depth_ptr[y*w + x];
        if (z>0.0) {
            float x0 = (float(x) - cx) * fx_inv;
            float y0 = (float(y) - cy) * fy_inv;
            centroid += R*Vec3f(x0 * z, y0 * z, z) + t;
            ++counter;
        }
    }

    
    return centroid / float(counter);
}

void normalize_GT_poses(std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>>& poses, const int first_frame)
{
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> normalized_poses;
    Mat4f base_pose = poses[first_frame];
    for(size_t i = first_frame; i<poses.size(); i++){
        Mat4f T = base_pose.inverse()*poses[i];
        normalized_poses.push_back(T);
    }
    poses.clear();
    poses = normalized_poses;
}



//! To selected limited number of frames if there are too many input as key frames.
void sampleKeyFrame(std::vector<int>& key_frames, std::vector<std::string>& key_stamps, std::vector<std::shared_ptr<cv::Mat>>& key_images, std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>>& key_poses, int max_num){
    if (key_frames.size() < max_num ){
        return;
    }
    // int step = int(std::round(static_cast<float>(valid_frames.size())/static_cast<float>(max_num)));
    max_num -= 1;
    float step = static_cast<float>(key_frames.size()) / static_cast<float>(max_num);
    std::vector<int> frames;
    std::vector<std::string> stamps;
    std::vector<std::shared_ptr<cv::Mat>> images;
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> poses;
    float idx = 0;
    for(int count = 0; count < max_num; count++){
        int i = static_cast<int>(idx);
        frames.push_back(key_frames[i]);
        stamps.push_back(key_stamps[i]);
        images.push_back(key_images[i]);
        poses.push_back(key_poses[i]);
        idx+=step;
    }
    frames.push_back(key_frames.back()); //we need the last frame for resize the visibility vector
    stamps.push_back(key_stamps.back());
    images.push_back(key_images.back());
    poses.push_back(key_poses.back());
    
    key_frames = frames;
    key_stamps = stamps;
    key_poses = poses;
    key_images = images;
}