//============================================================================
// Name        : Optimizer.h
// Author      : Lu Sang
// Date        : 10/2021
// License     : GNU General Public License
// Description : header of class Optimizer
//============================================================================

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "mat.h"
// #include "sdf_tracker/VolumetricGradPixelSdf.h"
#include "sdf_tracker/VolumetricGradSdf.h"
#include "OptimizerSettings.h"

#include <Eigen/StdVector>
#include <Eigen/Sparse>
// #include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>



class Optimizer {

    // variables:
protected: 
    size_t num_frames_;
    size_t num_voxels_;
    float voxel_size_;
    float voxel_size_inv_;
    Mat3f K_; // camera intrinsics
    std::string save_path_;

    VolumetricGradSdf* tSDF_;
    // std::vector<std::vector<bool>> vis_; // visibility per voxel and frame

    std::vector<int> frame_idx_; // indices of keyframes to be taken into account
    std::vector<std::shared_ptr<cv::Mat>> images_; // vector of pointers to keyframes
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_; // poses from tracking part
    std::vector<std::string> key_stamps_;

    // std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> > light_; // length 4 for first order and length 9 for second order

    std::vector<int> surface_points_;
    // std::vector<int> band_voxels_;


    OptimizerSettings* settings_;

    // pose related
    Mat3f getRotation(const int frame_id)
    {
        return poses_[frame_id].topLeftCorner(3,3);
    }

    Vec3f getTranslation(const int frame_id)
    {
        return poses_[frame_id].topRightCorner(3,1);
    }

    cv::Mat getImage(const int frame_id) const
    {
        return *(images_[frame_id]);
    }

    void getSurfaceVoxel();
  

    //update functions
    void updateAlbedo(const int lin_idx, float delta_r, float delta_g, float delta_b);
    void updateAlbedo(Eigen::VectorXf& delta_r);
    void updateGrad();
    void updateDist(Eigen::VectorXf& delta_d, bool updateGrad);
    void updatePose(const int frame_id, const Mat3f& R, const Vec3f& t, const Vec6f& xi);
    void updatePose(Eigen::VectorXf& delta_xi);

    //regularizer energy functions
    virtual float getPSEnergy() = 0;
    float getNormalEnergy();
    float getAlbedoRegEnergy();
    float getLaplacianEnergy();

    // compute weight and loss
    Eigen::VectorXf computeWeight(Eigen::VectorXf r);
    float computeLoss(Eigen::VectorXf r);

    //useful functions
    bool getIntensity(const Vec3i& idx, const SdfVoxel& v, const Mat3f& R, const Vec3f& t, const cv::Mat& img, Vec3f& intensity);

    // regularizer related
    float computeDistLaplacian(const SdfVoxel& v, const Vec3i& idx);

    std::pair<Vec3f, Vec3f> computeDistGrad(const SdfVoxel& v, const Vec3i& idx);
    void distRegJacobian(const SdfVoxel& v, const Vec3i& idx, std::vector<float>& Jr_d);

    void albedoRegJacobian(const SdfVoxel& v, const Vec3i& idx, std::vector<Vec3f>& Jr_r);
    std::pair<Mat3f, Vec3f> computeAlbedoGrad(const SdfVoxel& v, const Vec3i& idx);

    Vec3f normalJacobian(const SdfVoxel& v, const Vec3i& idx);
    Vec3f normalJacobian(Vec3f& grad, Vec3f& direction, bool lag);

    
    std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> distRegJacobian();
    std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> distLaplacianJacobian();
    std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> albedoRegJacobian();
   
    
    
    bool ifValidDirection(const Vec3i& idx, const int direction, const int pos);

    //auxilary functions

    float getTotalEnergy(float E, float E_n, float E_l, float E_r, std::ofstream& file);

    void setPoses(Eigen::VectorXf& xi);
    Eigen::VectorXf stackPoses();
    void getAlldsitance(Eigen::VectorXf& Distance);
    void setAlldsitance(Eigen::VectorXf& Distance);
    void getAllalbedo(Eigen::VectorXf& rho);
    void setAllalbedo(Eigen::VectorXf& rho);

    public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
    Optimizer(VolumetricGradSdf* tSDF,
                const float voxel_size,
                const Mat3f& K,
                std::string save_path,
                OptimizerSettings* settings

        );

    

    void setImages(std::vector<std::shared_ptr<cv::Mat>> images)
    {
            images_ = images;
    }

    void setPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& pose)
    {
        poses_ = pose;
    }

    void setKeyframes( std::vector<int>& keyframes)
    {
        frame_idx_ = keyframes;
    }
    void setKeytimestamps(std::vector<std::string>& keystamps)
    {
        key_stamps_ = keystamps;
    }

    // void setSavePath(std::string output){
    //     save_path_ = output;
    // }

    //debug functions
    void check_tsdf(bool sample);
    void check_vis_map(bool sample);
    void check_albedo(bool sample);
    void compare_normal(bool sample);

    virtual void init() = 0;
    void select_vis();
    void initAlbedo();
    bool savePoses(std::string filename);
    bool extract_mesh(std::string filename);
    bool save_pointcloud(std::string filename);
    bool save_white_mesh(std::string filename);
    bool saveSDF(std::string filename);
    // bool save_info();

    void subsampling();

    virtual bool alternatingOptimize(bool light, bool albedo, bool distance, bool pose){
        return false;
    }

};


#endif // PS_OPTIMIZER_H_
