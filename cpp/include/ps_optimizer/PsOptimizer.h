//============================================================================
// Name        : PsOptimizer.h
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : header of class PsOptimizer
//============================================================================

#ifndef PS_OPTIMIZER_H_
#define PS_OPTIMIZER_H_

#include "mat.h"
#include "Optimizer.h"


#include <Eigen/StdVector>
#include <Eigen/Sparse>
// #include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>



class PsOptimizer : public Optimizer {

    

    std::vector<Eigen::VectorXf> light_; // length 4 for first order and length 9 for second order // for each image there is a light


// private method:

    // image related
    Eigen::VectorXf SH(const Vec3f& n, const int order);
    Vec3f renderedIntensity(const SdfVoxel& v, const Vec3i& idx, const int frame_idx);
    bool computeResidual(const SdfVoxel& v, const Vec3i& idx, const int frame_id, Vec3f& r);

    //residual matrix
    std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > computeResidual();

    // energy related 
    virtual float getPSEnergy();
  
    // jacobian related
    bool poseJacobian(const Vec3i& idx, const SdfVoxel& v, int frame_id, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3,6>& J_c);
   
    float rhoJacobian(const SdfVoxel& v, const int frame_id);
    float rhoJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id);

    

    void lightJacobian(const SdfVoxel& v, const int frame_id, std::vector<Eigen::VectorXf>& J_l);
    void lightJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id, std::vector<Eigen::VectorXf>& J_l);

    bool distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d);
    bool numerical_distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d);
    // bool distJacobian_numeric(SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, Vec3f& J_d);
   
    // jacobian matrix
    Eigen::SparseMatrix<float> lightJacobian();
    Eigen::SparseMatrix<float> albedoJacobian();
    Eigen::SparseMatrix<float> distJacobian();
    Eigen::SparseMatrix<float> poseJacobian();

    
    void optimizeAlbedoAll(bool albedo_reg, float damping);
    void optimizeLightAll();
    void optimizeDistAll(bool normal_reg, bool laplacian_reg, float damping);
    void optimizePosesAll(float damping);

    public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
    PsOptimizer(VolumetricGradSdf* tSDF,
                const float voxel_size,
                const Mat3f& K,
                std::string save_path,
                OptimizerSettings* settings

        );


    // optimize
    virtual void init();
    virtual bool alternatingOptimize(bool light, bool albedo, bool distance, bool pose);

};


#endif // PS_OPTIMIZER_H_