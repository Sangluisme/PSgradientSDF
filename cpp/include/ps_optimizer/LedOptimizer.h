//============================================================================
// Name        : LedOptimizer.h
// Author      : Lu Sang
// Date        : 04/2022
// License     : GNU General Public License
// Description : header of class LedOptimizer
//============================================================================

#ifndef LED_OPTIMIZER_H_
#define LED_OPTIMIZER_H_

#include "mat.h"
#include "Optimizer.h"


#include <Eigen/StdVector>
#include <Eigen/Sparse>
// #include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>



class LedOptimizer : public Optimizer {

// private variables: 
   Vec3f light_ = Vec3f::Ones();

// private method:
    // energy related 
    virtual float getPSEnergy();
    void computeLightIntensive();
    Vec3f renderedIntensity(const SdfVoxel& v, const Vec3i& idx, const int frame_idx);
    bool computeResidual(const SdfVoxel& v, const Vec3i& idx, const int frame_id, Vec3f& r);

    //Jacobian function
    bool poseJacobian(const Vec3i& idx, const SdfVoxel& v, int frame_id, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3,6>& J_c);
    Vec3f rhoJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id);
    Vec3f LightJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id);
    bool distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d);
    bool numerical_distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d);
    //set jacobian matrix
    Eigen::SparseMatrix<float> albedoJacobian();
     Eigen::SparseMatrix<float> lightJacobian();
    Eigen::SparseMatrix<float> poseJacobian();
    Eigen::SparseMatrix<float> distJacobian();
    std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > computeResidual();

    //optimization function
    void optimizeLightAll(float damping);
    void optimizeAlbedoAll(bool albedo_reg, float damping);
    void optimizeDistAll(bool normal_reg, bool laplacian_reg, float damping);
    void optimizePosesAll(float damping);

    public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
    LedOptimizer(VolumetricGradSdf* tSDF,
                const float voxel_size,
                const Mat3f& K,
                std::string save_path,
                OptimizerSettings* settings

        );


    // optimize
    virtual void init();
    virtual bool alternatingOptimize(bool light, bool albedo, bool distance, bool pose);


};

#endif // LED_OPTIMIZER_H_
