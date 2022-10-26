//============================================================================
// Name        : RigidPointOptimizerOmp.cpp
// Author      : Christiane Sommer
// Date        : 01/2020
// License     : GNU General Public License
// Description : implementation of class RigidPointOptimizer, with OpenMP
//============================================================================

#include "RigidPointOptimizer.h"
#include <omp.h>

#pragma omp declare reduction (vecsum : Eigen::Vector<float, 6> : omp_out += omp_in) initializer(omp_priv = Eigen::Vector<float, 6>::Zero())
#pragma omp declare reduction (matsum : Eigen::Matrix<float, 6, 6> : omp_out += omp_in) initializer(omp_priv = Eigen::Matrix<float, 6, 6>::Zero())

bool RigidPointOptimizer::optimize_sampled(const cv::Mat &depth, const Mat3f K, size_t sampling) {

    const float z_min = 0., z_max = 3.5;
    const int w = depth.cols;
    const int h = depth.rows;
    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    const float fx_inv = 1.f / fx;
    const float fy_inv = 1.f / fy;
    const float* depth_ptr = (const float*)depth.data;
    
    for (size_t k=0; k<num_iterations_; ++k) {

        Mat3f R = pose_.rotationMatrix();
        Vec3f t = pose_.translation();
        
        float E = 0; // cost
        Vec6f g = Vec6f::Zero(); // gradient
        Mat6f H = Mat6f::Zero(); // approximate Hessian
        
        size_t counter = 0;
        
        omp_set_num_threads(4);
        # pragma omp parallel for shared(R, t, depth_ptr, tSDF_) reduction(+: counter, E) reduction(vecsum: g) reduction(matsum: H)
        for (int y=0; y<h; y+=sampling) for (int x=0; x<w; x+=sampling) {
        
            const float z = depth_ptr[y*w + x];
            if (z<=z_min || z>=z_max) continue;
            
            const float x0 = (float(x) - cx) * fx_inv;
            const float y0 = (float(y) - cy) * fy_inv;
            Vec3f point(x0*z, y0*z, z);
            point = R * point + t;
            
            float w0 = tSDF_->weights(point);
            if (w0>0) {
                Vec3f grad_curr;
                float phi0 = tSDF_->tsdf(point, &grad_curr);
                E += phi0 * phi0;
                Vec6f grad_xi;
                grad_xi << grad_curr, point.cross(grad_curr);
                g += phi0 * grad_xi;
                H += grad_xi * grad_xi.transpose();
                ++counter;
            }
        }
        
        Vec6f xi = Vec6f::Zero();
        if (counter == 0)
            return false;

        xi = damping_ * H.llt().solve(g); // Gauss-Newton
        
        if (xi.squaredNorm() < conv_threshold_sq_) {
            std::cout << "... Convergence after " << k << " iterations!" << std::endl;
            return true;
        }
        
        // update pose
        pose_ = SE3::exp(-xi) * pose_;
    }
    
    return false;
}

float RigidPointOptimizer::energy(const cv::Mat &depth, const Mat3f K, SE3 pose) {

    Mat3f R = pose.rotationMatrix();
    Vec3f t = pose.translation();
    
    float E = 0; // cost
    
    size_t counter = 0;
    
    float z_min = 0., z_max = 1./0.;
    int w = depth.cols;
    int h = depth.rows;
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);
    float fx_inv = 1.f / fx;
    float fy_inv = 1.f / fy;
    const float* depth_ptr = (const float*)depth.data;
    
    int sampling = 1;
    
    for (int y=0; y<h; y+=sampling) for (int x=0; x<w; x+=sampling) {
    
        float z = depth_ptr[y*w + x];
        if (z<=z_min) continue;
        
        float x0 = (float(x) - cx) * fx_inv;
        float y0 = (float(y) - cy) * fy_inv;
        Vec3f point(x0*z, y0*z, z);
        point = R.transpose() * (point - t);
        
        float w0 = tSDF_->weights(point);
        if (w0>0) {
            float phi0 = tSDF_->tsdf(point);
            E += phi0*phi0;
            ++counter;
        }
      
    }
    
    return .5*E;

}
