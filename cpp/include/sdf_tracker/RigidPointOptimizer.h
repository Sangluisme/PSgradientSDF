//============================================================================
// Name        : RigidPointOptimizer.h
// Author      : Christiane Sommer
// Date        : 11/2018
// License     : GNU General Public License
// Description : header file for class RigidPointOptimizer
//============================================================================

#ifndef RIGID_POINT_OPTIMIZER_H_
#define RIGID_POINT_OPTIMIZER_H_

// includes
#include <iostream>
//class includes
#include "RigidOptimizer.h"

/**
 * class declaration
 */
class RigidPointOptimizer : public RigidOptimizer {

public:

// constructors / destructor

    RigidPointOptimizer(Sdf* tSDF) :
        RigidOptimizer(tSDF)
    {}
    
    RigidPointOptimizer(int num_iterations, float conv_threshold, float damping, Sdf* tSDF) :
        RigidOptimizer(num_iterations, conv_threshold, damping, tSDF)
    {}
    
    ~RigidPointOptimizer() {}
    
    bool optimize_sampled(const cv::Mat &depth, const Mat3f K, size_t sampling); //, size_t num_iterations = num_iterations_);
    
// member functions
    
    bool optimize(const cv::Mat &depth, const Mat3f K) {
        // tSDF_->increase_counter();
        return optimize_sampled(depth, K, 1);
    }
    
    float energy(const cv::Mat &depth, const Mat3f K, SE3 pose = SE3());

};

#endif // RIGID_POINT_OPTIMIZER_H_
