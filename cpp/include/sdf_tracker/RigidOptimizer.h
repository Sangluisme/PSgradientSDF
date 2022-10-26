//============================================================================
// Name        : RigidOptimizer.h
// Author      : Christiane Sommer
// Date        : 10/2018
// License     : GNU General Public License
// Description : header file for class RigidOptimizer
//============================================================================

#ifndef RIGID_OPTIMIZER_H_
#define RIGID_OPTIMIZER_H_

// includes
#include <iostream>
#include <opencv2/core/core.hpp>
#include "mat.h"
//class includes
#include "Sdf.h"

/**
 * class declaration
 */
class RigidOptimizer {

protected:

// variables
    
    int num_iterations_;
    float conv_threshold_;
    float conv_threshold_sq_;
    float damping_;
    
    Sdf* tSDF_;
    
    SE3 pose_ = SE3();

public:

// constructors / destructor

    RigidOptimizer(Sdf* tSDF) :
        num_iterations_(50),
        conv_threshold_(1e-3),
        conv_threshold_sq_(conv_threshold_ * conv_threshold_),
        damping_(1.),
        tSDF_(tSDF)
    {}
    
    RigidOptimizer(int num_iterations, float conv_threshold, float damping, Sdf* tSDF) :
        num_iterations_(num_iterations),
        conv_threshold_(conv_threshold),
        conv_threshold_sq_(conv_threshold_ * conv_threshold_),
        damping_(damping),
        tSDF_(tSDF)
    {}
    
    virtual ~RigidOptimizer() {}
    
// member functions
    
    void set_num_iterations(int num_iterations) {
        num_iterations_ = num_iterations;
    }
    
    void set_conv_threshold(float conv_threshold) {
        conv_threshold_ = conv_threshold;
        conv_threshold_sq_ = conv_threshold_ * conv_threshold_;
    }
    
    void set_damping(float damping) {
        damping_  = damping;
    }
    
    void set_pose(SE3 pose) { pose_ = pose; }
    
    SE3 pose() {
        return pose_;
    }
    
    // virtual bool optimize() {}
    virtual bool optimize(const cv::Mat &depth, const Mat3f K) = 0;
    
    // virtual float energy(SE3 pose = SE3()) {}
    virtual float energy(const cv::Mat &depth, const Mat3f K, SE3 pose = SE3()) = 0;

};

#endif // RIGID_OPTIMIZER_H_
