//============================================================================
// Name        : VoxelGrid.h
// Author      : Christiane Sommer
// Modified    : Lu Sang
// Date        : 09/2021, original on 10/2018
// License     : GNU General Public License
// Description : header file for class VoxelGrid
//============================================================================

#ifndef VOXEL_GRID_H_
#define VOXEL_GRID_H_

// includes
#include <iostream>
#include <fstream>
#include "mat.h"
// library includes
// #include <opencv/cv.h>
#include <opencv2/core/core.hpp>

/**
 * class declaration
 */
class VoxelGrid {

protected:

// variables

    Vec3i grid_dim_; // voxel grid resolution
    size_t num_voxels_;
    float voxel_size_; // (linear) size of one voxel in m
    Vec3f shift_; // position of voxel grid center in camera coordinates
    Vec3f origin_; // voxel grid origin, i.e. position of voxel[0]
    
// methods
    
    Vec3f voxel2world(Vec3i index) const {
        return origin_ + voxel_size_ * index.cast<float>();
    }
    
    Vec3f voxel2world(int i, int j, int k) const {
        return origin_ + voxel_size_ * Vec3f(i, j, k);
    }

    Vec3f vox2float(int i, int j, int k)
    {
        return voxel_size_ * Vec3f(i, j, k);
    }

    Vec3f vox2float(const Vec3i idx) const {
        return voxel_size_ * idx.cast<float>();
    }

    Vec3f world2voxelf(Vec3f point) const {
        return (point - origin_) / voxel_size_;
    }
    
    Vec3f world2voxelf(float x, float y, float z) const {
        return (Vec3f(x, y, z) - origin_) / voxel_size_;
    }
    
    Vec3i world2voxel(Vec3f point) const {
        Vec3f tmp = world2voxelf(point) + Vec3f(0.5, 0.5, 0.5);
        return tmp.cast<int>(); // round
    }
    
    Vec3i world2voxel(float x, float y, float z) const {
        Vec3f tmp = world2voxelf(x, y, z) + Vec3f(0.5, 0.5, 0.5);
        return tmp.cast<int>(); // round
    }
    
    float interp3(const float* grid, Vec3f point, float extrap = 1./0.) const;
    
    float interp3(const float* grid, float x, float y, float z, float extrap = 1./0.) const {
        return interp3(grid, Vec3f(x, y, z), extrap);
    }

    int idx2line(const Vec3i idx) const{
        return idx[0] + idx[1]*grid_dim_[0] + idx[2]*grid_dim_[0]*grid_dim_[1];
    }

    int idx2line(const int i, const int j, const int k) const{
        Vec3i idx(i,j,k);
        return idx[0] + idx[1]*grid_dim_[0] + idx[2]*grid_dim_[0]*grid_dim_[1];
    }

    Vec3i line2idx(const int lin_idx){
        int rest = lin_idx;
        int k = rest/(grid_dim_[0]*grid_dim_[1]);
        rest -= k*grid_dim_[0]*grid_dim_[1];
        int j = rest/grid_dim_[0];
        rest -= j*grid_dim_[0];
        int i = rest;
        Eigen::Vector3i tmp(i,j,k);
        return tmp;
    }
    
    int nearest_index(Vec3f point) const;
    
    int nearest_index(float x, float y, float z) const {
        return nearest_index(Vec3f(x, y, z));
    }
    
    float eval(float* grid, size_t i, size_t j, size_t k) {
        size_t lin_index = i + j * grid_dim_[0] + k * grid_dim_[0] * grid_dim_[1];
        return grid[lin_index];
    }
    
    void slice_x(float* grid, size_t i, cv::Mat &slice);
    void slice_y(float* grid, size_t j, cv::Mat &slice);
    void slice_z(float* grid, size_t k, cv::Mat &slice);
    void write(float* grid, std::string file = "tmp_vol.dat");
    
public:

// constructors / destructor

    VoxelGrid() :
        grid_dim_(Vec3i(64, 64, 64)),
        voxel_size_(0.005),
        shift_(Vec3f(0.,0.,1.)),
        origin_(shift_ - 0.5*voxel_size_*grid_dim_.cast<float>()),
        num_voxels_(grid_dim_[0]*grid_dim_[1]*grid_dim_[2])
    {}
    
    VoxelGrid(Vec3i grid_dim, float voxel_size, Vec3f shift) :
        grid_dim_(grid_dim),
        voxel_size_(voxel_size),
        shift_(shift),
        origin_(shift_ - 0.5*voxel_size*grid_dim_.cast<float>()),
        num_voxels_(grid_dim_[0]*grid_dim_[1]*grid_dim_[2])
    {}

    void set_voxel_size(float voxel_size){
        voxel_size_ = voxel_size;
    }

    void set_grid_dim(Vec3i grid_dim){
        grid_dim_ = grid_dim;
    }

    void grid_subsample()
    {
        voxel_size_ *= 0.5;
        grid_dim_ = 2*grid_dim_; 
        origin_ = shift_ - 0.5*voxel_size_*grid_dim_.cast<float>() - 0.5*voxel_size_*Vec3f::Ones();
        num_voxels_ = grid_dim_[0]*grid_dim_[1]*grid_dim_[2];
    }
    
    virtual ~VoxelGrid() {}

};

#endif // VOXEL_GRID_H_
