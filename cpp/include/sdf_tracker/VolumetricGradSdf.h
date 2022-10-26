//============================================================================
// Name        : VolumetricGradSdf.h
// Author      : Christiane Sommer
// Date        : 09/2019
// License     : GNU General Public License
// Description : header file for class VolumetricGradSdf
//============================================================================

#ifndef VOLUMETRIC_GRAD_SDF_H_
#define VOLUMETRIC_GRAD_SDF_H_

// includes
#include <iostream>
//#include <fstream>
// class includes
#include "VoxelGrid.h"
#include "Sdfvoxel.h"
#include "Sdf.h"

/**
 * class declaration
 */
class VolumetricGradSdf : public VoxelGrid, public Sdf {

// friend class
friend class Optimizer;
friend class PsOptimizer;
friend class GsOptimizer;
friend class LedOptimizer;
friend class ImgRender;

// struct SdfVoxel {
//     float dist = 0;
//     Vec3f grad = Vec3f::Zero();
//     float weight = 0;
// };

// variables

    SdfVoxel* tsdf_;
    // std::vector<std::vector<bool>>* vis_;
    std::vector<bool>* vis_;
    
    void init();
    void subsampling();
    
public:

// constructors / destructor
    
    VolumetricGradSdf() :
        VoxelGrid(),
        Sdf()
    {
        init();
    }
    
    VolumetricGradSdf(Vec3i grid_dim, float voxel_size, Vec3f shift) :
        VoxelGrid(grid_dim, voxel_size, shift),
        Sdf()
    {
        init();
    }
    
    VolumetricGradSdf(Vec3i grid_dim, float voxel_size, Vec3f shift, float T) :
        VoxelGrid(grid_dim, voxel_size, shift),
        Sdf(T)
    {
        init();
    }
    
    ~VolumetricGradSdf();
    
// methods
    
    virtual float tsdf(Vec3f point, Vec3f* grad_ptr) const {
        int I = nearest_index(point);
        if (I < 0) {
            if (grad_ptr)
                (*grad_ptr) = Vec3f::Zero();
            return T_;
        }
        const SdfVoxel& v = tsdf_[I];
        if (grad_ptr)
            (*grad_ptr) = v.grad.normalized();
        return v.dist + v.grad.normalized().dot(voxel2world(world2voxel(point)) - point);
    }
    
    virtual float weights(Vec3f point) const {
        int I = nearest_index(point);
        if (I < 0) return 0;
        return tsdf_[I].weight;
    }
    
    virtual void update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst);

    SdfVoxel getVoxel(Vec3i& idx){
        return tsdf_[idx2line(idx)];
    }

    SdfVoxel getVoxel(const int i, const int j, const int k){
        Vec3i idx(i,j,k);
        return tsdf_[idx2line(idx)];
    }

// visualization / debugging
    
    virtual bool extract_mesh(std::string filename);

    virtual bool extract_pc(std::string filename);

    virtual bool saveSDF(std::string filename);

    virtual void check_vis_map();

    virtual bool save_normal(const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst, std::string filename);

    Vec8f subsample(const SdfVoxel& v);

};

#endif // VOLUMETRIC_GRAD_SDF_H_
