//============================================================================
// Name        : VoxelGrid.cpp
// Author      : Christiane Sommer
// Date        : 10/2018
// License     : GNU General Public License
// Description : implementation of class VoxelGrid
//============================================================================

#include "VoxelGrid.h"
    
float VoxelGrid::interp3(const float* grid, Vec3f point, float extrap) const {

    Vec3f float_idx = world2voxelf(point);
    const float i = float_idx[0], j = float_idx[1], k = float_idx[2];

    // remove out of volume cases
    if (i<=0 || j<=0 || k<=0 || i>=(grid_dim_[0]-1) || j>=(grid_dim_[1]-1) || k>=(grid_dim_[2]-1))
        return extrap;
    
    const int im = static_cast<int>(i);
    const int jm = static_cast<int>(j);
    const int km = static_cast<int>(k);
    const float dx = i-im;
    const float dy = j-jm;
    const float dz = k-km;
    
    int I = im + jm * grid_dim_[0] + km * grid_dim_[0] * grid_dim_[1]; // linear index i-dx, j-dy, k-dz
    const float V_imjmkm = grid[I];
    I += 1; // i+1-dx , j-dy , k-dz
    const float V_ipjmkm = grid[I];
    I += grid_dim_[0]; // i+1-dx , j+1-dy , k-dz
    const float V_ipjpkm = grid[I];
    I -= 1; // i-dx , j+1-dy , k-dz
    const float V_imjpkm = grid[I];
    I += grid_dim_[0]*grid_dim_[1]; // i-dx , j+1-dy , k+1-dz
    const float V_imjpkp = grid[I];
    I -= grid_dim_[0]; // i-dx , j-dy , k+1-dz
    const float V_imjmkp = grid[I];
    I += 1; // i+1-dx , j-dy , k+1-dz
    const float V_ipjmkp = grid[I];
    I += grid_dim_[0]; // i+1-dx , j+1-dy , k+1-dz
    const float V_ipjpkp = grid[I];
    
    // interpolate in z-direction
    const float V_imjm = (1-dz) * V_imjmkm + dz * V_imjmkp;
    const float V_imjp = (1-dz) * V_imjpkm + dz * V_imjpkp;
    const float V_ipjm = (1-dz) * V_ipjmkm + dz * V_ipjmkp;
    const float V_ipjp = (1-dz) * V_ipjpkm + dz * V_ipjpkp;
    // next level: y-direction
    const float V_im = (1-dy) * V_imjm + dy * V_imjp;
    const float V_ip = (1-dy) * V_ipjm + dy * V_ipjp;
    // final interpolated value: x-direction
    return (1-dx) * V_im + dx * V_ip;
    
}

int VoxelGrid::nearest_index(Vec3f point) const {

    Vec3f float_idx = world2voxelf(point);
    const float i = float_idx[0], j = float_idx[1], k = float_idx[2];

    // remove out of volume cases
    if (i<=0 || j<=0 || k<=0 || i>=(grid_dim_[0]-1) || j>=(grid_dim_[1]-1) || k>=(grid_dim_[2]-1))
        return -1;
    
    const int im = static_cast<int>(i + 0.5);
    const int jm = static_cast<int>(j + 0.5);
    const int km = static_cast<int>(k + 0.5);
    
    return im + jm * grid_dim_[0] + km * grid_dim_[0] * grid_dim_[1]; // linear index of closest voxel
    
}

// TODO: re-implement slice functions in a more efficient way!

void VoxelGrid::slice_x(float* grid, size_t i, cv::Mat &slice) {

    slice = cv::Mat(grid_dim_[1], grid_dim_[2], CV_32FC1);
    std::ofstream outfile("../../../results/tmp_xslice.dat");
    for (size_t k=0; k<grid_dim_[2]; ++k) for (size_t j=0; j<grid_dim_[1]; ++j) {
        float val = eval(grid, i, k, j);
        slice.at<float>(k, j) = val;
        outfile << val << "\t";
        
    }
    outfile.close();

}


void VoxelGrid::slice_y(float* grid, size_t j, cv::Mat &slice) {

    slice = cv::Mat(grid_dim_[2], grid_dim_[0], CV_32FC1);
    std::ofstream outfile("../../../results/tmp_yslice.dat");
    for (size_t i=0; i<grid_dim_[0]; ++i) for (size_t k=0; k<grid_dim_[2]; ++k) {
        float val = eval(grid, i, k, j);
        slice.at<float>(i, k) = val;
        outfile << val << "\t";
    }
    outfile.close();

}

void VoxelGrid::slice_z(float* grid, size_t k, cv::Mat &slice) {

    slice = cv::Mat(grid_dim_[0], grid_dim_[1], CV_32FC1);
    std::ofstream outfile("../../../results/tmp_zslice.dat");
    for (size_t j=0; j<grid_dim_[1]; ++j) for (size_t i=0; i<grid_dim_[0]; ++i) {
        float val = eval(grid, i, k, j);
        slice.at<float>(j, i) = val;
        outfile << val << "\t";
    }
    outfile.close();

}

void VoxelGrid::write(float* grid, std::string file) {

    std::ofstream outfile("../../../results/" + file);
    for (int I=0; I<num_voxels_; ++I) {
        outfile << grid[I] << "\t";
    }
    outfile.close();
}


