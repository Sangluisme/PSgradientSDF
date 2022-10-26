//============================================================================
// Name        : VolumetricGradPixelSdf.cpp
// Author      : Christiane Sommer
// Modified    : Lu Sang
// Date        : 09/2021, original on 10/2018
// License     : GNU General Public License
// Description : implementation of class VolumetricGradPixelSdf
//============================================================================

#include "VolumetricGradPixelSdf.h"
#include "normals/NormalEstimator.h"
// #include "normals/NormalSetup.h"
#include "mesh/MarchingCubes.h"


void VolumetricGradPixelSdf::init() {
    
    SdfVoxel v;
    v.dist = T_;
    v.grad = Vec3f::Zero();
    v.weight = 0;
    v.r = 0.0;
    v.g = 0.0;
    v.b = 0.0;
    
    // allcoate grids
    tsdf_     = new SdfVoxel[num_voxels_];
    // initialize grids
    std::fill_n(tsdf_, num_voxels_, v);

    // allocate vis map
    // vis_.resize(num_voxels_);
    std::vector<bool> vis;
    vis_ = new std::vector<bool>[num_voxels_];
    std::fill_n(vis_, num_voxels_, vis);
    
    std::cout << "Number of voxels: " << num_voxels_ << std::endl
              << "Memory used: " << num_voxels_ * 5. * sizeof(float) / (1024.*1024.) << "MB" << std::endl;

}

VolumetricGradPixelSdf::~VolumetricGradPixelSdf() {

    // free grid memory
    delete[] tsdf_;
    // overwrite pointer
    tsdf_ = nullptr;
    // free vis map memory
    delete[] vis_;

}

void VolumetricGradPixelSdf::update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst) {
    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    
    size_t lin_index = 0;
    
    cv::Mat nx, ny, nz, med_depth;
    // cv::medianBlur(depth, med_depth, 5); // median filtering
    
    if (!NEst) { // TODO: implement on-the-go normal estimation?
        std::cerr << "No normal estimation possible - cannot update SDF volume!" << std::endl;
        return;
    }
    
    // NEst->compute(med_depth, nx, ny, nz);
    NEst->compute(depth, nx, ny, nz);
    
    const Mat3f R = pose.rotationMatrix();
    const Mat3f Rt = pose.rotationMatrix().transpose();
    const Vec3f t = pose.translation();
    
    cv::Mat* x0_ptr = NEst->x0_ptr();
    cv::Mat* y0_ptr = NEst->y0_ptr(); 
    cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
    const float* x_hom_ptr = (const float*)x0_ptr->data;
    const float* y_hom_ptr = (const float*)y0_ptr->data;
    const float* hom_inv_ptr = (const float*)n_sq_inv_ptr->data;
    const float* z_ptr = (const float*)depth.data;
    const float* zm_ptr = (const float*)med_depth.data;
    
    const float* nx_ptr = (const float*)nx.data;
    const float* ny_ptr = (const float*)ny.data;
    const float* nz_ptr = (const float*)nz.data;
    
    const int factor = std::ceil(T_ / voxel_size_);

    size_t vis_length;
    for (size_t m = 0; m < depth.rows; m++) for (size_t n = 0; n < depth.cols; n++) {
    
        const size_t idx = m * depth.cols + n;
        
        const float z = z_ptr[idx];
        const cv::Vec3f z_color = color.at<cv::Vec3f>(m, n);
        
        // if (z <= z_min_ || z >= z_max_ || z != zm_ptr[idx])
        if (z <= z_min_ || z >= z_max_ )//|| z != z_ptr[idx])
            continue;

        const Vec3f xy_hom(x_hom_ptr[idx], y_hom_ptr[idx], 1.);
        const Vec3f Rpt(z * R * xy_hom + t);
        const Vec3f normal(nx_ptr[idx], ny_ptr[idx], nz_ptr[idx]);
        const Vec3f Rn(R * normal);
        
        if (normal.squaredNorm() < .1) // invalid normal
            continue;
        
        if (normal.dot(xy_hom) * normal.dot(xy_hom)  * hom_inv_ptr[idx] < .1 *.25) // normal direction too far from viewing ray direction (>75.5°)
            continue;
        
        for (float k = -factor; k <= factor; ++k) { // loop along ray

            Vec3f point = (z + k*voxel_size_) * R * xy_hom + t; // convert point into Sdf coordinate system
           
            const Vec3i vi = world2voxel(point);
            // std::cout << vi.transpose() << std::endl;
          
            if (vi[0] < 0 || vi[0] >= grid_dim_[0] || vi[1] < 0 || vi[1] >= grid_dim_[1] || vi[2] < 0 || vi[2] >= grid_dim_[2])
                continue;
            size_t lin_index = vi[0] + grid_dim_[0] * vi[1] + grid_dim_[0] * grid_dim_[1] * vi[2];
            point = Rt * (voxel2world(vi) - t);
            // const float sdf = point[2] - z;
            const float sdf = z - point[2]; //positive out side and negative inside, to be easy to calculate the gradient of distance
            // Vec3f point = Rpt + k * voxel_size_ * Rn; // convert point into Sdf coordinate system
            // const Vec3i vi = world2voxel(point);
            // if (vi[0] < 0 || vi[0] >= grid_dim_[0] || vi[1] < 0 || vi[1] >= grid_dim_[1] || vi[2] < 0 || vi[2] >= grid_dim_[2])
            //     continue;
            // size_t lin_index = vi[0] + grid_dim_[0] * vi[1] + grid_dim_[0] * grid_dim_[1] * vi[2];
            // const float sdf = Rn.dot(voxel2world(vi) - Rpt);
            const float w = weight(sdf);
            std::vector<bool>& vis = vis_[lin_index];
            if (w>0) {
                SdfVoxel& v = tsdf_[lin_index];
                v.weight += w;
                v.dist += (truncate(sdf) - v.dist) * w / v.weight;
                v.grad -= w * Rn; // * normal; // normals are inward-pointing! so add the oppsite sign

                v.r += (w*(z_color[2] - v.r) - v.r) / 2;
                v.g += (w*(z_color[1] - v.g) - v.g) / 2;
                v.b += (w*(z_color[0] - v.b) - v.b) / 2;

                vis.resize(counter_);
                vis.push_back(true);
                vis_length = vis.size();
                // }
            }
        }
    
    }
    std::cout << "-------------------------->Current frame counter: " << counter_ << std::endl;
    std::cout << "-------------------------->Current vis length: " << vis_length << std::endl;  // DEBUG    
}

bool VolumetricGradPixelSdf::extract_mesh(std::string filename) {

    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;

    size_t lin_index = 0;
    for (int k=0; k<grid_dim_[2]; ++k) for (int j=0; j<grid_dim_[1]; ++j) for (int i=0; i<grid_dim_[0]; ++i) {
        const SdfVoxel& v = tsdf_[lin_index];

         if (v.weight < 10) {
            ++lin_index;
            continue;
        }

        if (i < xmin) xmin = i;
        if (i > xmax) xmax = i;
        if (j < ymin) ymin = j;
        if (j > ymax) ymax = j;
        if (k < zmin) zmin = k;
        if (k > zmax) zmax = k;
        ++lin_index;
    }
    std::cout << "Size limits:" << std::endl
            << xmin << "\t" << xmax << std::endl
            << ymin << "\t" << ymax << std::endl
            << zmin << "\t" << zmax << std::endl;

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    float* dist = new float[num_voxels];
    float* weights = new float[num_voxels];
    unsigned char* red = new unsigned char[num_voxels];
    unsigned char* green = new unsigned char[num_voxels];
    unsigned char* blue = new unsigned char[num_voxels];

    size_t pos = 0;
    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i,j,k);
        size_t lin_idx = idx2line(idx);
        dist[pos] = -tsdf_[lin_idx].dist;
        weights[pos] = tsdf_[lin_idx].weight;
        red[pos] = int(255*tsdf_[lin_idx].r);
        green[pos] = int(255*tsdf_[lin_idx].g);
        blue[pos] = int(255*tsdf_[lin_idx].b);
        pos++;
    }
 
    
    // call marching cubes
    // MarchingCubesNoColor mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));

    // mc.computeIsoSurface(dist, weights);
    // bool success = mc.savePly(filename);

    MarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    mc.computeIsoSurface(dist, weights, red, green, blue);
    bool success = mc.savePly(filename);
    
    // delete temporary arrays  
    delete[] dist;
    delete[] weights;
    
    return success;
}

bool VolumetricGradPixelSdf::extract_pc(std::string filename) {

    std::vector<Vec6f> points_normals;
    
    size_t lin_index = 0;
    
    for (size_t k=0; k<grid_dim_[2]; ++k) for (size_t j=0; j<grid_dim_[1]; ++j) for (size_t i=0; i<grid_dim_[0]; ++i) {
    
        const SdfVoxel& v = tsdf_[lin_index];
        
        if (v.weight < 10) {
            ++lin_index;
            continue;
        }
        
        Vec3f g = v.grad.normalized();
        Vec3f d = v.dist * g;
        float voxel_size_2 = .5 * voxel_size_;
        if (std::fabs(d[0]) < voxel_size_2 && std::fabs(d[1]) < voxel_size_2 && std::fabs(d[2]) < voxel_size_2) {
            Vec6f pn;
            // pn.segment<3>(0) = voxel2world(i, j, k) - d;
            pn.segment<3>(0) = vox2float(i,j,k) - d;
            pn.segment<3>(3) = g;
            points_normals.push_back(pn);
        }
        ++lin_index;
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points_normals.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property float nx" << std::endl;
    plyFile << "property float ny" << std::endl;
    plyFile << "property float nz" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (const Vec6f& p : points_normals) {
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
    }
    
    plyFile.close();

    return true;
}

// this code is to save sdf and compare with the SDFGen generated GT sdf from mesh obj
bool VolumetricGradPixelSdf::saveSDF(std::string filename){
    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;

    size_t lin_index = 0;
    for (int k=0; k<grid_dim_[2]; ++k) for (int j=0; j<grid_dim_[1]; ++j) for (int i=0; i<grid_dim_[0]; ++i) {
        const SdfVoxel& v = tsdf_[lin_index];

        if (std::fabs(v.dist) > std::sqrt(3)*voxel_size_) { 
            ++lin_index;
            continue;
        }

        if (i < xmin) xmin = i;
        if (i > xmax) xmax = i;
        if (j < ymin) ymin = j;
        if (j > ymax) ymax = j;
        if (k < zmin) zmin = k;
        if (k > zmax) zmax = k;
        ++lin_index;
    }
    // std::cout << "Size limits:" << std::endl
    //         << xmin << "\t" << xmax << std::endl
    //         << ymin << "\t" << ymax << std::endl
    //         << zmin << "\t" << zmax << std::endl;

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    std::ofstream sdfFile;
    sdfFile.open(filename.c_str());
    if (!sdfFile.is_open())
        return false;

    sdfFile << dim[0] << " " << dim[1] << " " << dim[2] << "\n";
 
    //compute bounding box

    // Vec3f origin = - voxel_size_*Vec3i(xmin, ymin, zmin).cast<float>();
    Vec3f bottom = Vec3i(xmin, ymin, zmin).cast<float>()*voxel_size_;
    Vec3f up = Vec3i(xmax, ymax, zmax).cast<float>()*voxel_size_;

    std::cout << "Bounding box size:  (" << bottom.transpose() << ") to (" << up.transpose() << ") with dimensions " << dim.transpose() << std::endl; 
   
    sdfFile << bottom[0] << " " << bottom[1] << " " << bottom[2] << "\n";
    sdfFile << voxel_size_ << "\n";

    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i,j,k);
        size_t lin_idx = idx2line(idx);
        sdfFile << -tsdf_[lin_idx].dist << "\n";
    }

    sdfFile.close();
    return true;

}

// calculate the subsampled indx and grad, 
Vec8f VolumetricGradPixelSdf::subsample(const SdfVoxel& v)
{
    Vec8f d;
    Vec3f grad = v.grad.normalized();
    float dist = v.dist;

    const float voxel_size_4 = 0.25 * voxel_size_;
    d[0] = dist + voxel_size_4 * (-grad[0] - grad[1] - grad[2]);
    d[1] = dist + voxel_size_4 * ( grad[0] - grad[1] - grad[2]);
    d[2] = dist + voxel_size_4 * (-grad[0] + grad[1] - grad[2]);
    d[3] = dist + voxel_size_4 * ( grad[0] + grad[1] - grad[2]);
    d[4] = dist + voxel_size_4 * (-grad[0] - grad[1] + grad[2]);
    d[5] = dist + voxel_size_4 * ( grad[0] - grad[1] + grad[2]);
    d[6] = dist + voxel_size_4 * (-grad[0] + grad[1] + grad[2]);
    d[7] = dist + voxel_size_4 * ( grad[0] + grad[1] + grad[2]);

    return d;
}