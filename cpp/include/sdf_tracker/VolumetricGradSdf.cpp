//============================================================================
// Name        : VolumetricGradSdf.cpp
// Author      : Christiane Sommer
// Modified    : Lu Sang
// Date        : 09/2021, origin on 10/2018
// License     : GNU General Public License
// Description : implementation of class VolumetricGradSdf
//============================================================================

#include "VolumetricGradSdf.h"
#include "normals/NormalEstimator.h"
#include "mesh/MarchingCubes.h"

void VolumetricGradSdf::init() {
    
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
    // initialize visibility map
    // vis_.resize(num_voxels_);

    std::vector<bool> vis;
    vis_ = new std::vector<bool>[num_voxels_];
    std::fill_n(vis_, num_voxels_, vis);
    
    std::cout << "Number of voxels: " << num_voxels_ << std::endl
              << "Memory used: " << num_voxels_ * 5. * sizeof(float) / (1024.*1024.) << "MB" << std::endl;

}

VolumetricGradSdf::~VolumetricGradSdf() {

    // free grid memory
    delete[] tsdf_;
    delete[] vis_;
    // overwrite pointer
    tsdf_     = nullptr;
    vis_    = nullptr;

}

void VolumetricGradSdf::update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst) {

    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    
    const float z_min = 0., z_max = 1./0.;
    
    // size_t lin_index = 0;
    
    cv::Mat nx, ny, nz, med_depth;
    cv::medianBlur(depth, med_depth, 5); // median filtering
    
    if (!NEst) { // TODO: implement on-the-go normal estimation?
        std::cerr << "No normal estimation possible - cannot update SDF volume!" << std::endl;
        return;
    }
    
    NEst->compute(depth, nx, ny, nz);
    cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
    
    const Mat3f R = pose.rotationMatrix();
    const Mat3f Rt = R.transpose();
    const Vec3f t = pose.translation();
    
    size_t vis_size;

  
    for (size_t k=0; k<grid_dim_[2]; ++k) for (size_t j=0; j<grid_dim_[1]; ++j) for (size_t i=0; i<grid_dim_[0]; ++i) {
        
        size_t lin_index = i + grid_dim_[0] * j + grid_dim_[0] * grid_dim_[1] * k;

        Vec3f point = voxel2world(i, j, k);
        point = Rt * (point - t);
        if (point[2]<0.f)
            continue;

        const int n = static_cast<int>(cx + fx * point[0] / point[2] + 0.5); // +0.5 is needed for proper rounding
        const int m = static_cast<int>(cy + fy * point[1] / point[2] + 0.5);

        
        if ((n<0) || (n>=depth.cols) || (m<0) || (m>=depth.rows))
            continue;

        const float z = depth.at<float>(m, n); // no/nn interpolation
        const cv::Vec3f z_color = color.at<cv::Vec3f>(m, n);
        
        if (z <= z_min_ || z >= z_max_ )//|| z != med_depth.at<float>(m,n))
            continue;

        // const float sdf = point[2] - z; // point-to-point, negative outside, positive inside
        const float sdf = z - point[2]; //negative inside, positive outside
        const float w = weight(sdf);
        if (w == 0) 
            continue;

        const float nx_i = nx.at<float>(m, n);
        const float ny_i = ny.at<float>(m, n);
        const float nz_i = nz.at<float>(m, n);

        Vec3f normal(nx_i, ny_i, nz_i);
        if (normal.squaredNorm() < .1) // invalid normal
            continue;
        
        Vec3f xy_hom = (1. / point[2]) * point;
        if (normal.dot(xy_hom) * normal.dot(xy_hom) * n_sq_inv_ptr->at<float>(m, n) < .25 * .25) // normal direction too far from viewing ray direction (>75.5°)
            continue;

        SdfVoxel& v = tsdf_[lin_index];
        std::vector<bool>& vis = vis_[lin_index];
        
        v.weight += w;
        v.dist += (truncate(sdf) - v.dist) * w / v.weight;
        v.grad -= w * R * Vec3f(nx_i, ny_i, nz_i); // normals are inward-pointing!

        v.r += (z_color[2] - v.r) * w / v.weight;
        v.g += (z_color[1] - v.g) * w / v.weight;
        v.b += (z_color[0] - v.b) * w / v.weight;

        vis.resize(counter_);
        vis.push_back(true);
        
        vis_size = vis.size();
            
    }
        // ++lin_index;
    std::cout << " --------------------------------------------------current counter  " << counter_ << std::endl;
    std::cout << " --------------------------------------------------current vis size " << vis_size<<std::endl;
}

bool VolumetricGradSdf::save_normal(const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst, std::string filename)
{
    //DEBUG
    // std::ofstream normal_file;
    // normal_file.open((filename + ".txt").c_str());

    // if(!normal_file.is_open()){
    //     std::cout << " can't save normal file!" << std::endl;
    //     return false; 
    // }

    cv::Mat output(cv::Size(depth.cols, depth.rows), CV_32FC3, cv::Scalar(.0, .0, .0));


    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    
    const float z_min = 0., z_max = 1./0.;
    
    // size_t lin_index = 0;
    
    cv::Mat nx, ny, nz, med_depth;
    cv::medianBlur(depth, med_depth, 5); // median filtering
    
    if (!NEst) { // TODO: implement on-the-go normal estimation?
        std::cerr << "No normal estimation possible - cannot update SDF volume!" << std::endl;
        return false;
    }
    
    NEst->compute(depth, nx, ny, nz);
    cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
    
    const Mat3f R = pose.rotationMatrix();
    const Mat3f Rt = R.transpose();
    const Vec3f t = pose.translation();
    
  
    for (size_t k=0; k<grid_dim_[2]; ++k) for (size_t j=0; j<grid_dim_[1]; ++j) for (size_t i=0; i<grid_dim_[0]; ++i) {
        
        size_t lin_index = i + grid_dim_[0] * j + grid_dim_[0] * grid_dim_[1] * k;

        Vec3f point = voxel2world(i, j, k);
        point = Rt * (point - t);
        if (point[2]<0.f)
            continue;

        const int n = static_cast<int>(cx + fx * point[0] / point[2] + 0.5); // +0.5 is needed for proper rounding
        const int m = static_cast<int>(cy + fy * point[1] / point[2] + 0.5);

        
        if ((n<0) || (n>=depth.cols) || (m<0) || (m>=depth.rows))
            continue;

        const float z = depth.at<float>(m, n); // no/nn interpolation
        
        if (z <= z_min_ || z >= z_max_ )//|| z != med_depth.at<float>(m,n))
            continue;

        // const float sdf = point[2] - z; // point-to-point, negative outside, positive inside
        const float sdf = z - point[2]; //negative inside, positive outside
        const float w = weight(sdf);
        if (w == 0) 
            continue;

        const float nx_i = nx.at<float>(m, n);
        const float ny_i = ny.at<float>(m, n);
        const float nz_i = nz.at<float>(m, n);

        // Vec3f N(nx_i, ny_i, nz_i);
        // N = N.normalized();
        // // N = R.transpose()*N;

        cv::Vec3f& color = output.at<cv::Vec3f>(m, n);
        color[0] = (-nx_i + 1.0f)/2.0f*255.0f;
        color[1] = (-ny_i + 1.0f)/2.0f*255.0f;
        color[2] = (-nz_i + 1.0f)/2.0f*255.0f;
        // color[0] = (-N[0] + 1.0f)/2.0f*255.0f;
        // color[1] = (-N[1] + 1.0f)/2.0f*255.0f;
        // color[2] = (-N[2] + 1.0f)/2.0f*255.0f;

        // normal_file << n << " " << m << " " << -N[0] << " " << -N[1] << " " << -N[2] << "\n";
    }

    // normal_file.close();
    cv::Mat image;
    output.convertTo(image, CV_8UC3);

    cv::imwrite(filename + "normal_map.png", image);
    std::cout << "==== DEBUG: normal map for frame saved!" << std::endl;

    return true;

}

bool VolumetricGradSdf::extract_mesh(std::string filename) {

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

        //  if (v.weight < 5) {
        //     ++lin_index;
        //     continue;
        // }
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

    // MarchingCubes mc;
    // mc.set_resolution(dim[0], dim[1], dim[2]);
    // mc.init_all();

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

        // mc.set_data(tsdf_[lin_idx].dist, i ,j ,k);
    }
 
    // mc.run();
    // mc.clean_temps();

    // mc.writePLY(filename.c_str());

    // mc.clean_all();
    // call marching cubes
   
    MarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    mc.computeIsoSurface(dist, weights, red, green, blue);
    bool success = mc.savePly(filename);
    
    // delete temporary arrays  
    delete[] dist;
    delete[] weights;
    
    return success;

}

bool VolumetricGradSdf::extract_pc(std::string filename) {

    std::vector<Eigen::Matrix<float, 9 ,1> > points_normals_colors;
    
    size_t lin_index = 0;
    
    for (size_t k=0; k<grid_dim_[2]; ++k) for (size_t j=0; j<grid_dim_[1]; ++j) for (size_t i=0; i<grid_dim_[0]; ++i) {
    
        const SdfVoxel& v = tsdf_[lin_index];
        std::vector<bool> vis = vis_[lin_index];
        auto sum = std::count(vis.begin(), vis.end(), true);
        
        if ((std::fabs(v.dist) > std::sqrt(3.0)*voxel_size_) || (sum < 1)) {
            ++lin_index;
            continue;
        }
        
        Vec3f g = v.grad.normalized();
        Vec3f d = v.dist * g;
        float voxel_size_2 = .5 * voxel_size_;
        // if (std::fabs(d[0]) < voxel_size_2 && std::fabs(d[1]) < voxel_size_2 && std::fabs(d[2]) < voxel_size_2) {
            Eigen::Matrix<float, 9, 1> pnc;
            pnc.segment<3>(0) = vox2float(i,j,k) - d;
            pnc.segment<3>(3) = g;
            pnc.segment<3>(6) = Vec3f(v.r, v.g, v.b);
            points_normals_colors.push_back(pnc);
        // }
        ++lin_index;
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points_normals_colors.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property float nx" << std::endl;
    plyFile << "property float ny" << std::endl;
    plyFile << "property float nz" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (const Eigen::Matrix<float, 9, 1>& p : points_normals_colors) {
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << int(255 * p[6]) << " " << int(255 * p[7]) << " " << int(255 * p[8]) << std::endl;
    }
    
    plyFile.close();
    std::cout << "total " << points_normals_colors.size() << "surface points are written." << std::endl;
    return true;
}

// this code is to save sdf and compare with the SDFGen generated GT sdf from mesh obj
bool VolumetricGradSdf::saveSDF(std::string filename){
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
    Vec3f origin = - voxel_size_*Vec3i(xmin, ymin, zmin).cast<float>();

    Vec3f bottom = Vec3i(xmin, ymin, zmin).cast<float>()*voxel_size_;
    Vec3f up = Vec3i(xmax, ymax, zmax).cast<float>()*voxel_size_ ;

    std::cout << "Bounding box size:  (" << bottom.transpose() << ") to (" << up.transpose() << ")  with dimensions " << dim.transpose() << std::endl; 
   
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


void VolumetricGradSdf::check_vis_map()
{   
    int count = 0;
    for (size_t lin_idx = 0; lin_idx < num_voxels_; lin_idx++){
        std::vector<bool>& vis = vis_[lin_idx];
        SdfVoxel& v = tsdf_[lin_idx];
        if(std::fabs(v.dist) < std::sqrt(3.)*voxel_size_){
        
            if(count%100 == 0){
                std::cout << "voxel " << lin_idx << ":\t" << std::endl;
                for(size_t frame = 0; frame < vis.size(); frame++){
                        std::cout << " frame " << frame << " vis: " << vis[frame] <<" ";
                }
                std::cout << std::endl;
               
            }
            count++;
        }
    }
                
}


// calculate the subsampled indx and grad, 
Vec8f VolumetricGradSdf::subsample(const SdfVoxel& v)
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

    // d[0] = dist + voxel_size_4 * (-grad[0] - grad[1] - grad[2] - 1);
    // d[1] = dist + voxel_size_4 * ( grad[0] - grad[1] - grad[2] - 1);
    // d[2] = dist + voxel_size_4 * (-grad[0] + grad[1] - grad[2] - 1);
    // d[3] = dist + voxel_size_4 * ( grad[0] + grad[1] - grad[2] - 1);
    // d[4] = dist + voxel_size_4 * (-grad[0] - grad[1] + grad[2] - 1);
    // d[5] = dist + voxel_size_4 * ( grad[0] - grad[1] + grad[2] - 1);
    // d[6] = dist + voxel_size_4 * (-grad[0] + grad[1] + grad[2] - 1);
    // d[7] = dist + voxel_size_4 * ( grad[0] + grad[1] + grad[2] - 1);
    return d;
}