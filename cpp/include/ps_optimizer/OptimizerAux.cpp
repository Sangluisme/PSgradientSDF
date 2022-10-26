//============================================================================
// Name        : OptimizerAux.cpp
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : implementation of Auxilary function of Optimizer
//============================================================================

#include "Optimizer.h"
#include "sdf_tracker/Sdfvoxel.h"
#include "mesh/MarchingCubes.h"
#include "mesh/MarchingCubesNoColor.h"
#include "mesh/GradMarchingCubes.h"
#include "Auxilary.h"
// #include "Timer.h"

Mat3x8i subindexMatrix()
{
    Mat3x8i Sub;
    Sub << 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 1, 1, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1;
    return Sub;

}

Mat3x8i getSubVoxelIndex(Vec3i idx){
    Mat3x8i sub_idx = 2*idx.replicate(1,8) + subindexMatrix();

    return sub_idx;
}

Eigen::Matrix<int, 8, 1> getFlattenIndex(Mat3x8i& index, Vec3i grid_dim){
    Eigen::Matrix<int, 8, 1> lin_idx;
    for(size_t i = 0 ; i < 8; i++){
        lin_idx[i] = index(0,i) + index(1,i)*grid_dim[0] + index(2,i)*grid_dim[0]*grid_dim[1];
    }
    return lin_idx;
}



void Optimizer::getAlldsitance(Eigen::VectorXf& Distance)
{
    // Distance.clear();
    Distance.resize(surface_points_.size());
    int row = 0;
    for (auto& lin_idx: surface_points_){
        const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
        Distance[row] = v.dist;
        row++;
    }
}

void Optimizer::setAlldsitance(Eigen::VectorXf& Distance)
{
    size_t row = 0;
    for (auto& lin_idx: surface_points_){
        tSDF_->tsdf_[lin_idx].dist = Distance[row] ;
        row++;
    }
}

void Optimizer::getAllalbedo(Eigen::VectorXf& rho)
{
    rho.resize(surface_points_.size()*3);
    size_t row = 0;
    for(auto& lin_idx: surface_points_){
        rho[3*row] = tSDF_->tsdf_[lin_idx].r; 
        rho[3*row+1] = tSDF_->tsdf_[lin_idx].g;
        rho[3*row+2] = tSDF_->tsdf_[lin_idx].b; 
        row++;
    }
}

void Optimizer::setAllalbedo(Eigen::VectorXf& rho)
{
    size_t row = 0;
    for(auto& lin_idx: surface_points_){
        tSDF_->tsdf_[lin_idx].r = rho[3*row];
        tSDF_->tsdf_[lin_idx].g = rho[3*row+1];
        tSDF_->tsdf_[lin_idx].b = rho[3*row+2];
        row++;
    }
}

Eigen::VectorXf Optimizer::stackPoses()
{
    Eigen::VectorXf xi(6*num_frames_);
    for(size_t frame = 0; frame < num_frames_; frame++){
        Vec6f xi_cur = SE3(poses_[frame]).log();
        // std::cout << xi_cur.transpose() << std::endl;
        xi.segment(6*frame, 6) = xi_cur;
    }
    return xi;
}

void Optimizer::setPoses(Eigen::VectorXf& xi)
{   
    for(size_t frame = 0; frame< num_frames_; frame++){
        Vec6f xi_cur = xi.segment(frame*6, 6);
        poses_[frame].topRightCorner(3,1) = xi_cur.head(3);
        poses_[frame].topLeftCorner(3,3) = SO3::exp(xi_cur.tail(3)).matrix();
    }
}

// -------------- update function ---------------------------
void Optimizer::updateAlbedo(const int lin_idx, float delta_r, float delta_g, float delta_b)
{
	float r = tSDF_->tsdf_[lin_idx].r - delta_r;
	tSDF_->tsdf_[lin_idx].r = std::min(std::max(r, 0.0f), 1.0f);

	float g = tSDF_->tsdf_[lin_idx].g - delta_g;
	tSDF_->tsdf_[lin_idx].g = std::min(std::max(g, 0.0f), 1.0f);

	float b = tSDF_->tsdf_[lin_idx].b - delta_b;
	tSDF_->tsdf_[lin_idx].b = std::min(std::max(b, 0.0f), 1.0f);
}

void Optimizer::updateAlbedo(Eigen::VectorXf& delta_r)
{
	int row = 0;
	int count = 0;
	for(auto& lin_idx: surface_points_){
		float r = tSDF_->tsdf_[lin_idx].r - delta_r[3*row];
		float g = tSDF_->tsdf_[lin_idx].g - delta_r[3*row+1];
		float b = tSDF_->tsdf_[lin_idx].b - delta_r[3*row+2];

        // std::cout << "r: " << r << " g: " << g << " b: " << b << "\n";
		// updateAlbedo(lin_idx, r, g, b);
		if(r > 0.0 && r < 1.0  )
		{
			tSDF_->tsdf_[lin_idx].r = r;
			count++;
		}
		if(g > 0.0 && g < 1.0 )
		{
			tSDF_->tsdf_[lin_idx].g = g;
			count++;
		}
		if(b > 0.0 && b < 1.0 )
		{
			tSDF_->tsdf_[lin_idx].b = b;
			count++;
		}
		row++;
	}
	//DEBUG
	std::cout << "-----DEBUG: total " << count << "(" << static_cast<float>(count)/static_cast<float>(3*row) << ") voxel color has valid update." << std::endl; 
}

void Optimizer::updateGrad()
{
	for(auto& lin_idx: surface_points_){
		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
		Vec3i idx = tSDF_->line2idx(lin_idx);
		Vec3f grad = computeDistGrad(v, idx).first;
		tSDF_->tsdf_[lin_idx].grad = grad;
	}
}

void Optimizer::updateDist(Eigen::VectorXf& delta_d, bool update_grad)
{
	int count_small = 0;
	int count = 0;
	int row = 0;
	for(auto& lin_idx: surface_points_){

		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];

		// Eigen::SparseVector<float>::OuterIterator it(delta_d, lin_idx);
		float d = delta_d[row];
		if(std::abs(d)<std::sqrt(3)*voxel_size_){
			tSDF_->tsdf_[lin_idx].dist -= d;
			count++;
		}

		if(std::fabs(d) <= 0.001*voxel_size_)count_small++; //count for too small update of distance
		// if(row%5000 == 0){std::cout << "voxel " << lin_idx << " delta d: " << d << std::endl;}
		row++;
	}

	std::cout << "----- DEBUG: total " << count << " voxels have valid update, (" << static_cast<float>(count)/static_cast<float>(surface_points_.size()) << ") are update.(" << static_cast<float>(count_small)/static_cast<float>(count) << ") are too small." << std::endl;

    if (update_grad){
        updateGrad();
    }
}

void Optimizer::updatePose(const int frame_id, const Mat3f& R, const Vec3f& t, const Vec6f& xi)
{
	poses_[frame_id].topRightCorner(3, 1) = t - xi.head(3);
	poses_[frame_id].topLeftCorner(3, 3) = R * SO3::exp(-xi.tail(3)).matrix();
}

void Optimizer::updatePose(Eigen::VectorXf& delta_xi)
{
	for(size_t frame = 0; frame < num_frames_; frame++){
		const Mat3f R = getRotation(frame);
		const Vec3f t = getTranslation(frame);
		const Vec6f& xi = delta_xi.segment(6*frame, 6);
		// std::cout << "frame " << frame << " xi " << xi.transpose() << std::endl;
		updatePose(frame, R, t, xi);
	}
}

bool Optimizer::getIntensity(const Vec3i& idx, const SdfVoxel& v, const Mat3f& R, const Vec3f& t, const cv::Mat& img, Vec3f& intensity)
{
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

    Mat3f Rt = R.transpose();
    Vec3f point = Rt * (tSDF_->voxel2world(idx) - v.dist * v.grad.normalized() - t);
	// Vec3f point = Rt * (tSDF_->voxel2world(idx) - t);


    const float z_inv = 1. / point[2];
    const float z_inv_sq = z_inv * z_inv;
	float m = fx * point[0] * z_inv + cx;
    float n = fy * point[1] * z_inv + cy;

	if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
        return false;
    }

    intensity = interpolateImage(n, m, img);
	return true;

}


//--------------------------------------------- some useful functions ----------------------------------------------------------------------------------------------

//! first extract the surface points then create a band around the surface points
void Optimizer::getSurfaceVoxel()
{
    surface_points_.clear();
    int count = 0;
    for (size_t k=0; k<tSDF_->grid_dim_[2]; ++k) for (size_t j=0; j<tSDF_->grid_dim_[1]; ++j) for (size_t i=0; i<tSDF_->grid_dim_[0]; ++i) {
        int lin_idx = tSDF_->idx2line(i,j,k);
        const SdfVoxel& v = tSDF_->tsdf_[lin_idx];

        std::vector<bool> vis = tSDF_->vis_[lin_idx];
        auto sum = std::count(vis.begin(), vis.end(), true);
        

        if (std::fabs(v.dist) <= std::sqrt(3.0)*voxel_size_ && (sum >= 1)) {
            surface_points_.push_back(lin_idx);
            count++;
        }
    }
    if(count != surface_points_.size()) std::cout << "wrong size surface point size!" << std::endl;
    std::cout << "surface points index are computed. Total " << count << " voxels are near surface." << std::endl;

}

float Optimizer::getTotalEnergy(float E, float E_n, float E_l, float E_r, std::ofstream& file)
{
    float E_total =  E + settings_->reg_weight_n*E_n + settings_->reg_weight_l*E_l + settings_->reg_weight_rho*E_r;
    std::cout << "PS energy: " << E << "\t normal reg energy: " << settings_->reg_weight_n*E_n <<  "\t laplacian reg energy: " << settings_->reg_weight_l*E_l << "\t rho reg energy: " << settings_->reg_weight_rho*E_r << "\t total energy: " << E_total << std::endl;

    if(file.is_open()){
        file <<  "PS energy: " << E << "\t normal reg energy: " << settings_->reg_weight_n*E_n <<  "\t laplacian reg energy: " << settings_->reg_weight_l*E_l << "\t rho reg energy: " << settings_->reg_weight_rho*E_r << "\t total energy: " << E_total << std::endl;
    }

    return E_total;
}






//----------------------------------------- save results function ---------------------------------------------------------------------------------------------------

bool Optimizer::extract_mesh(std::string filename)
{
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
    for (int k=0; k<tSDF_->grid_dim_[2]; ++k) for (int j=0; j<tSDF_->grid_dim_[1]; ++j) for (int i=0; i<tSDF_->grid_dim_[0]; ++i) {
    // for(auto& lin_index: surface_points_){
        const SdfVoxel& v = tSDF_->tsdf_[lin_index];
        // const Vec3i idx = tSDF_->line2idx(lin_index);
        // int i = idx[0];
        // int j = idx[1];
        // int k = idx[2];

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

    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    // std::cout << "voxel num:" << num_voxels << std::endl;

    float* dist = new float[num_voxels];
    float* weights = new float[num_voxels];
    Vec3f* grads = new Vec3f[num_voxels];
    unsigned char* red = new unsigned char[num_voxels];
    unsigned char* green = new unsigned char[num_voxels];
    unsigned char* blue = new unsigned char[num_voxels];

    size_t pos = 0;
    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i,j,k);
        size_t lin_idx = tSDF_->idx2line(idx);
        dist[pos] = -tSDF_->tsdf_[lin_idx].dist;
		// auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx);
		weights[pos] = tSDF_->tsdf_[lin_idx].weight;
        // grads[pos] = tSDF_->tsdf_[lin_idx].grad;
        red[pos] = int(255*tSDF_->tsdf_[lin_idx].r);
        green[pos] = int(255*tSDF_->tsdf_[lin_idx].g);
        blue[pos] = int(255*tSDF_->tsdf_[lin_idx].b);

        pos++;
    }

    MarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    mc.computeIsoSurface(dist, weights, red, green, blue);
    // GradMarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    // mc.computeIsoSurface(dist, weights, grads, red, green, blue);
   
    bool success = mc.savePly( save_path_ + filename + "_mesh.ply");

	if (!success){std::cout << "couldn't save mesh " <<save_path_ << filename  << std::endl;}

    // delete temporary arrays  
    if (dist)    delete[] dist;
    if (weights) delete[] weights;
    if (grads)   delete[] grads;
    if (red)     delete[] red;
    if (green)   delete[] green;
    if (blue)    delete[] blue;


	return success;
}

bool Optimizer::save_white_mesh(std::string filename)
{
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
    for (int k=0; k<tSDF_->grid_dim_[2]; ++k) for (int j=0; j<tSDF_->grid_dim_[1]; ++j) for (int i=0; i<tSDF_->grid_dim_[0]; ++i) {
    // for(auto& lin_index: surface_points_){
        const SdfVoxel& v = tSDF_->tsdf_[lin_index];
        // const Vec3i idx = tSDF_->line2idx(lin_index);
        // int i = idx[0];
        // int j = idx[1];
        // int k = idx[2];

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

    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    // std::cout << "voxel num:" << num_voxels << std::endl;

    float* dist = new float[num_voxels];
    float* weights = new float[num_voxels];
    Vec3f* grads = new Vec3f[num_voxels];
    unsigned char* red = new unsigned char[num_voxels];
    unsigned char* green = new unsigned char[num_voxels];
    unsigned char* blue = new unsigned char[num_voxels];

    std::fill_n(red, num_voxels, int(255));
    std::fill_n(green, num_voxels, int(255));
    std::fill_n(blue, num_voxels, int(255));

    size_t pos = 0;
    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i,j,k);
        size_t lin_idx = tSDF_->idx2line(idx);
        dist[pos] = -tSDF_->tsdf_[lin_idx].dist;
		// auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx);
		weights[pos] = tSDF_->tsdf_[lin_idx].weight;
        // grads[pos] = tSDF_->tsdf_[lin_idx].grad;
        // red[pos] = int(255*tSDF_->tsdf_[lin_idx].r);
        // green[pos] = int(255*tSDF_->tsdf_[lin_idx].g);
        // blue[pos] = int(255*tSDF_->tsdf_[lin_idx].b);

        pos++;
    }

    MarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    mc.computeIsoSurface(dist, weights, red, green, blue);
    // GradMarchingCubes mc(dim, voxel_size_ * dim.template cast<float>(), -voxel_size_ * Vec3f(xmin, ymin, zmin));
    // mc.computeIsoSurface(dist, weights, grads, red, green, blue);
   
    bool success = mc.savePly( save_path_ + filename + "_obj.ply");

	if (!success){std::cout << "couldn't save obj " <<save_path_ << filename  << std::endl;}

    // delete temporary arrays  
    if (dist)    delete[] dist;
    if (weights) delete[] weights;
    if (grads)   delete[] grads;
    if (red)     delete[] red;
    if (green)   delete[] green;
    if (blue)    delete[] blue;


	return success;
}

bool Optimizer::save_pointcloud(std::string filename)
{
    std::vector<Eigen::Matrix<float, 9 ,1> > points_normals_colors;
    
    size_t lin_index = 0;
    
    // for (size_t k=0; k<tSDF_->grid_dim_[2]; ++k) for (size_t j=0; j<tSDF_->grid_dim_[1]; ++j) for (size_t i=0; i<tSDF_->grid_dim_[0]; ++i) {
    for(auto lin_index: surface_points_){
        
		const SdfVoxel& v = tSDF_->tsdf_[lin_index];
		const Vec3i idx = tSDF_->line2idx(lin_index);
        if(std::abs(v.dist) < std::sqrt(3)*voxel_size_){
            Vec3f g = v.grad.normalized();
            Vec3f d = v.dist * g;
            float voxel_size_2 = .5 * voxel_size_;

            Eigen::Matrix<float, 9, 1> pnc;
            pnc.segment<3>(0) = tSDF_->vox2float(idx) - d;
            pnc.segment<3>(3) = g;
            pnc.segment<3>(6) = Vec3f(v.r, v.g, v.b);
            points_normals_colors.push_back(pnc);
        }
        
    }    

    std::ofstream plyFile;
    plyFile.open((save_path_+filename + "_pointcloud.ply").c_str());
    if (!plyFile.is_open()){
        std::cout << " can't save point cloud!" << std::endl;
        return false;
    }
        
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
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] <<" " << int(255 * p[6]) << " " << int(255 * p[7]) << " " << int(255 * p[8]) << std::endl;
    }
    
    plyFile.close();

    // std::cout << " seccussfully saved point cloud: " << save_path_ << filename << "_point cloud.ply !" <<std::endl;

    return true;
}

bool Optimizer::saveSDF(std::string filename)
{
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
    for (int k=0; k<tSDF_->grid_dim_[2]; ++k) for (int j=0; j<tSDF_->grid_dim_[1]; ++j) for (int i=0; i<tSDF_->grid_dim_[0]; ++i) {
        const SdfVoxel& v = tSDF_->tsdf_[lin_index];

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
    sdfFile.open( (save_path_ + filename).c_str());
    if (!sdfFile.is_open())
        return false;

    sdfFile << dim[0] << " " << dim[1] << " " << dim[2] << "\n";
 
    //compute bounding box

    // Vec3f origin = - voxel_size_*Vec3i(xmin, ymin, zmin).cast<float>();
    Vec3f bottom = Vec3i(xmin, ymin, zmin).cast<float>()*voxel_size_;
    Vec3f up = Vec3i(xmax, ymax, zmax).cast<float>()*voxel_size_;

    std::cout << "Bounding box size:  (" << bottom.transpose() << ") to (" << up.transpose() << ") with dimensions " << dim.transpose() << std::endl; 
   
    sdfFile <<  bottom[0] << " " << bottom[1] << " " << bottom[2] << "\n";
    sdfFile  << voxel_size_ << "\n";

    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i,j,k);
        size_t lin_idx = tSDF_->idx2line(idx);
        sdfFile << -tSDF_->tsdf_[lin_idx].dist << "\n";
    }

    sdfFile.close();
    return true;
}


bool Optimizer::savePoses(std::string filename)
{
	std::ofstream posefile;
	posefile.open((save_path_ + filename + ".txt").c_str());
	if(!posefile.is_open()){
        std::cout << "couldn't save optimized poses! " << std::endl;
    	return false;

    }
	for(size_t i= 0; i < num_frames_; i++){
		Eigen::Quaternion<float> q(getRotation(i));
		Vec3f t = getTranslation(i);
		posefile << key_stamps_[i] << " " << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

	}
	posefile.close();
	// std::cout << "poses file is successfully saved!" << std::endl;
	return true;

}

// bool Optimizer::save_info()
// {
//     std::ofstream file;
// 	file.open((save_path_ + "parameters_info.txt").c_str());
// 	if(!file.is_open()){
//         std::cout << "couldn't save optimized poses! " << std::endl;
//     	return false;

//     }
//     file << "dampling: " << settings_->damping << "\n "
//     << "loss function: " << settings_->loss << "\n"
//     << "lambda: " << settings_->lambda << "\n "
//     << "rho reg weight: " << settings_->reg_weight_rho << "\n"
//     << "normal reg weight: " << settings_->reg_weight_n << "\n"
//     << "laplace reg weight: " << settings_->reg_weight_l;
//     file.close();

//     return true;
// }


void Optimizer::subsampling()
{
    SdfVoxel* tsdf_sub;
    SdfVoxel v;
    v.dist = tSDF_->T_;
    v.grad = Vec3f::Zero();
    v.weight = 0;
    v.r = 0.5;
    v.g = 0.5;
    v.b = 0.5;

    int num_voxels_new = 8*tSDF_->num_voxels_;

    tsdf_sub    = new SdfVoxel[num_voxels_new];
    // initialize grids
    std::fill_n(tsdf_sub, num_voxels_new, v);

    std::vector<bool>* vis_sub;
    std::vector<bool> vis;
    vis_sub = new std::vector<bool>[num_voxels_new];
    std::fill_n(vis_sub, num_voxels_new, vis);

    for (size_t k=0; k<tSDF_->grid_dim_[2]; ++k) for (size_t j=0; j<tSDF_->grid_dim_[1]; ++j) for (size_t i=0; i<tSDF_->grid_dim_[0]; ++i) {
        
        int lin_idx = tSDF_->idx2line(i,j,k);
        const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
        // if(std::fabs(v.dist)>std::sqrt(3)*voxel_size_){
        //     continue;
        // }
        std::vector<bool> vis = tSDF_->vis_[lin_idx];
        if(v.dist == tSDF_->T_)continue;
        Vec8f d = tSDF_->subsample(v);
        Mat3x8i indx = getSubVoxelIndex(Vec3i(i,j,k));
        Eigen::Matrix<int, 8, 1> lin_sub_idx = getFlattenIndex(indx, 2*tSDF_->grid_dim_);
        for (size_t sub_i = 0; sub_i < 8; sub_i++){
            SdfVoxel& v_sub = tsdf_sub[lin_sub_idx[sub_i]];
            v_sub.dist = d[sub_i];
            v_sub.grad = v.grad;
            v_sub.weight = v.weight;
            v_sub.r = v.r;
            v_sub.g = v.g;
            v_sub.b = v.b;

            vis_sub[lin_sub_idx[sub_i]] = vis;
        }

    }

    // tSDF_->tsdf_ = new SdfVoxel[num_voxels_new];
    tSDF_->tsdf_ = tsdf_sub;
    tSDF_->vis_ = vis_sub;

    // delete[] tsdf_sub;
    // delete[] vis_sub;

    tSDF_->grid_subsample();
    num_voxels_ = num_voxels_new;
    voxel_size_ *= 0.5;
    voxel_size_inv_ = 1.0/voxel_size_;

    getSurfaceVoxel();
    std::cout << "sub-sampling: voxel size is " << voxel_size_ << ". voxel dim is " << tSDF_->grid_dim_.transpose() << std::endl;
}