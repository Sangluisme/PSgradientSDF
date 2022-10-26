//============================================================================
// Name        : Optimizer.cpp
// Author      : Lu Sang
// Date        : 10/2021
// License     : GNU General Public License
// Description : implementation of class Optimizer
//============================================================================

#include "Optimizer.h"
#include "sdf_tracker/Sdfvoxel.h"

#include "Timer.h"

// ============================= class functions =====================================

Optimizer::Optimizer(VolumetricGradSdf* tSDF,
					const float voxel_size,
                    const Mat3f& K,
					std::string save_path,
					OptimizerSettings* settings):
                    tSDF_(tSDF),
                    voxel_size_(voxel_size),
                    voxel_size_inv_(1.f / voxel_size),
                    K_(K),
					save_path_(save_path),
                    settings_(settings)
{}

//! select vis_map according to key frame (might not be necessary)
void Optimizer::select_vis(){
	
	// for(auto& lin_idx: surface_points_){
	for (size_t k=0; k<tSDF_->grid_dim_[2]; ++k) for (size_t j=0; j<tSDF_->grid_dim_[1]; ++j) for (size_t i=0; i<tSDF_->grid_dim_[0]; ++i) {
        int lin_idx = tSDF_->idx2line(i,j,k);
		std::vector<bool> vis = tSDF_->vis_[lin_idx];
		std::vector<bool> vis_select;
		for(size_t frame = 0; frame< num_frames_; frame++){
			if((frame_idx_[frame])<vis.size())
				vis_select.push_back(vis[frame_idx_[frame]]);
		}
		vis_select.resize(num_frames_);
		tSDF_->vis_[lin_idx] = vis_select;
	}

	getSurfaceVoxel();

}

// initalize albedo using average color of voxels in each keyframe
void Optimizer::initAlbedo()
{
	for(auto& lin_idx: surface_points_){

		SdfVoxel& v = tSDF_->tsdf_[lin_idx];
		std::vector<bool>& vis = tSDF_->vis_[lin_idx];

		int count = 0;
		Vec3f rho = Vec3f::Zero();
		for(size_t frame = 0; frame < num_frames_; frame++){
			Vec3f intensity;
			Mat3f R = getRotation(frame);
			Vec3f t = getTranslation(frame);
			cv::Mat img = getImage(frame);
			Vec3i idx = tSDF_->line2idx(lin_idx);
			if(!vis[frame]|| (!getIntensity(idx, v, R, t, img, intensity)))
				continue;
			
			rho += intensity;
			count++;
		}
		if(count)
		{
			// rho = rho.array()/static_cast<float>(count))
			tSDF_->tsdf_[lin_idx].r = rho[0]/static_cast<float>(count);
			tSDF_->tsdf_[lin_idx].g = rho[1]/static_cast<float>(count);
			tSDF_->tsdf_[lin_idx].b = rho[2]/static_cast<float>(count);
		}
			
	}

}



//! computer normal regularizer energy
float Optimizer::getNormalEnergy()
{
	float E = 0;

	for(auto& lin_idx: surface_points_){
		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
	
		Vec3i idx = tSDF_->line2idx(lin_idx);
		Vec3f n = computeDistGrad(v, idx).first;
		// Vec3f n = v.grad;
		float E_cur = n.norm() - 1;
		E += E_cur * E_cur;
	}
	// Eigen::VectorXf r_reg = distRegJacobian().second;

	// std::cout << "total energy " << E << std::endl;
	return E/static_cast<float>(surface_points_.size());
}

//! compute the laplacian regularizer energy
float Optimizer::getLaplacianEnergy()
{
	float E = 0.0;

	for(auto& lin_idx: surface_points_){

		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
		Vec3i idx = tSDF_->line2idx(lin_idx);
		float E_cur = computeDistLaplacian(v, idx);
		E += E_cur*E_cur;
	}

	return E/static_cast<float>(surface_points_.size());
}

//! compute albedo regularizer energy
float Optimizer::getAlbedoRegEnergy()
{
	float E = 0;
	for(auto& lin_idx: surface_points_){
		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
	
		Vec3i idx = tSDF_->line2idx(lin_idx);
		Mat3f n_r = computeAlbedoGrad(v, idx).first;
	
		float E_cur = n_r.rowwise().norm().sum();
		E += E_cur;
	}

	return E/static_cast<float>(surface_points_.size());
}


//! compute robust loss function weights to be used in IRLS
Eigen::VectorXf Optimizer::computeWeight(Eigen::VectorXf r)
{
	Eigen::VectorXf w;
	switch (settings_->loss) {
		case LossFunction::CAUCHY :
			return 1.0 / (1.0 + (r.array()/settings_->lambda).square());
		case LossFunction::TUKEY :
			w = (1.0 - (r.array() / settings_->lambda).square()).square();
			return (r.array().square() < settings_->lambda_sq).select(w, 0.0);
		case LossFunction::HUBER :
			w = settings_->lambda * r.cwiseInverse().cwiseAbs();
			return (r.array().square() < settings_->lambda_sq).select(1.0, w);
		case LossFunction::TRUNC_L2 :
			w = Eigen::VectorXf(r.size());
			w.setOnes();
			return (r.array().square() < settings_->lambda_sq).select(w, 0.0);
		default :
			w = Eigen::VectorXf(r.size());
			w.setOnes();
			return w;
	}
}

// //! compute robust loss function
float Optimizer::computeLoss(Eigen::VectorXf r)
{
	Eigen::VectorXf tmp, tmp1, tmp2;
	switch (settings_->loss) {
		case LossFunction::CAUCHY :
			tmp = (1.0 + (r.array() / settings_->lambda).square()).log();
			return tmp.sum();
		case LossFunction::TUKEY : 
			tmp = 1.0 - (1.0 - (r.array() / settings_->lambda).square()).cube();
			tmp = (r.array().square() < settings_->lambda_sq).select(tmp, 1.0);
			return tmp.sum();
		case LossFunction::HUBER :
			tmp = Eigen::VectorXf(r.size());
			tmp.setOnes();
			tmp1 = .5 * r.array().square();
			tmp2 = settings_->lambda * (r.cwiseAbs() - 0.5 * settings_->lambda * tmp);
			return ((r.array().square() < settings_->lambda_sq).select(tmp1, tmp2)).sum();
		case LossFunction::TRUNC_L2:
			return r.cwiseMax(-settings_->lambda).cwiseMin(settings_->lambda).squaredNorm();
		default :
			return r.squaredNorm();
	}
}






// -------------------------------- Jacobians w.r.t regularizer energy --------------------------------------------------
//! compute normal regularizer distance jacobian E_n = (||\nabla d(v)||-1)^2
//! return a std::vector contain dirivtive of dist w.r.t of d0, d1, d2, d3
void Optimizer::distRegJacobian(const SdfVoxel& v, const Vec3i& idx, std::vector<float>& Jr_d)
{
	if(Jr_d.size()!=4){Jr_d.clear(); Jr_d.resize(4);}
	// center voxels d0
	std::pair<Vec3f, Vec3f> tmp = computeDistGrad(v, idx);
	Vec3f grad = tmp.first;
	Vec3f n_d = -voxel_size_inv_* tmp.second;
	Jr_d[0] = grad.transpose()*n_d;

	// d1, d2, d3
	for(size_t i=0; i<3; i++){
		Vec3f direction = Vec3f::Zero();
		direction[i] += voxel_size_inv_*tmp.second[i];
		Jr_d[i+1] = grad.transpose()*direction;
	}

	if(grad.norm() > 0.0){
		for(size_t i=0;i<4;i++){
			Jr_d[i]/= grad.norm();
		}
	}
	
}

//! compute nabla albedo jacobian E_r = \sum ||\nabla rho(v) || ^2
void Optimizer::albedoRegJacobian(const SdfVoxel& v, const Vec3i& idx, std::vector<Vec3f>& Jr_r)
{
	if(Jr_r.size()!=4){Jr_r.clear(); Jr_r.resize(4);}

	std::pair<Mat3f, Vec3f> tmp = computeAlbedoGrad(v, idx);
	Mat3f grad = tmp.first;
	Vec3f r_d = -voxel_size_inv_*tmp.second;
	Vec3f grad_norm = grad.rowwise().norm();
	Jr_r[0] = grad * r_d;

	for(size_t i = 0; i<3; i++){
		Vec3f direction = Vec3f::Zero();
		direction[i] += voxel_size_inv_*tmp.second[i];
		Jr_r[i+1] = grad*direction;
	}

	for(size_t i = 0; i < 3; i++){
		if(grad_norm[i] != 0.0){
			Jr_r[0][i] /= grad_norm[i];
			Jr_r[1][i] /= grad_norm[i];
			Jr_r[2][i] /= grad_norm[i];
			Jr_r[3][i] /= grad_norm[i];
		}
	}
}


//--------------------------------- other related functions --------------------------------------------------------------------

//! compute distance jacobian w.r.t normal
Vec3f Optimizer::normalJacobian(const SdfVoxel& v, const Vec3i& idx)
{
	
	std::pair<Vec3f, Vec3f> tmp = computeDistGrad(v, idx);
	Vec3f grad = tmp.first;
	Vec3f n_d = -voxel_size_inv_* tmp.second;

	float N_inv = 1.0 / std::max(grad.norm(), 0.001f);
	// Vec3f J_n = N_inv * n_d - std::pow(N_inv,3) * (n_d.cwiseProduct(grad));
	float dN_inv_d = std::pow(N_inv,3) * (n_d.transpose()*grad).value();
	Vec3f J_n = N_inv * n_d -  dN_inv_d* grad;


	return J_n;
}


//! compute the normal jacobian w.r.t distance
Vec3f Optimizer::normalJacobian(Vec3f& grad, Vec3f& direction, bool lag)
{
	Vec3f n_d = -voxel_size_inv_* direction;
	float N_inv = 1.0 / std::max(grad.norm(), 0.001f);
	Vec3f J_n;
	if(lag){
		J_n =  N_inv * n_d;
	}
	else{	
		float dN_inv_d = std::pow(N_inv,3) * (n_d.transpose()*grad).value();
		J_n = N_inv * n_d - dN_inv_d * grad;
	}

	return J_n;

}

//! compute voxel normal using n = gradient of distance function n = (d-d_1, d-d_2, d-d_3)
std::pair<Vec3f, Vec3f> Optimizer::computeDistGrad(const SdfVoxel& v, const Vec3i& idx)
{
	// x-direction
	float direction_x; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 0)){ //check forward
		direction_x = 1.0;
	}
	// else if(ifValidDirection(idx, -1, 0)){ //check backward
	// 		direction_x = -1.0;
	// }
	else{
		direction_x = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with x direction!" <<  std::endl;
	}

	Vec3i idx_x = idx;
	idx_x[0] += (int)direction_x;
	int idx_x_lin = tSDF_->idx2line(idx_x);
	SdfVoxel& v_x = tSDF_->tsdf_[idx_x_lin];

	// y-direction
	float direction_y; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 1)){ //check forward
		direction_y = 1.0;
	}
	// else if(ifValidDirection(idx, -1, 1)){ //check backward
	// 		direction_y = -1.0;
	// }
	else{
		direction_y = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with y direction!" <<  std::endl;
	}

	Vec3i idx_y = idx;
	idx_y[1] += (int)direction_y;
	int idx_y_lin = tSDF_->idx2line(idx_y);
	SdfVoxel& v_y = tSDF_->tsdf_[idx_y_lin];

	// z-direction
	float direction_z; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 2)){ //check forward
		direction_z = 1.0;
	}
	// else if(ifValidDirection(idx, -1, 2)){ //check backward
	// 		direction_z = -1.0;
	// }
	else{
		direction_z = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with z direction!" <<  std::endl;
	}

	Vec3i idx_z = idx;
	idx_z[2] += (int)direction_z;
	int idx_z_lin = tSDF_->idx2line(idx_z);
	SdfVoxel& v_z = tSDF_->tsdf_[idx_z_lin];

	//DEBUG
	// std::cout <<"v: " << idx.transpose() << " \t x: " << idx_x.transpose() << " \t y: " << idx_y.transpose() << " \t z: " << idx_z.transpose() << std::endl;

	Vec3f n(direction_x*(v_x.dist - v.dist), direction_y*(v_y.dist - v.dist), direction_z*(v_z.dist - v.dist));
	Vec3f pos(direction_x, direction_y, direction_z);

	// std::cout << " n without step size: " << n.transpose() << std::endl;

	n = n * voxel_size_inv_;

	// std::cout << " n with step size: " << n.transpose() << std::endl;
	// std::cout << " n with squared step size: " << (voxel_size_inv_*n).transpose() << std::endl;

	if ((pos.array() == 0.0).any()){
		n = v.grad.normalized();
		pos = Vec3f::Zero();
	}
	
	std::pair<Eigen::Vector3f, Eigen::Vector3f> tmp(n, pos);
	return tmp;

}

 
//! compute dist laplacian
float Optimizer::computeDistLaplacian(const SdfVoxel& v, const Vec3i& idx)
{
	Vec3i idx_ = idx;
	idx_[0] += 1;
	SdfVoxel& v_x1 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	idx_[0] -= 2;
	SdfVoxel& v_x0 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	float dxx = v_x1.dist + v_x0.dist - 2 * v.dist;

	idx_ = idx;
	idx_[1] += 1;
	SdfVoxel& v_y1 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	idx_[1] -= 2;
	SdfVoxel& v_y0 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	float dyy = v_y1.dist + v_y0.dist - 2 * v.dist;

	idx_ = idx;
	idx_[2] += 1;
	SdfVoxel& v_z1 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	idx_[2] -= 2;
	SdfVoxel& v_z0 = tSDF_->tsdf_[tSDF_->idx2line(idx_)];
	float dzz = v_z1.dist + v_z0.dist - 2 * v.dist;

	return (dxx + dyy + dzz)*voxel_size_inv_*voxel_size_inv_;

}

//! compute the albedo jacobian
std::pair<Mat3f, Vec3f> Optimizer::computeAlbedoGrad(const SdfVoxel& v, const Vec3i& idx)
{
	// x-direction
	float direction_x; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 0)){ //check forward
		direction_x = 1.0;
	}
	else{
		direction_x = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with x direction!" <<  std::endl;
	}

	Vec3i idx_x = idx;
	idx_x[0] += (int)direction_x;
	int idx_x_lin = tSDF_->idx2line(idx_x);
	SdfVoxel& v_x = tSDF_->tsdf_[idx_x_lin];

	// y-direction
	float direction_y; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 1)){ //check forward
		direction_y = 1.0;
	}
	else{
		direction_y = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with y direction!" <<  std::endl;
	}

	Vec3i idx_y = idx;
	idx_y[1] += (int)direction_y;
	int idx_y_lin = tSDF_->idx2line(idx_y);
	SdfVoxel& v_y = tSDF_->tsdf_[idx_y_lin];

	// z-direction
	float direction_z; // determine forward or backward finite difference
	if(ifValidDirection(idx, 1, 2)){ //check forward
		direction_z = 1.0;
	}
	else{
		direction_z = -1.0;
		// std::cout << " warning: voxel "<< idx.transpose() << " with z direction!" <<  std::endl;
	}

	Vec3i idx_z = idx;
	idx_z[2] += (int)direction_z;
	int idx_z_lin = tSDF_->idx2line(idx_z);
	SdfVoxel& v_z = tSDF_->tsdf_[idx_z_lin];

	//DEBUG
	// std::cout <<"v: " << idx.transpose() << " \t x: " << idx_x.transpose() << " \t y: " << idx_y.transpose() << " \t z: " << idx_z.transpose() << std::endl;

	Vec3f r(direction_x*(v_x.r - v.r), direction_y*(v_y.r - v.r), direction_z*(v_z.r - v.r));
	Vec3f g(direction_x*(v_x.g - v.g), direction_y*(v_y.g - v.g), direction_z*(v_z.g - v.g));
	Vec3f b(direction_x*(v_x.b - v.b), direction_y*(v_y.b - v.b), direction_z*(v_z.b - v.b));

	Mat3f Jr;
	Jr.row(0) = r.array()*voxel_size_inv_;
	Jr.row(1) = g.array()*voxel_size_inv_;
	Jr.row(2) = b.array()*voxel_size_inv_;

	Vec3f pos(direction_x, direction_y, direction_z);

	std::pair<Eigen::Matrix3f, Eigen::Vector3f> tmp(Jr, pos);
	return tmp;

}

bool Optimizer::ifValidDirection(const Vec3i& idx, const int direction, const int pos){

	if(idx[pos] + direction > tSDF_->grid_dim_[pos])
		return false;

	Vec3i idx_new = idx;
	idx_new[pos] += direction;
	size_t lin_idx = tSDF_->idx2line(idx_new);
	if(std::find(surface_points_.begin(), surface_points_.end(), lin_idx) == surface_points_.end())
		return false;

	return true;
}

//-------------------------------------------------------------regularizer-------------------------------------------------------------------------------------
std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> Optimizer::distRegJacobian()
{
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
	Eigen::SparseMatrix<float> Jr_d(surface_points_.size(), surface_points_.size());
	Eigen::VectorXf residual;
	residual.resize(surface_points_.size());
	
    // auto row; // prepare some thing
	Vec3f direction;

	int row = 0;
	for(auto& lin_idx: surface_points_){

		Vec3i idx = tSDF_->line2idx(lin_idx);
		SdfVoxel& v = tSDF_->tsdf_[lin_idx];
		// std::vector<bool> vis = tSDF_->vis_[lin_idx];

		// prepare for store dist jacobian for d0, d1, d2, d3
		std::vector<float> J_voxel;
		J_voxel.resize(4);

		distRegJacobian(v, idx, J_voxel);
		auto col = row;
		Eigen::Triplet<float> Tri(row, col, J_voxel[0]);
		
		tripleVector.push_back(Tri);

		std::pair<Vec3f, Vec3f> tmp = computeDistGrad(v, idx);
		Vec3f grad = tmp.first;
		Vec3f direction = tmp.second;
		

		float r = grad.norm()-1;
		residual[row] = r;
		// std::cout <<"row " << row <<  " col: " << col << " j_di: " << J_voxel[0] << "r: " << r <<  std::endl;

		if (!(direction.array() == 0).any()){
			
			for(size_t i = 0; i<3; i++){
				Vec3i idx_ = idx;
				idx_[i] += (int)direction[i];
				int lin_idx_ = tSDF_->idx2line(idx_);

				auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx_);

				if(it != surface_points_.end()){
					col = std::distance(surface_points_.begin(), it);
					Eigen::Triplet<float> Tri(row, col, J_voxel[i+1]);
					// std::cout <<"row " << row <<  " col " << i << ": " << col << " j_di: " << J_voxel[i+1] <<  std::endl;
					tripleVector.push_back(Tri);
				}
			}
		}

		row++;
	}

	Jr_d.setFromTriplets(tripleVector.begin(), tripleVector.end());
	std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> pair(Jr_d, residual);
	return pair;
}

//! compute the regularizer for E_d = \sum || \laplacian d(v) ||^2
std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> Optimizer::distLaplacianJacobian()
{
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
	Eigen::SparseMatrix<float> Jl_d(surface_points_.size(), surface_points_.size());
	Eigen::VectorXf residual;
	residual.resize(surface_points_.size());

	float voxel_size_inv_2 = voxel_size_inv_*voxel_size_inv_;

	int row = 0;
	for(auto& lin_idx: surface_points_){

		const Vec3i idx = tSDF_->line2idx(lin_idx);
		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];

		float r = computeDistLaplacian(v, idx);
		residual[row] = r;

		auto col = row;
		Eigen::Triplet<float> Tri(row, col, -6*voxel_size_inv_2);
		tripleVector.push_back(Tri);
		for(int i = 0; i < 3; i++){
			Vec3i idx_ = idx;
			idx_[i] += 1;
			int lin_idx_ = tSDF_->idx2line(idx_);
			auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx_);

			if(it != surface_points_.end()){
				col = std::distance(surface_points_.begin(), it);
				Eigen::Triplet<float> Tri(row, col, voxel_size_inv_2);
			}

			idx_[i] -= 2;
			lin_idx_ = tSDF_->idx2line(idx_);
			it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx_);

			if(it != surface_points_.end()){
				col = std::distance(surface_points_.begin(), it);
				Eigen::Triplet<float> Tri(row, col, voxel_size_inv_2);
			}

		}

		row++;
	}

	Jl_d.setFromTriplets(tripleVector.begin(), tripleVector.end());
	std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> pair(Jl_d, residual);
	return pair;

}

//! compute the regularizer for albedo E_r = \sum_v ||\nabla rho(v)||^2
std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> Optimizer::albedoRegJacobian()
{
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
	Eigen::SparseMatrix<float> Jr_r(surface_points_.size()*3, surface_points_.size()*3);
	Eigen::VectorXf residual;
	residual.resize(surface_points_.size()*3);

	int row = 0;
	for(auto& lin_idx: surface_points_){

		const Vec3i idx = tSDF_->line2idx(lin_idx);
		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];

		std::pair<Mat3f, Vec3f> pair = computeAlbedoGrad(v, idx);
		Vec3f direction = pair.second;
		Vec3f r = pair.first.rowwise().norm();

		residual.segment(3*row, 3) = r;

		std::vector<Vec3f> Jr_r;
		albedoRegJacobian(v, idx, Jr_r);

		auto col = row;
		Eigen::Triplet<float> Tri0(3*row, 3*row, Jr_r[0][0]); //r
		Eigen::Triplet<float> Tri1(3*row+1, 3*row+1, Jr_r[0][1]); //g
		Eigen::Triplet<float> Tri2(3*row+2, 3*row+1, Jr_r[0][2]); //b
		tripleVector.push_back(Tri0);
		tripleVector.push_back(Tri1);
		tripleVector.push_back(Tri2);

		for(size_t i = 0; i<3; i++){
			Vec3i idx_ = idx;
			idx_[i] += (int)direction[i];
			int lin_idx_ = tSDF_->idx2line(idx_);

			auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx_);

			if(it != surface_points_.end()){
				col = std::distance(surface_points_.begin(), it);
				Eigen::Triplet<float> Tri0(3*row, 3*col, Jr_r[i+1][0]); //r
				Eigen::Triplet<float> Tri1(3*row+1, 3*col+1, Jr_r[i+1][1]); //g
				Eigen::Triplet<float> Tri2(3*row+2, 3*col+2, Jr_r[i+1][2]); //b
				tripleVector.push_back(Tri0);
				tripleVector.push_back(Tri1);
				tripleVector.push_back(Tri2);
			}
		}
		
		row++;
	}

	Jr_r.setFromTriplets(tripleVector.begin(), tripleVector.end());
	std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> pair(Jr_r, residual);
	return pair;
}