//============================================================================
// Name        : LedOptimizerJa.cpp
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : implementation of class LedOptimizer for jacobian calculation
//============================================================================

#include "LedOptimizer.h"
#include "Auxilary.h"
#include "sdf_tracker/Sdfvoxel.h"
#include "Timer.h"

// =========================== rendered image ==========================
Vec3f LedOptimizer::renderedIntensity(const SdfVoxel& v, const Vec3i& idx, const int frame_idx)
{
	Mat3f R = getRotation(frame_idx);
    Vec3f t = getTranslation(frame_idx);
    Vec3f point = R.transpose() * (tSDF_->voxel2world(idx) - v.dist * v.grad.normalized() - t);
	Vec3f n = computeDistGrad(v, idx).first.normalized();
	// Vec3f n = v.grad.normalized();
	float Irradiance = - n.transpose()* R * point;
	float light_dist = std::pow(point.norm(),3);
	Irradiance /= light_dist;

	Vec3f rendered(v.r*light_[0]*Irradiance, v.g*light_[1]*Irradiance, v.b*light_[2]*Irradiance);
	return rendered;

}

// ============================= class functions =====================================
bool LedOptimizer::poseJacobian(const Vec3i& idx, const SdfVoxel& v, int frame_id, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3,6>& J_c)
{
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	// get subvoxel
	cv::Mat img = getImage(frame_id);
   	Vec3f point = R.transpose() * (tSDF_->voxel2world(idx) - v.dist * v.grad.normalized() - t);
	float m = fx * point[0]/point[2] + cx;
    float n = fy * point[1]/point[2] + cy;

	if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
		return false;
	}

	Eigen::Matrix<float,3,2> image_grad;
	image_grad.col(0) = computeImageGradient(n, m, img, 0); //3x2
	image_grad.col(1) = computeImageGradient(n, m, img, 1);

	const float z_inv = 1./point[2];
	const float z_inv_sq = z_inv * z_inv;
	Eigen::Matrix<float, 2, 3> pi_grad = Eigen::Matrix<float, 2, 3>::Zero();
	pi_grad(0,0) = fx * z_inv;
	pi_grad(0,2) = -fx * point[0] * z_inv_sq;
	pi_grad(1,1) = fy * z_inv;
	pi_grad(1,2) = -fy * point[1] * z_inv_sq;

	Mat3f image_pi_grad = image_grad * pi_grad;

	Vec3f normal = v.grad.normalized();
	float l = std::pow(point.norm(),3);
	Mat3f LED_t_grad, LED_R_grad;
	LED_t_grad.row(0) = - (v.r * light_[0] / l) * normal.transpose();
	LED_t_grad.row(1) = - (v.g * light_[1] / l) * normal.transpose();
	LED_t_grad.row(2) = - (v.b * light_[2] / l)* normal.transpose();

	float dl = std::pow(point.norm(),5);
	Vec3f dl_dR = (skew(point)*point).transpose();

	LED_R_grad.row(0) = -3*v.r * light_[0] / dl * (normal.transpose()* R * point).value() *dl_dR.transpose();
	LED_R_grad.row(1) = -3*v.g * light_[1] / dl * (normal.transpose()* R * point).value() *dl_dR.transpose();
	LED_R_grad.row(2) = -3*v.b * light_[2]/ dl * (normal.transpose()* R * point).value() *dl_dR.transpose();

	J_c.block<3, 3>(0, 0) = -image_pi_grad * R.transpose() + LED_t_grad;
    J_c.block<3, 3>(0, 3) = image_pi_grad * skew(point) + LED_R_grad;

    return true;
}


//! compute alebdo jacobian 
Vec3f LedOptimizer::rhoJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id)
{
	Mat3f R = getRotation(frame_id);
    Vec3f t = getTranslation(frame_id);
    Vec3f n = v.grad.normalized();
	// Vec3f n = computeDistGrad(v, idx).first;
    Vec3f point = R.transpose() * (tSDF_->voxel2world(idx) - v.dist * n- t);

	float reflectence = n.transpose()* R * point;
	float light_dist = std::pow( point.norm(), 3);
	reflectence /=light_dist;
	// reflectence *= 10.;

	return reflectence*light_;
}

Vec3f LedOptimizer::LightJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id)
{
	Mat3f R = getRotation(frame_id);
    Vec3f t = getTranslation(frame_id);
    Vec3f n = v.grad.normalized();
	// Vec3f n = computeDistGrad(v, idx).first;
    Vec3f point = R.transpose() * (tSDF_->voxel2world(idx) - v.dist * n- t);

	float reflectence = n.transpose()* R * point;
	float light_dist = std::pow( point.norm(), 3);
	reflectence /=light_dist;
	// reflectence *= 10.;

	return reflectence*Vec3f(v.r, v.g, v.b);
}

bool LedOptimizer::distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d)
{
    if(J_d.size()!=5){J_d.clear(); J_d.resize(5);}

	// dI(pi(x(v)))/dv
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	cv::Mat img = getImage(frame_id);
	
   	Vec3f point = R.transpose() * (tSDF_->voxel2world(idx) - v.dist * v.grad.normalized() - t);
	float m = fx * point[0]/point[2] + cx;
    float n = fy * point[1]/point[2] + cy;

	if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
		return false;
	}

	Eigen::Matrix<float,3,2> image_grad;
	image_grad.col(0) = computeImageGradient(n, m, img, 0); //3x2
	image_grad.col(1) = computeImageGradient(n, m, img, 1);

	const float z_inv = 1./point[2];
	const float z_inv_sq = z_inv * z_inv;
	Eigen::Matrix<float, 2, 3> pi_grad = Eigen::Matrix<float, 2, 3>::Zero();
	pi_grad(0,0) = fx * z_inv;
	pi_grad(0,2) = -fx * point[0] * z_inv_sq;
	pi_grad(1,1) = fy * z_inv;
	pi_grad(1,2) = -fy * point[1] * z_inv_sq;

	Mat3f image_pi_grad = image_grad * pi_grad;
	// dx/dv = g + d*dg/dv
	std::pair<Vec3f, Vec3f> tmp = computeDistGrad(v, idx);
	Vec3f grad = tmp.first;
	
	bool lag = false;
	Vec3f dn_d0 = normalJacobian(grad, tmp.second, lag);
	
	Vec3f n_d1 = Vec3f::Zero();
	n_d1[0] += tmp.second[0];
	Vec3f dn_d1 = normalJacobian(grad, n_d1, lag);

	n_d1.setZero();
	n_d1[1] += tmp.second[1];
	Vec3f dn_d2 = normalJacobian(grad, n_d1, lag);

	n_d1.setZero();
	n_d1[2] += tmp.second[2];
	Vec3f dn_d3 = normalJacobian(grad, n_d1, lag);

    Vec3f dx_d = -v.grad.normalized() - v.dist * dn_d0; // grad = -n

	// dI_dd = dI_dpi * dpi_dx * dx_dd
	Vec3f dI = image_pi_grad * R.transpose() * dx_d; // 3 x 1 for 3 channels

	Vec3f dI1 = image_pi_grad * R.transpose() * (-v.dist * dn_d1);
	Vec3f dI2 = image_pi_grad * R.transpose() * (-v.dist * dn_d2);
	Vec3f dI3 = image_pi_grad * R.transpose() * (-v.dist * dn_d3);

    // M part
    float dm_d0 = (dn_d0.transpose() * R * point + grad.normalized().transpose() * dx_d ).value();
    float dm_d1 = (dn_d1.transpose() * R * point + grad.normalized().transpose() * (-v.dist * dn_d1) ).value();
    float dm_d2 = (dn_d2.transpose() * R * point + grad.normalized().transpose() * (-v.dist * dn_d2) ).value();
    float dm_d3 = (dn_d3.transpose() * R * point + grad.normalized().transpose() * (-v.dist * dn_d3) ).value();

    // attenuation radius 
    float radius = std::pow(point.norm(), 3);


	// jacobian of radius part
	float dm_d0_2 = -3 * (point.transpose() * R.transpose() * dx_d).value() / std::pow(point.norm(), 5);
	float dm_d1_2 = -3 * (point.transpose() * R.transpose() * (-v.dist * dn_d1)).value() / std::pow(point.norm(), 5);
	float dm_d2_2 = -3 * (point.transpose() * R.transpose() * (-v.dist * dn_d2)).value() / std::pow(point.norm(), 5);
	float dm_d3_2 = -3 * (point.transpose() * R.transpose() * (-v.dist * dn_d3)).value() / std::pow(point.norm(), 5);

    Vec3f dR(v.r * light_[0], v.g * light_[1], v.b * light_[2]);


	dm_d0 = dm_d0 / radius + dm_d0_2 * (grad.normalized().transpose() * R *point).value();
	dm_d1 = dm_d1 / radius + dm_d1_2 * (grad.normalized().transpose() * R *point).value();
	dm_d2 = dm_d2 / radius + dm_d2_2 * (grad.normalized().transpose() * R *point).value();
	dm_d3 = dm_d3 / radius + dm_d3_2 * (grad.normalized().transpose() * R *point).value();

    
    Vec3f J_d0, J_d1, J_d2, J_d3;
    J_d0 =  dI + dR*dm_d0;
    J_d1 =  dI1 + dR*dm_d1;
    J_d2 =  dI2 + dR*dm_d2;
    J_d3 =  dI3 + dR*dm_d3;

    
	J_d[0] = (J_d0);
	J_d[1] = (J_d1);
	J_d[2] = (J_d2);
	J_d[3] = (J_d3);
	J_d[4] = (tmp.second); // push_back direction for compute col in distJacobian

	return true;

}


//----------------------numercial dist jacobian ----------------------------------
bool LedOptimizer::numerical_distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d)
{
	cv::Mat img = getImage(frame_id);
	Vec3f intensity;
	if(!getIntensity(idx, v, R, t, img, intensity))
		return false;
	
	Vec3f f1 = intensity - renderedIntensity(v, idx, frame_id);
	SdfVoxel v2 = v;
	float step_size = voxel_size_*0.1;
	v2.dist += step_size;
	// Vec3f step = Vec3f::Random();
	// v2.grad += step;
	Vec3f intensity2;

	if(!getIntensity(idx, v2, R, t, img, intensity2))
		return false;
	
	Vec3f f2 = intensity - renderedIntensity(v2, idx, frame_id);

	J_d.clear();
	J_d.push_back((f2 - f1)/ step_size);

	return true;
}

//---------------------------------- albedo related for using albedo regularizer ----------------------------------------------
//! compute albedo jacobian, should be diagonal matrix
Eigen::SparseMatrix<float> LedOptimizer::albedoJacobian()
{
	
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
    size_t total_num = surface_points_.size()*num_frames_*3;
	Eigen::SparseMatrix<float> J_r(total_num, surface_points_.size()*3);
	
	int row;
    for(size_t frame_id = 0; frame_id < num_frames_; frame_id++){
		row = 0;

		// DEBUG for saving normal image
		Mat3f R = getRotation(frame_id);
		Vec3f t = getTranslation(frame_id);
		cv::Mat img = getImage(frame_id);
		int h = img.rows;
		int w = img.cols;
		
	    for(auto& lin_idx: surface_points_){

            SdfVoxel& v = tSDF_->tsdf_[lin_idx];
            Vec3i idx = tSDF_->line2idx(lin_idx);

            std::vector<bool> vis = tSDF_->vis_[lin_idx];

           
			if(!vis[frame_id]){ //if the voxel is not visible in this frame, then skip the voxel for this frame
                row++;
				continue;
			}

			Vec3f Jr = rhoJacobian(v, idx, frame_id);
			// Vec3f Jr_vec = light_ * Jr;
			auto row_cur = 3*surface_points_.size()*frame_id + 3*row;

			for(size_t i = 0; i < 3; i++){
				Eigen::Triplet<float> Tri(row_cur+i, 3*row+i, Jr[i]);
				tripleVector.push_back(Tri);
			}

			row++;
		}
	}

	J_r.setFromTriplets(tripleVector.begin(), tripleVector.end());
	return J_r;

}

Eigen::SparseMatrix<float> LedOptimizer::lightJacobian()
{
	
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
    size_t total_num = surface_points_.size()*num_frames_*3;
	Eigen::SparseMatrix<float> J_l(total_num, 3);
	
	int row;
    for(size_t frame_id = 0; frame_id < num_frames_; frame_id++){
		row = 0;

		// DEBUG for saving normal image
		Mat3f R = getRotation(frame_id);
		Vec3f t = getTranslation(frame_id);
		cv::Mat img = getImage(frame_id);
		int h = img.rows;
		int w = img.cols;
		
	    for(auto& lin_idx: surface_points_){

            SdfVoxel& v = tSDF_->tsdf_[lin_idx];
            Vec3i idx = tSDF_->line2idx(lin_idx);

            std::vector<bool> vis = tSDF_->vis_[lin_idx];

           
			if(!vis[frame_id]){ //if the voxel is not visible in this frame, then skip the voxel for this frame
                row++;
				continue;
			}

			Vec3f Jl = LightJacobian(v, idx, frame_id);
			// Vec3f Jr_vec = light_ * Jr;
			auto row_cur = 3*surface_points_.size()*frame_id + 3*row;

			for(size_t i = 0; i < 3; i++){
				Eigen::Triplet<float> Tri(row_cur+i, i, Jl[i]);
				tripleVector.push_back(Tri);
			}

			row++;
		}
	}

	J_l.setFromTriplets(tripleVector.begin(), tripleVector.end());
	return J_l;

}


//--------------------------------- pose jacobian for all frames -------------------------------------------------------------------
//! compute pose jacobian sparse matrix
Eigen::SparseMatrix<float> LedOptimizer::poseJacobian()
{
    std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
	size_t total_num = surface_points_.size()*num_frames_*3; 
	
	Eigen::SparseMatrix<float> J_c(total_num, num_frames_*6);

	int row;

	for(int frame = 0; frame < num_frames_; frame++){
		row = 0;
		const Mat3f R = getRotation(frame);
		const Vec3f t = getTranslation(frame);

		for(auto& lin_idx: surface_points_){

            const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
            Vec3i idx = tSDF_->line2idx(lin_idx);

            std::vector<bool> vis = tSDF_->vis_[lin_idx];

           
			if(!vis[frame]){ //if the voxel is not visible in this frame, then skip the voxel for this frame
                row++;
				continue;
			}

			auto row_cur = 3*frame*surface_points_.size()+3*row;
			auto col_cur = frame*6;
			Eigen::Matrix<float,3,6> J_c;

			if(!poseJacobian(idx, v, frame, R , t, J_c)){
				row++;
				continue;
			}

			// load triplet for r_ijc
			for(size_t j = 0; j < 3; j++)for(size_t i = 0; i < 6; i++){
				Eigen::Triplet<float> Tri(row_cur+j, col_cur+i, J_c(j,i));
				tripleVector.push_back(Tri);
			}

			row++;
		}
	}
	J_c.setFromTriplets(tripleVector.begin(), tripleVector.end());
	if(total_num != 3*num_frames_*row){std::cout << "rows number not correct!" << std::endl;}
	return J_c;
}


//---------------------------------- distance related consider coupling of all voxels ----------------------------------------------
//! compute distance jacobian, consider all related voxels
Eigen::SparseMatrix<float> LedOptimizer::distJacobian()
{
	
	std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
    size_t total_num = surface_points_.size()*num_frames_*3; 

    // auto row; // prepare some thing
	
	int row;
    for(size_t frame_id = 0; frame_id < num_frames_; frame_id++){
		row = 0;
	    for(auto& lin_idx: surface_points_){

            SdfVoxel& v = tSDF_->tsdf_[lin_idx];
            Vec3i idx = tSDF_->line2idx(lin_idx);

            std::vector<bool> vis = tSDF_->vis_[lin_idx];

           
			if(!vis[frame_id]){ //if the voxel is not visible in this frame, then skip the voxel for this frame
                row++;
				continue;
			}

			Vec3f intensity, direction;
			std::vector<Vec3f> J, J_n;
			Mat3f R = getRotation(frame_id);
			Vec3f t = getTranslation(frame_id);
			// cv::Mat img = getImage(frame_id);

			if(!distJacobian(v, idx, R, t, frame_id, J)){ //if voxel is projected outside of the boundary, then skip
                row++;
				continue;
			}


			direction = J[4]; // store the direction to caculate the col
		
		
            // Jacobain for d0 -- row: lin_idx, col: lin_idx (diagonal)
            auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx);
            auto col = std::distance(surface_points_.begin(), it);

			auto row_cur = 3*frame_id*surface_points_.size() + 3*row;
		    Eigen::Triplet<float> Tri0(row_cur, col, J[0][0]);
            Eigen::Triplet<float> Tri1(row_cur+1, col, J[0][1]);
            Eigen::Triplet<float> Tri2(row_cur+2, col, J[0][2]);
		    tripleVector.push_back(Tri0);
            tripleVector.push_back(Tri1);
            tripleVector.push_back(Tri2);


		    // Jacobian for di -- rowL lin_idx, col: voxel i idx
			for(size_t i=0; i<3; i++){
				Vec3i idx_ = idx;
				idx_[i] += (int)direction[i];
				int lin_idx_ = tSDF_->idx2line(idx_);
				// std::cout <<"lin idx_ " << i << ": " << lin_idx_ <<std::endl;
				auto it = std::find(surface_points_.begin(), surface_points_.end(), lin_idx_);
				if(it != surface_points_.end()){
					col = std::distance(surface_points_.begin(), it);
					// std::cout <<"row " << row <<  " col " << i << ": " << col << " j_di: " << J_voxel[i+1] <<  std::endl;
					Eigen::Triplet<float> Tri0(row_cur, col, J[i+1][0]);
                    Eigen::Triplet<float> Tri1(row_cur+1, col, J[i+1][1]);
                    Eigen::Triplet<float> Tri2(row_cur+2, col, J[i+1][2]);
					tripleVector.push_back(Tri0);
                    tripleVector.push_back(Tri1);
                    tripleVector.push_back(Tri2);
				}
			}

            row++;
        }	
	}

	Eigen::SparseMatrix<float> J_d(total_num, surface_points_.size());
	J_d.setFromTriplets(tripleVector.begin(), tripleVector.end());
	// if(total_num != 3*num_frames_*row){std::cout << "rows number not correct!" << std::endl;}
	return J_d;
}

//-------------------------------------------------------------------- residual ---------------------------------------------------------------------------------
//! compute the residual r_{ic}(v_j) = I_ic(v_j) - rho(v_j)<n(v_j),l_ic>
// arrange order: image i voxel 1(r,g,b), 2(r,g,b), ... 
std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > LedOptimizer::computeResidual()
{
	Eigen::VectorXf residual;
	Eigen::SparseMatrix<float> weight;

    size_t total_num = surface_points_.size()*num_frames_*3; //3 is the r,g,b channels 

	weight.resize(total_num,total_num);
	// weight.reserve(num_voxels_);
	residual.resize(total_num);

	residual.setZero();
	int row;
    for(size_t frame_id = 0; frame_id < num_frames_; frame_id++){
		row = 0;
	    for(auto& lin_idx: surface_points_){

            SdfVoxel& v = tSDF_->tsdf_[lin_idx];
            Vec3i idx = tSDF_->line2idx(lin_idx);

            std::vector<bool> vis = tSDF_->vis_[lin_idx];
            
            if(!vis[frame_id]){ //if the voxel is not visible in this frame, then skip the voxel for this frame
                row++;
                continue;
            }
		
            // weight.coeffRef(row, row) = 0;

            Vec3f intensity;
            Mat3f R = getRotation(frame_id);
            Vec3f t = getTranslation(frame_id);
            cv::Mat img = getImage(frame_id);

            if(!getIntensity(idx, v, R , t, img, intensity)){ //if voxel is projected outside of the boundary, then skip
                row++;
                continue;
            }

            Vec3f r = intensity - renderedIntensity(v, idx, frame_id);
            Vec3f w = computeWeight(r);
			auto row_cur = 3*frame_id*surface_points_.size()+3*row;
            weight.coeffRef(row_cur, row_cur) = w[0];
            weight.coeffRef(row_cur+1, row_cur+1) = w[1];
            weight.coeffRef(row_cur+2, row_cur+2) = w[2];
            residual.segment(row_cur, 3) = r;
           
            row++;
        }
	}
	
	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > tmp(residual, weight);
	return tmp;

}
