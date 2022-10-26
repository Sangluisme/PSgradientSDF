//============================================================================
// Name        : PsOptimizer.cpp
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : implementation of class PsOptimizer
//============================================================================

#include "PsOptimizer.h"
#include "sdf_tracker/Sdfvoxel.h"
#include "Auxilary.h"
#include "opencv2/opencv.hpp"
// #include "Timer.h"

//-------------------------------------------some useful functions -----------------------------------------------------

Eigen::VectorXf PsOptimizer::SH(const Vec3f& n, const int order){
	Eigen::VectorXf sh_n;
	if (order == 1){
		sh_n.resize(4);
		sh_n << 1.0, n;
	}
	if(order == 2){
		sh_n.resize(9);
		sh_n << 1.0, n, n[0]*n[1], n[0]*n[2], n[1]*n[2], n[0]*n[0]-n[1]*n[1], n[0]*n[0]-n[2]*n[2];
	}
	return sh_n;
}

Vec3f PsOptimizer::renderedIntensity(const SdfVoxel& v, const Vec3i& idx, const int frame_idx)
{
	Mat3f R = getRotation(frame_idx);
	Vec3f n = computeDistGrad(v, idx).first;
	// Vec3f n = v.grad;
	float Irradiance = light_[frame_idx].transpose()*SH(n.normalized(), settings_->order);

	Vec3f rendered(v.r*Irradiance, v.g*Irradiance, v.b*Irradiance);
	return rendered;

}

bool PsOptimizer::computeResidual(const SdfVoxel& v, const Vec3i& idx, const int frame_id, Vec3f& r)
{
    Vec3f intensity, J_d;
    Mat3f R = getRotation(frame_id);
    Vec3f t = getTranslation(frame_id);
    cv::Mat img = getImage(frame_id);

    if(!getIntensity(idx, v, R , t, img, intensity)){ //if voxel is projected outside of the boundary, then skip
        return false;
    }

    r = intensity - renderedIntensity(v, idx, frame_id);
    return true;

}


// -------------------- Jacobians w.r.t PS energy --------------------------------------------------
//! Jacobian w.r.t. camera poses
bool PsOptimizer::poseJacobian(const Vec3i& idx, const SdfVoxel& v, int frame_id, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3,6>& J_c)
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

	// Mat3f sh_grad;
	// if (settings_->order == 1){
	// 	sh_grad.row(0) = v.r * light_[frame_id].tail(3).transpose()*skew(v.grad.normalized());
	// 	sh_grad.row(1) = v.g * light_[frame_id].tail(3).transpose()*skew(v.grad.normalized());
	// 	sh_grad.row(2) = v.b * light_[frame_id].tail(3).transpose()*skew(v.grad.normalized());
	// }
	// if (settings_->order == 2){
	// 	Vec3f RTn = R.transpose()*v.grad.normalized();
	// 	Eigen::Matrix<float, 3, 9> dSH_dnhat;
	// 	dSH_dnhat << 0, 1, 0, 0, RTn[1], RTn[2], 0, 2*RTn[0], 2*RTn[0],
	// 				 0, 0, 1, 0, RTn[0], 0, RTn[2], -2*RTn[1], 0,
	// 				 0, 0, 0, 1, 0, RTn[0], RTn[1], 0, -2*RTn[2];
	// 	sh_grad.row(0) = v.r * light_[frame_id].transpose()*dSH_dnhat.transpose()*skew(v.grad.normalized());
	// 	sh_grad.row(1) = v.g * light_[frame_id].transpose()*dSH_dnhat.transpose()*skew(v.grad.normalized());
	// 	sh_grad.row(2) = v.b * light_[frame_id].transpose()*dSH_dnhat.transpose()*skew(v.grad.normalized());
	// }

	J_c.block<3, 3>(0, 0) = -image_pi_grad * R.transpose();
    J_c.block<3, 3>(0, 3) = image_pi_grad * skew(point);

	// J_c << -image_pi_grad * R.transpose(), image_pi_grad * skew(point); // TODO: double-check for minus

	return true;
}

//! compute alebdo jacobian 
float PsOptimizer::rhoJacobian(const SdfVoxel& v, const int frame_id)
{
	Mat3f R = getRotation(frame_id);
	return -light_[frame_id].transpose()*SH(v.grad.normalized(), settings_->order);
}

float PsOptimizer::rhoJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id)
{
	Mat3f R = getRotation(frame_id);
	Vec3f n = computeDistGrad(v, idx).first;
	return -light_[frame_id].transpose()*SH(n.normalized(), settings_->order);
}

//! compute light jacobian
void PsOptimizer::lightJacobian(const SdfVoxel& v, const int frame_id, std::vector<Eigen::VectorXf>& J_l)
{
	J_l.clear();
	// Vec3f RTn = getRotation(frame_id).transpose()*v.grad.normalized();
	Vec3f n = v.grad.normalized();
	
	J_l.push_back(-v.r * SH(n, settings_->order));
	J_l.push_back(-v.g * SH(n, settings_->order));
	J_l.push_back(-v.b * SH(n, settings_->order));
	// J_l -= v.b * SH(v.grad.normalized(), settings_->order);
	// J_l -= v.g * SH(v.grad.normalized(), settings_->order);
}

void PsOptimizer::lightJacobian(const SdfVoxel& v, const Vec3i& idx, const int frame_id, std::vector<Eigen::VectorXf>& J_l)
{
	Vec3f n = computeDistGrad(v, idx).first;
	// Vec3f RTn = getRotation(frame_id).transpose()*n.normalized();

	// J_l = -v.r * SH(n.normalized(), settings_->order);
	// J_l -= v.b * SH(n.normalized(), settings_->order);
	// J_l -= v.g * SH(n.normalized(), settings_->order);
	J_l.clear();
	J_l.push_back(-v.r * SH(n, settings_->order));
	J_l.push_back(-v.g * SH(n, settings_->order));
	J_l.push_back(-v.b * SH(n, settings_->order));
}

//! compute distance jacobian for the center voxel d0 and d1, d2, d3
bool PsOptimizer::distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d)
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
	n_d1[0] -= tmp.second[0];
	Vec3f dn_d1 = normalJacobian(grad, n_d1, lag);

	n_d1.setZero();
	n_d1[1] -= tmp.second[1];
	Vec3f dn_d2 = normalJacobian(grad, n_d1, lag);

	n_d1.setZero();
	n_d1[2] -= tmp.second[2];
	Vec3f dn_d3 = normalJacobian(grad, n_d1, lag);
	


	Vec3f dx_d = -v.grad.normalized() - v.dist * dn_d0; // grad = -n

	// dI_dd = dI_dpi * dpi_dx * dx_dd
	Vec3f dI = image_pi_grad * R.transpose() * dx_d; // 3 x 1 for 3 channels

	Vec3f dI1 = image_pi_grad * R.transpose() * (-v.dist * dn_d1);
	Vec3f dI2 = image_pi_grad * R.transpose() * (-v.dist * dn_d2);
	Vec3f dI3 = image_pi_grad * R.transpose() * (-v.dist * dn_d3);

	
	Vec3f J_d0, J_d1, J_d2, J_d3;
	if (settings_->order == 1){
		Vec3f dr_r, dr_g, dr_b;
		dr_r = v.r * light_[frame_id].tail(3);
		dr_g = v.g * light_[frame_id].tail(3);
		dr_b = v.b * light_[frame_id].tail(3);

		Mat3f dR;
		dR.row(0) = dr_r.transpose();
		dR.row(1) = dr_g.transpose();
		dR.row(2) = dr_b.transpose();

		J_d0 = dI - dR*dn_d0;
		J_d1 = dI1 - dR*dn_d1;
		J_d2 = dI2 - dR*dn_d2;
		J_d3 = dI3 - dR*dn_d3;
	}

	if (settings_->order == 2){
		Eigen::Matrix<float, 9, 1> dSH_d0, dSH_d1, dSH_d2, dSH_d3;
		Eigen::Matrix<float, 3, 9> dSH_dnhat;

		Vec3f RTn = grad.normalized();

		dSH_dnhat << 0, 1, 0, 0, RTn[1], RTn[2], 0, 2*RTn[0], 2*RTn[0],
					 0, 0, 1, 0, RTn[0], 0, RTn[2], -2*RTn[1], 0,
					 0, 0, 0, 1, 0, RTn[0], RTn[1], 0, -2*RTn[2];

		dSH_d0 = dSH_dnhat.transpose() * dn_d0;
		dSH_d1 = dSH_dnhat.transpose() * dn_d1;
		dSH_d2 = dSH_dnhat.transpose() * dn_d2;
		dSH_d3 = dSH_dnhat.transpose() * dn_d3;

		// dSH_d0 = (R * dSH_dnhat).transpose() * dn_d0;
		// dSH_d1 = R * dSH_dnhat.transpose() * dn_d1;
		// dSH_d2 = R * dSH_dnhat.transpose() * dn_d2;
		// dSH_d3 = R * dSH_dnhat.transpose() * dn_d3;

		
		Eigen::Matrix<float, 9, 1> dr_r, dr_g, dr_b;
		dr_r = v.r * light_[frame_id];
		dr_g = v.g * light_[frame_id];
		dr_b = v.b * light_[frame_id];

		Eigen::Matrix<float,3,9> dR;
		dR.row(0) = dr_r.transpose();
		dR.row(1) = dr_g.transpose();
		dR.row(2) = dr_b.transpose();

	
		J_d0 = dI - dR*dSH_d0;
		J_d1 = dI1 - dR*dSH_d1;
		J_d2 = dI2 - dR*dSH_d2;
		J_d3 = dI3 - dR*dSH_d3;
	}

	

	J_d[0] = (J_d0);
	J_d[1] = (J_d1);
	J_d[2] = (J_d2);
	J_d[3] = (J_d3);
	J_d[4] = (tmp.second); // push_back direction for compute col in distJacobian

	return true; 
}

// ------------------------- nummerical dist Jacobian ---------------------------------

bool PsOptimizer::numerical_distJacobian(const SdfVoxel& v, const Vec3i& idx, Mat3f& R, Vec3f& t, const int frame_id, std::vector<Vec3f>& J_d)
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


//--------------------------------- light jacobian for all frames -------------------------------------------------------------------
//! compute the light jacobian 
Eigen::SparseMatrix<float> PsOptimizer::lightJacobian()
{
    std::vector<Eigen::Triplet<float> > tripleVector; // to construct sparse matirx
   
    size_t basis; //SH 1 or SH 2
    if(settings_->order == 1)basis = 4;
    if(settings_->order == 2)basis = 9;

    size_t total_num = surface_points_.size()*num_frames_*3;  // total number of residual

	Eigen::SparseMatrix<float> Jl(total_num, basis*num_frames_);

	int row_total;
	for(int frame = 0; frame < num_frames_; frame++)
    {
		int row = 0;
        int col = 0; // starting column of the light jacobian
        for(auto& lin_idx: surface_points_){

            std::vector<bool> vis = tSDF_->vis_[lin_idx];
            const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
			const Vec3i idx = tSDF_->line2idx(lin_idx);

            if(!vis[frame]){row++; continue;} //check visibility of the voxel in current frame

            std::vector<Eigen::VectorXf> J_l; 
            // lightJacobian(v, idx, frame, J_l);
			lightJacobian(v, frame, J_l);
			auto row_cur = frame*surface_points_.size()*3 + 3*row;
			// std::cout << "row: " << row << " row_cur: " << row_cur << std::endl;
            for(int i = 0; i < basis; i++){
                Eigen::Triplet<float> tri0(row_cur, basis*frame+col+i, J_l[0][i]); //r
                Eigen::Triplet<float> tri1(row_cur+1,  basis*frame+col+i, J_l[1][i]); //g
                Eigen::Triplet<float> tri2(row_cur+2,  basis*frame+col+i, J_l[2][i]); //b
                tripleVector.push_back(tri0);
                tripleVector.push_back(tri1);
                tripleVector.push_back(tri2);
            }

            row++;
			row_total = row_cur;
        }
    }

    Jl.setFromTriplets(tripleVector.begin(), tripleVector.end());
    if(total_num != row_total+3){std::cout << "rows number not correct! row should be " << total_num << " but it is " << row_total + 2 << std::endl;}
    return Jl;
	
}

//---------------------------------- albedo related for using albedo regularizer ----------------------------------------------
//! compute albedo jacobian, should be diagonal matrix
Eigen::SparseMatrix<float> PsOptimizer::albedoJacobian()
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

			// float Jr = rhoJacobian(v, idx, frame_id);
			float Jr = rhoJacobian(v, frame_id);
			auto row_cur = 3*surface_points_.size()*frame_id + 3*row;

			for(size_t i = 0; i < 3; i++){
				Eigen::Triplet<float> Tri(row_cur+i, 3*row+i, Jr);
				tripleVector.push_back(Tri);
			}

			row++;
		}
	}

	J_r.setFromTriplets(tripleVector.begin(), tripleVector.end());
	return J_r;

}


//--------------------------------- pose jacobian for all frames -------------------------------------------------------------------
//! compute pose jacobian sparse matrix
Eigen::SparseMatrix<float> PsOptimizer::poseJacobian()
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
Eigen::SparseMatrix<float> PsOptimizer::distJacobian()
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

			// numerical_distJacobian(v, idx, R, t, frame_id, J_n);
			// std::cout << "voxel: "<< idx << "\n" 
			// << "analytical gradient: " << J[0].transpose() << "\n" 
			// << "nummerical gradient: " << J_n[0].transpose() << std::endl;

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
std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > PsOptimizer::computeResidual()
{
	Eigen::VectorXf residual;
	Eigen::SparseMatrix<float> weight;

    size_t basis; //SH 1 or SH 2
    if(settings_->order == 1)basis = 4;
    if(settings_->order == 2)basis = 9;

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
            // if(std::isnan(r.sum())|| std::isinf(r.sum())){std::cout << "nan value r, intensity is: " << intensity.transpose() << std::endl; }
            row++;
        }
	}
	

	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > tmp(residual, weight);
	return tmp;

}



