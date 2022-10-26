//============================================================================
// Name        : PsOptimizer.cpp
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : implementation of class PsOptimizer
//============================================================================

#include "PsOptimizer.h"
#include "sdf_tracker/Sdfvoxel.h"
#include "Timer.h"

// ============================= class functions =====================================

PsOptimizer::PsOptimizer(VolumetricGradSdf* tSDF,
					const float voxel_size,
                    const Mat3f& K,
					std::string save_path,
					OptimizerSettings* settings):
			Optimizer(tSDF, voxel_size, K, save_path, settings)
{
	init();
}

void PsOptimizer::init()
{	
	num_frames_ = frame_idx_.size();
	num_voxels_ = tSDF_->grid_dim_[0]*tSDF_->grid_dim_[1]*tSDF_->grid_dim_[2];
	// initialize lighting vectors for each frame
	Vec3f s(0.0,0.0,-1.0);
	
	for(size_t frame = 0; frame < num_frames_; frame++){
		Mat3f R = getRotation(frame);
		Eigen::VectorXf l = SH(R*s, settings_->order);
		l[0] = 0.02;
		light_.push_back(l);
	}
	

	select_vis();
	
}


//------------------ energy function--------------------------------------------

float PsOptimizer::getPSEnergy()
{
	float E = 0;

	for(auto& lin_idx: surface_points_){

		const SdfVoxel& v = tSDF_->tsdf_[lin_idx];
		std::vector<bool> vis = tSDF_->vis_[lin_idx];
		
		for(size_t frame_id = 0; frame_id < num_frames_; frame_id++){
			if(!vis[frame_id]){ // vis for whole sequence, frame_idx_ for key frame
				continue;
			}

			Vec3i idx = tSDF_->line2idx(lin_idx);

			Mat3f R = getRotation(frame_id);
			Vec3f t = getTranslation(frame_id);
			cv::Mat img = getImage(frame_id);

			Vec3f intensity;
			if(!getIntensity(idx, v, R, t, img, intensity)){
				continue;
			}

			E += computeLoss(intensity - renderedIntensity(v, idx, frame_id));
		}

	}

	return E/static_cast<float>(surface_points_.size());
}




//------------------------------- optimization function ---------------------------------------------------------------------------------

void PsOptimizer::optimizeAlbedoAll(bool albedo_reg, float damping)
{
	Eigen::SparseMatrix<float> J_r = albedoJacobian();
	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > pair = computeResidual();
	Eigen::VectorXf r = pair.first;
	Eigen::SparseMatrix<float> w = pair.second;

	Eigen::SparseMatrix<float> H = J_r.transpose()*w*J_r;
	Eigen::VectorXf b = J_r.transpose()*w*r;

	if(albedo_reg){
		std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> tmp = albedoRegJacobian();
		Eigen::SparseMatrix<float> Jr_r = tmp.first;
		Eigen::VectorXf r_r = tmp.second;
		H += settings_->reg_weight_rho*Jr_r.transpose()*Jr_r;
		b += settings_->reg_weight_rho*Jr_r.transpose()*r_r;
	}

	if(damping!=0.0){
		H.diagonal() += damping*H.diagonal();
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
   
	solver.compute(H);
	if(solver.info() != Eigen::Success ){
		std::cout << "decomposition failed! " << std::endl;
	}
    Eigen::VectorXf delta_r = solver.solve(b);
	if(solver.info()!=Eigen::Success) {
          std::cout << "solver failed: " << solver.info() << "\n";
	}

	if(solver.info()==Eigen::Success){
		updateAlbedo(delta_r);
	}
}


void PsOptimizer::optimizeDistAll(bool normal_reg, bool laplacian_reg, float damping)
{
	// bool normal_reg = false;

	Eigen::SparseMatrix<float> J_d = distJacobian();
	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > tmp = computeResidual();
	Eigen::VectorXf r = tmp.first;
	Eigen::SparseMatrix<float> w = tmp.second;

	Eigen::SparseMatrix<float> H = J_d.transpose()*w*J_d;
	Eigen::VectorXf b = J_d.transpose()*w*r;

	if(normal_reg){
		std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> pair = distRegJacobian();
		Eigen::SparseMatrix<float> Jr_d = pair.first;
		Eigen::VectorXf r_n = pair.second;
		H += settings_->reg_weight_n*(Jr_d.transpose()*Jr_d);
		b += settings_->reg_weight_n*(Jr_d.transpose()*r_n);
	}

	if(laplacian_reg){
		std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> pair = distLaplacianJacobian();
		Eigen::SparseMatrix<float> Jl_d = pair.first;
		Eigen::VectorXf r_n = pair.second;
		H += settings_->reg_weight_l*(Jl_d.transpose()*Jl_d);
		b += settings_->reg_weight_l*(Jl_d.transpose()*r_n);
	}

	if(damping!=0.0){
		H.diagonal() += damping*H.diagonal();
	}

	
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
   
	solver.compute(H);
	if(solver.info() != Eigen::Success ){
		std::cout << "decomposition failed! " << std::endl;
	}
    Eigen::VectorXf delta_d = solver.solve(b);
	if(solver.info()!=Eigen::Success) {
          std::cout << "solver failed: " << solver.info() << "\n";
	}
	
	if(solver.info()==Eigen::Success){
		updateDist(delta_d, true);
	}
	 
}


void PsOptimizer::optimizeLightAll()
{
	Eigen::SparseMatrix<float> Jl = lightJacobian();
	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > pair = computeResidual();
	Eigen::VectorXf r = pair.first;
	Eigen::SparseMatrix<float> w = pair.second;
	Eigen::SparseMatrix<float> H = Jl.transpose()*w*Jl;
	Eigen::VectorXf b = Jl.transpose()*w*r;

	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
   
	solver.compute(H);
	if(solver.info() != Eigen::Success ){
		std::cout << "decomposition failed! " << std::endl;
	}

    Eigen::VectorXf delta_l = solver.solve(b);
	if(solver.info()!=Eigen::Success) {
          std::cout << "solver failed: " << solver.info() << "\n";
	}
  	
	size_t basis; //SH 1 or SH 2
    if(settings_->order == 1)basis = 4;
    if(settings_->order == 2)basis = 9;
	for(size_t frame = 0; frame < num_frames_; frame++){
		light_[frame] -= delta_l.segment(frame*basis, basis);
	}

}


//! optimize camera poses
void PsOptimizer::optimizePosesAll(float damping)
{
	Eigen::SparseMatrix<float> Jc = poseJacobian();
	std::pair<Eigen::VectorXf, Eigen::SparseMatrix<float> > pair = computeResidual();
	Eigen::VectorXf r = pair.first;
	Eigen::SparseMatrix<float> w = pair.second;
	Eigen::SparseMatrix<float> H = Jc.transpose()*w*Jc;
	Eigen::VectorXf b = Jc.transpose()*w*r;

	
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
	if(damping!=0.0){
		H.diagonal() += damping*H.diagonal();
	}
   
	solver.compute(H);
	if(solver.info() != Eigen::Success ){
		std::cout << "decomposition failed! " << std::endl;
	}

    Eigen::VectorXf delta_xi = solver.solve(b);
	if(solver.info()!=Eigen::Success) {
          std::cout << "solver failed: " << solver.info() << "\n";
	}

	updatePose(delta_xi);

}



// ---------------------------- optimization function -----------------------------------
bool PsOptimizer::alternatingOptimize(bool light, bool albedo, bool distance, bool pose)
{

	// save iter detail
	std::ofstream file;
	file.open((save_path_ + "optimizer_doc.txt").c_str());

	bool normal_reg = false;
	bool albedo_reg = false;
	bool laplacian_reg = false;

	if(settings_->reg_weight_n != 0.0) normal_reg = true;
	if(settings_->reg_weight_rho != 0.0) albedo_reg = true;
	if(settings_->reg_weight_l != 0.0) laplacian_reg = true;
	
	std::cout << "albation study settings: \n" <<
				"light: " << light << "\n" << "albedo: " << albedo << "\n" << "distance: " << distance << "\n" << "pose: " << pose << std::endl;
	
	float damping = settings_->damping;

	file << "albation study settings: \t" <<
			"light: " << light << "\t" << "albedo: " << albedo << "\t" << "distance: " << distance << "\t" << "pose: " << pose << "\n" << "num of key frame: " << num_frames_ << " \n total voxels: " << num_voxels_  << "\n";


	Timer T;

	T.tic();
	// getSurfaceVoxel(); // compute surface points.
	initAlbedo();
	T.toc("initial albedo");

	float E, E_total;
	float E_n = 0.0;
	float E_r = 0.0;
	float E_l = 0.0;
	E = getPSEnergy();
	if(normal_reg) {
		E_n = getNormalEnergy();
		settings_->reg_weight_n *= E / E_n; //normalize the weight
	}
	if(albedo_reg) E_r = getAlbedoRegEnergy();
	
	if(laplacian_reg){ 
		E_l = getLaplacianEnergy();
		settings_->reg_weight_l *= E / E_l;
		if(settings_->upsample)laplacian_reg=false;
	}


	E_total = getTotalEnergy(E, E_n, E_l, E_r, file);

	float rel_diff;
	std::vector<float> E_vec;

	E_vec.push_back(E_total);

	std::cout << "======================== start alternating optimization ================================= " << std::endl
		<< "======> total key frame:\t " << num_frames_ << std::endl
		<< "======> total voxels: \t" << num_voxels_ << std::endl
		<< "======> inital PS energy:" << E << "\t normal reg energy: " << settings_->reg_weight_n*E_n << "\t laplacian reg energy: " << settings_->reg_weight_l*E_l << " \t rho reg energy: " << settings_->reg_weight_rho*E_r << std::endl
		<< "================================================================================================= " << std::endl;

	int iter = 0;

	while(iter < settings_->max_it){
		if(albedo){
			
			T.tic();
			optimizeAlbedoAll(albedo_reg, damping);
			// optimizeAlbedo();
			T.toc("albedo optimize");

			E = getPSEnergy();
			if(albedo_reg) E_r = getAlbedoRegEnergy();
			std::cout << "===> [" << iter  << "]: after albedo optimization: ";

			file << "===> [" << iter  << "]: after albedo optimization: \n";
			E_total = getTotalEnergy(E, E_n, E_l, E_r, file);
		}

		if(light){
			T.tic();
			optimizeLightAll();
			T.toc("lights optimize");
			E = getPSEnergy();
			std::cout << "===> [" << iter  << "]: after lights optimization: ";
			
			file << "===> [" << iter  << "]: after light optimization: \n";
			E_total = getTotalEnergy(E, E_n, E_l, E_r, file);

			
		}

		if(distance){	
			
			T.tic();
			optimizeDistAll(normal_reg, laplacian_reg, damping);
			T.toc("dist optimize");

			E = getPSEnergy();
			if(normal_reg) E_n = getNormalEnergy();
			if(laplacian_reg) E_l = getLaplacianEnergy();

			std::cout << "===> [" << iter  << "]: after distance optimization: ";
			
			file << "===> [" << iter  << "]: after distance optimization: \n";
			E_total = getTotalEnergy(E, E_n, E_l, E_r, file);

		}

		if(pose){
			T.tic();
			optimizePosesAll(damping);
			T.toc("poses optimize");

			E = getPSEnergy();
			std::cout << "===> [" << iter  << "]: after pose optimization: ";

			file << "===> [" << iter  << "]: after pose optimization: \n";
			E_total = getTotalEnergy(E, E_n, E_l, E_r, file);

		}

		E_vec.push_back(E_total);
		
		rel_diff = abs(E_vec.end()[-2] - E_total)/E_vec.end()[-2];
		std::cout << "===> [" << iter << "]: relative diff " << rel_diff << std::endl; 
		file << "===> [" << iter << "]: relative diff " << rel_diff << "\n";

		if(rel_diff < settings_->conv_threshold)
		{
			std::cout << "===> [" << iter  << "]: converged!" <<std::endl;
			file << "===> [" << iter  << "]: converged! \n";
			save_pointcloud("final_refined");
			extract_mesh("final_refined");
			return true;
		}

		if(E_vec.end()[-2] < E_total )
		{
			save_pointcloud("final_refined");
			extract_mesh("final_refined");
			std::cout << "===> [" << iter  << "]: diverged!" <<std::endl;
			file << "===> [" << iter  << "]: diverged!\n";
			return false;
		}
		
		if(iter == 5 && settings_->upsample){

			// if upsampled, need laplacian to smooth a bit
			// maybe we can remove it
			if(settings_->reg_weight_l==0.0)settings_->reg_weight_l = 1.0;
			laplacian_reg = true;

			T.tic();
			subsampling();
			T.toc("upsample");
		
			save_pointcloud("upsample_after_" + std::to_string(iter));//DEBUG
			extract_mesh("upsample_after_" + std::to_string(iter));//DEBUG

			file << "===> [" << iter  << "]: after pose optimization: \n";
			
			E_l = getLaplacianEnergy();
			settings_->reg_weight_l *= E / E_l;
	
			E_total = getTotalEnergy(E, E_n, E_l, E_r, file);

			E_vec.push_back(E_total);

		}

		if (iter > 15 && settings_->upsample){
			settings_->reg_weight_l = 0.0;
		}


		++iter;

		// save after each 3 iterations for debug
		if(iter%3 == 0){
			savePoses("after_poses_opt_" + std::to_string(iter));
			save_pointcloud("after_iter_" + std::to_string(iter));
			extract_mesh("after_iter_" + std::to_string(iter));
		}
	
	}

	return false;
}