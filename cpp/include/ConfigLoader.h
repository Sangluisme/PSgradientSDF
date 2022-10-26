//============================================================================
// Name        : ConfigLoader.h
// Author      : Lu Sang
// Date        : 04/2022
// License     : GNU General Public License
// Description : load config json file 
//============================================================================

#include "ps_optimizer/OptimizerSettings.h"
#include "sdf_tracker/TrackingSettings.h"
#include "img_loader/img_loader.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool LoadConfig(const std::string& filepath, TrackingSettings** trac_settings, OptimizerSettings** opt_settings, ImageLoader** Loader, json& config)
{
    std::ifstream file;
    file.open(filepath.c_str());
    if(!file.is_open()){
        std::cout << "can't load config file!" << std::endl;
        return false;
    }

    // read json file 
    file >> config;

    //check necessary argument
    if(!config.contains("input") || !config.contains("output") || ! config.contains("datatype")){
        std::cout << "missing necessary input arguments (input/out folder/datatype) in config file!" << std::endl;
        return false;
    }

    std::string input = config.at("input");
    std::string output = config.at("output");
    std::string datatype = config.at("datatype");

    ImageLoader* loader;

    // parse datatype 
    DataType DT;
    if (datatype == "tum") {
        DT = DataType::TUM_RGBD;
        loader = new TumrgbdLoader(input);
    }
    else if (datatype == "led" || datatype == "synth") {
        DT = DataType::SYNTH;
        loader = new SynthLoader(input);
    }
    else if (datatype == "intrinsic3d" || datatype == "multiview") {
        DT = DataType::MULTIVIEW;
        loader = new MultiviewLoader(input);
    }
    else {
        std::cerr << "Your specified dataset type is not supported (yet)." << std::endl;
        return false;
    }

    // consrtuct tracking settings
    TrackingSettings* trac_ = new TrackingSettings(input, output, DT);
    *Loader = loader;

    // check rest input
    if(config.contains("pose filename"))trac_->pose_file_ = config.at("pose filename");
    if(config.contains("first")) trac_->first_ = config.at("first");
    if(config.contains("last")) trac_->last_ = config.at("last");
    if(config.contains("voxel size")) trac_->voxel_size_ = config.at("voxel size");
    if(config.contains("truncation factor")) trac_->truncation_factor_ = config.at("truncation factor");
    if(config.contains("sharpness threshold")) trac_->sharpness_threshold_ = config.at("sharpness threshold");
    if(config.contains("zmin"))trac_->zmin_ = config.at("zmin");
    if(config.contains("zmax"))trac_->zmax_ = config.at("zmax");

    *trac_settings = trac_;

    // parse optimizer argument
    float reg_r, reg_n, reg_l;
    std::string loss_func, mtype;
    LossFunction loss_fun;

    // construct optimizer settings
    OptimizerSettings* opt_ = new OptimizerSettings();
    
    if(config.contains("model type")){
        mtype = config.at("model type");
    
        // parse model type
        ModelType Model;
        if(mtype == "SH1"){
            Model = ModelType::SH1;
            opt_->model = Model;
            opt_->order = 1;
        }
        else if(mtype == "SH2"){
            Model = ModelType::SH2;
            opt_->model = Model;
            opt_->order = 2;
        }
        else if(mtype == "LED"){
            Model = ModelType::LED;
            opt_->model = Model;
        }
        else
        {
            std::cerr << "Your specified model type is not supported (yet)." << std::endl;
            return false;
        }
    }

    // parse loss function
    if(config.contains("loss function")){
        loss_func = config.at("loss function");
        LossFunction Loss;
        if(loss_func == "cauchy"){
            Loss = LossFunction::CAUCHY;
            opt_->loss = Loss;
        }
        else if(loss_func == "l2"){
            Loss = LossFunction::L2;
            opt_->loss = Loss;
        }
        else if(loss_func == "huber"){
            Loss = LossFunction::HUBER;
            opt_->loss = Loss;
        }
        else if(loss_func == "trunc_l2"){
            Loss == LossFunction::TRUNC_L2;
            opt_->loss = Loss;
        }
        else if(loss_func == "tukey"){
            Loss == LossFunction::TUKEY;
            opt_->loss = Loss;
        }
        else
        {
            std::cerr << "Your specified loss function type is not supported (yet)." << std::endl;
            return false;
        }

    }

    //check the rest settings
    if(config.contains("reg albedo"))opt_->reg_weight_rho = config.at("reg albedo");
    if(config.contains("reg norm"))opt_->reg_weight_n =config.at("reg norm");
    if(config.contains("reg laplacian")) opt_->reg_weight_l = config.at("reg laplacian");
    if(config.contains("max iter")) opt_->max_it = config.at("max iter");
    if(config.contains("damping")) opt_->damping = config.at("damping");
    if(config.contains("converge threshold")) opt_->conv_threshold = config.at("converge threshold");
    if(config.contains("upsample")) opt_->upsample = config.at("upsample");


    if(config.contains("lambda")){
        float lambda = config.at("lambda");
        opt_->lambda = lambda;
        opt_->lambda_sq = lambda * lambda;
    }

    *opt_settings = opt_;

    // save this config file to results folder for reference
    // std::string save_path = (output + "config.json").c_str();
    std::ofstream save_conf(output + "saved_config.json", std::fstream::out);
    if(!save_conf.is_open()){
        std::cout << "could not save config file." << std::endl;
    }
    save_conf << std::setw(4) <<  config;


    return true;
    
}