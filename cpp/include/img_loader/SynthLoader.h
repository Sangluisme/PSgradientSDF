//============================================================================
// Name        : SynthLoader.h
// Author      : Christiane Sommer, Lu Sang
// Date        : 03/2021
// License     : GNU General Public License
// Description : class imageLoader
//============================================================================

#ifndef SYNTH_LOADER_H_
#define SYNTH_LOADER_H_

#include <iomanip>
#include "ImageLoader.h"

class SynthLoader : public ImageLoader {

private:

    size_t counter;

public:

    SynthLoader() :
        ImageLoader(1./1000, true),
        counter(1)
    {}
    
    SynthLoader(const std::string& path) :
        ImageLoader(1./1000, path, true),
        counter(1)
    {}
    
    ~SynthLoader() {}
    
    bool load_next(cv::Mat& color, cv::Mat& depth) {

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(3) << counter;
        
        timestamp_rgb_ = ss.str();
        timestamp_depth_ = timestamp_rgb_;

        const std::string filename = timestamp_rgb_ + ".png";
        
    
        if (!load_depth("depth/" + filename, depth))
            return false;
        
            
        
        if (!load_color("rgb/" + filename, color))
            return false;
        
        ++counter;
        
        return true;
    }

    bool load_reflectance(cv::Mat& albedo, cv::Mat& depth){
        
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(3) << counter;
        
        std::string filename = ss.str() + ".png";
        if (!load_albedo("albedo/" + filename, albedo))
            return false;

        if (!load_depth("depth/" + filename, depth))
            return false;

        ++counter;
        
        return true;

    }

    bool load_keyframe(cv::Mat& color, cv::Mat& depth, const int frame) {
        
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(3) << frame+1;

        timestamp_rgb_ = ss.str();
        timestamp_depth_ = timestamp_rgb_;

        const std::string filename = timestamp_rgb_ + ".png";
        timestamps_depth_.push_back(timestamp_depth_);
        timestamps_rgb_.push_back(timestamp_rgb_);

        if (!load_depth("depth/" + filename, depth))
            return false;
        
        if (!load_color("albedo/" + filename, color))
            return false;
        
        std::cout<<"image:" << filename << " is loaded!" << std::endl;
        return true;
    }

    void reset_counter()
    {
        counter = 1;
    }

    void reset()
    {
        counter = 1;
        timestamps_depth_.clear();
        timestamps_rgb_.clear();
    }

};

#endif // SYNTH_LOADER_H_
