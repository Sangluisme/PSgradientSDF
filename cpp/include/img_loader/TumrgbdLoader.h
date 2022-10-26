//============================================================================
// Name        : TumrgbdLoader.h
// Author      : Christiane Sommer
// Date        : 10/2019
// License     : GNU General Public License
// Description : class TumrgbdLoader
//============================================================================

#ifndef TUMRGBD_LOADER_H_
#define TUMRGBD_LOADER_H_

#include "ImageLoader.h"
#include <fstream>
#include <sstream>

class TumrgbdLoader : public ImageLoader {

private:

    std::ifstream depth_file_;
    std::ifstream color_file_;
    std::ifstream assoc_file_;
    
    void init() {
        depth_file_.open(path_ + "depth.txt");
        color_file_.open(path_ + "rgb.txt");
        assoc_file_.open(path_ + "associated.txt");

        return;
    }

public:

    TumrgbdLoader() :
        ImageLoader(1./5000, false)
    {
        init();
    }
    
    TumrgbdLoader(const std::string& path) :
        ImageLoader(1./5000, path, false)
    {
        init();
    }
    
    ~TumrgbdLoader() {
        depth_file_.close();
        color_file_.close();
        assoc_file_.close();
    }
    
    // bool load_next(cv::Mat& color, cv::Mat& depth) {
                
    //     std::string line, tmp, filename;
        
    //     line = "#";
    //     while (line.at(0) == '#') {
    //         if(!std::getline(depth_file_, line))
    //             return false;
    //     }
        
    //     std::istringstream dss(line);
    //     dss >> timestamp_ >> filename;
        
    //     if (!load_depth(filename, depth))
    //         return false;        
        
    //     line = "#";
    //     while (line.at(0) == '#') {
    //         if(!std::getline(color_file_, line))
    //             return false;
    //     }
        
    //     std::istringstream css(line);
    //     css >> tmp >> filename;

    //     if (!load_color(filename, color))
    //         return false;
        
    //     return true;
    // }

     bool load_next(cv::Mat& color, cv::Mat& depth) {
                
        std::string line, tmp, rgb_filename, depth_filename;
        
        line = "#";
        while (line.at(0) == '#') {
            if(!std::getline(assoc_file_, line))
                return false;
        }
        
        std::istringstream dss(line);
        dss >> timestamp_rgb_ >> rgb_filename >> timestamp_depth_ >> depth_filename;
        
        // std::cout << "load image " << timestamp_rgb_ << std::endl;
        
        timestamps_depth_.push_back(timestamp_depth_);
        timestamps_rgb_.push_back(timestamp_rgb_);
        
        if (!load_depth(depth_filename, depth))
            return false;        
        
        // line = "#";
        // while (line.at(0) == '#') {
        //     if(!std::getline(color_file_, line))
        //         return false;
        // }
        
        // std::istringstream css(line);
        // css >> tmp >> filename;

        if (!load_color(rgb_filename, color))
            return false;
        
        // std::cout << "loaded image rgb: " << rgb_filename << "\t depth: " << depth_filename << std::endl;

        return true;
    }

    void reset() {
        depth_file_.close();
        color_file_.close();
        assoc_file_.close();
        depth_file_.open(path_ + "depth.txt");
        color_file_.open(path_ + "rgb.txt");
        assoc_file_.open(path_ + "associated.txt");
        timestamps_depth_.clear();
        timestamps_rgb_.clear();
    }

    void reset_counter(){};
    bool load_reflectance(cv::Mat& albedo, cv::Mat& depth){return false;}

};

#endif // TUMRGBD_LOADER_H_
