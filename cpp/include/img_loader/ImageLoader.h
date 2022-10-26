//============================================================================
// Name        : ImageLoader.h
// Author      : Lu sang, Christiane Sommer
// Date        : 10/2019, 06/2022
// License     : GNU General Public License
// Description : class ImageLoader
//============================================================================

#ifndef IMAGE_LOADER_H_
#define IMAGE_LOADER_H_

// includes
#include <string>
#include <vector>
#include <fstream>
// library includes
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageLoader {

protected:
    
    Eigen::Matrix3f K_;
    
    const float unit_; // in meters
    const std::string path_; // path to directory containing data
    std::string timestamp_rgb_, timestamp_depth_; // current timestamp
    bool consecutive_numbers_ ;
    std::vector<std::string> timestamps_depth_, timestamps_rgb_;


public:
    
 
    ImageLoader() :
        unit_(1.),
        path_(""),
        timestamp_rgb_(""),
        timestamp_depth_(""),
        consecutive_numbers_(false)
    {}
    
    ImageLoader(float unit, bool consecutive_numbers) :
        unit_(unit),
        path_(""),
        timestamp_rgb_(""),
        timestamp_depth_(""),
        consecutive_numbers_(consecutive_numbers)
    {}
    
    ImageLoader(const std::string& path, bool consecutive_numbers) :
        unit_(1.),
        path_(path),
        timestamp_rgb_(""),
        timestamp_depth_(""),
        consecutive_numbers_(consecutive_numbers)
    {}
    
    ImageLoader(float unit, const std::string& path, bool consecutive_numbers) :
        unit_(unit),
        path_(path),
        timestamp_rgb_(""),
        timestamp_depth_(""),
        consecutive_numbers_(consecutive_numbers)
    {}
    
    virtual ~ImageLoader() {}
    
    Eigen::Matrix3f K() const {
        return K_;
    }
    
    std::string rgb_timestamp() {
        return timestamp_rgb_;
    }

    std::string depth_timestamp() {
        return timestamp_depth_;
    }

    
    std::string getDepthTimestamp(const int i){
        return timestamps_depth_[i];
    }

    std::string getRgbTimestamp(const int i){
        return timestamps_rgb_[i];
    }

    std::vector<std::string> getDepthTimestamps(){
		return timestamps_depth_;
	}

    std::vector<std::string> getRgbTimestamps(){
		return timestamps_rgb_;
	}

    void setDepthTimestamps(std::vector<std::string> timestamps){
        timestamps_depth_ = timestamps;
    }

    void setRgbTimestamps(std::vector<std::string> timestamps){
        timestamps_rgb_ = timestamps;
    }
    
    bool load_intrinsics(const std::string& filename = "intrinsics.txt") {

        if (filename.empty())
            return false;

        std::ifstream infile(path_ + filename);
        if (!infile.is_open())
            return false;

        // load intrinsic camera matrix
        float tmp = 0;
        for (size_t i=0; i<3; ++i) for (size_t j=0; j<3; ++j) {
                infile >> tmp;
                K_(i, j) = tmp;
        }
        
        infile.close();

        return true;
    }

    bool load_depth(const std::string& filename, cv::Mat& depth) {

        if (filename.empty()) {
            std::cerr << "Error: missing filename" << std::endl;
            return false;
        }
        
        // fill/read 16 bit depth image
        cv::Mat depthIn = cv::imread(path_ + filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        if (depthIn.empty()) {
            std::cerr << "Error: empty depth image " << path_ + filename << std::endl;
            return false;
        }
        depthIn.convertTo(depth, CV_32FC1, unit_);

        return true;
    }

    bool load_gray(const std::string& filename, cv::Mat& gray){
        if (filename.empty()) {
            std::cerr << "Error: missing filename" << std::endl;
            return false;
        }
        
        // load gray
        cv::Mat imgGray = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        // convert gray to float
        imgGray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);

        if(gray.empty()) {
            std::cerr << "Error: empty gray scale image " << path_ + filename << std::endl;
            return false;
        }
        return true;

    }
    
    bool load_color(const std::string& filename, cv::Mat& color) {

        if (filename.empty()) {
            std::cerr << "Error: missing filename" << std::endl;
            return false;
        }
        
        // load color
        cv::Mat imgColor = cv::imread(path_ + filename);

        if(imgColor.channels()==1)
        {
            cv::cvtColor(imgColor, imgColor, cv::COLOR_GRAY2BGR);
        }
        imgColor.convertTo(color, CV_32FC3, 1.0f / 255.0f);
        if (color.empty()) {
            std::cerr << "Error: empty color image " << path_ + filename << std::endl;
            return false;
        }

        return true;
    }

    bool load_albedo(const std::string& filename, cv::Mat& albedo) {

         if (filename.empty()) {
            std::cerr << "Error: missing filename" << std::endl;
            return false;
        }
        
        // load color
        cv::Mat imgColor = cv::imread(path_ + filename);
        if(imgColor.empty())return false;
        
        if(imgColor.channels()==1)
        {
            cv::cvtColor(imgColor, imgColor, cv::COLOR_GRAY2BGR);
        }

        imgColor.convertTo(albedo, CV_32FC3, 1.0f / 255.0f);

        if (albedo.empty()) {
            std::cerr << "Error: empty color image " << path_ + filename << std::endl;
            return false;
        }

        return true;
    }

    bool consecutive_numbers(){
        return consecutive_numbers_;
    }
    
    virtual bool load_next(cv::Mat &color, cv::Mat &depth) = 0;


    virtual bool load_keyframe(cv::Mat &color, cv::Mat &depth, const int frame) {
        // only implement if consecutive_numbers_ == true
        return false;
    }

    bool load_pose(std::string& filename, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& poses)
    {
        
        std::ifstream file;
        file.open((path_ + filename).c_str());
        if(!file.is_open()){
            std::cout << "can't load poses!" << std::endl; 
            return false;
        }
        std::string line;
        while (std::getline(file, line)){
            float timestamp;
            Eigen::Vector3f translation;
            Eigen::Quaternionf q;
            std::stringstream s(line);
            s >> timestamp >> translation[0] >> translation[1] >> translation[2] >> q.x() >> q.y() >> q.z() >> q.w();
            if (q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z() < 0.99) {
                std::cerr << "pose " << timestamp << " has invalid rotation" << std::endl;
            }
            Eigen::Matrix4f tmp = Eigen::Matrix4f::Identity();
            tmp.topRightCorner(3,1) = translation;
            Eigen::Matrix3f rot = q.toRotationMatrix();
            tmp.topLeftCorner(3,3) = rot;
            poses.push_back(tmp);
        }

        if (poses.size()==0) return false;

        return true;

    }


    virtual void reset() = 0;
    virtual void reset_counter() = 0;
    virtual bool load_reflectance(cv::Mat& albedo, cv::Mat& depth) = 0;

};

#endif // IMAGE_LOADER_H_
