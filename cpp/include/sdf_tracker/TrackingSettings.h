#ifndef TRACKING_SETTINGS_
#define TRACKING_SETTINGS_

#include <iostream>
#include <fstream>

// parse dataset type
enum class DataType {
    TUM_RGBD,
    SYNTH,
    MULTIVIEW
};

struct TrackingSettings {
    std::string input_;
    std::string output_;
    std::string pose_file_;
    DataType datatype_;
    size_t first_;
    size_t last_;
    float voxel_size_;
    float truncation_factor_;
    float zmin_;
    float zmax_;
    float sharpness_threshold_;
    TrackingSettings(std::string& input, std::string& output, DataType datatype):
        input_(input),
        output_(output),
        pose_file_("pose.txt"),
        datatype_(datatype),
        first_(0),
        last_(std::numeric_limits<size_t>::max()),
        voxel_size_(0.02),
        truncation_factor_(5),
        zmin_(0.5),
        zmax_(3.5),
        sharpness_threshold_(0.5)
        {}

};

#endif // TRACKING_SETTINGS_