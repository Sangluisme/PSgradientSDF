#ifndef SDFVOXEL_H_
#define SDFVOXEL_H_

#include "mat.h"

struct SdfVoxel {
    float dist = 0;
    Vec3f grad = Vec3f::Zero();
    float weight = 0;
    float r = 1.0;
    float g = 1.0;
    float b = 1.0;
};

#endif //SDFVOXEL_H