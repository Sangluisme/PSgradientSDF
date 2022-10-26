#ifndef MARCHING_CUBES_NO_COLOR_H
#define MARCHING_CUBES_NO_COLOR_H

#include <vector>

#include "mat.h"

#include <map>

class MarchingCubesNoColor
{
public:
    MarchingCubesNoColor(const Vec3i &dimensions, const Vec3f &size, const Vec3f &origin);

    ~MarchingCubesNoColor();

    bool computeIsoSurface(const float* tsdf, const float* weights, float isoValue = 0.0f);

    bool savePly(const std::string &filename) const;

protected:

    static int edgeTable[256];
    
    static int triTable[256][16];

    inline int computeLutIndex(int i, int j, int k, float isoValue);

    Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue);

    Vec3f getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue);

    Vec3b getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue);

    void computeTriangles(int cubeIndex, const Vec3f edgePoints[12]);

    inline unsigned int addVertex(const Vec3f &v);

    Vec3f voxelToWorld(int i, int j, int k) const;

    std::vector<Vec3f> vertices_;
    std::vector<Vec3i> faces_;
    Vec3i dim_;
    Vec3f size_;
    Vec3f voxelSize_;
    Vec3f origin_;

    const float* tsdf_;
    const float* weights_;
};

#endif
