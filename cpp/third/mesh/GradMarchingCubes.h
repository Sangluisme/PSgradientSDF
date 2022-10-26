#ifndef GRAD_MARCHING_CUBES_H
#define GRAD_MARCHING_CUBES_H

#include <vector>

#include "mat.h"

#include <map>

class GradMarchingCubes
{
public:
    GradMarchingCubes(const Vec3i &dimensions, const Vec3f &size, const Vec3f &origin);

    ~GradMarchingCubes();

    bool computeIsoSurface(const float* tsdf, const float* weights, const unsigned char* red, const unsigned char* green, const unsigned char* blue, float isoValue = 0.0f);
    bool computeIsoSurface(const float* tsdf, const float* weights, const Vec3f* grads, const unsigned char* red, const unsigned char* green, const unsigned char* blue, float isoValue = 0.0f);

    bool savePly(const std::string &filename) const;

protected:

    static int edgeTable[256];
    
    static int triTable[256][16];

    inline int computeLutIndex(int i, int j, int k, float isoValue);

    Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue);

    Vec3f getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue);
    Vec3f getVertex(int i1, int j1, int k1, float isoValue);

    Vec3b getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue);
    Vec3b getColor(int x1, int y1, int z1, float isoValue);

    void computeTriangles(int cubeIndex, const Vec3f edgePoints[12], const Vec3b edgeColors[12]);

    inline unsigned int addVertex(const Vec3f &v, const Vec3b &c);

    Vec3f voxelToWorld(int i, int j, int k) const;

    std::vector<Vec3f> vertices_;
    std::vector<Vec3b> colors_;
    std::vector<Vec3i> faces_;
    Vec3i dim_;
    Vec3f size_;
    Vec3f voxelSize_;
    Vec3f origin_;

    const float* tsdf_;
    const float* weights_;
    const Vec3f* grads_;
    const unsigned char* red_;
    const unsigned char* green_;
    const unsigned char* blue_;
};

#endif
