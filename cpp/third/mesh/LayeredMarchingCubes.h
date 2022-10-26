#ifndef LAYERED_MARCHING_CUBES_H
#define LAYERED_MARCHING_CUBES_H

#include <vector>

#include "mat.h"

#include <map>

#include "sdf_voxel/SdfVoxel.h"

class LayeredMarchingCubes
{
public:

    using voxel_phmap = phmap::parallel_node_hash_map<Vec3i, SdfVoxelHr,
                                                    phmap::priv::hash_default_hash<Vec3i>,
                                                    phmap::priv::hash_default_eq<Vec3i>,
                                                    Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxelHr>>>;

    // LayeredMarchingCubes(const Vec3i &dimensions, const Vec3f &size);

    LayeredMarchingCubes(const Vec3f &voxelSize);

    ~LayeredMarchingCubes();

    bool computeIsoSurface(const voxel_phmap* sdf_map, float isoValue = 0.0f);

    bool savePly(const std::string &filename) const;

protected:

    static int edgeTable[256];
    
    static int triTable[256][16];

    inline int computeLutIndex(int i, int j, int k, float isoValue);

    void copyLayer(int z, const voxel_phmap* sdf_map);

    inline void zeroWeights(int x, int y, int z);

    inline void copyCube(int x, int y, int z, const float w,
                        const Eigen::Matrix<float, 8, 1>& d,
                        const Eigen::Matrix<float, 8, 1>& r,
                        const Eigen::Matrix<float, 8, 1>& g,
                        const Eigen::Matrix<float, 8, 1>& b
                        );

    inline void setVoxel(const size_t off, const size_t corner, const float w,
                        const Eigen::Matrix<float, 8, 1>& d,
                        const Eigen::Matrix<float, 8, 1>& r,
                        const Eigen::Matrix<float, 8, 1>& g,
                        const Eigen::Matrix<float, 8, 1>& b
                        );

    Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue);

    Vec3f getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue);

    Vec3b getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue);

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
    Vec3i min_;

    // layers
    float* tsdf_;
    float* weights_;
    unsigned char* red_;
    unsigned char* green_;
    unsigned char* blue_;
};

#endif
