#include "LayeredMarchingCubes.h"

#include <fstream>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include<limits>

typedef boost::tuple<double, double, double> tuple3;


// LayeredMarchingCubes::LayeredMarchingCubes(const Vec3i &dimensions, const Vec3f &size) :
//     dim_(dimensions),
//     size_(size),
//     voxelSize_(Vec3f(size_.cwiseQuotient(dim_.cast<float>()))),
//     tsdf_(0),
//     weights_(0),
//     red_(0),
//     green_(0),
//     blue_(0)
// {
// }

LayeredMarchingCubes::LayeredMarchingCubes(const Vec3f &voxelSize) :
    voxelSize_(voxelSize),
    origin_(Vec3f::Zero()),
    tsdf_(0),
    weights_(0),
    red_(0),
    green_(0),
    blue_(0)
{
    // TODO: what happens with dim_ and size_? (unspecified at this point)
}


LayeredMarchingCubes::~LayeredMarchingCubes()
{
    if (tsdf_)    delete[] tsdf_;
    if (weights_) delete[] weights_;
    if (red_)     delete[] red_;
    if (green_)   delete[] green_;
    if (blue_)    delete[] blue_;
}


// To find which edges are intersected by the surface, we find the edges in a (trivial) table.
// Table giving the edges intersected by the surface:
int LayeredMarchingCubes::edgeTable[256] = { 0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,
        0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96,
        0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f,
        0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5,
        0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a,
        0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
        0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759,
        0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56,
        0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
        0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5,
        0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a,
        0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453,
        0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69,
        0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,
        0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
        0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435,
        0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a,
        0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393,
        0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };


// This table gives the edges forming triangles for the surface.
int LayeredMarchingCubes::triTable[256][16] = { { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1 }, { 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1 },
        { 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
                8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
                2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 }, { 3,
                11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 }, {
                3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        { 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 }, { 3, 9,
                0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 }, { 9, 8,
                10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4,
                7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
                4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
                2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 3,
                4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 }, { 9, 2,
                10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 }, { 2, 10, 9,
                2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 }, { 8, 4, 7, 3, 11,
                2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 4, 7, 11, 2,
                4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 }, { 9, 0, 1, 8, 4, 7,
                2, 3, 11, -1, -1, -1, -1, -1, -1, -1 }, { 4, 7, 11, 9, 4, 11,
                9, 11, 2, 9, 2, 1, -1, -1, -1, -1 }, { 3, 10, 1, 3, 11, 10, 7,
                8, 4, -1, -1, -1, -1, -1, -1, -1 }, { 1, 11, 10, 1, 4, 11, 1,
                0, 4, 7, 11, 4, -1, -1, -1, -1 }, { 4, 7, 8, 9, 0, 11, 9, 11,
                10, 11, 0, 3, -1, -1, -1, -1 }, { 4, 7, 11, 4, 11, 9, 9, 11,
                10, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 4, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 4, 0, 8, 3, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 5, 4, 1, 5, 0, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 8, 5, 4, 8, 3, 5, 3, 1,
                5, -1, -1, -1, -1, -1, -1, -1 }, { 1, 2, 10, 9, 5, 4, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 0, 8, 1, 2, 10, 4, 9, 5,
                -1, -1, -1, -1, -1, -1, -1 }, { 5, 2, 10, 5, 4, 2, 4, 0, 2, -1,
                -1, -1, -1, -1, -1, -1 }, { 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4,
                8, -1, -1, -1, -1 }, { 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1,
                -1, -1, -1, -1, -1 }, { 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1,
                -1, -1, -1, -1 }, { 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1,
                -1, -1, -1 }, { 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1,
                -1, -1, -1 }, { 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1,
                -1, -1 }, { 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1,
                -1 }, { 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1,
                -1 }, { 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1 },
        { 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 }, { 0, 7, 8,
                0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 }, { 1, 5, 3, 3,
                5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 7, 8, 9,
                5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 }, { 10, 1, 2, 9, 5,
                0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 }, { 8, 0, 2, 8, 2, 5, 8,
                5, 7, 10, 5, 2, -1, -1, -1, -1 }, { 2, 10, 5, 2, 5, 3, 3, 5, 7,
                -1, -1, -1, -1, -1, -1, -1 }, { 7, 9, 5, 7, 8, 9, 3, 11, 2, -1,
                -1, -1, -1, -1, -1, -1 }, { 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,
                11, -1, -1, -1, -1 }, { 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7,
                -1, -1, -1, -1 }, { 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1,
                -1, -1, -1, -1 }, { 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1,
                -1, -1, -1 }, { 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10,
                0, -1 },
        { 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 }, { 11, 10, 5,
                7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 10, 6, 5,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8,
                3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 0,
                1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 8,
                3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 }, { 1, 6, 5,
                2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 6, 5,
                1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 }, { 9, 6, 5, 9,
                0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 }, { 5, 9, 8, 5, 8,
                2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 }, { 2, 3, 11, 10, 6, 5,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 0, 8, 11, 2, 0,
                10, 6, 5, -1, -1, -1, -1, -1, -1, -1 }, { 0, 1, 9, 2, 3, 11, 5,
                10, 6, -1, -1, -1, -1, -1, -1, -1 }, { 5, 10, 6, 1, 9, 2, 9,
                11, 2, 9, 8, 11, -1, -1, -1, -1 }, { 6, 3, 11, 6, 5, 3, 5, 1,
                3, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 11, 0, 11, 5, 0, 5, 1,
                5, 11, 6, -1, -1, -1, -1 }, { 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5,
                9, -1, -1, -1, -1 }, { 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1,
                -1, -1, -1, -1 }, { 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1 }, { 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1,
                -1, -1, -1 }, { 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1,
                -1, -1 },
        { 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 }, { 6, 1, 2, 6,
                5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 }, { 1, 2, 5, 5, 2,
                6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 }, { 8, 4, 7, 9, 0, 5, 0,
                6, 5, 0, 2, 6, -1, -1, -1, -1 }, { 7, 3, 9, 7, 9, 4, 3, 2, 9,
                5, 9, 6, 2, 6, 9, -1 }, { 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1,
                -1, -1, -1, -1, -1 }, { 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11,
                -1, -1, -1, -1 }, { 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1,
                -1, -1, -1 }, { 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10,
                6, -1 },
        { 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 }, { 5, 1, 11,
                5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 }, { 0, 5, 9, 0, 6,
                5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 }, { 6, 5, 9, 6, 9, 11, 4, 7,
                9, 7, 11, 9, -1, -1, -1, -1 }, { 10, 4, 9, 6, 4, 10, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1 }, { 4, 10, 6, 4, 9, 10, 0, 8,
                3, -1, -1, -1, -1, -1, -1, -1 }, { 10, 0, 1, 10, 6, 0, 6, 4, 0,
                -1, -1, -1, -1, -1, -1, -1 }, { 8, 3, 1, 8, 1, 6, 8, 6, 4, 6,
                1, 10, -1, -1, -1, -1 }, { 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1,
                -1, -1, -1, -1, -1 }, { 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1,
                -1, -1, -1 }, { 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1 }, { 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1,
                -1, -1 }, { 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1,
                -1, -1 }, { 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1,
                -1 }, { 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
        { 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 }, { 9, 6, 4, 9,
                3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 }, { 8, 11, 1, 8, 1, 0,
                11, 6, 1, 9, 1, 4, 6, 4, 1, -1 }, { 3, 11, 6, 3, 6, 0, 0, 6, 4,
                -1, -1, -1, -1, -1, -1, -1 }, { 6, 4, 8, 11, 6, 8, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1 }, { 7, 10, 6, 7, 8, 10, 8, 9, 10,
                -1, -1, -1, -1, -1, -1, -1 }, { 0, 7, 3, 0, 10, 7, 0, 9, 10, 6,
                7, 10, -1, -1, -1, -1 }, { 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8,
                0, -1, -1, -1, -1 }, { 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1,
                -1, -1, -1, -1 }, { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1,
                -1, -1 }, { 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
        { 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 }, { 7, 3, 2,
                6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 2, 3, 11,
                10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 }, { 2, 0, 7, 2, 7,
                11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 }, { 1, 8, 0, 1, 7, 8, 1,
                10, 7, 6, 7, 10, 2, 3, 11, -1 }, { 11, 2, 1, 11, 1, 7, 10, 6,
                1, 6, 7, 1, -1, -1, -1, -1 }, { 8, 9, 6, 8, 6, 7, 9, 1, 6, 11,
                6, 3, 1, 3, 6, -1 }, { 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0,
                -1, -1, -1, -1 }, { 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1 }, { 3, 0, 8, 11, 7, 6, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1 }, { 0, 1, 9, 11, 7, 6, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1 }, { 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1,
                -1, -1, -1, -1, -1 }, { 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1,
                -1, -1, -1, -1, -1 }, { 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1,
                -1, -1, -1, -1, -1 }, { 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8,
                -1, -1, -1, -1 }, { 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1 }, { 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1,
                -1, -1, -1 }, { 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1,
                -1, -1 },
        { 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 }, { 10, 7, 6, 10,
                1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 }, { 10, 7, 6, 1, 7,
                10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 }, { 0, 3, 7, 0, 7, 10, 0,
                10, 9, 6, 10, 7, -1, -1, -1, -1 }, { 7, 6, 10, 7, 10, 8, 8, 10,
                9, -1, -1, -1, -1, -1, -1, -1 }, { 6, 8, 4, 11, 8, 6, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 6, 11, 3, 0, 6, 0, 4, 6,
                -1, -1, -1, -1, -1, -1, -1 }, { 8, 6, 11, 8, 4, 6, 9, 0, 1, -1,
                -1, -1, -1, -1, -1, -1 }, { 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3,
                6, -1, -1, -1, -1 }, { 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1,
                -1, -1, -1, -1 }, { 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1,
                -1, -1, -1 }, { 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1,
                -1, -1 },
        { 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 }, { 8, 2, 3, 8,
                4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 }, { 0, 4, 2, 4, 6,
                2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 9, 0, 2, 3,
                4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 }, { 1, 9, 4, 1, 4, 2, 2,
                4, 6, -1, -1, -1, -1, -1, -1, -1 }, { 8, 1, 3, 8, 6, 1, 8, 4,
                6, 6, 10, 1, -1, -1, -1, -1 }, { 10, 1, 0, 10, 0, 6, 6, 0, 4,
                -1, -1, -1, -1, -1, -1, -1 }, { 4, 6, 3, 4, 3, 8, 6, 10, 3, 0,
                3, 9, 10, 9, 3, -1 }, { 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1,
                -1, -1, -1, -1 }, { 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1,
                -1, -1, -1 }, { 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1,
                -1, -1 }, { 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1,
                -1, -1 }, { 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1,
                -1 },
        { 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 }, { 3, 4, 8,
                3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 }, { 7, 2, 3, 7, 6, 2,
                5, 4, 9, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 4, 0, 8, 6, 0,
                6, 2, 6, 8, 7, -1, -1, -1, -1 }, { 3, 6, 2, 3, 7, 6, 1, 5, 0,
                5, 4, 0, -1, -1, -1, -1 }, { 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8,
                5, 1, 5, 8, -1 }, { 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1,
                -1, -1, -1 }, { 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,
                -1 }, { 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
        { 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 }, { 6, 9, 5,
                6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 }, { 3, 6, 11,
                0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 }, { 0, 11, 8, 0, 5,
                11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 }, { 6, 11, 3, 6, 3, 5,
                5, 3, 1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 2, 10, 9, 5, 11, 9,
                11, 8, 11, 5, 6, -1, -1, -1, -1 }, { 0, 11, 3, 0, 6, 11, 0, 9,
                6, 5, 6, 9, 1, 2, 10, -1 }, { 11, 8, 5, 11, 5, 6, 8, 0, 5, 10,
                5, 2, 0, 2, 5, -1 }, { 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3,
                -1, -1, -1, -1 }, { 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1,
                -1, -1 }, { 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1,
                -1 }, { 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 }, { 1,
                5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
                3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 }, { 10, 1, 0,
                10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 }, { 0, 3, 8, 5, 6,
                10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 10, 5, 6, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 5, 10,
                7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 5,
                10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 }, { 5, 11,
                7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 }, { 10, 7,
                5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 }, { 11, 1, 2,
                11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 3, 1,
                2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 }, { 9, 7, 5, 9, 2, 7,
                9, 0, 2, 2, 11, 7, -1, -1, -1, -1 }, { 7, 5, 2, 7, 2, 11, 5, 9,
                2, 3, 2, 8, 9, 8, 2, -1 }, { 2, 5, 10, 2, 3, 5, 3, 7, 5, -1,
                -1, -1, -1, -1, -1, -1 }, { 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2,
                5, -1, -1, -1, -1 }, { 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2,
                -1, -1, -1, -1 }, { 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5,
                2, -1 }, { 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1 }, { 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1,
                -1 },
        { 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 }, { 9, 8, 7,
                5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 5, 8, 4,
                5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 }, { 5, 0, 4,
                5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 }, { 0, 1, 9, 8,
                4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 }, { 10, 11, 4, 10,
                4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 }, { 2, 5, 1, 2, 8, 5, 2,
                11, 8, 4, 5, 8, -1, -1, -1, -1 }, { 0, 4, 11, 0, 11, 3, 4, 5,
                11, 2, 11, 1, 5, 1, 11, -1 }, { 0, 2, 5, 0, 5, 9, 2, 11, 5, 4,
                5, 8, 11, 8, 5, -1 }, { 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1 }, { 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4,
                -1, -1, -1, -1 }, { 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1,
                -1, -1, -1 }, { 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9,
                -1 }, { 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
        { 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 4, 5,
                1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 8, 4, 5,
                8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 }, { 9, 4, 5, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4, 11, 7, 4, 9,
                11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 3, 4, 9,
                7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 }, { 1, 10, 11, 1, 11,
                4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 }, { 3, 1, 4, 3, 4, 8, 1,
                10, 4, 7, 4, 11, 10, 11, 4, -1 }, { 4, 11, 7, 9, 11, 4, 9, 2,
                11, 9, 1, 2, -1, -1, -1, -1 }, { 9, 7, 4, 9, 11, 7, 9, 1, 11,
                2, 11, 1, 0, 8, 3, -1 }, { 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1,
                -1, -1, -1, -1, -1 }, { 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4,
                -1, -1, -1, -1 }, { 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1,
                -1, -1, -1 }, { 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7,
                -1 }, { 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
        { 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4, 9,
                1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 }, { 4, 9, 1,
                4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 }, { 4, 0, 3, 7, 4,
                3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4, 8, 7, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 10, 8, 10,
                11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 0, 9, 3,
                9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 }, { 0, 1, 10, 0,
                10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 }, { 3, 1, 10, 11,
                3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 2, 11, 1,
                11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 }, { 3, 0, 9, 3, 9,
                11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 }, { 0, 2, 11, 8, 0, 11,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 2, 11, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 2, 3, 8, 2, 8,
                10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 }, { 9, 10, 2, 0, 9,
                2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 2, 3, 8, 2, 8,
                10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 }, { 1, 10, 2, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 3, 8, 9, 1,
                8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 9, 1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 3, 8, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } };


// bool LayeredMarchingCubes::computeIsoSurface(const float* tsdf, const float* weights, const unsigned char* red, const unsigned char* green, const unsigned char* blue, float isoValue)
// input: pointer/reference to
    // phmap::parallel_node_hash_map<Vec3i, SdfVoxelHr,
    //     phmap::priv::hash_default_hash<Vec3i>,
    //     phmap::priv::hash_default_eq<Vec3i>,
    //     Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxelHr>>>
bool LayeredMarchingCubes::computeIsoSurface(const LayeredMarchingCubes::voxel_phmap* sdf_map, float isoValue)
{
    // if (!tsdf || !weights || !red || !green || !blue)
        // return false;
    if (!sdf_map)
        return false;

    // TODO: compute dimensions (and, from that, size)
    int pos_inf = std::numeric_limits<int>::max();
    int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;
    for (auto v : *sdf_map) {
        if (v.first[0] < xmin) xmin = v.first[0];
        if (v.first[0] > xmax) xmax = v.first[0];
        if (v.first[1] < ymin) ymin = v.first[1];
        if (v.first[1] > ymax) ymax = v.first[1];
        if (v.first[2] < zmin) zmin = v.first[2];
        if (v.first[2] > zmax) zmax = v.first[2];
    }

    min_ = Vec3i(xmin, ymin, zmin);
    origin_ = -min_.cast<float>().cwiseProduct(voxelSize_);
    dim_ = 2 * Vec3i(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1); // each coarse voxel has 2 x 2 x 2 fine voxels
    size_ = 0.5 * Vec3f(voxelSize_.cwiseProduct(dim_.cast<float>()));

    size_t num_voxels = dim_[0] * dim_[1] * 4;

    tsdf_ = new float[num_voxels];
    std::fill_n(tsdf_, num_voxels, 0.);
    weights_ = new float[num_voxels];
    std::fill_n(weights_, num_voxels, 0.);
    red_ = new unsigned char[num_voxels];
    std::fill_n(red_, num_voxels, 0);
    green_ = new unsigned char[num_voxels];
    std::fill_n(green_, num_voxels, 0);
    blue_ = new unsigned char[num_voxels];
    std::fill_n(blue_, num_voxels, 0);

    vertices_.clear();
    colors_.clear();
    faces_.clear();

    Vec3f edgePoints[12];
    Vec3b edgeColors[12];
    int edgeIndices[12][6];

    copyLayer(0, sdf_map);
    for (int z = 0; z < dim_[2]-2; z++)
    {
        if (z%2) {
            copyLayer(z+1, sdf_map);
        }
        for (int y = 0; y < dim_[1]-2; y++)
        {
            for (int x = 0; x < dim_[0]-2; x++)
            {
                int cubeindex = computeLutIndex(x, y, z, isoValue);
                if (cubeindex != 0 && cubeindex != 255)
                {
                    if (edgeTable[cubeindex] & 1)
                    {
                        // interpolate between vertices 0 and 1
                        edgePoints[0] = getVertex(x + 1, y + 1, z, x + 1, y, z, isoValue);
                        edgeIndices[0][0] = x + 1;
                        edgeIndices[0][1] = y + 1;
                        edgeIndices[0][2] = z;
                        edgeIndices[0][3] = x + 1;
                        edgeIndices[0][4] = y;
                        edgeIndices[0][5] = z;
                        edgeColors[0] = getColor(x + 1, y + 1, z, x + 1, y, z, isoValue);
                    }

                    if (edgeTable[cubeindex] & 2)
                    {
                        // interpolate between vertices 1 and 2
                        edgePoints[1] = getVertex(x + 1, y, z, x, y, z, isoValue);
                        edgeIndices[1][0] = x + 1;
                        edgeIndices[1][1] = y;
                        edgeIndices[1][2] = z;
                        edgeIndices[1][3] = x;
                        edgeIndices[1][4] = y;
                        edgeIndices[1][5] = z;
                        edgeColors[1] = getColor(x + 1, y, z, x, y, z, isoValue);
                    }

                    if (edgeTable[cubeindex] & 4)
                    {
                        // interpolate between vertices 2 and 3
                        edgePoints[2] = getVertex(x, y, z, x, y + 1, z, isoValue);
                        edgeIndices[2][0] = x;
                        edgeIndices[2][1] = y;
                        edgeIndices[2][2] = z;
                        edgeIndices[2][3] = x;
                        edgeIndices[2][4] = y + 1;
                        edgeIndices[2][5] = z;
                        edgeColors[2] = getColor(x, y + 1, z, x, y, z, isoValue);
                    }

                    if (edgeTable[cubeindex] & 8)
                    {
                        // interpolate between vertices 3 and 0
                        edgePoints[3] = getVertex(x, y + 1, z, x + 1, y + 1, z, isoValue);
                        edgeIndices[3][0] = x;
                        edgeIndices[3][1] = y + 1;
                        edgeIndices[3][2] = z;
                        edgeIndices[3][3] = x + 1;
                        edgeIndices[3][4] = y + 1;
                        edgeIndices[3][5] = z;
                        edgeColors[3] = getColor(x + 1, y + 1, z, x, y + 1, z, isoValue);
                    }

                    if (edgeTable[cubeindex] & 16)
                    {
                        // interpolate between vertices 4 and 5
                        edgePoints[4] = getVertex(x + 1, y + 1, z + 1, x + 1, y, z + 1, isoValue);
                        edgeIndices[4][0] = x + 1;
                        edgeIndices[4][1] = y + 1;
                        edgeIndices[4][2] = z + 1;
                        edgeIndices[4][3] = x + 1;
                        edgeIndices[4][4] = y;
                        edgeIndices[4][5] = z + 1;
                        edgeColors[4] = getColor(x + 1, y + 1, z + 1, x + 1, y, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 32)
                    {
                        // interpolate between vertices 5 and 6
                        edgePoints[5] = getVertex(x + 1, y, z + 1, x, y, z + 1, isoValue);
                        edgeIndices[5][0] = x + 1;
                        edgeIndices[5][1] = y;
                        edgeIndices[5][2] = z + 1;
                        edgeIndices[5][3] = x;
                        edgeIndices[5][4] = y;
                        edgeIndices[5][5] = z + 1;
                        edgeColors[5] = getColor(x + 1, y, z + 1, x, y, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 64)
                    {
                        // interpolate between vertices 6 and 7
                        edgePoints[6] = getVertex(x, y, z + 1, x, y + 1, z + 1, isoValue);
                        edgeIndices[6][0] = x;
                        edgeIndices[6][1] = y;
                        edgeIndices[6][2] = z + 1;
                        edgeIndices[6][3] = x;
                        edgeIndices[6][4] = y + 1;
                        edgeIndices[6][5] = z + 1;
                        edgeColors[6] = getColor(x, y + 1, z + 1, x, y, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 128)
                    {
                        // interpolate between vertices 7 and 4
                        edgePoints[7] = getVertex(x, y + 1, z + 1, x + 1, y + 1, z + 1, isoValue);
                        edgeIndices[7][0] = x;
                        edgeIndices[7][1] = y + 1;
                        edgeIndices[7][2] = z + 1;
                        edgeIndices[7][3] = x + 1;
                        edgeIndices[7][4] = y + 1;
                        edgeIndices[7][5] = z + 1;
                        edgeColors[7] = getColor(x + 1, y + 1, z + 1, x, y + 1, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 256)
                    {
                        // interpolate between vertices 0 and 4
                        edgePoints[8] = getVertex(x + 1, y + 1, z, x + 1, y + 1, z + 1, isoValue);
                        edgeIndices[8][0] = x + 1;
                        edgeIndices[8][1] = y + 1;
                        edgeIndices[8][2] = z;
                        edgeIndices[8][3] = x + 1;
                        edgeIndices[8][4] = y + 1;
                        edgeIndices[8][5] = z + 1;
                        edgeColors[8] = getColor(x + 1, y + 1, z, x + 1, y + 1, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 512)
                    {
                        // interpolate between vertices 1 and 5
                        edgePoints[9] = getVertex(x + 1, y, z, x + 1, y, z + 1, isoValue);
                        edgeIndices[9][0] = x + 1;
                        edgeIndices[9][1] = y;
                        edgeIndices[9][2] = z;
                        edgeIndices[9][3] = x + 1;
                        edgeIndices[9][4] = y;
                        edgeIndices[9][5] = z + 1;
                        edgeColors[9] = getColor(x + 1, y, z, x + 1, y, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 1024)
                    {
                        // interpolate between vertices 2 and 6
                        edgePoints[10] = getVertex(x, y, z, x, y, z + 1, isoValue);
                        edgeIndices[10][0] = x;
                        edgeIndices[10][1] = y;
                        edgeIndices[10][2] = z;
                        edgeIndices[10][3] = x;
                        edgeIndices[10][4] = y;
                        edgeIndices[10][5] = z + 1;
                        edgeColors[10] = getColor(x, y, z, x, y, z + 1, isoValue);
                    }

                    if (edgeTable[cubeindex] & 2048)
                    {
                        // interpolate between vertices 3 and 7
                        edgePoints[11] = getVertex(x, y + 1, z, x, y + 1, z + 1, isoValue);
                        edgeIndices[11][0] = x;
                        edgeIndices[11][1] = y + 1;
                        edgeIndices[11][2] = z;
                        edgeIndices[11][3] = x;
                        edgeIndices[11][4] = y + 1;
                        edgeIndices[11][5] = z + 1;
                        edgeColors[11] = getColor(x, y + 1, z, x, y + 1, z + 1, isoValue);
                    }

                    computeTriangles(cubeindex, edgePoints, edgeColors);
                }
            }
        }
    }

    return true;
}

// Copy one z-layer from voxel map to 3D layer volume
// Input: z coordinate
void LayeredMarchingCubes::copyLayer(int z, const LayeredMarchingCubes::voxel_phmap* sdf_map)
{
    // TODO: check voxel indices (int coarse vs. fine!)
    const auto end = sdf_map->end();
    for (int y = 0; y < dim_[1]; y+=2)
    {
        for (int x = 0; x < dim_[0]; x+=2)
        {
            auto finder = sdf_map->find(Vec3i(x/2, y/2, z/2) + min_);
            if (finder != end) {
                const SdfVoxelHr& voxel = finder->second;
                copyCube(x, y, z, voxel.weight, voxel.d, voxel.r, voxel.g, voxel.b);
            }
            else {
                zeroWeights(x, y, z);
            }
        }
    }
}

// Set weights of one coarse 2 x 2 x 2 voxel to zero
// Input: voxel indices x, y, z
inline void LayeredMarchingCubes::zeroWeights(int x, int y, int z) {
    size_t off = ((z%4) * dim_[1] + y) * dim_[0] + x; // linear index of first corner voxel
    weights_[off] = 0; // corner 1
    ++off;
    weights_[off] = 0; // corner 2
    off += dim_[0];
    weights_[off] = 0; // corner 4
    --off;
    weights_[off] = 0; // corner 3
    off += dim_[1] * dim_[0];
    weights_[off] = 0; // corner 7
    off -= dim_[0];
    weights_[off] = 0; // corner 5
    ++off;
    weights_[off] = 0; // corner 6
    off += dim_[0];
    weights_[off] = 0; // corner 8
}

// Set one coarse 2 x 2 x 2 voxel in 3D layer volume
// Input: voxel indices x, y, z, voxel data w, d, r, g, b
inline void LayeredMarchingCubes::copyCube(int x, int y, int z, const float w,
                                            const Eigen::Matrix<float, 8, 1>& d,
                                            const Eigen::Matrix<float, 8, 1>& r,
                                            const Eigen::Matrix<float, 8, 1>& g,
                                            const Eigen::Matrix<float, 8, 1>& b
                                            ) {
    size_t off = ((z%4) * dim_[1] + y) * dim_[0] + x; // linear index of first corner voxel
    setVoxel(off, 0, w, d, r, g, b); // corner 1
    ++off;
    setVoxel(off, 1, w, d, r, g, b); // corner 2
    off += dim_[0];
    setVoxel(off, 3, w, d, r, g, b); // corner 4
    --off;
    setVoxel(off, 2, w, d, r, g, b); // corner 3
    off += dim_[1] * dim_[0];
    setVoxel(off, 6, w, d, r, g, b); // corner 7
    off -= dim_[0];
    setVoxel(off, 4, w, d, r, g, b); // corner 5
    ++off;
    setVoxel(off, 5, w, d, r, g, b); // corner 6
    off += dim_[0];
    setVoxel(off, 7, w, d, r, g, b); // corner 8
}

// Set one (fine) voxel in 3D layer volume
// Input: linear index off of voxel, index of corner, w, d, r, g, b
inline void LayeredMarchingCubes::setVoxel(const size_t off, const size_t corner, const float w,
                                            const Eigen::Matrix<float, 8, 1>& d,
                                            const Eigen::Matrix<float, 8, 1>& r,
                                            const Eigen::Matrix<float, 8, 1>& g,
                                            const Eigen::Matrix<float, 8, 1>& b
                                            ) {
    weights_[off] = w;
    tsdf_[off] = d[corner];
    red_[off] = static_cast<unsigned char>(r[corner] * 255);
    green_[off] = static_cast<unsigned char>(g[corner] * 255);
    blue_[off] = static_cast<unsigned char>(b[corner] * 255);
}


// Finding which vertices are below zero.
// Input: Integers i,j and k
// Output: A vector with integer with value between 0 and 256.
inline int LayeredMarchingCubes::computeLutIndex(int i, int j, int k, float isoValue)
{
    size_t offZ = dim_[0] * dim_[1];
    size_t offY = dim_[0];

    size_t off1   = (k%4) * offZ + (j + 1) * offY + (i + 1);
    size_t off2   = (k%4) * offZ + j * offY + (i + 1);
    size_t off4   = (k%4) * offZ + j * offY + i;
    size_t off8   = (k%4) * offZ + (j + 1) * offY + i;
    size_t off16  = ((k + 1)%4) * offZ + (j + 1) * offY + (i + 1);
    size_t off32  = ((k + 1)%4) * offZ + j * offY + (i + 1);
    size_t off64  = ((k + 1)%4) * offZ + j * offY + i;
    size_t off128 = ((k + 1)%4) * offZ + (j + 1) * offY + i;

    // determine cube index for lookup table
    int cubeIdx = 0;
    // check if behind the surface
    if (!(weights_[off1] == 0.0f ||
        weights_[off2] == 0.0f ||
        weights_[off4] == 0.0f ||
        weights_[off8] == 0.0f ||
        weights_[off16] == 0.0f ||
        weights_[off32] == 0.0f ||
        weights_[off64] == 0.0f ||
        weights_[off128] == 0.0f))
    {
        if (tsdf_[off1] > isoValue)
            cubeIdx |= 1;
        if (tsdf_[off2] > isoValue)
            cubeIdx |= 2;
        if (tsdf_[off4] > isoValue)
            cubeIdx |= 4;
        if (tsdf_[off8] > isoValue)
            cubeIdx |= 8;
        if (tsdf_[off16] > isoValue)
            cubeIdx |= 16;
        if (tsdf_[off32] > isoValue)
            cubeIdx |= 32;
        if (tsdf_[off64] > isoValue)
            cubeIdx |= 64;
        if (tsdf_[off128] > isoValue)
            cubeIdx |= 128;
    }

    return cubeIdx;
}


Vec3f LayeredMarchingCubes::interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue)
{
    if (std::fabs(isoValue - tsdf0) < 1e-7)
        return val0;
    if (std::fabs(isoValue - tsdf1) < 1e-7)
        return val1;
    if (std::fabs(tsdf0 - tsdf1) < 1e-7)
        return val0;

    double mu = (isoValue - tsdf0) / (tsdf1 - tsdf0);
    if(mu > 1.0)
        mu = 1.0;
    else if (mu < 0)
        mu = 0.0;

    Vec3f val;
    val[0] = val0[0] + mu * (val1[0] - val0[0]);
    val[1] = val0[1] + mu * (val1[1] - val0[1]);
    val[2] = val0[2] + mu * (val1[2] - val0[2]);
    return val;
}


Vec3f LayeredMarchingCubes::getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue)
{
    float v1 = tsdf_[(k1%4) * dim_[0] * dim_[1] + j1 * dim_[0] + i1];
    Vec3f p1 = voxelToWorld(i1, j1, k1);
    float v2 = tsdf_[(k2%4) * dim_[0] * dim_[1] + j2 * dim_[0] + i2];
    Vec3f p2 = voxelToWorld(i2, j2, k2);
    return interpolate(v1, v2, p1, p2, isoValue);
}


Vec3b LayeredMarchingCubes::getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue)
{
    Vec3b c(0, 0, 0);
    // linear interpolation
    size_t idx1 = (z1%4) * dim_[0] * dim_[1] + y1 * dim_[0] + x1;
    size_t idx2 = (z2%4) * dim_[0] * dim_[1] + y2 * dim_[0] + x2;
    float v1 = tsdf_[idx1];
    float v2 = tsdf_[idx2];
    // TODO: double-check, seems wrong with the green and blue indices
    Vec3b c1b(red_[idx1], green_[idx1+1], blue_[idx1+2]);
    Vec3b c2b(red_[idx2], green_[idx2+1], blue_[idx2+2]);
    Vec3f c1 = c1b.cast<float>() / 255.0;
    Vec3f c2 = c2b.cast<float>() / 255.0;
    Vec3f cVal = interpolate(v1, v2, c1, c2, isoValue);
    // convert double to unsigned char
    c = (cVal * 255.0).cast<unsigned char>();
    return c;
}


void LayeredMarchingCubes::computeTriangles(int cubeIndex, const Vec3f edgePoints[12], const Vec3b edgeColors[12])
{
    std::vector<Vec3f> pts;
    pts.resize(3);
    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
    {
        Vec3f p1 = edgePoints[triTable[cubeIndex][i]];
        pts[0] = p1;
        Vec3f p2 = edgePoints[triTable[cubeIndex][i + 1]];
        pts[1] = p2;
        Vec3f p3 = edgePoints[triTable[cubeIndex][i + 2]];
        pts[2] = p3;

        if (p1 != p2 && p1 != p3 && p2 != p3)
        {
            // add vertices
            Vec3i vIdx;
            for (int t = 0; t < 3; ++t)
            {
                Vec3b c = edgeColors[triTable[cubeIndex][i + t]];
                vIdx[t] = addVertex(pts[t], c);
            }

            // add face
            Vec3i faceVerts(vIdx[0], vIdx[1], vIdx[2]);
            faces_.push_back(faceVerts);
        }
    }
}


inline unsigned int LayeredMarchingCubes::addVertex(const Vec3f &v, const Vec3b &c)
{
    // add vertex
    unsigned int vIdx = vertices_.size();
    vertices_.push_back(v);
    colors_.push_back(c);
    return vIdx;
}


Vec3f LayeredMarchingCubes::voxelToWorld(int i, int j, int k) const
{
    Vec3f pt = 0.5 * (Vec3i(i, j, k).cast<float>().cwiseProduct(voxelSize_)) - origin_;
    return pt;
}


bool LayeredMarchingCubes::savePly(const std::string &filename) const
{
    if (vertices_.empty())
        return false;

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;

    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << vertices_.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "element face " << (int)faces_.size() << std::endl;
    plyFile << "property list uchar int vertex_indices" << std::endl;
    plyFile << "end_header" << std::endl;

    // write vertices
    for (size_t i = 0; i < vertices_.size(); i++)
    {
        plyFile << vertices_[i][0] << " " << vertices_[i][1] << " " << vertices_[i][2];
        plyFile << " " << (int)colors_[i][0] << " " << (int)colors_[i][1] << " " << (int)colors_[i][2];
        plyFile << std::endl;
    }

    // write faces
    for (size_t i = 0; i < faces_.size(); i++)
    {
        plyFile << "3 " << (int)faces_[i][0] << " " << (int)faces_[i][1] << " " << (int)faces_[i][2] << std::endl;
    }

    plyFile.close();

    return true;
}
