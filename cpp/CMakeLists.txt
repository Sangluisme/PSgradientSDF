CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

PROJECT( voxelPS )
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# ROOTs
SET(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third/build-ceres-solver/")

# path to CMakeModules
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

########## INCLUDED LIBRARIES ##########

# Eigen
#SET(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third/eigen/include/eigen3")
SET(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third/eigen/")
# FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES( ${EIGEN_INCLUDE_DIR})


# OpenCV
FIND_PACKAGE( OpenCV REQUIRED )
MESSAGE(STATUS "Found OpenCV headers in: ${OpenCV_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# Sophus
SET(SOPHUS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third/Sophus/")

# INCLUDE_DIRECTORIES( third/Sophus/ )
MESSAGE(STATUS "sophus dir in: ${SOPHUS_INCLUDE_DIR}")
# FIND_PACKAGE(Sophus REQUIRED)
INCLUDE_DIRECTORIES( ${SOPHUS_INCLUDE_DIR})


# CLI, command line parser
INCLUDE_DIRECTORIES( third/CLI11/include/ )

# nlohmann json parser
INCLUDE_DIRECTORIES(third/json/include/)

########## CUSTOM LIBRARIES ##########

INCLUDE_DIRECTORIES( include/
                     third/
                    )

# marching cubes
SET( LIB_NAME mc_lib )
ADD_LIBRARY( ${LIB_NAME}
    third/mesh/GradMarchingCubes.cpp
    third/mesh/MarchingCubes.cpp
    third/mesh/MarchingCubesNoColor.cpp
    )
TARGET_COMPILE_OPTIONS( ${LIB_NAME} PRIVATE -std=c++17 )



# SDF-Tracking stuff
SET( LIB_NAME sdf_tracker_lib )
ADD_LIBRARY( ${LIB_NAME}
    include/sdf_tracker/VoxelGrid.cpp
    include/sdf_tracker/VolumetricGradSdf.cpp
    # include/sdf_tracker/VolumetricGradPixelSdf.cpp
   include/sdf_tracker/RigidPointOptimizer.cpp
    )
TARGET_LINK_LIBRARIES( ${LIB_NAME}
    ${OpenCV_LIBS}
    mc_lib
#    ${CERES_LIBRARIES} # only needed if RigidPointOptimizerCeres.cpp included
    )
TARGET_COMPILE_OPTIONS( ${LIB_NAME} PRIVATE -std=c++17 )

# Color Optimization
SET( LIB_NAME ps_lib )
ADD_LIBRARY( ${LIB_NAME}
    include/ps_optimizer/Optimizer.cpp
    include/ps_optimizer/OptimizerAux.cpp
    include/ps_optimizer/PsOptimizer.cpp
    include/ps_optimizer/PsOptimizerJa.cpp
    include/ps_optimizer/LedOptimizer.cpp
    include/ps_optimizer/LedOptimizerJa.cpp
    )
TARGET_LINK_LIBRARIES( ${LIB_NAME}
    ${OpenCV_LIBS}
    mc_lib
    sdf_tracker_lib
    )


SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" )

                 
########## DIRECTORIES ##########
ADD_SUBDIRECTORY(voxel_ps)

