CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

SET(EXECUTABLE_OUTPUT_PATH ../../voxel_ps/bin)
SET(SUBPROJECT_NAME voxelPS)
# volumetric tracking - different options

ADD_EXECUTABLE( ${SUBPROJECT_NAME}
    src/main_ps.cpp
)

TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME}
    sdf_tracker_lib
    ps_lib
    ${OpenCV_LIBS}
)

TARGET_COMPILE_OPTIONS( ${SUBPROJECT_NAME} PRIVATE -std=c++17 )

