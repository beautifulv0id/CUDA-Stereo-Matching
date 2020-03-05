cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

project(GPU_Elas)

set(PRJ_DIR ${CMAKE_CURRENT_SOURCE_DIR})

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PRJ_DIR}/bin)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PRJ_DIR}/bin)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${PRJ_DIR}/bin)

find_package( CUDA QUIET REQUIRED )
find_package( OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

include_directories(/usr/includes/cub)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_30,code=sm_30
    )

add_definitions(-DREAL_TYPE_FP32)

#FILE(GLOB_RECURSE INCLUDES "src/*.h")

CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
    src/Main.cu
    src/Elas.cu
    src/triangle.cpp

    src/Kernels/KerComputeDisparity.cu
    src/Kernels/KerDescriptor.cu
    src/Kernels/KerGrid.cu
    src/Kernels/KerSupportMatches.cu
    src/Kernels/KerLeftRightConsistencyCheck.cu

    src/delaunay/HashTable.cpp
    src/delaunay/RandGen.cpp
    src/delaunay/Visualizer.cpp
    src/delaunay/gDel2D/CPU/PredWrapper.cpp
    src/delaunay/gDel2D/CPU/predicates.cpp
    src/delaunay/gDel2D/GPU/GpuDelaunay.cu
    src/delaunay/gDel2D/GPU/ThrustWrapper.cu
    src/delaunay/gDel2D/GPU/KerPredicates.cu
    src/delaunay/gDel2D/GPU/KerDivision.cu
    src/delaunay/gDel2D/GPU/SmallCounters.cu
    ${INCLUDES}
    )


set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})



