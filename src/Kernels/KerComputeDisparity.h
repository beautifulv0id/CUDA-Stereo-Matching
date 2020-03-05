#ifndef __KERCOMPUTEDISPARITY_H__
#define __KERCOMPUTEDISPARITY_H__
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "KernelsCommon.h"

// computes the mean disparities
__global__ void
kerComputeDepthApproximation
(Ker2DArray<int32_t> d_can_,
KerArray<Point2> can_vec_,
KerArray<Tri> Tri_,
bool right_image,
Ker2DArray<float> D_approx_
);

// computes the remaining disparities minimizing the energy function
__global__ void
kerComputeDisparity
(
Ker2DArray<int32_t> D_grid_,
uint8_t* desc0_,
uint8_t* desc1_,
float* d_approx_,
bool right_image,
Ker2DArray<float> D_
);

// solve linear system M*x=B, results are written to B
__device__ bool
kerSolve
(
FLOAT A[3][3],
FLOAT B[3][1],
FLOAT eps=1e-20
);

// updates min_val and  min_d if the matching costs of the pixels
// associated with the descriptors are lower than the current min_val
__device__ void
kerUpdatePosteriorMinimum
(
uint8_t* desc0_block_addr,
uint8_t* desc1_block_addr,
const int32_t &d,
int32_t& val,
int32_t& min_val,
int32_t& min_d
);

// updates min_val and  min_d if the matching costs of the pixels
// associated with the descriptors are lower than the current min_val
// a start value for the costs has to be provided in val
__device__ void
kerUpdatePosteriorMinimum
(
uint8_t* desc0_block_addr,
uint8_t* desc1_block_addr,
const int32_t &d,
const int32_t &w,
int32_t& val,
int32_t& min_val,
int32_t& min_d
);

#endif
