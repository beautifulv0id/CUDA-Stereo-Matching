#ifndef __KERSUPPORTMATCHES_H__
#define __KERSUPPORTMATCHES_H__
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GPU/HostToKernel.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "../Utils.h"


// computes disparities for candidates on a 5x5 grid in can_ the disparities of
// candidates that do not have enough texture or can not be mapped from
// left-to-right and right-to-left are set to -1
__global__ void
kerComputeSupportMatches(
uint8_t *desc0_,
uint8_t* desc1_,
Ker2DArray<int32_t> can_,
int32_t width,
int32_t height,
bool right_image
);

// removes inconsistent support points
__global__ void
removeInconsistentSupportPoints
(
Ker2DArray<int32_t> d_can_arr
);

// removes redundant support points horizontally
__global__ void
removeRedundantSupportPointsHorizontal
(
Ker2DArray<int32_t> d_can_arr
);

// removes redundant support points vertically
__global__ void
removeRedundantSupportPointsVertical
(
Ker2DArray<int32_t> d_can_arr
);

// sets positions in flag to 1 if the corresponding support point in can_ is valid
__global__ void
kerFlag
(
Ker2DArray<int32_t> can_,
int32_t* flag
);

// creates the arrays sp0_ and sp1_ holding the support point coordinates
__global__ void
kerCompact
(
int32_t* d_prefix_sum,
Ker2DArray<int32_t> d_can_,
Point2* sp0_,
Point2* sp1_
);


#endif
