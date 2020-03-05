#ifndef __KERGRID_H__
#define __KERGRID_H__
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"

// load the disparities values from the support points in d_can to the disparity grid
__global__ void
kerLoadGridValues
(
Ker2DArray<int32_t> d_can_,
Ker2DArray<uint8_t> temp0_,
Ker2DArray<uint8_t> temp1_,
int32_t grid_width,
int32_t grid_height
);

// diffuse the grid and aggregate disparity values
__global__ void
kerCreateGird
(
Ker2DArray<int32_t> d_grid_,
Ker2DArray<uint8_t> temp_
);

#endif
