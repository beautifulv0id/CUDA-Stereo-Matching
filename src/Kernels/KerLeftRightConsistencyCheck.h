#ifndef __KERLEFTRIGHTCONSISTENCYCHECK_H__
#define __KERLEFTRIGHTCONSISTENCYCHECK_H__
#include <stdint.h>
#include "../delaunay/gDel2D/GpuDelaunay.h"

// apply left/right consistency check to input images disp0_ and disp1_
// results are written to disp0_cpy_ and disp1_cpy_
__global__ void
kerLeftRightConsistencyCheck
(
    Ker2DArray<float> disp0_,
    Ker2DArray<float> disp1_,
    Ker2DArray<float> disp0_cpy_,
    Ker2DArray<float> disp1_cpy_
);

#endif


