#ifndef __KERDESCRIPTOR_H__
#define __KERDESCRIPTOR_H__
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"

// compute horizontal and vertical sobel filtered images
__global__ void
kerComputeSobel(
Ker2DArray<uint8_t> I_,
uint8_t* I_du,
uint8_t* I_dv
);

// compute descriptor from horizontal and vertical sobel filtered images
__global__ void
kerComputeDescriptor(
uint8_t *I_du,
uint8_t* I_dv,
uint8_t *I_desc,
int32_t width,
int32_t height
);


#endif
