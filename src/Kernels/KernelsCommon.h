#ifndef __KERNELSCOMMON_H__
#define __KERNELSCOMMON_H__
#include <stdint.h>
#include "cuda.h"
#include "vector_types.h"

// ELAS parameters
#define CANDIDATE_STEPSIZE 5
#define DISP_MIN 0
#define DISP_MAX 255
#define SUPPORT_TEXTURE 10
#define SUPPORT_THRESHOLD 0.85
#define ICON_WINDOW_SIZE 5
#define INCON_THRESHOLD 5
#define INCON_MIN_SUPPORT 5
#define REDUN_MAX_DIST 5
#define REDUN_THRESHOLD 1
#define LR_THRESHOLD 2
#define SPECKLE_SIM_THRESHOLD 1
#define SPECKLE_SIZE 200
#define INTERPOL_GAP_WIDTH 3
#define GRID_SIZE 20
#define SIGMA 1
#define GAMMA 3
#define BETA 0.02
#define SRADIUS 2
#define MATCH_TEXTURE 0

typedef double FLOAT;      // double precision
#define SWAP(a,b) {temp=a;a=b;b=temp;}


// functions that are shared among the kernels
__forceinline__ __device__ uint32_t getLocalThreadIdx()
{
    return threadIdx.x;
}

__forceinline__ __device__ uint32_t getLocalBlockIdx()
{
    return blockIdx.x;
}

__forceinline__ __device__ uint32_t getGlobThreadIdx()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__forceinline__ __device__ uint32_t getGlobThreadIdx_U()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__forceinline__ __device__ uint32_t getGlobThreadIdx_V()
{
    return threadIdx.y + blockIdx.y * blockDim.y;
}

__forceinline__ __device__ uint32_t numThreads_U()
{
    return blockDim.x * gridDim.x;
}

__forceinline__ __device__ uint32_t numThreads_V()
{
    return blockDim.y * gridDim.y;
}

__forceinline__ __device__ uint32_t getImageOffset(const int32_t& u,const int32_t& v,
                                                   const int32_t& width)
{
    return v*width+u;
}

__forceinline__ __device__ uint32_t getAddressOffestGrid
(
        const int32_t& u,
        const int32_t& v,
        const int32_t& d,
        const int32_t& width,
        const int32_t& disp_num
)
{
    return (v*width+u)*disp_num+d;
}

__forceinline__ __device__ uint32_t getAddressOffestGrid
(
        const int32_t& pos,
        const int32_t& d,
        const int32_t& disp_num
)
{
    return pos*disp_num+d;
}

/* reads 16 bytes from global mem pointer global_desc_8 to shared mem pointer local_desc_8 */
__forceinline__ __device__ void
loadDescriptorBlock
(
uint8_t *global_desc_8,
uint8_t *local_desc_8
)
{
    int4* global_desc_int4 = (int4*) global_desc_8;
    int4* local_desc_int4 = (int4*) local_desc_8;

    *local_desc_int4 = *global_desc_int4;
}




#endif
