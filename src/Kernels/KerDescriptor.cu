#include "KerDescriptor.h"
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "KernelsCommon.h"

// compute horizontal and vertical sobel filtered images
__global__ void
kerComputeSobel( Ker2DArray<uint8_t> img_, uint8_t* img_du, uint8_t* img_dv )
{
    // declare variables
    uint32_t width = img_._width;
    uint32_t height = img_._height;

    uint8_t* img = img_._arr;

    uint32_t u = getGlobThreadIdx_U();
    uint32_t v = getGlobThreadIdx_V();

    float sum_du, sum_dv;
    float du, dv;

    // apply 3x3 sobel mask
    if ( u > 0 && v > 0 && u < width-1 && v < height-1) {
        sum_du = (img[getImageOffset(u-1, v-1, width)])
                  + ( 2*img[getImageOffset(u-1, v, width)])
                  + (   img[getImageOffset(u-1, v+1, width)])
                  + (-1*img[getImageOffset(u+1, v-1, width)])
                  + (-2*img[getImageOffset(u+1, v, width)])
                  + (-1*img[getImageOffset(u+1, v+1, width)]);

        sum_dv = (img[getImageOffset(u-1, v-1, width)])
                  + ( 2*img[getImageOffset(  u, v-1, width)])
                  + (   img[getImageOffset(u+1, v-1, width)])
                  + (-1*img[getImageOffset(u-1, v+1, width)])
                  + (-2*img[getImageOffset(  u, v+1, width)])
                  + (-1*img[getImageOffset(u+1, v+1, width)]);

        // scale and clamp values
        du = (uint8_t) (min(max(-128.f, sum_du/4), 127.f) + 128);
        dv = (uint8_t) (min(max(-128.f, sum_dv/4), 127.f) + 128);

        // save the results to the arrays of the filtered images
        img_du[getImageOffset(u, v, width)] = du;
        img_dv[getImageOffset(u, v, width)] = dv;

    }

}

// compute descriptor from horizontal and vertical sobel filtered images
__global__ void
kerComputeDescriptor(
uint8_t *img_du,
uint8_t* img_dv,
uint8_t *desc,
int width,
int height
)
{
    // declare variables
    uint u = getGlobThreadIdx_U();
    uint v = getGlobThreadIdx_V();

    int4 *desc_int4 = (int4*) (desc+(v*width+u)*16);

    uint32_t addr_v0,addr_v1,addr_v2,addr_v3,addr_v4;

    addr_v2 = v*width;
    addr_v0 = addr_v2-2*width;
    addr_v1 = addr_v2-1*width;
    addr_v3 = addr_v2+1*width;
    addr_v4 = addr_v2+2*width;

    int4 tmp;
    uint8_t* tmp_ptr = (uint8_t*) &tmp;

    // aggregate descriptor to tmp and write to the descriptor array
    if(u > 2 && u < width - 3 && v > 2 && v < height - 3){
        *(tmp_ptr++) = *(img_du+addr_v0+u+0);
        *(tmp_ptr++) = *(img_du+addr_v1+u-2);
        *(tmp_ptr++) = *(img_du+addr_v1+u+0);
        *(tmp_ptr++) = *(img_du+addr_v1+u+2);
        *(tmp_ptr++) = *(img_du+addr_v2+u-1);
        *(tmp_ptr++) = *(img_du+addr_v2+u+0);
        *(tmp_ptr++) = *(img_du+addr_v2+u+0);
        *(tmp_ptr++) = *(img_du+addr_v2+u+1);
        *(tmp_ptr++) = *(img_du+addr_v3+u-2);
        *(tmp_ptr++) = *(img_du+addr_v3+u+0);
        *(tmp_ptr++) = *(img_du+addr_v3+u+2);
        *(tmp_ptr++) = *(img_du+addr_v4+u+0);
        *(tmp_ptr++) = *(img_dv+addr_v1+u+0);
        *(tmp_ptr++) = *(img_dv+addr_v2+u-1);
        *(tmp_ptr++) = *(img_dv+addr_v2+u+1);
        *(tmp_ptr++) = *(img_dv+addr_v3+u+0);

        *desc_int4 = tmp;
    }

}
