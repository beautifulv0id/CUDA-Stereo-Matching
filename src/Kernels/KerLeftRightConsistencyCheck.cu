#include "KerSupportMatches.h"
#include <stdint.h>
#include "KernelsCommon.h"

// apply left/right consistency check to input images disp0_ and disp1_
// results are written to disp0_cpy_ and disp1_cpy_
__global__ void
kerLeftRightConsistencyCheck(Ker2DArray<float> disp0_, Ker2DArray<float> disp1_,
                             Ker2DArray<float> disp0_cpy_, Ker2DArray<float> disp1_cpy_)
{
    // declare variables
    int32_t u = getGlobThreadIdx_U();
    int32_t v = getGlobThreadIdx_V();

    float* disp0 = disp0_._arr;
    float* disp1 = disp1_._arr;
    float* disp0_cpy = disp0_cpy_._arr;
    float* disp1_cpy = disp1_cpy_._arr;

    int32_t width = disp0_._width;
    int32_t height = disp0_._height;

    if (u < width && v < height) {
        // loop variables
        uint32_t addr,addr_warp;
        float    u_warp_1,u_warp_2,d1,d2;

        // compute address (u,v) and disparity value
        addr     = getImageOffset(u,v,width);
        d1       = *(disp0_cpy+addr);
        d2       = *(disp1_cpy+addr);

        u_warp_1 = (float)u-d1;
        u_warp_2 = (float)u+d2;

        // check if left disparity is valid
        if (d1>=0 && u_warp_1>=0 && u_warp_1<width) {

            // compute warped _image address
            addr_warp = getImageOffset((int32_t)u_warp_1,v,width);

            // if check failed
            if (fabs(*(disp1_cpy+addr_warp)-d1)>LR_THRESHOLD)
                *(disp0+addr) = -1;

            // set invalid
        } else {
            *(disp0+addr) = -1;
        }

        // check if right disparity is valid
        if (d2>=0 && u_warp_2>=0 && u_warp_2<width) {

            // compute warped _image address
            addr_warp = getImageOffset((int32_t)u_warp_2,v,width);

            // if check failed
            if (fabs(*(disp0_cpy+addr_warp)-d2)>LR_THRESHOLD)
                *(disp1+addr) = -1;

            // set invalid
        } else
            *(disp1+addr) = -1;

    }
}
