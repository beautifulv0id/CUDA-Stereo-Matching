#include "KerSupportMatches.h"
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GPU/HostToKernel.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "../Utils.h"
#include "KernelsCommon.h"

// computes disparities for candidates on a 5x5 grid and stores them in can_.
// disparities of candidates that do not have enough texture or can not be
// mapped from left-to-right and right-to-left are set to -1
__global__ void
kerComputeSupportMatches(
uint8_t *desc0,
uint8_t *desc1,
Ker2DArray<int32_t> can_,
int32_t width,
int32_t height,
bool right_image
)
{
    // declare variables
    int32_t tpc = blockDim.x;
    int32_t tidx = threadIdx.x;
    int32_t u_can = getGlobThreadIdx_U() / tpc;
    int32_t v_can = getGlobThreadIdx_V();

    int32_t u = u_can * CANDIDATE_STEPSIZE;
    int32_t v = v_can * CANDIDATE_STEPSIZE;

    int32_t* can = can_._arr;
    int32_t can_width = can_._width;

    const int32_t x_step      = 2;
    const int32_t y_step      = 2;
    const int32_t desc_window = 3;

    int32_t desc_offset[4];
    desc_offset[0] = -16*x_step-(16*width)*y_step;
    desc_offset[1] = +16*x_step-(16*width)*y_step;
    desc_offset[2] = -16*x_step+(16*width)*y_step;
    desc_offset[3] = +16*x_step+(16*width)*y_step;

    // declare and initialiize pointers to
    // associated position in shared memory
    extern __shared__ int32_t shared[];
    int32_t* min_1_E_p = &shared[4*tidx+0];
    int32_t* min_1_d_p = &shared[4*tidx+1];
    int32_t* min_2_E_p = &shared[4*tidx+2];
    int32_t* min_2_d_p = &shared[4*tidx+3];

    // set defaults
    *min_1_E_p = 32767;
    *min_1_d_p = -1;
    *min_2_E_p = 32767;
    *min_2_d_p = -1;

    if (u >= x_step+desc_window && u <= width-x_step-desc_window-1
        && v >= y_step+desc_window && v <= height-y_step-desc_window-1) {
        int32_t d_left;
        // if right image wrap u-coordinate
        if(right_image){
            d_left = can[v_can*can_width+u_can];
            if(d_left < 0) return;
            u = u-d_left;
        }

        // compute desc and start addresses
        int32_t  line_offset = 16*width*v;
        uint8_t *desc0_line_addr,*desc1_line_addr;
        if (!right_image) {
            desc0_line_addr = desc0+line_offset;
            desc1_line_addr = desc1+line_offset;
        } else {
            desc0_line_addr = desc1+line_offset;
            desc1_line_addr = desc0+line_offset;
        }

        // compute I1 block start addresses
        uint8_t* desc0_block_addr = desc0_line_addr+16*u;
        uint8_t* desc1_block_addr;
        uint8_t desc_block[16];


        // require at least some texture
        int32_t sum = 0;
        loadDescriptorBlock(desc0_block_addr, desc_block);
        for (int32_t i=0; i<16; i++)
            sum += abs((int32_t)(*(desc_block+i))-128);
        if (sum<SUPPORT_TEXTURE){
                can[v_can*can_width+u_can] = -1;
                return;
        }

        // load current descriptor block
        uint8_t desc0_block_arr[4][16];
        for(int32_t i=0; i<4; i++)
            loadDescriptorBlock(desc0_block_addr+desc_offset[i], desc0_block_arr[i]);


        // best match
        int16_t min_1_E = 32767;
        int16_t min_1_d = -1;
        int16_t min_2_E = 32767;
        int16_t min_2_d = -1;

        int32_t disp_min_valid = max(DISP_MIN, 0);
        int32_t disp_max_valid = DISP_MAX;

        // compute maximal valid disparity
        if (!right_image) {
            disp_max_valid = min(DISP_MAX, u - desc_window - x_step);
        } else {
            disp_max_valid = min(DISP_MAX, width - u - desc_window - x_step);
        }

        if ( disp_max_valid - disp_min_valid < 10 ) {
            can[v_can*can_width+u_can] = -1;
            return;
        }

        // declare match energy for each disparity
        int32_t u_warp;
        int32_t d = disp_min_valid+tidx;

        for(;d<=disp_max_valid; d+=tpc){
            // warp u coordinate
            if (!right_image) u_warp = u-d;
            else              u_warp = u+d;

            // compute I2 block start addresses
            desc1_block_addr = desc1_line_addr+16*u_warp;

            sum = 0;
            for (int32_t i=0; i<4; i++) {
                loadDescriptorBlock(desc1_block_addr+desc_offset[i], desc_block);
                for(int32_t j=0; j<16; j++)
                    sum += abs( ((int32_t) desc0_block_arr[i][j])
                               -((int32_t) *(desc_block+j)) );
            }

            // best + second best match
            if (sum<min_1_E) {
                min_2_E = min_1_E;
                min_2_d = min_1_d;
                min_1_E = sum;
                min_1_d = d;
            } else if (sum<min_2_E) {
                min_2_E = sum;
                min_2_d = d;
            }
        }

        *min_1_E_p = min_1_E;
        *min_1_d_p = min_1_d;
        *min_2_E_p = min_2_E;
        *min_2_d_p = min_2_d;

        __syncthreads();

        if(tidx == 0){
            int32_t tmp_E, tmp_d;
            for(int32_t i = 0; i < 4*tpc; i+=2){
                tmp_E = shared[i];
                tmp_d = shared[i+1];
                if (tmp_E<min_1_E) {
                    min_2_E = min_1_E;
                    min_2_d = min_1_d;
                    min_1_E = tmp_E;
                    min_1_d = tmp_d;
                } else if (tmp_E<min_2_E) {
                    min_2_E = tmp_E;
                    min_2_d = tmp_d;
                }
            }

            // check if best and second best match are available
            // and if matching ratio is sufficient
            if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<SUPPORT_THRESHOLD*(float)min_2_E) {
                if(!right_image) {
                    can[v_can*can_width+u_can] = min_1_d;
                } else if(abs(d_left - min_1_d) > LR_THRESHOLD) {
                    can[v_can*can_width+u_can] = -1;
                }
            }
            else {
                can[v_can*can_width+u_can] = -1;
            }
        }

    } // end if current candidate is valid
}

// removes inconsistent support points
__global__ void
removeInconsistentSupportPoints(Ker2DArray<int32_t> can_)
{
    // declare variables
    int32_t x_can = getGlobThreadIdx_U();
    int32_t y_can = getGlobThreadIdx_V();

    int32_t* d_can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    int32_t d_can_offset = (y_can*can_width)+x_can;

    if (x_can < can_width && y_can < can_width) {
        int32_t d = d_can[d_can_offset];

        if (d >= 0) {
            int32_t support = 0;
            int32_t window_offset;

            // horizontal window iterarion
            for (int32_t x_can_2=x_can-ICON_WINDOW_SIZE;
                 x_can_2 <= x_can+ICON_WINDOW_SIZE; x_can_2++) {

                //vertical window iteration
                for (int32_t y_can_2=y_can-ICON_WINDOW_SIZE;
                     y_can_2 <= y_can+ICON_WINDOW_SIZE; y_can_2++) {

                    if (x_can_2>=0 && x_can_2<can_width
                        && y_can_2>=0 && y_can_2<can_height) {
                        window_offset = y_can_2*can_width+x_can_2;
                        int32_t d2 = d_can[window_offset];

                        if (d2>=0 && abs(d-d2)<=INCON_THRESHOLD)
                            support++;
                    }
                }
            }
            // invalidate support point if number of supporting points is too low
            if (support<INCON_MIN_SUPPORT)
                d_can[d_can_offset] = -1;
        }
    }
}

// removes redundant support points horizontally
__global__ void
removeRedundantSupportPointsHorizontal(Ker2DArray<int32_t> can_)
{
    // declare variables
    int32_t v_can = getGlobThreadIdx_V();

    int32_t* can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    // parameters
    int32_t redun_dir_u[2] = {-1,1};

    if (v_can < can_height) {
        int32_t d_can_offset;
        int32_t d;
        for (int32_t u_can = 0; u_can < can_width; u_can++) {
            d_can_offset = v_can*can_width+u_can;
            d = can[d_can_offset];

            if (d >= 0) {
                // check all directions for redundancy
                bool redundant = true;

                int32_t d2_can_offset;
                for (int32_t i=0; i<2; i++) {
                    // search for support
                    int32_t u_can_2 = u_can;
                    int32_t d2;

                    bool support = false;
                    for (int32_t j=0; j<REDUN_MAX_DIST; j++) {
                        u_can_2 += redun_dir_u[i];
                        if (u_can_2<0 || u_can_2>=can_width) {
                            break;
                        }
                        d2_can_offset = (v_can*can_width)+u_can_2;
                        d2 = can[d2_can_offset];
                        if (d2>=0 && abs(d-d2)<=REDUN_THRESHOLD) {
                            support = true;
                            break;
                        }
                    }

                    // if we have no support => point is not redundant
                    if (!support) {
                        redundant = false;
                        break;
                    }
                }

                if (redundant)
                    can[d_can_offset] = -1;
            }
        }
    }
}

// removes redundant support points vertically
__global__ void
removeRedundantSupportPointsVertical(Ker2DArray<int32_t> can_)
{
    // declare variables
    int32_t u_can = getGlobThreadIdx_U();

    int32_t* can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    // parameters
    int32_t redun_dir_v[2] = {-1,1};

    if (u_can < can_width) {
        int32_t d_can_offset;
        int32_t d;
        for (int32_t v_can = 0; v_can < can_height; v_can++) {
            d_can_offset = v_can*can_width+u_can;
            d = can[d_can_offset];

            if (d >= 0) {
                // check all directions for redundancy
                bool redundant = true;

                int32_t d2_can_offset;
                for (int32_t i=0; i<2; i++) {
                    // search for support
                    int32_t v_can_2 = v_can;
                    int32_t d2;

                    bool support = false;
                    for (int32_t j=0; j<REDUN_MAX_DIST; j++) {
                        v_can_2 += redun_dir_v[i];
                        if (v_can_2<0 || v_can_2>=can_height) {
                            break;
                        }
                        d2_can_offset = (v_can_2*can_width)+u_can;
                        d2 = can[d2_can_offset];
                        if (d2>=0 && abs(d-d2)<=REDUN_THRESHOLD) {
                            support = true;
                            break;
                        }
                    }

                    // if we have no support => point is not redundant
                    if (!support) {
                        redundant = false;
                        break;
                    }
                }

                if (redundant)
                    can[d_can_offset] = -1;
            }
        }
    }
}


// sets positions in flag to 1 if the corresponding support point in can_ is valid
__global__ void
kerFlag
(
Ker2DArray<int32_t> can_,
int32_t* flag
)
{
    // declare variables
    int32_t u = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t v = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t* can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    // flag if disparity is valid
    if(u < can_width && v < can_height){
        if(can[v*can_width+u] >= 0)
            flag[v*can_width+u] = 1;
    }
}




// creates the arrays sp0_ and sp1_ holding the support point coordinates
__global__ void
kerCompact
(
int32_t* d_prefix_sum,
Ker2DArray<int32_t> can_,
Point2* d_sp0,
Point2* d_sp1
)
{
    // declare variables
    int32_t u = getGlobThreadIdx_U();
    int32_t v = getGlobThreadIdx_V();

    int32_t* can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    if(u < can_width && v < can_height){
        int32_t d;
        int32_t offset = v*can_width+u;
        int32_t s[2];

        s[0] = d_prefix_sum[offset];
        s[1] = d_prefix_sum[offset+1];

        // store coordinates in sp arrays
        if(s[0] != s[1]){
            d = can[offset];
            Point2 p;
            p._p[0] = u*CANDIDATE_STEPSIZE;
            p._p[1] = v*CANDIDATE_STEPSIZE;

            d_sp0[s[0]] = p;
            p._p[0] -= d;
            d_sp1[s[0]] = p;
        }
    }

}
