#include "KerComputeDisparity.h"
#include <stdint.h>
#include <math.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "KernelsCommon.h"

// computes the mean disparities
__global__ void
kerComputeDepthApproximation(Ker2DArray<int32_t> can_, KerArray<Point2> sp_,
                             KerArray<Tri> tri_arr, bool right_image,
                             Ker2DArray<float> mean_)
{
    int32_t tid = getGlobThreadIdx();

    Tri* tri_ = tri_arr._arr;
    int32_t num_tri = tri_arr._num;

    Point2* sp = sp_._arr;

    int32_t plane_radius = (int32_t)max((float)ceilf(SIGMA*SRADIUS),(float)2.0);

    if (tid < num_tri) {
        Tri tri = tri_[tid];
        float t_a, t_b, t_c;

        int32_t* can      = can_._arr;
        float*   mean   = mean_._arr;
        int32_t  can_width = can_._width;
        int32_t  width       = mean_._width;

        // compute disparity plane
        // get triangle corner indices
        int32_t c1 = tri._v[0];
        int32_t c2 = tri._v[1];
        int32_t c3 = tri._v[2];

        // declare matrix A and vector b
        FLOAT A[3][3];
        FLOAT b[3][1];

        // declare vectors for triangle corners
        int32_t tri_u[3];
        int32_t tri_v[3];
        int32_t tri_d[3];

        // compute Matrix A for linear system of left triangle
        Point2 p1 = sp[c1];    tri_u[0] = p1._p[0];  tri_v[0] = p1._p[1];
        Point2 p2 = sp[c2];    tri_u[1] = p2._p[0];  tri_v[1] = p2._p[1];
        Point2 p3 = sp[c3];    tri_u[2] = p3._p[0];  tri_v[2] = p3._p[1];

        // get corresponding depth values
        tri_d[0] = can[(tri_v[0]*can_width+tri_u[0])/CANDIDATE_STEPSIZE];
        tri_d[1] = can[(tri_v[1]*can_width+tri_u[1])/CANDIDATE_STEPSIZE];
        tri_d[2] = can[(tri_v[2]*can_width+tri_u[2])/CANDIDATE_STEPSIZE];

        // compute Matrix A for linear system of left/right triangle accordingly
        if(!right_image){
            A[0][0] = tri_u[0];
            A[1][0] = tri_u[1];
            A[2][0] = tri_u[2];
        } else {
            A[0][0] = tri_u[0] - tri_d[0];
            A[1][0] = tri_u[1] - tri_d[1];
            A[2][0] = tri_u[2] - tri_d[2];
        }

        A[0][1] = tri_v[0]; A[0][2] = 1;
        A[1][1] = tri_v[1]; A[1][2] = 1;
        A[2][1] = tri_v[2]; A[2][2] = 1;

        // compute vector b for linear system (containing the disparities)
        b[0][0] = tri_d[0];
        b[1][0] = tri_d[1];
        b[2][0] = tri_d[2];

        if(kerSolve(A, b)){
            t_a = b[0][0];
            t_b = b[1][0];
            t_c = b[2][0];
        } else {
            t_a = 0;
            t_b = 0;
            t_c = 0;
        }

        // loop variables
        float plane_a,plane_b,plane_c,plane_d;

        plane_a = t_a;
        plane_b = t_b;
        plane_c = t_c;

        // sort triangle corners wrt. u (ascending)
        if (right_image) {
            tri_u[0] -= tri_d[0];
            tri_u[1] -= tri_d[1];
            tri_u[2] -= tri_d[2];
        }

        int32_t tri_u_temp;
        int32_t tri_v_temp;
        for (uint32_t j=0; j<3; j++) {
            for (uint32_t k=0; k<j; k++) {
                if (tri_u[k]>tri_u[j]) {
                    tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
                    tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
                }
            }
        }

        // rename corners
        float A_u = (float) tri_u[0]; float A_v = (float) tri_v[0];
        float B_u = (float) tri_u[1]; float B_v = (float) tri_v[1];
        float C_u = (float) tri_u[2]; float C_v = (float) tri_v[2];

        // compute straight lines connecting triangle corners (*_a => slope, *_b => y-axis intercept)
        float AB_a = 0; float AC_a = 0; float BC_a = 0;
        if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
        if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
        if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
        float AB_b = A_v-AB_a*A_u;
        float AC_b = A_v-AC_a*A_u;
        float BC_b = B_v-BC_a*B_u;

        int32_t d_plane;

        // first part (triangle corner A->B)
        if ((int32_t)(A_u)!=(int32_t)(B_u)) {
            for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++) {
                int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
                int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
                for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++) {
                    d_plane = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
                    mean[v*width+u] = d_plane;
                }
            }
        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_u)!=(int32_t)(C_u)) {
            for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++) {
                int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
                int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
                for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++) {
                    d_plane = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
                    mean[v*width+u] = d_plane;
                }
            }
        }
    } // end if valid tri
}

// computes the remaining disparities minimizing the energy function
__global__ void
kerComputeDisparity(Ker2DArray<int32_t> grid_, uint8_t* desc0, uint8_t* desc1,
                    float* mean, bool right_image, Ker2DArray<float> disp_)
{
    // declare variables
    int32_t tpb = blockDim.x;
    int32_t u = getGlobThreadIdx_U();
    int32_t v = getGlobThreadIdx_V();

    float*  disp      = disp_._arr;
    int32_t width   = disp_._width;
    int32_t height  = disp_._height;

    const int32_t desc_window = 3;
    const int32_t disp_num    = DISP_MAX+1;

    // read desc row to shared mem
    extern __shared__ uint8_t shared[];
    uint8_t* shared_ptr = &shared[16*u];


    if(!right_image){
        int32_t pos = u;
        while (pos < width) {
            shared_ptr = &shared[16*pos];
            loadDescriptorBlock(desc1+(v*width+pos)*16,shared_ptr);
            pos+=tpb;
        }

    } else {
        int32_t pos = u;
        while (pos < width) {
            shared_ptr = &shared[16*pos];
            loadDescriptorBlock(desc0+(v*width+pos)*16,shared_ptr);
            pos+=tpb;
        }
    }
    __syncthreads();

    int32_t plane_radius = (int32_t)max((float)ceilf(SIGMA*SRADIUS),(float)2.0);
    int32_t* grid = grid_._arr;
    int32_t grid_width = grid_._width/(DISP_MAX+2);

    if(v > desc_window && v < height - desc_window ) {
        int32_t  line_offset = 16*width*max(min(v,height-3),2);
        uint8_t *I1_line_addr,*I2_line_addr;
        if (!right_image) {
            I1_line_addr = desc0+line_offset;
            I2_line_addr = desc1+line_offset;
        } else {
            I1_line_addr = desc1+line_offset;
            I2_line_addr = desc0+line_offset;
        }

        for(; u < width; u+=tpb){
            if (u > desc_window &&  u < width - desc_window) {
                int32_t d_plane = mean[v*width+u];
                // only compute pixles that are within a triangle
                if (d_plane > 0) {

                    // compute line start address
                    // compute I1 block start address
                    uint8_t* I1_block_addr = I1_line_addr+16*u;
                    uint8_t* I2_block_addr;

                    // read block to local memory
                    uint8_t I1_block_arr[16];
                    loadDescriptorBlock(I1_block_addr, I1_block_arr);

                    uint8_t I2_block_arr[16];


                    // does this patch have enough texture?
                    int32_t sum = 0;
                    for (int32_t i=0; i<16; i++)
                        sum += abs((int32_t)(I1_block_arr[i])-128);
                    if (sum<MATCH_TEXTURE)
                        return;

                    int32_t d_plane_min = max(d_plane-plane_radius,0);
                    int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

                    // get grid pointer
                    int32_t  grid_x    = (int32_t)floor((float)u/(float)GRID_SIZE);
                    int32_t  grid_y    = (int32_t)floor((float)v/(float)GRID_SIZE);
                    uint32_t grid_offset = getAddressOffestGrid(grid_x,grid_y,0,grid_width,DISP_MAX+2);
                    int32_t  num_grid  = *(grid+grid_offset);
                    int32_t* grid_block_addr = grid+grid_offset+1;

                    // loop variables
                    int32_t d_curr, u_warp, val;
                    int32_t min_val = 10000;
                    int32_t min_d   = -1;

                    int32_t delta_d;
                    int32_t P;
                    float two_sigma_squared = 2*SIGMA*SIGMA;

                    // minimize energy function
                    if (!right_image) {
                        for (int32_t i=0; i<num_grid; i++) {
                            d_curr = grid_block_addr[i];
                            if (d_curr<d_plane_min || d_curr>d_plane_max) {
                                u_warp = u-d_curr;
                                if (u_warp<desc_window || u_warp>=width-desc_window)
                                    continue;
                                loadDescriptorBlock(&shared[16*u_warp], I2_block_arr);
                                kerUpdatePosteriorMinimum(I2_block_arr,I1_block_arr,d_curr,val,min_val,min_d);
                            }
                        }
                        for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
                            u_warp = u-d_curr;
                            if (u_warp<desc_window || u_warp>=width-desc_window)
                                continue;
                            delta_d = abs(d_curr-d_plane);
                            P = (int32_t)((-logf(GAMMA+exp(-delta_d*delta_d/two_sigma_squared))
                                           +logf(GAMMA))/BETA);

                            loadDescriptorBlock(&shared[16*u_warp], I2_block_arr);
                            kerUpdatePosteriorMinimum(I2_block_arr,I1_block_arr,d_curr,
                                                      P,val,min_val,min_d);
                        }
                    } else {
                        for (int32_t i=0; i<num_grid; i++) {
                            d_curr = grid_block_addr[i];
                            if (d_curr<d_plane_min || d_curr>d_plane_max) {
                                u_warp = u+d_curr;
                                if (u_warp<desc_window || u_warp>=width-desc_window)
                                    continue;
                                loadDescriptorBlock(&shared[16*u_warp], I2_block_arr);
                                kerUpdatePosteriorMinimum(I2_block_arr,I1_block_arr,d_curr,
                                                          val,min_val,min_d);
                            }
                        }
                        for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
                            u_warp = u+d_curr;
                            if (u_warp<desc_window || u_warp>=width-desc_window)
                                continue;

                            delta_d = abs(d_curr-d_plane);
                            P = (int32_t)((-logf(GAMMA+exp(-delta_d*delta_d/two_sigma_squared))
                                           +logf(GAMMA))/BETA);

                            loadDescriptorBlock(&shared[16*u_warp], I2_block_arr);
                            kerUpdatePosteriorMinimum(I2_block_arr,I1_block_arr,d_curr,
                                                      P,val,min_val, min_d);
                        }
                    }

                    // address of disparity we want to compute
                    uint32_t d_addr;
                    d_addr = getImageOffset(u,v,width);

                    // set disparity value
                    if (min_d>=0) *(disp+d_addr) = min_d; // MAP value (min neg-Log probability)
                    else          *(disp+d_addr) = -1;    // invalid disparity

                }
            }
        }
    }
}



// solve linear system M*x=B, results are written to B
__device__ bool
kerSolve(FLOAT A[3][3], FLOAT B[3][1], FLOAT eps)
{
    int32_t ipiv[3];

    // loop variables
    int32_t i, icol, irow, j, k, l, ll;
    FLOAT big, dum, pivinv, temp;

    // initialize pivots to zero
    for (j=0;j<3;j++) ipiv[j]=0;

    // main loop over the columns to be reduced
    for (i=0;i<3;i++) {
        big=0.0;
        // search for a pivot element
        for (j=0;j<3;j++)
            if (ipiv[j]!=1)
                for (k=0;k<3;k++)
                    if (ipiv[k]==0)
                        if (fabs(A[j][k])>=big) {
                            big=fabs(A[j][k]);
                            irow=j;
                            icol=k;
                        }
        ++(ipiv[icol]);

        // We now have the pivot element, so we interchange rows, if needed, to put the pivot
        // element on the diagonal. The columns are not physically interchanged, only relabeled.
        if (irow != icol) {
            for (l=0;l<3;l++) SWAP(A[irow][l], A[icol][l])
                    for (l=0;l<1;l++) SWAP(B[irow][l], B[icol][l])
        }

        // check for singularity
        if (fabs(A[icol][icol]) < eps)
            return false;

        pivinv=1.0/A[icol][icol];
        A[icol][icol]=1.0;
        for (l=0;l<3;l++) A[icol][l] *= pivinv;
        for (l=0;l<1;l++) B[icol][l] *= pivinv;

        // Next, we reduce the rows except for the pivot one
        for (ll=0;ll<3;ll++)
            if (ll!=icol) {
                dum = A[ll][icol];
                A[ll][icol] = 0.0;
                for (l=0;l<3;l++) A[ll][l] -= A[icol][l]*dum;
                for (l=0;l<1;l++) B[ll][l] -= B[icol][l]*dum;
            }
    }

    return true;
}

__device__ void
kerUpdatePosteriorMinimum(uint8_t* desc0_block, uint8_t* desc1_block, const int32_t &d,
                          int32_t& val, int32_t& min_val, int32_t& min_d)
{
    val = 0;
    for (int32_t j=0; j<16; j++)
        val += abs(((int32_t) *(desc0_block+j))-((int32_t) *(desc1_block+j)));
    if (val<min_val) {
        min_val = val;
        min_d = d;
    }
}


__device__ void
kerUpdatePosteriorMinimum(uint8_t* desc0_block, uint8_t* desc1_block, const int32_t &d,
                          const int32_t &w, int32_t& val, int32_t& min_val, int32_t& min_d)
{
    val = w;
    for(int32_t j=0; j<16; j++)
        val += abs(((int32_t) *(desc0_block+j))-((int32_t) *(desc1_block+j)));
    if(val<min_val){
        min_val = val;
        min_d = d;
    }
}


