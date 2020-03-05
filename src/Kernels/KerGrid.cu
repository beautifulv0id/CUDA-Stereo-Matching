#include "KerGrid.h"
#include <stdint.h>
#include "../delaunay/gDel2D/GPU/MemoryManager.h"
#include "../delaunay/gDel2D/GpuDelaunay.h"
#include "KernelsCommon.h"

// load the disparities values from the support points in d_can to the disparity grid
__global__ void
kerLoadGridValues(Ker2DArray<int32_t> can_, Ker2DArray<uint8_t> temp0_,
                  Ker2DArray<uint8_t> temp1_, int32_t grid_width, int32_t grid_height)
{
    int32_t u_can = getGlobThreadIdx_U();
    int32_t v_can = getGlobThreadIdx_V();

    uint8_t* temp0 = temp0_._arr;
    uint8_t* temp1 = temp1_._arr;

    int32_t* can = can_._arr;
    int32_t can_width = can_._width;
    int32_t can_height = can_._height;

    if (u_can < can_width && v_can < can_height) {
        int32_t d_curr = *(can + getImageOffset(u_can, v_can, can_width));
        if(d_curr >= 0){
            int32_t d_min  = max(d_curr-1,0);
            int32_t d_max  = min(d_curr+1,DISP_MAX);

            int32_t u_img = u_can * CANDIDATE_STEPSIZE;
            int32_t v_img = v_can * CANDIDATE_STEPSIZE;

            int32_t u_grid0, u_grid1;
            for (int32_t d=d_min; d<=d_max; d++) {
                u_grid0 = floor((float) u_img/(float) GRID_SIZE);
                u_grid1 = floor((float) (u_img-d_curr)/(float) GRID_SIZE);

                int32_t v_grid = floor((float) v_img/(float)GRID_SIZE);

                int32_t offset;
                // point may potentially lay outside (corner points)
                if (v_grid>=0 && v_grid<grid_height) {
                    if (u_grid0>=0 && u_grid0<grid_width ) {
                    offset = getAddressOffestGrid(u_grid0, v_grid, d, grid_width, DISP_MAX+1);
                    *(temp0+offset) = 1;
                    }
                    if (u_grid1>=0 && u_grid1<grid_width ) {
                    offset = getAddressOffestGrid(u_grid1, v_grid, d, grid_width, DISP_MAX+1);
                    *(temp1+offset) = 1;
                    }
                }
            }
        }
    }
}

// diffuse the grid and aggregate disparity values
__global__ void
kerCreateGird(Ker2DArray<int32_t> grid_, Ker2DArray<uint8_t> temp_)
{
    int32_t u_grid = getGlobThreadIdx_U();
    int32_t v_grid = getGlobThreadIdx_V();

    uint8_t local_grid[DISP_MAX+1];

    int32_t* grid = grid_._arr;

    uint8_t* temp = temp_._arr;
    int32_t grid_width = temp_._width/(DISP_MAX+1);
    int32_t grid_height = temp_._height;

    if(u_grid>0 && u_grid<grid_width-1 && v_grid>0 && v_grid<grid_height-1){
        // diffuse grid
        const uint8_t* start_input, *end_input;
        start_input = temp+getAddressOffestGrid(u_grid,v_grid,0,grid_width,DISP_MAX+1);
        end_input = start_input + (grid_width+2)*(DISP_MAX+1);

        // diffusion pointers
        const int4* tl4_ptr = (int4*) (start_input + (-1*grid_width-1)*(DISP_MAX+1));
        const int4* tc4_ptr = (int4*) (start_input + (-1*grid_width+0)*(DISP_MAX+1));
        const int4* tr4_ptr = (int4*) (start_input + (-1*grid_width+1)*(DISP_MAX+1));
        const int4* cl4_ptr = (int4*) (start_input + (0*grid_width-1)*(DISP_MAX+1));
        const int4* cc4_ptr = (int4*) (start_input + (0*grid_width+0)*(DISP_MAX+1));
        const int4* cr4_ptr = (int4*) (start_input + (0*grid_width+1)*(DISP_MAX+1));
        const int4* bl4_ptr = (int4*) (start_input + (1*grid_width-1)*(DISP_MAX+1));
        const int4* bc4_ptr = (int4*) (start_input + (1*grid_width+0)*(DISP_MAX+1));
        const int4* br4_ptr = (int4*) (start_input + (1*grid_width+1)*(DISP_MAX+1));

        int4 tl4, tc4, tr4, cl4, cc4, cr4, bl4, bc4, br4;
        uint8_t *tl, *tc, *tr, *cl, *cc, *cr, *bl, *bc, *br;

        uint8_t* result = local_grid;

        // diffuse temporary grid
        for( ;br4_ptr<(int4*)end_input; tl4_ptr++, tc4_ptr++, tr4_ptr++, cl4_ptr++, cc4_ptr++, cr4_ptr++, bl4_ptr++, bc4_ptr++, br4_ptr++ ){
            tl4 = *tl4_ptr; tl = (uint8_t*) &tl4;
            tc4 = *tc4_ptr; tc = (uint8_t*) &tc4;
            tr4 = *tr4_ptr; tr = (uint8_t*) &tr4;
            cl4 = *cl4_ptr; cl = (uint8_t*) &cl4;
            cc4 = *cc4_ptr; cc = (uint8_t*) &cc4;
            cr4 = *cr4_ptr; cr = (uint8_t*) &cr4;
            bl4 = *bl4_ptr; bl = (uint8_t*) &bl4;
            bc4 = *bc4_ptr; bc = (uint8_t*) &bc4;
            br4 = *br4_ptr; br = (uint8_t*) &br4;

            for(int32_t i=0; i<16; i++){
                *result++ = *tl++ | *tc++ | *tr++ | *cl++ | *cc++ | *cr++ | *bl++ | *bc++ | *br++;
            }
        }


        // for all grid positions create disparity grid
        // start with second value (first is reserved for count)
        int32_t curr_ind = 1;
        // for all disparities do
        for(int32_t d=0; d<DISP_MAX+1; d++){

            // if yes => add this disparity to current cell
            if(*(local_grid+d) > 0){
                *(grid+getAddressOffestGrid(u_grid,v_grid,curr_ind,grid_width,DISP_MAX+2)) = d;
                curr_ind++;
            }
        }

        // set number of indices
        *(grid+getAddressOffestGrid(u_grid,v_grid,0,grid_width,DISP_MAX+2))=curr_ind-1;
    }
}
