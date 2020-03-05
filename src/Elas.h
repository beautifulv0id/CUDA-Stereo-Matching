#ifndef __ELAS_H__
#define __ELAS_H__

#include <stdint.h>
#include "delaunay/gDel2D/GPU/MemoryManager.h"
#include "delaunay/gDel2D/GpuDelaunay.h"

#include "Timer.h"
#include "Utils.h"

class Elas {

public:
    struct parameters {
        int32_t disp_min;               // min disparity
        int32_t disp_max;               // max disparity
        float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
        int32_t support_texture;        // min texture for support points
        int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
        int32_t incon_window_size;      // window size of inconsistent support point check
        int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
        int32_t incon_min_support;      // minimum number of consistent support points
        bool    add_corners;            // add support points at image corners with nearest neighbor disparities
        int32_t grid_size;              // size of neighborhood for additional support point extrapolation
        float   beta;                   // image likelihood parameter
        float   gamma;                  // prior constant
        float   sigma;                  // prior sigma
        float   sradius;                // prior sigma radius
        int32_t match_texture;          // min texture for dense matching
        int32_t lr_threshold;           // disparity threshold for left/right consistency check
        float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
        int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
        int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
        bool    filter_median;          // optional median filter (approximated)
        bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
        bool    postprocess_only_left;  // saves time by not postprocessing the right image
        bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
        // note: for this option D1 and D2 must be passed with size
        //       width/2 x height/2 (rounded towards zero)

        // constructor
        parameters () {
            disp_min              = 0;
            disp_max              = 255;
            support_threshold     = 0.95;
            support_texture       = 10;
            candidate_stepsize    = 5;
            incon_window_size     = 5;
            incon_threshold       = 5;
            incon_min_support     = 5;
            add_corners           = 1;
            grid_size             = 20;
            beta                  = 0.02;
            gamma                 = 5;
            sigma                 = 1;
            sradius               = 3;
            match_texture         = 0;
            lr_threshold          = 2;
            speckle_sim_threshold = 1;
            speckle_size          = 200;
            ipol_gap_width        = 5000;
            filter_median         = 1;
            filter_adaptive_mean  = 0;
            postprocess_only_left = 0;
            subsampling           = 0;
        }
    };


    Elas () {
        cudaGetDeviceProperties(&prop, 0);
    }

    ~Elas () {}

    void process
    (
        uint8_t* h_img0,
        uint8_t* h_img1,
        float* h_disp0,
        float* h_disp1,
        const int32_t* dims
    );

private:
    struct triangle {
        int32_t c1,c2,c3;
        float   t1a,t1b,t1c;
        float   t2a,t2b,t2c;
        triangle
        (
            int32_t c1,
            int32_t c2,
            int32_t c3
        ) :c1(c1),c2(c2),c3(c3){}
    }; 


    // main functions

    // computes the descriptor for an input image
    void computeDescriptor
    (
        Dev2DVector<uint8_t> &d_img,
        Dev2DVector<uint8_t> &d_desc
    );

    // computes support points on a 5x5 grid
    // that can be roboustly matched
    Dev2DVector<int32_t> computeSupportMatches
    (
        Dev2DVector<uint8_t>& d_desc0,
        Dev2DVector<uint8_t>& d_desc1
    );

    // computes the delauny triangluation on
    // a set of support point coordinates
    void computeDelaunyTrianguation
    (
        Point2DVec& d_sp,
        TriDVec &d_tri
    );

    // computes the disparity grid
    void createDistributionGrid
    (
        Dev2DVector<int32_t>& d_can,
        Dev2DVector<int32_t>& d_grid0,
        Dev2DVector<int32_t>& d_grid1
    );

    // computes the disparities the remaining pixels
    void computeDisparity
    (
        Dev2DVector<int32_t>& d_can,
        Point2DVec& d_sp,
        DevVector<Tri>& d_tri,
        Dev2DVector<int32_t>& d_grid,
        Dev2DVector<uint8_t>& d_desc0,
        Dev2DVector<uint8_t>& d_desc1,
        bool right_image,
        Dev2DVector<float>& d_disp
    );

    // applies a l/r consistency check on both images
    void leftRightConsistencyCheck
    (
        Dev2DVector<float>& d_disp0,
        Dev2DVector<float>& d_disp1
    );

    // helper functions
    void computeCanditateVectors
    (
        Dev2DVector<int32_t>& d_can,
        Point2DVec& d_sp0,
        Point2DVec& d_sp1
    );

    // parameter set
    parameters param;
    int32_t width,height,size;
    int32_t can_width, can_height;
    // profiling timer
    Timer timer;
    cudaDeviceProp prop;
};


#endif
