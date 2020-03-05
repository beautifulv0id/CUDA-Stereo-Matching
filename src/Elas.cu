#include "Elas.h"
#include <cub/cub.cuh>
#include "triangle.h"
#include "delaunay/gDel2D/GPU/MemoryManager.h"
#include "Utils.h"
#include "Kernels/KerDescriptor.h"
#include "Kernels/KerSupportMatches.h"
#include "Kernels/KerGrid.h"
#include "Kernels/KerComputeDisparity.h"
#include "Kernels/KerLeftRightConsistencyCheck.h"
#include "delaunay/gDel2D/GpuDelaunay.h"
#ifdef WINDOWS_VISUALIZATION
    #include "delaunay/Visualizer.h"
#endif
#include "thrust/copy.h"

using namespace std;

void Elas::process (uint8_t* h_img0,uint8_t* h_img1,float* h_disp0,float* h_disp1,const int32_t* dims){

#if PROFILE
    timer.reset();
#endif
    // get width, height and bytes per line
    width  = dims[0];
    height = dims[1];
    size = width*height;

    can_width = width / param.candidate_stepsize;
    can_height = height / param.candidate_stepsize;

    // LOAD IMAGES TO DEVICE
#if PROFILE
    timer.start("Upload Images");
#endif
    Dev2DVector<uint8_t> d_img0(width, height);
    Dev2DVector<uint8_t> d_img1(width, height);

    d_img0.copyFromHost(h_img0, size);
    d_img1.copyFromHost(h_img1, size);

    // COMPUTE DESCRIPsTOR
#if PROFILE
    timer.start("Descriptor");
#endif
    Dev2DVector<uint8_t> d_desc0, d_desc1;
    computeDescriptor(d_img0, d_desc0);
    computeDescriptor(d_img1, d_desc1);

    // COMPUTE SUPPORT MATCHES
#if PROFILE
    timer.start("Support Matches");
#endif
    Dev2DVector<int32_t> d_can = computeSupportMatches(d_desc0, d_desc1);

    // CREATE CANDIDATE VECTOR
    Point2DVec d_sp0, d_sp1;
    computeCanditateVectors(d_can, d_sp0, d_sp1);

    // COMPUTE DISTRIBUTION GRID
#if PROFILE
    timer.start("Distribution Grid");
#endif
    Dev2DVector<int32_t> d_grid0, d_grid1;
    createDistributionGrid(d_can, d_grid0, d_grid1);


#if GPU_ONLY
    // COMPUTE DELAUNY TRIANGULATION GPU
    GpuDel gpu_del;
    TriDVec tri_vec0_, tri_vec1_;

#if PROFILE
    timer.start("Delaunay Triangulation left");
#endif
    gpu_del.compute(can_vec0_, &tri_vec0_);

#if PROFILE
    timer.start("Delaunay Triangulation right");
#endif
    gpu_del.compute(can_vec1_, &tri_vec1_);

#else
    // COMPUTE DELAUNY TRIANGULATION CPU
#if PROFILE
    timer.start("CPU Delaunay Triangulation");
#endif
    TriDVec d_tri0, d_tri1;
    computeDelaunyTrianguation(d_sp0, d_tri0);
    computeDelaunyTrianguation(d_sp1, d_tri1);
#endif

    // COMPUTE DISPARITY
    Dev2DVector<float> d_disp0(width, height, -1);
    Dev2DVector<float> d_disp1(width, height, -1);

#if PROFILE
    timer.start("Compute Disparity left");
#endif
    computeDisparity(d_can, d_sp0, d_tri0, d_grid0, d_desc0, d_desc1, false, d_disp0);

#if PROFILE
    timer.start("Compute Disparity right");
#endif
    computeDisparity(d_can, d_sp0, d_tri1, d_grid1, d_desc0, d_desc1, true, d_disp1);

    // LEFT/RIGHT CONSISTENCY CHECK
#if PROFILE
    timer.start("left/right consistency check");
#endif
    leftRightConsistencyCheck(d_disp0, d_disp1);
    cudaDeviceSynchronize();

#if PROFILE
    timer.plot();
#endif

    cudaMemcpy(h_disp0, toKernelPtr(d_disp0) , size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_disp1, toKernelPtr(d_disp1) , size*sizeof(float), cudaMemcpyDeviceToHost);
}


/*****************************************************************************/
/*                                                                           */
/*   Computes the descriptor for an input image.                             */
/*   A descriptor vector is the concatenatoin of                             */
/*   16 features from the sobel filtered images.                             */
/*                                                                           */
/*****************************************************************************/

void Elas::computeDescriptor
(
    Dev2DVector<uint8_t> &d_img,
    Dev2DVector<uint8_t> &d_desc
)
{
    // allocate memory for horizontal/vertival sobel filter
    Dev2DVector<uint8_t> d_du(width, height);
    Dev2DVector<uint8_t> d_dv(width, height);

    // allocate memory for descriptor
    d_desc = Dev2DVector<uint8_t>(width*16, height);

    dim3 numThreads = dim3(32, 8, 1);
    dim3 numBlocks = dim3(iDivUp(width, numThreads.x),
                          iDivUp(height, numThreads.y), 1);

    // compute sobel filtered images
    kerComputeSobel<<<numBlocks, numThreads>>>
    (
        toKernel2DArray( d_img ),
        toKernelPtr( d_du ),
        toKernelPtr( d_dv )
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif

    // compute the descriptor
    kerComputeDescriptor<<<numBlocks, numThreads>>>
    (
        toKernelPtr( d_du ),
        toKernelPtr( d_dv ),
        toKernelPtr( d_desc ),
        width,
        height
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif
}


/*****************************************************************************/
/*                                                                           */
/*   compute support matches on a 5x5 grid that can be robustly matched      */
/*                                                                           */
/*****************************************************************************/
Dev2DVector<int32_t> Elas::computeSupportMatches
(
Dev2DVector<uint8_t>& d_desc0,
Dev2DVector<uint8_t>& d_desc1
)
{
        // allocate memory for 2D candidate vector
        Dev2DVector<int32_t> d_can(can_width, can_height, -1);
        dim3 num_threads = dim3(32, 1, 1);
        dim3 num_blocks = dim3(iDivUp(can_width, num_threads.x),
                               iDivUp(can_height, num_threads.y), 1);

        num_threads = dim3(32, 1, 1);
        num_blocks = dim3(can_width, can_height, 1);
        int32_t shared_mem = 4*num_threads.x*sizeof(int32_t);

        // compute support candidates from left to right
        kerComputeSupportMatches<<<num_blocks, num_threads, shared_mem>>>
        (
            toKernelPtr( d_desc0 ),
            toKernelPtr( d_desc1 ),
            toKernel2DArray( d_can ),
            width,
            height,
            false
        );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif


        // check if support candidates can be matched from right-to-left as well
        kerComputeSupportMatches<<<num_blocks, num_threads, shared_mem>>>
        (
            toKernelPtr( d_desc0 ),
            toKernelPtr( d_desc1 ),
            toKernel2DArray( d_can ),
            width,
            height,
            true
        );

#if NDEBUG
#else
        CHECK_LAUNCH_ERROR();
#endif

        num_threads = dim3(32, 1, 1);
        num_blocks = dim3(iDivUp(can_width, num_threads.x),
                          iDivUp(can_height, num_threads.y), 1);

        // remove inconsistent support points (eg. points that
        // have dissimilar disparity values in their surrounding area)
        removeInconsistentSupportPoints<<<num_blocks, num_threads>>>
        (
            toKernel2DArray( d_can )
        );

#if NDEBUG
#else
        CHECK_LAUNCH_ERROR();
#endif

        num_threads = dim3(1, 32, 1);
        num_blocks = dim3(1, iDivUp(can_height, num_threads.y), 1);


        // remove points that have supporting (eg. similar)
        // points along their horitontal sournding
        removeRedundantSupportPointsHorizontal<<<num_blocks, num_threads>>>
        (
            toKernel2DArray( d_can )
        );

#if NDEBUG
#else
        CHECK_LAUNCH_ERROR();
#endif

        num_threads = dim3(32, 1, 1);
        num_blocks = dim3(iDivUp(can_width, num_threads.x), 1, 1);

        // remove points that have supporting (eg. similar)
        // points along their vertical sournding
        removeRedundantSupportPointsVertical<<<num_blocks, num_threads>>>
        (
            toKernel2DArray( d_can )
        );

#if NDEBUG
#else
        CHECK_LAUNCH_ERROR();
#endif

        return d_can;
}

void Elas::computeDelaunyTrianguation(Point2DVec &d_sp, TriDVec &d_tri)
{
    Point2* h_sp = (Point2*) malloc((d_sp.size()) * sizeof(Point2));
    thrust::copy(d_sp.begin(), d_sp.end(), h_sp);

    // input/output structure for triangulation
    struct TRILIB::triangulateio in, out;

    in.numberofpoints = d_sp.size();
    in.pointlist = reinterpret_cast<float*>( h_sp );

    h_sp = NULL;

    in.numberofpointattributes = 0;
    in.pointattributelist      = NULL;
    in.pointmarkerlist         = NULL;
    in.numberofsegments        = 0;
    in.numberofholes           = 0;
    in.numberofregions         = 0;
    in.regionlist              = NULL;

    // outputs
    out.pointlist              = NULL;
    out.pointattributelist     = NULL;
    out.pointmarkerlist        = NULL;
    out.trianglelist           = NULL;
    out.triangleattributelist  = NULL;
    out.neighborlist           = NULL;
    out.segmentlist            = NULL;
    out.segmentmarkerlist      = NULL;
    out.edgelist               = NULL;
    out.edgemarkerlist         = NULL;

    // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
    char parameters[] = "zQB";
    TRILIB::triangulate(parameters, &in, &out, NULL);

    Tri* h_tri = reinterpret_cast<Tri*>( out.trianglelist );

    d_tri = TriDVec(out.numberoftriangles);
    d_tri.copyFromHost(h_tri, out.numberoftriangles);

    // free memory used for triangulation
    free(in.pointlist);
    free(out.pointlist);
    free(out.trianglelist);
}



/*****************************************************************************/
/*                                                                           */
/*   move support points from image representation to vector representation  */
/*                                                                           */
/*****************************************************************************/

void Elas::computeCanditateVectors
(
Dev2DVector<int32_t>& d_can,
Point2DVec& d_sp0,
Point2DVec& d_sp1
)
{
    // allocate memory for candidate vectors
    d_sp0 = Point2DVec(can_width*can_height);
    d_sp1 = Point2DVec(can_width*can_height);

    int32_t *d_prefix_sum_in, *d_prefix_sum_out;
    CUDA_SAFE_CALL( cudaMalloc(&d_prefix_sum_in, (width*height+1)*sizeof(int32_t)) );
    CUDA_SAFE_CALL( cudaMalloc(&d_prefix_sum_out, (width*height+1)*sizeof(int32_t)) );
    CUDA_SAFE_CALL( cudaMemset(d_prefix_sum_in,0, (width*height+1)*sizeof(int32_t)) );
    CUDA_SAFE_CALL( cudaMemset(d_prefix_sum_out,0, (width*height+1)*sizeof(int32_t)) );

    dim3 num_threads = dim3(16, 16, 1);
    dim3 num_blocks = dim3(iDivUp(can_width, num_threads.x), iDivUp(can_height, num_threads.y), 1);
    kerFlag<<<num_blocks, num_threads>>>(toKernel2DArray( d_can ), d_prefix_sum_in);

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
    d_prefix_sum_in, d_prefix_sum_out, width*height+1);
    // Allocate temporary storage
    CUDA_SAFE_CALL( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
    d_prefix_sum_in, d_prefix_sum_out,  width*height+1);

    int32_t num_sp;
    CUDA_SAFE_CALL( cudaMemcpy(&num_sp, &d_prefix_sum_out[width*height], sizeof(int32_t), cudaMemcpyDeviceToHost) );

    d_sp0.resize(num_sp);
    d_sp1.resize(num_sp);

    num_threads = dim3(32, 4, 1);
    num_blocks = dim3(iDivUp(can_width, num_threads.x), iDivUp(can_height, num_threads.y), 1);

    kerCompact<<<num_blocks, num_threads>>>
    (
        d_prefix_sum_out,
        toKernel2DArray( d_can ),
        toKernelPtr( d_sp0 ),
        toKernelPtr( d_sp1 )
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif
}

/************************************************************************************************/
/*                                                                                              */
/*  Calculates all occuring disparities in a 20x20 grid of the given support points             */
/*  Output: for each grid entry at index 0 the amount of disparities followed by the disparites */
/*                                                                                              */
/************************************************************************************************/

void Elas::createDistributionGrid
(
Dev2DVector<int32_t>& d_can,
Dev2DVector<int32_t>& d_grid0,
Dev2DVector<int32_t>& d_grid1
)
{
    // calculate grid dimensions
    int32_t grid_width = (int32_t)ceil((float)width/(float)param.grid_size);
    int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);

    // allocate distribution grid
    d_grid0 = Dev2DVector<int32_t>(grid_width * (param.disp_max+2), grid_height, 0);
    d_grid1 = Dev2DVector<int32_t>(grid_width * (param.disp_max+2), grid_height, 0);

    // allocate temporary memory
    Dev2DVector<uint8_t> d_temp0(grid_width * (param.disp_max+1), grid_height, 0);
    Dev2DVector<uint8_t> d_temp1(grid_width * (param.disp_max+1), grid_height, 0);

    dim3 num_threads = dim3(32, 4, 1);
    dim3 num_blocks = dim3(iDivUp(can_width, num_threads.x),
                           iDivUp(can_height, num_threads.y), 1);

    // load grid values from support points
    kerLoadGridValues<<<num_blocks, num_threads>>>
    (
        toKernel2DArray( d_can ),
        toKernel2DArray( d_temp0 ),
        toKernel2DArray( d_temp1 ),
        grid_width,
        grid_height
    );


#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif

    num_threads = dim3(32, 4, 1);
    num_blocks = dim3(iDivUp(grid_width, num_threads.x),
                      iDivUp(grid_height, num_threads.y), 1);

    // create the final grid
    kerCreateGird<<<num_blocks, num_threads>>>
    (
        toKernel2DArray( d_grid0 ),
        toKernel2DArray( d_temp0 )
    );

    kerCreateGird<<<num_blocks, num_threads>>>
    (
        toKernel2DArray( d_grid1 ),
        toKernel2DArray( d_temp1 )
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif

}


/************************************************************************************************/
/*                                                                                              */
/*  Computes the disparity for all pixels                                                       */
/*                                                                                              */
/************************************************************************************************/

void Elas::computeDisparity
(
Dev2DVector<int32_t>& d_can,
Point2DVec& d_sp,
DevVector<Tri>& d_tri,
Dev2DVector<int32_t>& d_grid,
Dev2DVector<uint8_t>& d_desc0,
Dev2DVector<uint8_t>& d_desc1,
bool right_image,
Dev2DVector<float>& d_disp
)
{
    dim3 num_threads(32, 1, 1);
    dim3 num_blocks(iDivUp(d_tri.size(), num_threads.x), 1, 1);

    Dev2DVector<float> d_mean(width, height, -1);

    // pre compute approximate disparity values
    kerComputeDepthApproximation<<<num_blocks, num_threads>>>
    (
        toKernel2DArray( d_can ),
        toKernelArray( d_sp ),
        toKernelArray( d_tri ),
        right_image,
        toKernel2DArray( d_mean )
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif


    num_threads = dim3(min(prop.maxThreadsPerBlock,width), 1, 1);
    num_blocks  = dim3(1, iDivUp(height, num_threads.y), 1);
    int32_t shared_mem = 16*width*sizeof(uint8_t);

    // compute exact disparity alues
    kerComputeDisparity<<<num_blocks, num_threads, shared_mem>>>
    (
        toKernel2DArray( d_grid),
        toKernelPtr( d_desc0 ),
        toKernelPtr( d_desc1 ),
        toKernelPtr( d_mean ),
        right_image,
        toKernel2DArray(d_disp)
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif

}


/************************************************************************************************/
/*                                                                                              */
/*  Applies a left/right consistency check to both images                                       */
/*                                                                                              */
/************************************************************************************************/

void Elas::leftRightConsistencyCheck
(
Dev2DVector<float> &d_disp0,
Dev2DVector<float> &d_disp1
)
{
    // create copy of disparity images
    Dev2DVector<float> d_disp0_cpy(width, height);
    thrust::copy(d_disp0.begin(), d_disp0.end(), d_disp0_cpy.begin());

    Dev2DVector<float> d_disp1_cpy(width, height);
    thrust::copy(d_disp1.begin(), d_disp1.end(), d_disp1_cpy.begin());

    dim3 num_threads(32, 1, 1);
    dim3 num_blocks(iDivUp(width, num_threads.x), iDivUp(height, num_threads.y), 1);

    // apply left/right consistency check
    kerLeftRightConsistencyCheck<<<num_blocks, num_threads>>>
    (
        toKernel2DArray( d_disp0 ),
        toKernel2DArray( d_disp1 ),
        toKernel2DArray( d_disp0_cpy ),
        toKernel2DArray( d_disp1_cpy )
    );

#if NDEBUG
#else
    CHECK_LAUNCH_ERROR();
#endif

}
