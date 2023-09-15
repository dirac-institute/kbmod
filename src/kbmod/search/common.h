/*
 * common.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef COMMON_H_
#define COMMON_H_

namespace search {

#ifdef HAVE_CUDA
    constexpr bool HAVE_GPU = true;
#else
    constexpr bool HAVE_GPU = false;
#endif

constexpr unsigned int MAX_KERNEL_RADIUS = 15;
constexpr unsigned short MAX_STAMP_EDGE = 64;
constexpr unsigned short CONV_THREAD_DIM = 32;
constexpr unsigned short THREAD_DIM_X = 128;
constexpr unsigned short THREAD_DIM_Y = 2;
constexpr unsigned short RESULTS_PER_PIXEL = 8;
constexpr float NO_DATA = -9999.0;

enum StampType { STAMP_SUM = 0, STAMP_MEAN, STAMP_MEDIAN };

/*
 * Data structure to represent an objects trajectory
 * through a stack of images
 */
struct trajectory {
    // Trajectory velocities
    float xVel;
    float yVel;
    // Likelyhood
    float lh;
    // Est. Flux
    float flux;
    // Origin
    short x;
    short y;
    // Number of images summed
    short obsCount;
};

// The position (in pixels) of a trajectory.
struct pixelPos {
    float x;
    float y;
};

/*
 * Linear approximation to the barycentric correction needed to transform a
 * pixel in the first image to a pixel in a consequent image. One struct needed
 * per image. Correction calculated in higher level code.
 */
struct baryCorrection {
    // linear coefficients of linear fit of pixel dependence
    float dx;
    float dxdx;
    float dxdy;
    float dy;
    float dydx;
    float dydy;
};

/* The parameters to use for the on device search. */

struct searchParameters {
    // Basic filtering paramets.
    int minObservations;
    float minLH;

    // Parameters for sigmaG filtering on device.
    bool do_sigmag_filter;
    float sGL_L;
    float sGL_H;
    float sigmaGCoeff;

    // Do barycentric corrections.
    bool useCorr;

    // Use a compressed image representation.
    int psiNumBytes;  // -1 (No encoding), 1 or 2
    int phiNumBytes;  // -1 (No encoding), 1 or 2

    // The bounds on which x and y pixels can be used
    // to start a search.
    int x_start_min;
    int x_start_max;
    int y_start_min;
    int y_start_max;

    // Provide debugging output.
    bool debug;
};

struct scaleParameters {
    float minVal;
    float maxVal;
    float scale;
};

// Search data on a per-image basis.
struct perImageData {
    int numImages = 0;

    float* imageTimes = nullptr;
    baryCorrection* baryCorrs = nullptr;

    scaleParameters* psiParams = nullptr;
    scaleParameters* phiParams = nullptr;
};

struct stampParameters {
    int radius = 10;
    StampType stamp_type = STAMP_SUM;
    bool do_filtering = false;

    // Thresholds on the location of the image peak.
    float center_thresh;
    float peak_offset_x;
    float peak_offset_y;

    // Limits on the moments.
    float m01_limit;
    float m10_limit;
    float m11_limit;
    float m02_limit;
    float m20_limit;
};

// Basic image moments use for analysis.
struct imageMoments {
    float m00;
    float m01;
    float m10;
    float m11;
    float m02;
    float m20;
};

} /* namespace search */

#endif /* COMMON_H_ */
