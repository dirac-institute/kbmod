/*
 * common.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef COMMON_H_
#define COMMON_H_

namespace kbmod {

constexpr unsigned int MAX_KERNEL_RADIUS = 15;
constexpr unsigned short CONV_THREAD_DIM = 32;
constexpr unsigned short POOL_THREAD_DIM = 32;
enum pool_method {POOL_MIN, POOL_MAX};
constexpr int REGION_RESOLUTION = 4;
constexpr unsigned short THREAD_DIM_X = 128;
constexpr unsigned short THREAD_DIM_Y = 2;
constexpr unsigned short RESULTS_PER_PIXEL = 8;
constexpr float NO_DATA = -9999.0;
constexpr float FLAGGED = -9999.5;

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
	unsigned short  x;
	unsigned short  y;
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
    int minObservations;
    float minLH;

    // Parameters for sigmaG filtering on device.
    bool doFilter;
    float sGL_L;
    float sGL_H;
    float sigmaGCoeff;

    // Use a compressed image representation.
    bool encodeImg;
    float minPsiVal;
    float maxPsiVal;
    float psiScale;
    int psiNumBytes;  // 1 or 2

    float minPhiVal;
    float maxPhiVal;
    float phiScale;
    int phiNumBytes;  // 1 or 2
};

// Trajectory used for searching max-pooled images
struct trajRegion {
	float ix;
	float iy;
	float fx;
	float fy;
	short depth;
	short obs_count;
	float likelihood;
	float flux;
};

} /* namespace kbmod */

#endif /* COMMON_H_ */
