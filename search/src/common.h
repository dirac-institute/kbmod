/*
 * common.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef COMMON_H_
#define COMMON_H_

//#include "PointSpreadFunc.h"

namespace kbmod {

constexpr unsigned int MAX_KERNEL_RADIUS = 15;
constexpr unsigned short CONV_THREAD_DIM = 32;
constexpr unsigned short POOL_THREAD_DIM = 32;
enum pool_method {POOL_MIN, POOL_MAX};
constexpr int REGION_RESOLUTION = 4;
constexpr unsigned short THREAD_DIM_X = 256;
constexpr unsigned short THREAD_DIM_Y = 2;
constexpr unsigned short RESULTS_PER_PIXEL = 4;
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
