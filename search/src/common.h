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

const unsigned int MAX_KERNEL_RADIUS = 15;
const unsigned short CONV_THREAD_DIM = 32;
const unsigned short POOL_THREAD_DIM = 32;
enum pool_method {POOL_MIN, POOL_MAX};
const int REGION_RESOLUTION = 8;
//const short POOL_MIN = 0;
//const short POOL_MAX = 1;
const unsigned short THREAD_DIM_X = 16;
const unsigned short THREAD_DIM_Y = 32;
const unsigned short RESULTS_PER_PIXEL = 4;
const float NO_DATA = -9999.99;

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
