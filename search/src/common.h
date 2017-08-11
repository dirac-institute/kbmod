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
const unsigned short THREAD_DIM_X = 16;
const unsigned short THREAD_DIM_Y = 32;
const unsigned short RESULTS_PER_PIXEL = 12;
const float MASK_FLAG = -9999.99;

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
	unsigned short obsCount;
};

// Trajectory used for searching max-pooled images
struct dtraj {
	short ix;
	short iy;
	short fx;
	short fy;
	char depth;
	float likelihood;
};

} /* namespace kbmod */

#endif /* COMMON_H_ */
