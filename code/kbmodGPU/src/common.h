/*
 * common.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef COMMON_H_
#define COMMON_H_


#define THREAD_DIM_X 16
#define THREAD_DIM_Y 32
#define RESULTS_PER_PIXEL 12
#define MASK_FLAG -9999.99

/*
 * Data structure to represent a PSF kernel
 */
struct psfMatrix {
	float *kernel;
	int dim;
	float sum;
};

/*
 * Data structure to represent a trajectory
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
	//int itCount;
};


#endif /* COMMON_H_ */
