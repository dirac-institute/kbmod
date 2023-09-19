/*
 * common.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef COMMON_H_
#define COMMON_H_

#ifdef HAVE_CUDA
    constexpr bool HAVE_GPU = true;
#else
    constexpr bool HAVE_GPU = false;
#endif

constexpr float NO_DATA = -9999.0;
constexpr unsigned int MAX_KERNEL_RADIUS = 15;

// The position (in pixels) on an image.
struct pixelPos {
    float x;
    float y;
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

#endif /* COMMON_H_ */
