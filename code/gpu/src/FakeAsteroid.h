/*
 * FakeAsteroid.h
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#ifndef FAKEASTEROID_H_
#define FAKEASTEROID_H_

//#include <random>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <omp.h>
#include <fitsio.h>
#include "GeneratorPSF.h"

class FakeAsteroid
{
	public:
		FakeAsteroid();
		void createImage(float *image, int width, int height,
			float xpos, float ypox, psfMatrix psf, float asteroidLevel, 
				    float backgroundLevel, float backgroundSigma);
		
		float generateGaussianNoise(float mu, float sigma);
};

#endif /* FAKEASTEROID_H_ */
