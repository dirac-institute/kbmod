/*
 * GeneratorPSF.h
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#ifndef GENERATORPSF_H_
#define GENERATORPSF_H_

#include <cmath>
#include <iostream>

struct psfMatrix {
	float *kernel;
	int dim;
};

class GeneratorPSF
{
	public:
		GeneratorPSF();
		psfMatrix createGaussian(float stdev);
		float printPSF(psfMatrix p, int debug);
	private:
		float gaussian(float x, float stdev);
};

#endif /* GENERATORPSF_H_ */
