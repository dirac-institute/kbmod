/*
 ============================================================================
 Name        : KBMOD CUDA
 Author      : Peter Whidden
 Description :
 ============================================================================
 */

#include <ctime>
#include <list>
#include <iostream>
#include "common.h"
#include "KBMOSearch.h"

using namespace kbmod;

int main(int argc, char* argv[])
{

	//std::clock_t setupA = std::clock();

	/* Create instances of psf and object generators */

	int debug = 1;

	PointSpreadFunc psf(1.0);
	psf.printPSF(debug);

	std::list<std::string> f;

	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535827.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535837.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535847.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535857.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535867.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535877.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535887.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535897.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535907.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535917.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535927.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535937.fits");
	f.push_back("~/cuda-workspace/fraser/chip_7/CORR40535947.fits");

	ImageStack imStack(f, debug);
	imStack.applyMasterMask(0xFFFFFF, 6);
	imStack.applyMaskFlags(0x000000);

	/*
	imStack.saveSci("../output/sci");
	imStack.saveMask("../output/mask");
	imStack.saveVar("../output/var");
	*/

	KBMOSearch search(imStack, psf);

	search.imageSaveLocation("../output/");

	search.gpu(0.1, 1.0, 150.0, 350.0);

	//search.saveResults("../output/testResults2.dat", 0.1);



	/*
	LayeredImage img("~/cuda-workspace/fraser/chip_7/CORR40535917.fits");
	img.saveSci("../output");
	img.convolve(psf);
	img.saveSci("../output/psi");
	*/

	// Pixel modification test //
	/*
	RawImage img("~/cuda-workspace/fraser/chip_7/CORR40535837.fits");
	std::cout << "img.isLoaded == " << img.isLoaded() << "\n";
	std::cout << "Getting SDataRef...\n";
	float *dummy = img.getSDataRef();
	std::cout << "img.isLoaded == " << img.isLoaded() << "\n";

	for (unsigned p=0; p<imStack.getPPI(); ++p)
	{
		dummy[p] = p == 2000*2000+1000 ? 100.0 : 0.0;
	}

	img.saveSci("../output/");
	*/

}





