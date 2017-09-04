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

	PointSpreadFunc psf(1.0);
	//psf.printPSF();
	/*
	std::vector<std::string> f;

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
	*/

	std::vector<LayeredImage> imgs;
	for (int i=0; i<10; i++) {
		imgs.push_back(LayeredImage("test"+std::to_string(i), 1000, 1000, 10.0, 60.0, float(i)*0.1));
	}

	for (int i=0; i<10; i++) {
		imgs[i].addObject(350.0+float(i)*3, 771.0+float(i)*3.5, 105.0, psf);
	}

	ImageStack imStack(imgs);
	//imStack.saveImages("./");
	//imStack.applyMasterMask(0xFFFFFF, 6);
	//imStack.applyMaskFlags(0x000000, {});

	KBMOSearch search = KBMOSearch(imStack, psf);
	search.setDebug(true);
	//std::cout << search.squareSDF(1.0, 0.0, 0.0, 0.5, 0.5) << "\n";
	auto res = search.regionSearch(30.0, 35.0, 4.0, 8.0, 3);
	std::cout << "\nix: " << res[0].ix << " iy: " << res[0].iy
			<< " lh: " << res[0].likelihood << "\n";
	/*
	imStack.saveSci("../output/sci");
	imStack.saveMask("../output/mask");
	imStack.saveVar("../output/var");
	*/

	//KBMOSearch search(imStack, psf);

	//search.gpu(10, 10, 0.1, 1.0, 150.0, 350.0, 4);

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





