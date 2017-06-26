/*
 * fitsutil.h
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef FITSUTIL_H_
#define FITSUTIL_H_

#include <fitsio.h>
#include <sstream>

void readFitsImg(const char *name, long pixelsPerImage, float *target)
{
	fitsfile *fptr;
	int nullval = 0;
	int anynull;
	int status = 0;

	if (fits_open_file(&fptr, name, READONLY, &status)) ffrprt(stderr, status);
	if (fits_read_img(fptr, TFLOAT, 1, pixelsPerImage,
			&nullval, target, &anynull, &status)) ffrprt(stderr, status);
	if (fits_close_file(fptr, &status)) ffrprt(stderr, status);

}

double readFitsMJD(const char *name)
{
	int status = 0;
        fitsfile *fptr;
	double time;
	if (fits_open_file(&fptr, name, READONLY,
			&status)) ffrprt(stderr, status);
	if (fits_read_key(fptr, TDOUBLE, "MJD", &time,
			NULL, &status)) ffrprt(stderr, status);
	if (fits_close_file(fptr, &status)) ffrprt(stderr, status);
	return time;
}

void writeFitsImg(const char *name, long *dimensions,
		          long pixelsPerImage, void *array)
{
	int status = 0;
	fitsfile *f;
    /* Create file with name */
	fits_create_file(&f, name, &status);

	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, array, &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void writeImageBatch(int imageCount, std::string psiPath, std::string phiPath,
					 int pixelsPerImage, float **psiImages, float **phiImages, long *dimensions)
{
	std::stringstream ss;
	for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
	{
		/* Create file name */
		ss << psiPath << "T";
		// Add leading zeros to filename
		if (writeIndex+1<100) ss << "0";
		if (writeIndex+1<10) ss << "0";
		ss << writeIndex+1 << "psi.fits";
		writeFitsImg(ss.str().c_str(), dimensions,
			pixelsPerImage, psiImages[writeIndex]);
		ss.str("");
		ss.clear();

		ss << phiPath << "T";
		if (writeIndex+1<100) ss << "0";
		if (writeIndex+1<10) ss << "0";
		ss << writeIndex+1 << "phi.fits";
		writeFitsImg(ss.str().c_str(), dimensions,
			pixelsPerImage, phiImages[writeIndex]);
		ss.str("");
		ss.clear();
	}
}


#endif /* FITSUTIL_H_ */
