/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

namespace search {

#ifdef HAVE_CUDA
    // Performs convolution between an image represented as an array of floats
    // and a PSF on a GPU device.
    extern "C" void deviceConvolve(float* sourceImg, float* resultImg, int width, int height, float* psfKernel,
                                   int psfSize, int psfDim, int psfRadius, float psfSum);
#endif

RawImage::RawImage() : width(0), height(0) { pixels = std::vector<float>(); }

// Copy constructor
RawImage::RawImage(const RawImage& old) {
    width = old.getWidth();
    height = old.getHeight();
    pixels = old.getPixels();
}

// Copy assignment
RawImage& RawImage::operator=(const RawImage& source) {
    width = source.width;
    height = source.height;
    pixels = source.pixels;
    return *this;
}

// Move constructor
RawImage::RawImage(RawImage&& source)
        : width(source.width), height(source.height), pixels(std::move(source.pixels)) {}

// Move assignment
RawImage& RawImage::operator=(RawImage&& source) {
    if (this != &source) {
        width = source.width;
        height = source.height;
        pixels = std::move(source.pixels);
    }
    return *this;
}

RawImage::RawImage(unsigned w, unsigned h) : height(h), width(w), pixels(w * h) {}

RawImage::RawImage(unsigned w, unsigned h, const std::vector<float>& pix) : width(w), height(h), pixels(pix) {
    assert(w * h == pix.size());
}

#ifdef Py_PYTHON_H
RawImage::RawImage(pybind11::array_t<float> arr) { setArray(arr); }

void RawImage::setArray(pybind11::array_t<float>& arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2) throw std::runtime_error("Array must have 2 dimensions.");

    width = info.shape[1];
    height = info.shape[0];
    float* pix = static_cast<float*>(info.ptr);

    pixels = std::vector<float>(pix, pix + getPPI());
}
#endif

void RawImage::saveToFile(const std::string& path, bool append) {
    int status = 0;
    fitsfile* f;

    // Create a new file if append is false or we cannot open
    // the specified file.
    if (!append || fits_open_file(&f, path.c_str(), READWRITE, &status)) {
        fits_create_file(&f, (path).c_str(), &status);
        fits_report_error(stderr, status);
    }

    // This appends a layer (extension) if the file exists)
    /* Create the primary array image (32-bit float pixels) */
    long dimensions[2];
    dimensions[0] = width;
    dimensions[1] = height;
    fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
    fits_report_error(stderr, status);

    /* Write the array of floats to the image */
    fits_write_img(f, TFLOAT, 1, getPPI(), pixels.data(), &status);
    fits_report_error(stderr, status);
    fits_close_file(f, &status);
    fits_report_error(stderr, status);
}

RawImage RawImage::createStamp(float x, float y, int radius, bool interpolate, bool keep_no_data) const {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");

    int dim = radius * 2 + 1;
    RawImage stamp(dim, dim);
    for (int yoff = 0; yoff < dim; ++yoff) {
        for (int xoff = 0; xoff < dim; ++xoff) {
            float pixVal;
            if (interpolate)
                pixVal = getPixelInterp(x + static_cast<float>(xoff - radius),
                                        y + static_cast<float>(yoff - radius));
            else
                pixVal = getPixel(static_cast<int>(x) + xoff - radius, static_cast<int>(y) + yoff - radius);
            if ((pixVal == NO_DATA) && !keep_no_data) pixVal = 0.0;
            stamp.setPixel(xoff, yoff, pixVal);
        }
    }
    return stamp;
}

void RawImage::convolve_cpu(const PointSpreadFunc& psf) {
    std::vector<float> result(width * height, 0.0);
    const int psfRad = psf.getRadius();
    const float psfTotal = psf.getSum();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Pixels with NO_DATA remain NO_DATA.
            if (pixels[y * width + x] == NO_DATA) {
                result[y * width + x] = NO_DATA;
                continue;
            }

            float sum = 0.0;
            float psfPortion = 0.0;
            for (int j = -psfRad; j <= psfRad; j++) {
                for (int i = -psfRad; i <= psfRad; i++) {
                    if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
                        float currentPixel = pixels[(y + j) * width + (x + i)];
                        if (currentPixel != NO_DATA) {
                            float currentPSF = psf.getValue(j + psfRad, i + psfRad);
                            psfPortion += currentPSF;
                            sum += currentPixel * currentPSF;
                        }
                    }
                }
            }
            result[y * width + x] = (sum * psfTotal) / psfPortion;
        }
    }

    // Copy the data into the pixels vector.
    const int ppi = width * height;
    for(int i = 0; i < ppi; ++i) {
        pixels[i] = result[i];
    }
}

void RawImage::convolve(PointSpreadFunc psf) {
    #ifdef HAVE_CUDA
        deviceConvolve(pixels.data(), pixels.data(), getWidth(), getHeight(), psf.kernelData(),
                       psf.getSize(), psf.getDim(), psf.getRadius(), psf.getSum());
    #else
        convolve_cpu(psf);
    #endif
}

void RawImage::applyMask(int flags, const std::vector<int>& exceptions, const RawImage& mask) {
    const std::vector<float>& maskPix = mask.getPixels();
    const int num_pixels = getPPI();
    assert(num_pixels == mask.getPPI());
    for (unsigned int p = 0; p < num_pixels; ++p) {
        int pixFlags = static_cast<int>(maskPix[p]);
        bool isException = false;
        for (auto& e : exceptions) isException = isException || e == pixFlags;
        if (!isException && ((flags & pixFlags) != 0)) pixels[p] = NO_DATA;
    }
}

/* This implementation of growMask is optimized for steps > 1
   (which is how the code is generally used. If you are only
   growing the mask by 1, the extra copy will be a little slower.
*/
void RawImage::growMask(int steps) {
    const int num_pixels = width * height;

    // Set up the initial masked vector that stores the number of steps
    // each pixel is from a masked pixel in the original image.
    std::vector<int> masked(num_pixels, -1);
    for (int i = 0; i < num_pixels; ++i) {
        if (pixels[i] == NO_DATA) masked[i] = 0;
    }

    // Grow out the mask one for each step.
    for (int itr = 1; itr <= steps; ++itr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int center = width * y + x;
                if (masked[center] == -1) {
                    // Mask pixels that are adjacent to a pixel masked during
                    // the last iteration only.
                    if ((x + 1 < width && masked[center + 1] == itr - 1) ||
                        (x - 1 >= 0 && masked[center - 1] == itr - 1) ||
                        (y + 1 < height && masked[center + width] == itr - 1) ||
                        (y - 1 >= 0 && masked[center - width] == itr - 1)) {
                        masked[center] = itr;
                    }
                }
            }
        }
    }

    // Mask the pixels in the image.
    for (std::size_t i = 0; i < num_pixels; ++i) {
        if (masked[i] > -1) {
            pixels[i] = NO_DATA;
        }
    }
}

std::vector<float> RawImage::bilinearInterp(float x, float y) const {
    // Linear interpolation
    // Find the 4 pixels (aPix, bPix, cPix, dPix)
    // that the corners (a, b, c, d) of the
    // new pixel land in, and blend into those

    // Returns a vector with 4 pixel locations
    // and their interpolation value

    // Top right
    float ax = x + 0.5;
    float ay = y + 0.5;
    float aPx = floor(ax);
    float aPy = floor(ay);
    float aAmount = (ax - aPx) * (ay - aPy);

    // Bottom right
    float bx = x + 0.5;
    float by = y - 0.5;
    float bPx = floor(bx);
    float bPy = floor(by);
    float bAmount = (bx - bPx) * (bPy + 1.0 - by);

    // Bottom left
    float cx = x - 0.5;
    float cy = y - 0.5;
    float cPx = floor(cx);
    float cPy = floor(cy);
    float cAmount = (cPx + 1.0 - cx) * (cPy + 1.0 - cy);

    // Top left
    float dx = x - 0.5;
    float dy = y + 0.5;
    float dPx = floor(dx);
    float dPy = floor(dy);
    float dAmount = (dPx + 1.0 - dx) * (dy - dPy);

    // make sure the right amount has been distributed
    float diff = std::abs(aAmount + bAmount + cAmount + dAmount - 1.0);
    if (diff > 0.01) std::cout << "warning: bilinearInterpSum == " << diff << "\n";
    // assert(std::abs(aAmount+bAmount+cAmount+dAmount-1.0)<0.001);
    return {aPx, aPy, aAmount, bPx, bPy, bAmount, cPx, cPy, cAmount, dPx, dPy, dAmount};
}

void RawImage::addPixelInterp(float x, float y, float value) {
    // Interpolation values
    std::vector<float> iv = bilinearInterp(x, y);

    addToPixel(iv[0], iv[1], value * iv[2]);
    addToPixel(iv[3], iv[4], value * iv[5]);
    addToPixel(iv[6], iv[7], value * iv[8]);
    addToPixel(iv[9], iv[10], value * iv[11]);
}

void RawImage::addToPixel(float fx, float fy, float value) {
    assert(fx - floor(fx) == 0.0 && fy - floor(fy) == 0.0);
    int x = static_cast<int>(fx);
    int y = static_cast<int>(fy);
    if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] += value;
}

float RawImage::getPixelInterp(float x, float y) const {
    if ((x < 0.0 || y < 0.0) || (x > static_cast<float>(width) || y > static_cast<float>(height)))
        return NO_DATA;
    std::vector<float> iv = bilinearInterp(x, y);
    float a = getPixel(iv[0], iv[1]);
    float b = getPixel(iv[3], iv[4]);
    float c = getPixel(iv[6], iv[7]);
    float d = getPixel(iv[9], iv[10]);
    float interpSum = 0.0;
    float total = 0.0;
    if (a != NO_DATA) {
        interpSum += iv[2];
        total += a * iv[2];
    }
    if (b != NO_DATA) {
        interpSum += iv[5];
        total += b * iv[5];
    }
    if (c != NO_DATA) {
        interpSum += iv[8];
        total += c * iv[8];
    }
    if (d != NO_DATA) {
        interpSum += iv[11];
        total += d * iv[11];
    }
    if (interpSum == 0.0) {
        return NO_DATA;
    } else {
        return total / interpSum;
    }
}

void RawImage::setAllPix(float value) {
    for (auto& p : pixels) p = value;
}

std::array<float, 2> RawImage::computeBounds() const {
    const int num_pixels = getPPI();
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    for (unsigned p = 0; p < num_pixels; ++p) {
        if (pixels[p] != NO_DATA) {
            minVal = std::min(minVal, pixels[p]);
            maxVal = std::max(maxVal, pixels[p]);
        }
    }

    // Assert that we have seen at least some valid data.
    assert(maxVal != -FLT_MAX);
    assert(minVal != FLT_MAX);

    // Set and return the result array.
    std::array<float, 2> res;
    res[0] = minVal;
    res[1] = maxVal;
    return res;
}

// The maximum value of the image and return the coordinates.
pixelPos RawImage::findPeak(bool furthest_from_center) const {
    int c_x = width / 2;
    int c_y = height / 2;

    // Initialize the variables for tracking the peak's location.
    pixelPos result = {0, 0};
    float max_val = pixels[0];
    float dist2 = c_x * c_x + c_y * c_y;

    // Search each pixel for the peak.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pix_val = pixels[y * width + x];
            if (pix_val > max_val) {
                max_val = pix_val;
                result.x = x;
                result.y = y;
                dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
            } else if (pix_val == max_val) {
                int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
                if ((furthest_from_center && (new_dist2 > dist2)) ||
                    (!furthest_from_center && (new_dist2 < dist2))) {
                    max_val = pix_val;
                    result.x = x;
                    result.y = y;
                    dist2 = new_dist2;
                }
            }
        }
    }

    return result;
}

// Find the basic image moments in order to test if stamps have a gaussian shape.
// It computes the moments on the "normalized" image where the minimum
// value has been shifted to zero and the sum of all elements is 1.0.
// Elements with NO_DATA are treated as zero.
imageMoments RawImage::findCentralMoments() const {
    const int num_pixels = width * height;
    const int c_x = width / 2;
    const int c_y = height / 2;

    // Set all the moments to zero initially.
    imageMoments res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Find the min (non-NO_DATA) value to subtract off.
    float min_val = FLT_MAX;
    for (int p = 0; p < num_pixels; ++p) {
        min_val = ((pixels[p] != NO_DATA) && (pixels[p] < min_val)) ? pixels[p] : min_val;
    }

    // Find the sum of the zero-shifted (non-NO_DATA) pixels.
    double sum = 0.0;
    for (int p = 0; p < num_pixels; ++p) {
        sum += (pixels[p] != NO_DATA) ? (pixels[p] - min_val) : 0.0;
    }
    if (sum == 0.0) return res;

    // Compute the rest of the moments.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int ind = y * width + x;
            float pix_val = (pixels[ind] != NO_DATA) ? (pixels[ind] - min_val) / sum : 0.0;

            res.m00 += pix_val;
            res.m10 += (x - c_x) * pix_val;
            res.m20 += (x - c_x) * (x - c_x) * pix_val;
            res.m01 += (y - c_y) * pix_val;
            res.m02 += (y - c_y) * (y - c_y) * pix_val;
            res.m11 += (x - c_x) * (y - c_y) * pix_val;
        }
    }

    return res;
}

RawImage createMedianImage(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].getWidth();
    int height = images[0].getHeight();
    for (auto& img : images) assert(img.getWidth() == width and img.getHeight() == height);

    RawImage result = RawImage(width, height);
    std::vector<float> pixArray(num_images);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int num_unmasked = 0;
            for (int i = 0; i < num_images; ++i) {
                // Only used the unmasked pixels.
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal != NO_DATA) && (!std::isnan(pixVal))) {
                    pixArray[num_unmasked] = pixVal;
                    num_unmasked += 1;
                }
            }

            if (num_unmasked > 0) {
                std::sort(pixArray.begin(), pixArray.begin() + num_unmasked);

                // If we have an even number of elements, take the mean of the two
                // middle ones.
                int median_ind = num_unmasked / 2;
                if (num_unmasked % 2 == 0) {
                    float ave_middle = (pixArray[median_ind] + pixArray[median_ind - 1]) / 2.0;
                    result.setPixel(x, y, ave_middle);
                } else {
                    result.setPixel(x, y, pixArray[median_ind]);
                }
            } else {
                // We use a 0.0 value if there is no data to allow for visualization
                // and value based filtering.
                result.setPixel(x, y, 0.0);
            }
        }
    }

    return result;
}

RawImage createSummedImage(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].getWidth();
    int height = images[0].getHeight();
    for (auto& img : images) assert(img.getWidth() == width and img.getHeight() == height);

    RawImage result = RawImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal == NO_DATA) || (std::isnan(pixVal))) pixVal = 0.0;
                sum += pixVal;
            }
            result.setPixel(x, y, sum);
        }
    }

    return result;
}

RawImage createMeanImage(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].getWidth();
    int height = images[0].getHeight();
    for (auto& img : images) assert(img.getWidth() == width and img.getHeight() == height);

    RawImage result = RawImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            float count = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal != NO_DATA) && (!std::isnan(pixVal))) {
                    count += 1.0;
                    sum += pixVal;
                }
            }

            if (count > 0.0) {
                result.setPixel(x, y, sum / count);
            } else {
                // We use a 0.0 value if there is no data to allow for visualization
                // and value based filtering.
                result.setPixel(x, y, 0.0);
            }
        }
    }

    return result;
}

} /* namespace search */
