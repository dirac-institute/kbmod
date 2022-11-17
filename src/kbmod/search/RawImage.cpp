/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

namespace search {

// Performs convolution between an image represented as an array of floats
// and a PSF on a GPU device.
extern "C" void deviceConvolve(float* sourceImg, float* resultImg, int width, int height, float* psfKernel,
                               int psfSize, int psfDim, int psfRadius, float psfSum);

// Performs pixel pooling on an image represented as an array of floats.
// on a GPU device.
extern "C" void devicePool(int sourceWidth, int sourceHeight, float* source, int destWidth, int destHeight,
                           float* dest, char mode, bool two_sided);

// Grow the mask by expanding masked pixels to their neighbors
// out for "steps" steps.
extern "C" void deviceGrowMask(int width, int height, float* source, float* dest, int steps);

RawImage::RawImage() {
    initDimensions(0, 0);
    pixels = std::vector<float>();
}

RawImage::RawImage(unsigned w, unsigned h) : pixels(w * h) { initDimensions(w, h); }

RawImage::RawImage(unsigned w, unsigned h, const std::vector<float>& pix) : pixels(pix) {
    assert(w * h == pix.size());
    initDimensions(w, h);
}

#ifdef Py_PYTHON_H
RawImage::RawImage(pybind11::array_t<float> arr) { setArray(arr); }

void RawImage::setArray(pybind11::array_t<float>& arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2) throw std::runtime_error("Array must have 2 dimensions.");

    initDimensions(info.shape[1], info.shape[0]);
    float* pix = static_cast<float*>(info.ptr);

    pixels = std::vector<float>(pix, pix + pixelsPerImage);
}
#endif

void RawImage::initDimensions(unsigned w, unsigned h) {
    width = w;
    height = h;
    pixelsPerImage = w * h;
}

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
    fits_write_img(f, TFLOAT, 1, pixelsPerImage, pixels.data(), &status);
    fits_report_error(stderr, status);
    fits_close_file(f, &status);
    fits_report_error(stderr, status);
}

RawImage RawImage::createStamp(float x, float y, int radius, bool interpolate, bool keep_no_data) const {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");

    int dim = radius * 2 + 1;
    RawImage stamp(dim, dim);
    for (int xoff = 0; xoff < dim; ++xoff) {
        for (int yoff = 0; yoff < dim; ++yoff) {
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

void RawImage::convolve(PointSpreadFunc psf) {
    deviceConvolve(pixels.data(), pixels.data(), getWidth(), getHeight(), psf.kernelData(), psf.getSize(),
                   psf.getDim(), psf.getRadius(), psf.getSum());
}

RawImage RawImage::pool(short mode, bool two_sided) {
    // Half the dimensions, rounded up
    int pooledWidth = (getWidth() + 1) / 2;
    int pooledHeight = (getHeight() + 1) / 2;
    RawImage pooledImage = RawImage(pooledWidth, pooledHeight);
    devicePool(getWidth(), getHeight(), pixels.data(), pooledWidth, pooledHeight, pooledImage.getDataRef(),
               mode, two_sided);
    return pooledImage;
}

float RawImage::extremeInRegion(int lx, int ly, int hx, int hy, short pool_mode) {
    // Pool over the region of the image.
    float extreme = NO_DATA;
    for (int y = ly; y <= hy; ++y) {
        for (int x = lx; x <= hx; ++x) {
            float pix = getPixel(x, y);
            if (pix != NO_DATA) {
                if (extreme == NO_DATA)
                    extreme = pix;
                else if ((pool_mode == POOL_MAX) && (pix > extreme))
                    extreme = pix;
                else if ((pool_mode == POOL_MIN) && (pix < extreme))
                    extreme = pix;
            }
        }
    }
    return extreme;
}

void RawImage::applyMask(int flags, const std::vector<int>& exceptions, const RawImage& mask) {
    const std::vector<float>& maskPix = mask.getPixels();
    assert(pixelsPerImage == mask.getPPI());
    for (unsigned int p = 0; p < pixelsPerImage; ++p) {
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
void RawImage::growMask(int steps, bool on_gpu) {
    if (on_gpu) {
        deviceGrowMask(width, height, pixels.data(), pixels.data(), steps);
        return;
    }

    const int num_pixels = width * height;

    // Set up the initial masked vector that stores the number of steps
    // each pixel is from a masked pixel in the original image.
    std::vector<int> masked(num_pixels, -1);
    for (int i = 0; i < num_pixels; ++i) {
        if (pixels[i] == NO_DATA) masked[i] = 0;
    }

    // Grow out the mask one for each step.
    for (int itr = 1; itr <= steps; ++itr) {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
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

void RawImage::maskObject(float x, float y, const PointSpreadFunc& psf) {
    const std::vector<float>& k = psf.getKernel();
    // *2 to mask extra area, to be sure object is masked
    int dim = psf.getDim() * 2;
    float initialX = x - static_cast<float>(psf.getRadius() * 2);
    float initialY = y - static_cast<float>(psf.getRadius() * 2);
    // Does x/y order need to be flipped?
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            maskPixelInterp(initialX + static_cast<float>(i), initialY + static_cast<float>(j));
        }
    }
}

void RawImage::maskPixelInterp(float x, float y) {
    std::vector<float> iv = bilinearInterp(x, y);

    setPixel(iv[0], iv[1], NO_DATA);
    setPixel(iv[3], iv[4], NO_DATA);
    setPixel(iv[6], iv[7], NO_DATA);
    setPixel(iv[9], iv[10], NO_DATA);
}

void RawImage::addToPixel(float fx, float fy, float value) {
    assert(fx - floor(fx) == 0.0 && fy - floor(fy) == 0.0);
    int x = static_cast<int>(fx);
    int y = static_cast<int>(fy);
    if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] += value;
}

void RawImage::setPixel(int x, int y, float value) {
    if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] = value;
}

float RawImage::getPixel(int x, int y) const {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return pixels[y * width + x];
    } else {
        return NO_DATA;
    }
}

bool RawImage::pixelHasData(int x, int y) const {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return pixels[y * width + x] != NO_DATA;
    } else {
        return false;
    }
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

float* RawImage::getDataRef() { return pixels.data(); }

const std::vector<float>& RawImage::getPixels() const { return pixels; }

std::array<float, 2> RawImage::computeBounds() const {
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    for (unsigned p = 0; p < pixelsPerImage; ++p) {
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

RawImage createMedianImage(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].getWidth();
    int height = images[0].getHeight();
    for (auto& img : images) assert(img.getWidth() == width and img.getHeight() == height);

    RawImage result = RawImage(width, height);
    std::vector<float> pixArray(num_images);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int num_unmasked = 0;
            for (int i = 0; i < num_images; ++i) {
                // Only used the unmasked pixels.
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal != NO_DATA) && (!isnan(pixVal))) {
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
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            float sum = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal == NO_DATA) || (isnan(pixVal))) pixVal = 0.0;
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
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            float sum = 0.0;
            float count = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pixVal = images[i].getPixel(x, y);
                if ((pixVal != NO_DATA) && (!isnan(pixVal))) {
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
