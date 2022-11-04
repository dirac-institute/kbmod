/*
 * PooledImage.cpp
 *
 * Created on: July 14, 2022
 */

#include "PooledImage.h"

namespace search {

PooledImage::PooledImage(const RawImage& org_image, int mode, bool pool_symmetric)
        : images(), pool_mode(mode), symmetric(pool_symmetric) {
    images.push_back(org_image);

    int last_ind = 0;
    while (images[last_ind].getPPI() > 1) {
        images.push_back(images[last_ind].pool(mode, symmetric));
        last_ind += 1;
    }
}

float PooledImage::getPixel(int depth, int x, int y) const { return images[depth].getPixel(x, y); }

float PooledImage::getMappedPixelAtDepth(int depth, int x, int y) const {
    int scale = 1 << depth;
    int mapped_x = x / scale;
    int mapped_y = y / scale;
    return images[depth].getPixel(mapped_x, mapped_y);
}

bool PooledImage::containsPixel(int depth, int x, int y, int pix_x, int pix_y) const {
    int scale = 1 << depth;
    int xs = x * scale;
    int xe = (x + 1) * scale - 1;
    int ys = y * scale;
    int ye = (y + 1) * scale - 1;
    return ((pix_x >= xs) && (pix_x <= xe) && (pix_y >= ys) && (pix_y <= ye));
}

std::array<float, 2> PooledImage::getPixelDistanceBounds(int depth, int x1, int y1, int x2, int y2) const {
    int scale = 1 << depth;

    float min_dx = (x1 == x2) ? 0 : scale * (fabs(x1 - x2) - 1);
    float max_dx = scale * (fabs(x1 - x2) + 1);

    float min_dy = (y1 == y2) ? 0 : scale * (fabs(y1 - y2) - 1);
    float max_dy = scale * (fabs(y1 - y2) + 1);

    std::array<float, 2> results;
    results[0] = sqrt(min_dx * min_dx + min_dy * min_dy);
    results[1] = sqrt(max_dx * max_dx + max_dy * max_dy);
    return results;
}

void PooledImage::repoolArea(float x, float y, float radius) {
    for (unsigned depth = 1; depth < images.size(); ++depth) {
        float scale = std::pow(2.0, static_cast<float>(depth));

        // The block to repool is +/- the width around (x, y).
        int minX = floor(static_cast<float>(x - radius) / scale);
        int maxX = ceil(static_cast<float>(x + radius) / scale);
        int minY = floor(static_cast<float>(y - radius) / scale);
        int maxY = ceil(static_cast<float>(y + radius) / scale);

        for (int px = minX; px <= maxX; ++px) {
            for (int py = minY; py <= maxY; ++py) {
                if (pool_mode == POOL_MAX) {
                    float value = -FLT_MAX;
                    float pixel = images[depth - 1].getPixel(px * 2, py * 2);
                    if (pixel != NO_DATA && pixel > value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2 + 1, py * 2);
                    if (pixel != NO_DATA && pixel > value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2, py * 2 + 1);
                    if (pixel != NO_DATA && pixel > value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2 + 1, py * 2 + 1);
                    if (pixel != NO_DATA && pixel > value) {
                        value = pixel;
                    }

                    if (value == -FLT_MAX) {
                        value = NO_DATA;
                    }
                    images[depth].setPixel(px, py, value);
                } else {
                    float value = FLT_MAX;
                    float pixel = images[depth - 1].getPixel(px * 2, py * 2);
                    if (pixel != NO_DATA && pixel < value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2 + 1, py * 2);
                    if (pixel != NO_DATA && pixel < value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2, py * 2 + 1);
                    if (pixel != NO_DATA && pixel < value) {
                        value = pixel;
                    }
                    pixel = images[depth - 1].getPixel(px * 2 + 1, py * 2 + 1);
                    if (pixel != NO_DATA && pixel < value) {
                        value = pixel;
                    }

                    if (value == FLT_MAX) {
                        value = NO_DATA;
                    }
                    images[depth].setPixel(px, py, value);
                }
            }
        }
    }
}

std::vector<PooledImage> PoolMultipleImages(const std::vector<RawImage>& imagesToPool, int mode,
                                            bool symmetric) {
    std::vector<PooledImage> destination;
    for (auto& i : imagesToPool) {
        destination.push_back(PooledImage(i, mode, symmetric));
    }
    return destination;
}

} /* namespace search */
