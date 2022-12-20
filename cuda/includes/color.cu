#ifndef COLOR_H
#define COLOR_H

#include "vec3.cu"

__device__ inline void write_color(const color &pixel_color, const int& index, const int &samples_per_pixel, unsigned char *image) {
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();
    float scale = 1.0 / samples_per_pixel;
    r = sqrtf(scale * r);
    g = sqrtf(scale * g);
    b = sqrtf(scale * b);
    image[index] = static_cast<unsigned char>(256 * clamp(r, 0.0, 0.999));
    image[index + 1] = static_cast<unsigned char>(256 * clamp(g, 0.0, 0.999));
    image[index + 2] = static_cast<unsigned char>(256 * clamp(b, 0.0, 0.999));
}

#endif