#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"
#include <iostream>

void write_color(const int &height, const int &width, const int &channels, int x, int y, unsigned char *data, color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    unsigned int uir = static_cast<unsigned int>(256 * clamp(r, 0.0, 0.999));
    unsigned int uig = static_cast<unsigned int>(256 * clamp(g, 0.0, 0.999));
    unsigned int uib = static_cast<unsigned int>(256 * clamp(b, 0.0, 0.999));

    data[y * width * channels + x * channels] = uir;
    data[y * width * channels + x * channels + 1] = uig;
    data[y * width * channels + x * channels + 2] = uib;


    // out << static_cast<unsigned int>(256 * clamp(r, 0.0, 0.999)) << ' '
    //     << static_cast<unsigned int>(256 * clamp(g, 0.0, 0.999)) << ' '
    //     << static_cast<unsigned int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif