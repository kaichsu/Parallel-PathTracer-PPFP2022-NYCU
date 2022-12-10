#include <iostream>
#include <string>

#include "../includes/rtweekend.hpp"
#include "../includes/hittable_list.hpp"
#include "../includes/sphere.hpp"
#include "../includes/color.hpp"
#include "../includes/vec3.hpp"
#include "../includes/ray.hpp"
#include "../includes/camera.hpp"
#include "../includes/material.hpp"
#include "../includes/scene.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image_write.h"

#include <omp.h>

color ray_color(const ray &r, const hittable &world, int depth)
{
    hit_record rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec)){
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color(0, 0, 0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{

      // Image
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int image_channels = 3;
    const int samples_per_pixel = 50;
    const int max_depth = 50;
    unsigned char *image_data = new unsigned char[image_height * image_width * image_channels];
    const std::string image_path = "./image.png";
    // World
    auto world = random_scene();

    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    
    int max_thread = omp_get_max_threads();
    std::cerr << "max_thread: " << max_thread << "\n";

    // split frame
    #pragma omp parallel for 
    for(int y = 0; y < image_height; ++y){
        // std::cerr << "\rScanlines remaining: " << y << '/' <<  image_height << std::flush;
        for(int x = 0; x < image_width; ++x){
            color pixel_color(0,0,0);
            // split ray
            // #pragma omp parallel for
            for(int s = 0; s < samples_per_pixel; ++s){
                auto u = (x + random_double()) / (image_width - 1);
                auto v = (y + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(image_height, image_width, image_channels, x, image_height - 1 - y, image_data, pixel_color, samples_per_pixel);
        }
    }
    stbi_write_png(image_path.c_str(), image_width, image_height, image_channels, image_data, 0);
    // std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    // for(int y = 0; y < image_height; ++y){
    //     for(int x = 0; x<image_width; ++x){
    //         for(int c=0; c<image_channels; ++c){
    //             std::cout << int(image_data[y * image_width * image_channels + x * image_width + c]) << ' ';
    //         }
    //         std::cout << "\n";
    //     }
    // }
    delete [] image_data;
}