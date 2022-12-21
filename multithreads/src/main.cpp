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
#include "../includes/parseArg.h"

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

int main(int argc, char** argv)
{

    // Image
    int image_width = 200;
    int image_height = 133;;
    int image_channels = 3;
    int samples_per_pixel = 50;
    int max_depth = 50;
    const char *image_path = "./image.png";
    int max_thread = 8;

    if(parse_arg(argc, argv, image_height, image_width, samples_per_pixel, max_depth, max_thread, &image_path) != 0) return -1;

    omp_set_num_threads(max_thread);

    const double aspect_ratio = static_cast<double>(image_width) / image_height;
    unsigned char *image_data = new unsigned char[image_height * image_width * image_channels];
    // World
    // auto world = random_scene();
    auto world = fixed_random_scene();

    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    

    std::cerr << "max_thread: " << omp_get_max_threads() << "\n";
    double start;
    double end;
    
    start = omp_get_wtime();
    // split frame
    #pragma omp parallel for
    for (int i = 0; i < image_height*image_width; i++) {
        int x = i % image_width;
        int y = i / image_width;
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
    end = omp_get_wtime();
    fprintf(stderr, "Work took %f seconds\n", end-start);
    stbi_write_png(image_path, image_width, image_height, image_channels, image_data, 0);
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