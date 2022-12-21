#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "../includes/material.cu"
#include "../includes/rtweekend.cu"
#include "../includes/hittable_list.cu"
#include "../includes/sphere.cu"
#include "../includes/ray.cu"
#include "../includes/color.cu"
#include "../includes/vec3.cu"
#include "../includes/utils.cu"
#include "../includes/parseArg.h" 
#include "../includes/camera.cu"
#include "../includes/scene.cu"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image_write.h"

#define BLOCKSIZE 8

__device__ color ray_color(const ray& r, curandState* rand_state, hittable_list **world, int max_depth) {
    ray cur_ray = r;
    ray scattered;
    color attenuation(1.0f, 1.0f, 1.0f);
    color next_attuenuation;
    hit_record rec;
    const float inf = infinity;
    for(int i=0; i < max_depth; ++i){
        if ((*world)->hit(cur_ray, 0.001, inf, rec)) {
            if(rec.mat_ptr->scatter(cur_ray, rec, next_attuenuation, scattered, rand_state)){
                cur_ray = scattered;
                attenuation = attenuation * next_attuenuation;
            } else {
                break; 
            }
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            float t = 0.5*(unit_direction.y() + 1.0);
            color light((1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0));
            return attenuation * light;
        }    
    }

    // reach max depth, shadow
    return color(0.0f,0.0f,0.0f);    
}

__global__ void render(
        unsigned char *image,
        int height, int width,
        int samples,
        int maxDepth,
        vec3 origin,
        vec3 lower_left_corner,
        vec3 horizontal,
        vec3 vertical,
        curandState *rand_state,
        camera **camera,
        hittable_list **world
    ){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y >= height or x >= width) return;

    int index = y * width + x;
    curandState local_rand_state = rand_state[index];

    int image_index = (height -1 - y) * width * 3 + x * 3;
    color pixel_color(0, 0, 0);
    for(int i=0; i<samples; ++i){
        float u = float(x + random_float(&local_rand_state)) / (width-1);
        float v = float(y + random_float(&local_rand_state)) / (height-1);
        ray r = (*camera)->get_ray(u, v, rand_state);
        pixel_color += ray_color(r, &local_rand_state, world, maxDepth);

    }
    write_color(pixel_color, image_index, samples, image);
}

int main(int argc, char **argv) {
    // Image
    int width = 200;
    int height = 133;
    int samples = 50;
    int max_depth = 50;
    float aspect_ratio = static_cast<float>(width) / height;
    const char *path = "./image.png";
    int view = 1;
    int uselessthread = 1;
    if(parse_arg(argc, argv, height, width, samples, max_depth, uselessthread, view, &path) != 0)
        return;

    // Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;
    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);
    point3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    hittable_list **d_world;
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 1 * sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list *)));
    if(view == 1){
        create_fixed_world<<<1, 1>>>(d_world);
    }
    else if(view == 2){
        create_ground_metal_world<<<1, 1>>>(d_world);
    }
    checkCudaErrors(cudaGetLastError());


    // Allocate memory
    unsigned char *h_image = new unsigned char[height * width * 3];
    unsigned char *d_image;
    checkCudaErrors(cudaMalloc((void **)&d_image, height * width * 3 * sizeof(unsigned char)));

    // invoke kernel function
    dim3 blocks(width/BLOCKSIZE+1,height/BLOCKSIZE+1);
    dim3 threads(BLOCKSIZE,BLOCKSIZE);
    
    // preapare camera

    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera)));
    camera_init<<<1, 1>>>(d_camera, lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // prepare random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, width * height * sizeof(curandState)));
    rand_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    float time;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
    render<<<blocks, threads>>>(d_image, height, width, samples, max_depth, origin, lower_left_corner, horizontal, vertical, d_rand_state, d_camera, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors( cudaEventElapsedTime(&time, start, stop) );
    fprintf(stderr, "Work took %f seconds\n", time/1e3);
    checkCudaErrors(cudaMemcpy(h_image, d_image, height * width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());


    // write image file
    stbi_write_png(path, width, height, 3, h_image, 0);

    // release source
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    delete [] h_image;

    
}
