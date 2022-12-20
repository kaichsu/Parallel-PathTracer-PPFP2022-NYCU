#ifndef RTWEEKEND_H
#define RTWEEKEND_H



#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

// Usings

using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__global__ void rand_init(int width, int height, curandState *rand_state){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= width || y >= height)
        return;
    int index = y * width + x;
    curand_init(1234, index, 0, &rand_state[index]);
}

__device__ __host__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ inline float random_float(curandState *rand_state) {
    // Returns a random real in [0,1).
    return 1. - curand_uniform(rand_state);
}

__device__ inline float random_float(float min, float max, curandState *rand_state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_float(rand_state);
}

__device__  __host__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}


// Common Headers
#include "ray.cu"
#include "vec3.cu"

#endif
