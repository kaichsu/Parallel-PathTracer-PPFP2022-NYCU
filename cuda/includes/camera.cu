#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.cu"
#include "ray.cu"
#include "vec3.cu"

class camera {
    public:
        __device__
        camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            float vfov, // vertical field-of-view in degrees
            float aspect_ratio,
            float aperture,
            float focus_dist
        ) {
            auto theta = degrees_to_radians(vfov);
            auto h = tanf(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperture / 2;
        }

        __device__
        ray get_ray(float s, float t, curandState *rand_state) const {
            vec3 rd = lens_radius * random_in_unit_disk(rand_state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(
                origin + offset,
                lower_left_corner + s*horizontal + t*vertical - origin - offset
            );
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
};

__global__ void camera_init(
        camera **cam, point3 lookfrom, point3 lookat, vec3 vup,
        float vfov, float aspect_ratio,
        float aperture, float focus_dist){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y != 0 or x != 0) return;
    *cam = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);
}

#endif