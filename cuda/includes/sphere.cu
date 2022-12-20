#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cu"
#include "vec3.cu"

class sphere : public hittable {
    public:
        __device__ sphere() {}
        __device__ sphere(point3 cen, float r, material* m)
            : center(cen), radius(r), mat_ptr(m) {};

        __device__ virtual bool hit(
            const ray& r, const float &t_min, const float &t_max, hit_record& rec) const override;

    public:
        point3 center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, const float &t_min, const float &t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    
    rec.normal = (rec.p - center) / radius;

    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif