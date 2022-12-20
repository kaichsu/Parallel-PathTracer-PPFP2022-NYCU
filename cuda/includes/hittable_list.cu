#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cu"
#include "vector.cu"


class hittable_list : public hittable {
    public:
        __device__
        hittable_list() {}

        __device__
        hittable_list(hittable* object) { add(object); }

        __device__ 
        void clear() { objects.clear(); }
        
        __device__ 
        void add(hittable* object) { objects.push_back(object); }

        __device__
        virtual bool hit(
            const ray& r, const float &t_min, const float &t_max, hit_record& rec) const override;

    public:
        vector<hittable*> objects;
};

__device__
bool hittable_list::hit(const ray& r, const float &t_min, const float& t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i=0; i<objects.size(); ++i){
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif