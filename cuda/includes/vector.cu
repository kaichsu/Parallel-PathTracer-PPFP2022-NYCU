#ifndef VECTOR_H
#define VECTOR_H

#include <algorithm>
#include <cuda.h>

template <typename T> class vector {
public:
    T *arr;
    size_t capacity;
    size_t arr_size;

public:
    __device__ vector() {
        arr = new T[1];
        capacity = 1;
        arr_size = 0;
    }
    __device__ ~vector() {
        delete [] arr;
    }
    __device__ void push_back(T data) {
        if (arr_size == capacity) {
            capacity *= 2;
            T *new_arr = new T[capacity];
            for (int i = 0; i < arr_size; ++i)
                new_arr[i] = arr[i];
            delete [] arr;
            arr = new_arr;
        }
        arr[arr_size++] = data;
    }
    __device__ void clear() {
        arr_size = 0;
    }
    __device__ size_t size() const {
        return arr_size;
    }
    __device__ T& operator [](int idx) {
        return arr[idx];
    }
    __device__ T operator [](int idx) const {
        return arr[idx];
    }
};

#endif