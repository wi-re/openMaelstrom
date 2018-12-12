#pragma once
/*
This file provides a cache for linear access for arrays in CUDA.
It assumes that there is one thread per array element in order to load
the data. The cache works using dynamic shared memory which is globally
defined in sm_cache[]. The cache provides an offset member that can be
used to use multiple caches or for custom shared memory data before or
after the cache. The cache only provides read access for the data.

Example usage:

auto pos = cache_array(arrays.position);
auto vol = cache_array(arrays.volume, pos.offset);
auto p = pos[idx];

Where the kernel should be launched with caches<float4,float> as the
shared memory function.

This functionality can be disabled by defining NO_CACHE
*/

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef NO_CACHE
// Global variable needed for dynamic shared memory functionality
extern __shared__ float sm_cache[];

/*
Provides simple memory access using shared memory for memory local
accesses to speed up the overall computational performance.
*/
template <typename T> struct cache {
  T *__restrict__ const shared_ptr;       // Pointer to the shared memory with the correct offset
  const T *__restrict__ const global_ptr; // Pointer to the global memory that is being cached
  const int32_t blockId;                  // Contains blockIdx.x * blockDim.x
  const int32_t dim;                      // Contains blockDim.x or the length of the cache
  const int32_t offset; // Contains sizeof(T) / sizeof(float) * blockDim.x plus the old offset

  // The code needs to be separated for GPU and CPU code. Note that the code does not perform any
  // caching on the CPU side
#ifdef __CUDA_ARCH__
  /*
  The constructor should be called with the global Pointer as the first argument and the Offset into
  sm_cache The constructor also loads the data into shared memory and performs a __syncthreads()!
  */
  template <typename U>
  hostDevice cache(U gl, int32_t off = 0)
      : shared_ptr((T *)(sm_cache + off)), global_ptr((T *)gl), blockId(blockIdx.x * blockDim.x),
        dim(blockDim.x), offset(off + sizeof(T) / sizeof(float) * blockDim.x) {

    shared_ptr[threadIdx.x] = global_ptr[blockId + threadIdx.x];
    __syncthreads();
  }

  // Reloads the data from global memory and performs a __syncthreads()!
  hostDevice void reload() {
    shared_ptr[threadIdx.x] = global_ptr[blockId + threadIdx.x];
    __syncthreads();
  }

  // Access the underlaying data, can be called with the actual global index
  // the transformation to the shared memory address is done internally
  hostDevice const T &operator[](int32_t idx) const {
    if (idx >= blockId && idx - blockId < blockDim.x)
      return shared_ptr[idx - blockId];
    else
      return global_ptr[idx];
  }
#else
  // This code is used to provide a compiling version for CPU code.
  template <typename U>
  hostDevice cache(U gl, int32_t = 0)
      : shared_ptr(nullptr), global_ptr((T *)gl), blockId(0), dim(0), offset(0) {}

  hostDevice void reload() {}

  hostDevice const T &operator[](int32_t idx) const { return global_ptr[idx]; }
#endif
};
#endif

/*
        Helper function to avoid having to call the constructor with a
        template argument. In C++17 this could be avoided using deduction
        guidelines but CUDA does not support them yet.
*/
template <typename T> hostDevice auto cache_array(T *ptr, int32_t offset = 0) {
  return cache<typename std::decay<T>::type>(ptr, offset);
}
/*
                Different name to make it more readable
*/
template <typename T> hostDevice auto cached(T *ptr, int32_t offset = 0) {
  return cache<typename std::decay<T>::type>(ptr, offset);
}
/*
        Helper function to create a cache with a different type than
        the type of the calling pointer
*/
template <typename U, typename T = U> hostDevice auto unit_cache(T *ptr, int32_t offset = 0) {
  return cache<typename std::decay<U>::type>(ptr, offset);
}

/*
        variadic_sizeof() provides a constexpr function to calculate
        the sizeof a parameter pack. Contrary to sizeof...(Ts) this
        does not provide the length of the parameter pack but the
        sum of the sizeof(T) of every argument. Could be simplified
        using C++17 fold expressions which are not yet supported in
        CUDA. Relies on recursive template calls
*/
constexpr auto variadic_sizeof() { return 0; }

template <typename T> constexpr auto variadic_sizeof() { return sizeof(T); }

template <typename T, typename U, typename... Ts> constexpr auto variadic_sizeof() {
  return sizeof(T) + variadic_sizeof<U, Ts...>();
}

/*
        Provides a functor that when called returns the variadic_sizeof the
        underlying types multiplied with the argument. This format is required
        for cudaOccupancyMaxPotentialBlockSizeVariableSMem used in the kernel
        Launcher. This cannot be done in a lambda function locally as a lambda
        function cannot yet be a __device__ AND __host__ function.
*/
template <typename... Ts> struct caches {
  template <typename T> hostDevice auto operator()(T &&in) { return in * variadic_sizeof<Ts...>(); }
};

/*
        Template partial specialization for no templates, technically not necessary
        but required to properly work on some platforms.
*/
template <> struct caches<> {
  template <typename T> hostDevice auto operator()(T &&in) { return in * 0; }
};
