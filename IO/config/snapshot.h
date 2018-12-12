#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <utility/identifier.h>
#include <utility/cuda/memory.h>
#include <vector>

namespace IO {
namespace config {
struct SnapShot {
  virtual void save() = 0;
  virtual void load() = 0;
  virtual void clear() = 0;
}; 

template <typename Ty> struct UniformSnap : public SnapShot {
  using T = typename Ty::type;
  T *original_ptr = Ty::ptr;
  T lastValue = *Ty::ptr;

  virtual void save() override { lastValue = *original_ptr; }
  virtual void load() override { *original_ptr = lastValue; }
  virtual void clear() override{};
};

template <typename Ty> struct ArraySnap : public SnapShot {
  using T = typename Ty::type;
  T *original_ptr = nullptr;
  T *last_ptr = nullptr;
  size_t allocSize;
  virtual void load() override {
    original_ptr = Ty::ptr;
    if (last_ptr == nullptr) {
      return;
    }
    cuda::memcpy(original_ptr, last_ptr, allocSize, cudaMemcpyHostToDevice);
  }
  virtual void save() override {
    original_ptr = Ty::ptr;
    if (original_ptr == nullptr)
      return;
    if (last_ptr == nullptr) {
      last_ptr = (T *)malloc(Ty::alloc_size);
      allocSize = Ty::alloc_size;
    }
    cuda::memcpy(last_ptr, original_ptr, allocSize, cudaMemcpyDeviceToHost);
  }
  virtual void clear() override {
    original_ptr = nullptr;
    if (last_ptr != nullptr)
      free(last_ptr);
    last_ptr = nullptr;
    allocSize = 0;
  }
};

extern std::vector<SnapShot *> snaps;
} // namespace config
} // namespace IO
