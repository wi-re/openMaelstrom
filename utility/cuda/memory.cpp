#include <cuda.h>
#include <string>
#include <iostream>
#include <utility/cuda.h>

cudaError_t cuda::sync(std::string message) {
  if (parameters::target{} != launch_config::pure_host) {
    cudaError_t err = cudaDeviceSynchronize();
    error_check(err, message);
    return err;
  }
  return cudaSuccess;
}

cudaError_t cuda::sync_quiet(std::string message) {
	if (parameters::target{} != launch_config::pure_host) {
		cudaError_t err = cudaDeviceSynchronize();
		error_check_quiet(err, message);
		return err;
	}
	return cudaSuccess;
}

cudaError_t cuda::error_check(std::string message) {
  if (parameters::target{} != launch_config::pure_host) {
    cudaError_t err = cudaGetLastError();
    error_check(err, message);
    return err;
  }
  return cudaSuccess;
}
#include <iostream>

cudaError_t cuda::error_check(cudaError_t err, std::string message) {
  if (err != cudaSuccess) {
    std::string error_string = message + " failed with " + cudaGetErrorString(err);
    std::cerr << error_string << std::endl;
    throw std::runtime_error(error_string.c_str());
    return err;
  }
  return cudaSuccess;
}

cudaError_t cuda::error_check_quiet(std::string message) {
	if (parameters::target{} != launch_config::pure_host) {
		cudaError_t err = cudaGetLastError();
		error_check_quiet(err, message);
		return err;
	}
	return cudaSuccess;
}

cudaError_t cuda::error_check_quiet(cudaError_t err, std::string) {
	if (err != cudaSuccess) {
		return err;
	}
	return cudaSuccess;
}
void *cuda::malloc(size_t alloc_size) {
  launch_config cfg = parameters::target{};
  void *ptr = nullptr;
  switch (cfg) {
  case launch_config::pure_host:
    ptr = ::malloc(alloc_size);
    if (ptr == nullptr) {
      std::string message =
          "Failed to allocate memory on host " + std::to_string(alloc_size) + "B\n";
      throw std::runtime_error(message.c_str());
    }
    break;
  default:
	  error_check(cudaAllocateMemory(&ptr, alloc_size),
		  "Failed to allocate memory on device " + std::to_string(alloc_size) + "B");
  }
  return ptr;
}
