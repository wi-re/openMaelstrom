#pragma once
#include <cuda.h>
#include <driver_types.h>
#include <utility/cuda.h>
namespace cuda {
cudaError_t sync(std::string message = "Synchronizing device");
cudaError_t sync_quiet(std::string message = "Synchronizing device");
} // namespace cuda