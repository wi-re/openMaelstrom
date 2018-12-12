#pragma once
#include <utility/cuda.h>
#include <utility/cuda/error_handling.h>

namespace cuda {
	void *malloc(size_t alloc_size);

	template <typename T> T *allocate(size_t size) { return (T *)cuda::malloc(size); }

	template <typename T, typename U>
	void memcpy(T *dst, U *src, size_t size, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
		launch_config cfg = parameters::target{};
		switch (cfg) {
		case launch_config::device: {
			cudaError_t err = cudaMemcpy(dst, src, size, kind);
			if (err != cudaSuccess) {
				std::stringstream sstream;
				sstream << "cudaMemcpy from " << src << " to " << dst << " of " << size << "B";
				error_check(err, sstream.str());
			}
			break;
		}
		default:
			::memcpy(dst, src, size);
			break;
		}
	}

	template <typename T> void Memset(T *ptr, uint32_t data, size_t size) {
		launch_config cfg = parameters::target{};
		switch (cfg) {
		case launch_config::device: {
			cudaError_t err = cudaMemset(ptr, data, size);
			if (err != cudaSuccess) {
				std::stringstream sstream;
				sstream << "cudaMemset of " << ptr << " to " << data << " of " << size << "B";
				error_check(err, sstream.str());
			}
			break;
		}
		default:
			::memset(ptr, data, size);
			break;
		}
	}

	template <typename T> void arrayMemset(uint32_t data) { Memset(T::ptr, data, T::alloc_size); }
}
