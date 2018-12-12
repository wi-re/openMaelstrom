#pragma once
#include <utility/algorithm.h>
#include <utility/identifier.h>

namespace algorithm {
	template <typename T, typename Func> auto find_if(launch_config cfg, T *input, int32_t n, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return thrust::find_if(thrust::host, input, input + n, fn) - input;
		}
		else {
			return thrust::find_if(thrust::device, input, input + n, fn) - input;
		}
	}
	template <typename T, typename Func> auto find_if(T *input, int32_t elements, Func fn) {
		return find_if(parameters::target{}, input, elements, fn);
	}

	template <typename T, typename Func>
	auto copy_if(launch_config cfg, T *input, T *output, int32_t n, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return thrust::copy_if(thrust::host, input, input + n, output, fn) - output;
		}
		else {
			return thrust::copy_if(thrust::device, input, input + n, output, fn) - output;
		}
	}
	template <typename T, typename Func> auto copy_if(T *input, T *output, int32_t elements, Func fn) {
		return copy_if(parameters::target{}, input, output, elements, fn);
	}

	template <typename T, typename Func>
	auto count_if(launch_config cfg, T *input, int32_t n, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return thrust::count_if(thrust::host, input, input + n, fn);
		}
		else {
			return thrust::count_if(thrust::device, input, input + n, fn);
		}
	}
	template <typename T, typename Func> auto count_if(T *input, int32_t elements, Func fn) {
		return count_if(parameters::target{}, input, elements, fn);
	}

	template <typename T, typename Func>
	void generate(launch_config cfg, T *input, int32_t n, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			thrust::generate(thrust::host, input, input + n, fn);
		}
		else {
			thrust::generate(thrust::device, input, input + n, fn);
		}
	}
	template <typename T, typename Func> void generate(T *input, int32_t elements, Func fn) {
		generate(parameters::target{}, input, elements, fn);
	}
}