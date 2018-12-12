#pragma once
#include <utility/algorithm.h>

namespace algorithm {
	template <typename T, typename U>
	void stable_sort_by_key(launch_config cfg, int32_t elements, T *keys, U *values) {
		if (cfg == launch_config::host) {
			cuda::sync();
			thrust::stable_sort_by_key(thrust::host, keys, keys + elements, values);
		}
		else {
			thrust::stable_sort_by_key(thrust::device, keys, keys + elements, values);
		}
	}
	template <typename T, typename U> void stable_sort_by_key(int32_t elements, T *keys, U *values) {
		stable_sort_by_key(parameters::target{}, elements, keys, values);
	}

	template <typename T, typename U>
	void sort_by_key(launch_config cfg, int32_t elements, T *keys, U *values) {
		if (cfg == launch_config::host) {
			cuda::sync();
			thrust::sort_by_key(thrust::host, keys, keys + elements, values);
		}
		else {
			thrust::sort_by_key(thrust::device, keys, keys + elements, values);
		}
	}
	template <typename T, typename U> void sort_by_key(int32_t elements, T *keys, U *values) {
		sort_by_key(parameters::target{}, elements, keys, values);
	}
}