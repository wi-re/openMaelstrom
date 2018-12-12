#pragma once
#include <utility/algorithm.h>

namespace algorithm {

	template <typename T, typename Func>
	T inclusive_scan(launch_config cfg, T *input, T *output, int32_t n, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return *thrust::inclusive_scan(thrust::host, input, input + n, output, fn);
		}
		else {
			return *thrust::inclusive_scan(thrust::device, input, input + n, output, fn);
		}
	}
	template <typename T, typename U, typename Func>
	value_unit<T, U> inclusive_scan(launch_config cfg, value_unit<T, U> *input,
		value_unit<T, U> *output, int32_t elements, Func fn) {
		return value_unit<T, U>{inclusive_scan(cfg, (T *)input, (T *)output, elements, fn)};
	}

	template <typename T, typename Func>
	T exclusive_scan(launch_config cfg, T *input, T *output, int32_t n, T initial, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return *thrust::exclusive_scan(thrust::host, input, input + n, output, initial, fn);
		}
		else {
			return *thrust::exclusive_scan(thrust::device, input, input + n, output, initial, fn);
		}
	}
	template <typename T, typename U, typename Func>
	value_unit<T, U> exclusive_scan(launch_config cfg, value_unit<T, U> *input,
		value_unit<T, U> *output, int32_t elements,
		value_unit<T, U> initial, Func fn) {
		return value_unit<T, U>{exclusive_scan(cfg, (T *)input, (T *)output, elements, initial.val, fn)};
	}

#define SCAN_WRAP(name, init, function)                                                            \
  template <typename T> T inclusive##name(launch_config cfg, T *input, T *output, int32_t n) {     \
    return inclusive_scan(cfg, input, output, n, function);                                        \
  }                                                                                                \
  template <typename T, typename U>                                                                \
  value_unit<T, U> inclusive##name(launch_config cfg, value_unit<T, U> *input,                     \
                                   value_unit<T, U> *output, int32_t n) {                          \
    return value_unit<T, U>{name(cfg, (T *)input, (T *)output, n)};                                \
  }                                                                                                \
  template <typename T> T inclusive##name(T *input, T *output, int32_t n) {                        \
    return name(parameters::target{}, input, output, n);                                           \
  }                                                                                                \
  template <typename T> T exclusive##name(launch_config cfg, T *input, T *output, int32_t n) {     \
    return exclusive_scan(cfg, input, output, n, init, function);                                  \
  }                                                                                                \
  template <typename T, typename U>                                                                \
  value_unit<T, U> exclusive##name(launch_config cfg, value_unit<T, U> *input,                     \
                                   value_unit<T, U> *output, int32_t n) {                          \
    return value_unit<T, U>{name(cfg, (T *)input, (T *)output, n)};                                \
  }                                                                                                \
  template <typename T> T exclusive##name(T *input, T *output, int32_t n) {                        \
    return name(parameters::target{}, input, output, n);                                           \
  }

	SCAN_WRAP(scan_min, vec<T>::max(),
		[] hostDevice(const T &a, const T &b) { return math::min(a, b); });
	SCAN_WRAP(scan_max, vec<T>::min(),
		[] hostDevice(const T &a, const T &b) { return math::max(a, b); });
	SCAN_WRAP(scan_sum, vec<T>::zero(), [] hostDevice(const T &a, const T &b) { return a + b; });


}