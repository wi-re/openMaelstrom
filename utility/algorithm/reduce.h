#pragma once
#include <utility/algorithm.h>

namespace algorithm {
	template <typename T, typename Func>
	T reduce(launch_config cfg, T *input, int32_t elements, T initial, Func fn) {
		if (cfg == launch_config::host) {
			cuda::sync();
			return thrust::reduce(thrust::host, input, input + elements, initial, fn);
		}
		else {
			return thrust::reduce(thrust::device, input, input + elements, initial, fn);
		}
	}
	template <typename T, typename U, typename Func>
	value_unit<T, U> reduce(launch_config cfg, value_unit<T, U> *input, int32_t elements,
		value_unit<T, U> initial, Func fn) {
		return value_unit<T, U>{reduce(cfg, (T *)input, elements, initial.val, fn)};
	}

#define REDUCTION_WRAP(name, init, function)                                                       \
  template <typename T> T name(launch_config cfg, T *input, int32_t n) {                           \
    return reduce(cfg, input, n, init, function);                                                  \
  }                                                                                                \
  template <typename T, typename U>                                                                \
  value_unit<T, U> name(launch_config cfg, value_unit<T, U> *input, int32_t n) {                   \
    return value_unit<T, U>{name(cfg, (T *)input, n)};                                             \
  }                                                                                                \
  template <typename T> T name(T *input, int32_t n) { return name(parameters::target{}, input, n); }
	REDUCTION_WRAP(reduce_min, vec<T>::max(),
		[] hostDevice(const T &a, const T &b) { return math::min(a, b); });
	REDUCTION_WRAP(reduce_max, -vec<T>::max(),
		[] hostDevice(const T &a, const T &b) { return math::max(a, b); });
	REDUCTION_WRAP(reduce_sum, vec<T>::zero(), [] hostDevice(const T &a, const T &b) { return a + b; });
	// REDUCTION_WRAP(reduce_avg, vec<T>::zero(),
	//               [n] hostDevice(const T &a, const T &b) { return a + b / ((base_t<T>)n); });

}