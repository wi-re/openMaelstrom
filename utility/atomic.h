#pragma once
template <typename T, typename... Rest> struct is_any : std::false_type {};

template <typename T, typename First> struct is_any<T, First> : std::is_same<T, First> {};

template <typename T, typename First, typename... Rest>
struct is_any<T, First, Rest...>
    : std::integral_constant<bool, std::is_same<T, First>::value || is_any<T, Rest...>::value> {};

#ifdef __CUDA_ARCH__
#define volatile__
#elif defined(_WIN32)
#define volatile__ volatile
#else
#define volatile__
#endif

#include <utility/atomic/linux_atomic.h>
#include <utility/atomic/windows_atomic.h>
#include <utility/atomic/cuda_atomic.h>
#include <utility/atomic/atomic_type.h>
