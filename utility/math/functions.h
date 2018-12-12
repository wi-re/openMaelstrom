#pragma once
#include <cmath>
#include <cstddef>
#include <math_constants.h>
#include <utility/math/get.h>
#include <utility/math/operators.h>

namespace math {
template <typename T> hostDeviceInline auto floorf(T&& lhs) {
  return fn(lhs, [](auto a) { return ::floorf(a); });
}
template <typename T> hostDeviceInline auto ceilf(T&& lhs) {
  return fn(lhs, [](auto a) { return ::ceilf(a); });
}
template <typename T> hostDeviceInline auto log2(T&& lhs) {
  return fn(lhs, [](auto a) { return ::log2f(a); });
}
/* tanf wrapper*/
template <typename T> hostDeviceInline auto atan(T&& lhs) {
  return fn(lhs, [](auto a) { return ::atan(a); });
}
/* cosf wrapper*/
template <typename T> hostDeviceInline auto cosf(T&& lhs) {
  return fn(lhs, [](auto a) { return ::cosf(a); });
}
/* sinf wrapper*/
template <typename T> hostDeviceInline auto sinf(T&& lhs) {
  return fn(lhs, [](auto a) { return ::sinf(a); });
}
template <typename T> hostDeviceInline auto expf(T&& lhs) {
  return fn(lhs, [](auto a) { return ::expf(a); });
}

template <typename T, typename U> hostDeviceInline auto max_impl(T&& lhs, U&& rhs) {
  return lhs > rhs ? lhs : rhs;
}
template <typename T, typename U> hostDeviceInline auto min_impl(T&& lhs, U&& rhs) {
  return lhs < rhs ? lhs : rhs;
}

template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) ||
                                      (dimension<T>::value == 0) || (dimension<U>::value == 0),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto max(T&& lhs, U&& rhs) {
  return fn(lhs, rhs,
            [](auto a,
				auto b) { return max_impl(a, b); });
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) ||
                                      (dimension<T>::value == 0) || (dimension<U>::value == 0),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto min(T&& lhs, U&& rhs) {
  return fn(lhs, rhs,
            [](auto a,
				auto b) { return min_impl(a, b); });
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) ||
                                      (dimension<T>::value == 0) || (dimension<U>::value == 0),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto pow(T&& lhs, U&& rhs) {
  return fn(lhs, rhs,
            [](auto a,
               auto b) { return powf(a, b); });
}
template <typename T> hostDeviceInline auto sqrt(T&& lhs) {
  return fn(lhs, [](auto a) { return sqrtf(a); });
}
template <typename T> hostDeviceInline auto abs(T&& lhs) {
  return fn(lhs, [](auto a) { return fabsf(a); });
}
template <typename T, typename U, typename V,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) ||
                                      (dimension<T>::value == 0) || (dimension<U>::value == 0),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto lerp(T&& lhs, U&& rhs, V&& t) {
  return fn(lhs, rhs,
            [t](auto a,
                auto b) { return a + t * (b - a); });
}
template <typename T, typename U, typename V,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) ||
                                      (dimension<T>::value == 0) || (dimension<U>::value == 0),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto clamp(T&& value, U&& min_value, V&& max_value) {
  return min(max(value, min_value), max_value);
}
template <typename T> hostDeviceInline auto sign(T&& value) {
  return fn(value, [](auto a) {
    using return_t = decltype(a);
    if (a == return_t(0))
      return return_t(0);
    else if (a > return_t(0))
      return return_t(1);
    else
      return return_t(-1);
  });
}
template <typename T> hostDeviceInline auto max_elem(T&& lhs) {
  auto res = max(weak_get<1>(lhs), max(weak_get<2>(lhs), max(weak_get<3>(lhs), weak_get<4>(lhs))));
  return res;
}
template <typename T> hostDeviceInline auto min_elem(T&& lhs) {
  auto res = min(weak_get<1>(lhs), min(weak_get<2>(lhs), min(weak_get<3>(lhs), weak_get<4>(lhs))));
  return res;
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto dot(T&& lhs, U&& rhs) {
  auto res = weak_get<1>(lhs) * weak_get<1>(rhs);
  res += weak_get<2>(lhs) * weak_get<2>(rhs);
  res += weak_get<3>(lhs) * weak_get<3>(rhs);
  res += weak_get<4>(lhs) * weak_get<4>(rhs);
  return res;
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value > 2 || dimension<U>::value > 2),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto dot3(T&& lhs, U&& rhs) {
  auto res = weak_get<1>(lhs) * weak_get<1>(rhs);
  res += weak_get<2>(lhs) * weak_get<2>(rhs);
  res += weak_get<3>(lhs) * weak_get<3>(rhs);
  return res;
}
template <typename T> hostDeviceInline auto length(T&& value) {
  return sqrtf(dot(value, value));
}
template <typename T> hostDeviceInline auto sqlength(T&& value) {
  return (dot(value, value));
}
template <typename T> hostDeviceInline auto length3(T&& value) {
  return sqrtf(dot3(value, value));
}
template <typename T> hostDeviceInline auto sqlength3(T&& value) {
  return (dot3(value, value));
}
template <typename T> hostDeviceInline auto normalize(T&& value) { return value / length(value); }
template <typename T> hostDeviceInline auto normalize3(T&& value) { return value / length3(value); }
template <typename T> hostDeviceInline auto floor(T&& value) {
  return fn(value, [](auto a) { return ::floor(a); });
}
template <typename T> hostDeviceInline auto ceil(T&& value) {
  return fn(value, [](auto a) { return ::ceil(a); });
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == dimension<U>::value) &&
                                      (dimension<T>::value > 2 || dimension<U>::value > 2),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto cross(T&& lhs, U&& rhs) {
  return return_type<T, U>{get<2>(lhs) * get<3>(rhs) - get<3>(lhs) * get<2>(rhs),
                           get<3>(lhs) * get<1>(rhs) - get<1>(lhs) * get<3>(rhs),
                           get<1>(lhs) * get<2>(rhs) - get<2>(lhs) * get<1>(rhs),
                           0.f};
}
template <typename T, typename U>
hostDeviceInline auto distance(T&& lhs, U&& rhs){
  return length(rhs - lhs);
}
template <typename T, typename U>
hostDeviceInline auto distance3(T&& lhs, U&& rhs){
  return length3(rhs - lhs);
}
template <typename T, typename U>
hostDeviceInline auto sqdistance(T&& lhs, U&& rhs){
  return sqlength(rhs - lhs);
}
template <typename T, typename U>
hostDeviceInline auto sqdistance3(T&& lhs, U&& rhs){
  return sqlength3(rhs - lhs);
}
template <typename T, typename U> hostDeviceInline auto reflect(T&& lhs, U&& rhs) {
  return lhs - 2.f * rhs * dot(rhs, lhs);
}

} // namespace math
