#pragma once
#include <cstdint>
#include <utility/macro.h>
#include <utility/math/macro.h>
#include <utility/math/traits.h>
#include <vector_types.h>
namespace math {
namespace getter {
template <typename T>
using elem_ref_t = typename std::add_lvalue_reference<typename std::remove_reference<
    typename std::remove_cv<decltype(std::declval<T>().x)>::type>::type>::type;


template <uint32_t idx, typename T,
          typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type * = nullptr>
hostDeviceInline auto&& get(T &&a)  {
  return a;
}
template <uint32_t idx, typename T,
	std::enable_if_t<idx == 1 && !(dimension_v<T> < 1)>* = nullptr>
hostDeviceInline auto&& get(T &&a) {
  return a.x;
}
template <uint32_t idx, typename T,
	std::enable_if_t<idx == 2 && !(dimension_v<T> < 2)>* = nullptr>
hostDeviceInline auto&& get(T &&a) {
  return a.y;
}
template <uint32_t idx, typename T,
	std::enable_if_t<idx == 3 && !(dimension_v<T> < 3)>* = nullptr>
hostDeviceInline auto&& get(T &&a) {
  return a.z;
}
template <uint32_t idx, typename T,
	std::enable_if_t<idx == 4 && !(dimension_v<T> < 4)>* = nullptr>
hostDeviceInline auto&& get(T &&a) {
  return a.w;
}

template <uint32_t idx, typename T,
          typename std::enable_if<(dimension<T>::value != 0 && dimension<T>::value < idx),
                                  std::nullptr_t>::type * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a.x) {
  return decltype(T::x)(0);
}
template <uint32_t idx, typename T,
          typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a) {
  return a;
}
template <uint32_t idx, typename T, greaterDimension<0, T> * = nullptr,
	std::enable_if_t<idx == 1> * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a.x) {
  return a.x;
}
template <uint32_t idx, typename T, greaterDimension<1, T> * = nullptr,
	std::enable_if_t<idx == 2> * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a.y) {
  return a.y;
}
template <uint32_t idx, typename T, greaterDimension<2, T> * = nullptr,
	std::enable_if_t<idx == 3> * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a.z) {
  return a.z;
}
template <uint32_t idx, typename T, greaterDimension<3, T> * = nullptr,
	std::enable_if_t<idx == 4> * = nullptr>
hostDeviceInline auto weak_get(T a) -> decltype(a.w) {
  return a.w;
}
}; // namespace getter
using namespace getter;
namespace conversion {
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == 4), std::nullptr_t>::type * = nullptr>
hostDeviceInline T to(U &&a) {
  return T{static_cast<decltype(weak_get<1>(std::declval<T>()))>(weak_get<1>(a)),
           static_cast<decltype(weak_get<2>(std::declval<T>()))>(weak_get<2>(a)),
           static_cast<decltype(weak_get<3>(std::declval<T>()))>(weak_get<3>(a)),
           static_cast<decltype(weak_get<4>(std::declval<T>()))>(weak_get<4>(a))};
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == 3), std::nullptr_t>::type * = nullptr>
hostDeviceInline T to(U &&a) {
  return T{static_cast<decltype(weak_get<1>(std::declval<T>()))>(weak_get<1>(a)),
           static_cast<decltype(weak_get<2>(std::declval<T>()))>(weak_get<2>(a)),
           static_cast<decltype(weak_get<3>(std::declval<T>()))>(weak_get<3>(a))};
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == 2), std::nullptr_t>::type * = nullptr>
hostDeviceInline T to(U &&a) {
  return T{static_cast<decltype(weak_get<1>(std::declval<T>()))>(weak_get<1>(a)),
           static_cast<decltype(weak_get<2>(std::declval<T>()))>(weak_get<2>(a))};
}
template <typename T, typename U,
          typename std::enable_if<(dimension<T>::value == 1), std::nullptr_t>::type * = nullptr>
hostDeviceInline T to(U &&a) {
  return T{static_cast<decltype(weak_get<1>(std::declval<T>()))>(weak_get<1>(a))};
}
}; // namespace conversion
using namespace conversion;
namespace function_helper {
template <typename T, typename U, typename C, std::enable_if_t<dimension_compatible<T, U, 4>>* = nullptr>
hostDeviceInline return_type<T, U> fn(T lhs, U rhs, C fn) {
  return return_type<T, U>{
      fn(weak_get<1>(lhs), weak_get<1>(rhs)), fn(weak_get<2>(lhs), weak_get<2>(rhs)),
      fn(weak_get<3>(lhs), weak_get<3>(rhs)), fn(weak_get<4>(lhs), weak_get<4>(rhs))};
}
template <typename T, typename U, typename C, std::enable_if_t<dimension_compatible<T, U, 3>>* = nullptr>
hostDeviceInline return_type<T, U> fn(T lhs, U rhs, C fn) {
  return return_type<T, U>{fn(weak_get<1>(lhs), weak_get<1>(rhs)),
                           fn(weak_get<2>(lhs), weak_get<2>(rhs)),
                           fn(weak_get<3>(lhs), weak_get<3>(rhs))};
}
template <typename T, typename U, typename C, std::enable_if_t<dimension_compatible<T, U, 2>>* = nullptr>
hostDeviceInline return_type<T, U> fn(T lhs, U rhs, C fn) {
  return return_type<T, U>{fn(weak_get<1>(lhs), weak_get<1>(rhs)),
                           fn(weak_get<2>(lhs), weak_get<2>(rhs))};
}
template <typename T, typename U, typename C, std::enable_if_t<dimension_compatible<T, U, 1>>* = nullptr>
hostDeviceInline return_type<T, U> fn(T lhs, U rhs, C fn) {
  return return_type<T, U>{fn(weak_get<1>(lhs), weak_get<1>(rhs))};
}
template <typename T, typename U, typename C, std::enable_if_t<dimension_compatible<T, U, 0>>* = nullptr>
hostDeviceInline T fn(T lhs, U rhs, C fn) {
  return fn(weak_get<1>(lhs), weak_get<1>(rhs));
}

template <typename T, typename C, std::enable_if_t<dimension_v<T> == 4>* = nullptr>
hostDeviceInline T fn(T lhs, C fn) {
  return T{fn(weak_get<1>(lhs)), fn(weak_get<2>(lhs)), fn(weak_get<3>(lhs)), fn(weak_get<4>(lhs))};
}
template <typename T, typename C, std::enable_if_t<dimension_v<T> == 3>* = nullptr>
hostDeviceInline T fn(T lhs, C fn) {
  return T{fn(weak_get<1>(lhs)), fn(weak_get<2>(lhs)), fn(weak_get<3>(lhs))};
}
template <typename T, typename C, std::enable_if_t<dimension_v<T> == 2>* = nullptr>
hostDeviceInline T fn(T lhs, C fn) {
  return T{fn(weak_get<1>(lhs)), fn(weak_get<2>(lhs))};
}
template <typename T, typename C, std::enable_if_t<dimension_v<T> == 1>* = nullptr>
hostDeviceInline T fn(T lhs, C fn) {
  return T{fn(weak_get<1>(lhs))};
}
template <typename T, typename C, std::enable_if_t<dimension_v<T> == 0>* = nullptr>
hostDeviceInline T fn(T lhs, C fn) {
  return fn(weak_get<1>(lhs));
}
}; // namespace function_helper
using namespace function_helper;

template <typename T,
          typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type * = nullptr>
hostDeviceInline auto at(T &&a, int32_t idx) -> decltype(a) {
  return a;
}
template <typename T, std::enable_if_t<dimension<T>::value == 1> * = nullptr>
hostDeviceInline auto at(T &&a, int32_t idx) -> decltype(a.x) {
  if (idx == 0)
    return a.x;
  return 0.f;
}
template <typename T, std::enable_if_t<dimension<T>::value == 2> * = nullptr, typename = void>
hostDeviceInline auto at(T &&a, int32_t idx) -> decltype(a.x) {
  if (idx == 0)
    return a.x;
  if (idx == 1)
    return a.y;
  return 0.f;
}
template <typename T, std::enable_if_t<dimension<T>::value == 3> * = nullptr, typename = void,
          typename = void>
hostDeviceInline auto at(T &&a, int32_t idx) -> decltype(a.x) {
  if (idx == 0)
    return a.x;
  if (idx == 1)
    return a.y;
  if (idx == 2)
    return a.z;
  return 0.f;
}
template <typename T, std::enable_if_t<dimension<T>::value == 4> * = nullptr, typename = void,
          typename = void, typename = void>
hostDeviceInline auto at(T &&a, int32_t idx) -> decltype(a.x) {
  if (idx == 0)
    return a.x;
  if (idx == 1)
    return a.y;
  if (idx == 2)
    return a.z;
  if (idx == 3)
    return a.w;
  return 0.f;
}
} // namespace math
