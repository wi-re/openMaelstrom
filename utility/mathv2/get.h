#pragma once
#include <cstdint>
#include <utility/macro.h>
#include <utility/mathv2/detection.h>
#include <utility/mathv2/tools.h>
#include <utility/mathv2/vector_types.h>

namespace math {
/* This set of functions returns either the value passed directly or, if the
 * type is a unit type, it returns the .val member of the given argument. This
 * is used to uniformly access the value itself (without considering units) in
 * order to make writing uniform methods for both options easier. */
template <typename T, enable_ty((is_unit<T>))>
hostDeviceInline constexpr auto &&getValue(T &&lhs) {
  return lhs.val;
}
template <typename T, enable_ty((!is_unit<T>))>
hostDeviceInline constexpr auto &&getValue(T &&lhs) {
  return lhs;
}
/* This set of functions returns the kth component of an argument.
 * INDICES START AT ONE to match the result from math::dimension_v.
 * These functions return auto&& which binds to references if the given argument
 * was properly forwarded allowing for manipulation of vectors using get.
 * Additionally the get method is disabled (via enable_if) for invalid component
 * access at compile time. Accessing a scalar value (dimension = 0) is valid for
 * all dimensions to allow scalar op vector operations. The functions with a _s
 * suffix return a type less element, e.g. get_s<1>{float4{...}) will return a
 * float type value. The functions without the _s suffix return either the type
 * directly, as in the previous example and for unit based types return the unit
 * base type, e.g. get<1>(float4_u<SI::m>{}) will return a float_u<SI::m> type
 * value. For unit based types and the non suffix version references are mapped
 * to the inner type, e.g. value_unit<float&,SI::m> to allow for reference
 * semantics and for non unit based types simple references are returned via
 * auto&&. */
// Set of functions which always return non unit based types
template <uint32_t idx, typename T, enable_ty((dimu_eq<T, 0>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a));
}
template <uint32_t idx, typename T, enable_ty((idx == 0 && !dimu_l<T, 0>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a)).x;
}
template <uint32_t idx, typename T, enable_ty((idx == 1 && !dimu_l<T, 1>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a)).x;
}
template <uint32_t idx, typename T, enable_ty((idx == 2 && !dimu_lt<T, 2>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a)).y;
}
template <uint32_t idx, typename T, enable_ty((idx == 3 && !dimu_lt<T, 3>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a)).z;
}
template <uint32_t idx, typename T, enable_ty((idx == 4 && !dimu_lt<T, 4>))>
hostDeviceInline auto &&get_s(T &&a) {
  return getValue(std::forward<T>(a)).w;
}
// Set of functions which return units for unit based arguments
// Overload for unit based types
template <uint32_t idx, typename T, enable_ty((is_unit<T>))>
hostDeviceInline auto get(T &&arg) {
  using U = std::decay_t<decltype(get_s<idx, T>(std::forward<T>(arg)))>;
  return value_unit<U &, SI::unit_of<T>>{get_s<idx, T>(std::forward<T>(arg))};
}
// Overload for non unit based types
template <uint32_t idx, typename T, enable_ty((!is_unit<T>))>
hostDeviceInline auto &&get(T &&arg) {
  return get_s<idx, T>(std::forward<T>(arg));
}
template <uint32_t idx, typename T> hostDeviceInline auto unit_get(T arg) {
  return get<idx>(arg);
}
template <uint32_t idx, typename T, typename U> hostDeviceInline void unit_assign(T &arg, U && rhs) {
  get<idx>(std::forward<T>(arg)) = rhs;
}

/* Weak get works similar to get but returns values only instead of references.
 * The same distinction of _s and no suffix also exists here, however if an
 * invalid component is accessed, e.g. weak_get<4>(float3{...}) the code will
 * still compile and weak_get in these cases will return a 0 of the element type
 * of the given argument. This also works for unit based types where the result
 * will have zero value but the correct unit.*/
// Set of functions which always return non unit based types
// For valid accesses the call is forwarded to get_s
template <uint32_t idx, typename T,
          enable_ty((!(dimu_neq<T, 0> && dimu_lt<T, idx>)))>
hostDeviceInline auto weak_get_s(T &&arg) {
  return get_s<idx>(std::forward<T>(arg));
}
// For invalid accesses a zero value of the correct type is generated
template <uint32_t idx, typename T,
          enable_ty((dimu_neq<T, 0> && dimu_lt<T, idx>))>
hostDeviceInline auto weak_get_s(T &&a) {
  return decltype(std::decay_t<decltype(getValue(std::forward<T>(a)))>::x)(0);
}
// Set of functions which return units for unit based arguments
// Overload for unit based types
template <uint32_t idx, typename T, enable_ty((is_unit<T>))>
hostDeviceInline auto weak_get(T &&arg) {
  using U = std::decay_t<decltype(weak_get_s<idx, T>(std::forward<T>(arg)))>;
  return value_unit<U, SI::unit_of<std::decay_t<T>>>{
      weak_get_s<idx, T>(std::forward<T>(arg))};
}
// Overload for non unit based types
template <uint32_t idx, typename T, enable_ty((!is_unit<T>))>
hostDeviceInline auto weak_get(T &&arg) {
  return weak_get_s<idx, T>(std::forward<T>(arg));
}
/* This set of functions is used to help with certain functions where accessing
 * indices based on a runtime parameter might be beneficial. An example of this
 * would be implementing a cast to method or iterating over a vector without
 * requiring a manual unroll of the loop before compiling. These functions
 * follow the same convention of a _s and a no suffix version with behaviour as
 * usual.*/
template <typename T> hostDeviceInline auto weak_at_s(T &&lhs, int32_t idx) {
  if (idx == 1)
    return weak_get_s<1>(std::forward<T>(lhs));
  if (idx == 2)
    return weak_get_s<2>(std::forward<T>(lhs));
  if (idx == 3)
    return weak_get_s<3>(std::forward<T>(lhs));
  if (idx == 4)
    return weak_get_s<4>(std::forward<T>(lhs));
  return weak_get_s<0>(std::forward<T>(lhs));
}
template <typename T> hostDeviceInline auto weak_at(T &&lhs, int32_t idx) {
  using Ty = decltype(weak_get<0>(std::forward<T>(lhs)));
  return Ty{weak_at_s(std::forward<T>(lhs), idx)};
}
/* This set of templates is used to implement functions on types. The basic
 * template is fn which is given a function, e.g. a lambda, and a set of
 * parameters aas a templlate paramter pack. The function and parameter pack are
 * perfectly forwarded to fn_is and an integer sequence from 0 to d (exclusive)
 * is created where d is the dimension of the result of common operations based
 * on the return_types template. The fn_is template then further forwards the
 * argument template parameter pack and unpacks the integer sequence. The result
 * of this would be for calling fn with a float4 (called a1):
 * - fn(fun, a1) -> fn_is(fun, [0,1,2,3], a1)
 * - fn_is(fun, [0,1,2,3], a1) -> float4{ calls...}
 * - calls are fn_iss<I>(fun, a1) once per I in [0,1,2,3]
 * The construction of the return type allows reconstruction of the type from 4
 * "flat" values. The fn_iss_indexed function then calls fun with 1+I to give
 * the function access to the index (e.g. for cross products or casting) and
 * weak_get<I+1> on each argument. This allows for very powerful and expressive
 * methods that all rely on these three meta functions.*/
namespace fn_detail {
// Set for indexed weak calling with return value per component
template <int32_t I, typename C, typename... Ts>
hostDeviceInline auto fn_iss_indexed(C &&fun, Ts &&... args) {
  return fun(1 + I, weak_get<1 + I>(args)...);
}
template <int32_t... I, typename T, typename C, typename... Ts>
hostDeviceInline auto fn_is_indexed(C &&fun, std::integer_sequence<T, I...>,
                                    Ts &&... args) {
  return return_types<Ts...>{
      fn_iss_indexed<I>(fun, std::forward<Ts>(args)...)...};
}
// Set for non indexed weak calling with return value per component
template <int32_t I, typename C, typename... Ts>
hostDeviceInline auto fn_iss(C &&fun, Ts &&... args) {
  return fun(weak_get<1 + I>(args)...);
}
template <int32_t... I, typename T, typename C, typename... Ts>
hostDeviceInline auto fn_is(C &&fun, std::integer_sequence<T, I...>,
                            Ts &&... args) {
  return return_types<Ts...>{fn_iss<I>(fun, std::forward<Ts>(args)...)...};
}
// Set for non indexed strong calling with no return value
template <int32_t I, typename C, typename... Ts>
hostDeviceInline auto fn_iss_void(C &&fun, Ts &&... args) {
  fun(get<1 + I>(std::forward<Ts>(args))...);
}
template <int32_t... I, typename T, typename C, typename... Ts>
hostDeviceInline auto fn_is_void(C &&fun, std::integer_sequence<T, I...>,
                                 Ts &&... args) {
  int32_t dummy[] = {
      0, ((void)fn_iss_void<I>(fun, std::forward<Ts>(args)...), 0)...};
      [&dummy]{}();
}
// Set for non indexed weak calling with accumulator function
template <int32_t I, typename C, typename... Ts>
hostDeviceInline auto fn_iss_acc(C &&fun, Ts &&... args) {
  return fun(weak_get<1 + I>(std::forward<Ts>(args))...);
}
template <int32_t... I, typename T, typename C2, typename C, typename... Ts>
hostDeviceInline auto fn_is_acc(C2 &&fun_sum, C &&fun,
                                std::integer_sequence<T, I...>, Ts &&... args) {
  return fun_sum(fn_iss_acc<I>(fun, std::forward<Ts>(args)...)...);
}
} // namespace fn_detail
// Used for indexed functions
template <typename... Ts, typename C, typename ret_t = return_types<Ts...>>
hostDeviceInline auto fn_indexed(C &&fun, Ts &&... lhs) -> ret_t {
  constexpr auto result_dim = dimension_v<return_types<Ts...>>;
  return fn_detail::fn_is_indexed(
      std::forward<C>(fun),
      std::make_integer_sequence<int32_t, (result_dim == 0 ? 1 : result_dim)>{},
      std::forward<Ts>(lhs)...);
}
// This set does not add the index to the calls
template <typename... Ts, typename C, typename ret_t = return_types<Ts...>>
hostDeviceInline auto fn(C &&fun, Ts &&... lhs) -> ret_t {
  constexpr auto result_dim = dimension_v<return_types<Ts...>>;
  return fn_detail::fn_is(
      std::forward<C>(fun),
      std::make_integer_sequence<int32_t, (result_dim == 0 ? 1 : result_dim)>{},
      std::forward<Ts>(lhs)...);
}
// This set does not return a value, useful for certain operators due to this,
// this set also uses get instead of weak_get.
template <typename... Ts, typename C, typename ret_t = return_types<Ts...>>
hostDeviceInline auto fn_void(C &&fun, Ts &&... lhs) {
  constexpr auto result_dim = dimension_v<return_types<Ts...>>;
  fn_detail::fn_is_void(
      std::forward<C>(fun),
      std::make_integer_sequence<int32_t, (result_dim == 0 ? 1 : result_dim)>{},
      std::forward<Ts>(lhs)...);
}
// Set of functions that takes a second function which is called with all
// partial results.
template <typename... Ts, typename C2, typename C,
          typename ret_t = return_types<Ts...>>
hostDeviceInline auto fn_acc(C2 &&fun_sum, C &&fun, Ts &&... lhs) {
  constexpr auto result_dim = dimension_v<return_types<Ts...>>;
  return fn_detail::fn_is_acc(
      std::forward<C2>(fun_sum), std::forward<C>(fun),
      std::make_integer_sequence<int32_t, (result_dim == 0 ? 1 : result_dim)>{},
      std::forward<Ts>(lhs)...);
}
} // namespace math