#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility/math/get.h>
#include <utility/math/functions.h>
#include <utility/math/operators.h>

#ifndef _WIN32
#include <float.h>
#include <limits.h>
#endif

template <typename T, uint32_t dim> struct vector_t;

#define TYPE_WRAP4(type_name, min_value, max_value, zero_value, one_value)                         \
  template <> struct vector_t<type_name, 4> {                                                      \
    using type = type_name##4;                                                                     \
    using base_type = type_name;                                                                   \
    static const uint32_t dim = 4;                                                                 \
    static constexpr type make(base_type a = zero_value, base_type b = zero_value,                 \
                               base_type c = zero_value, base_type d = zero_value) {               \
      return type{a, b, c, d};                                                                     \
    }                                                                                              \
    static constexpr type init(base_type a) { return make(a, a, a, a); }                           \
    static constexpr type min() { return make(min_value, min_value, min_value, min_value); }       \
    static constexpr type max() { return make(max_value, max_value, max_value, max_value); }       \
    static constexpr type one() { return make(one_value, one_value, one_value, one_value); }       \
    static constexpr type zero() { return make(zero_value, zero_value, zero_value, zero_value); }  \
  };
#define TYPE_WRAP3(type_name, min_value, max_value, zero_value, one_value)                         \
  template <> struct vector_t<type_name, 3> {                                                      \
    using type = type_name##3;                                                                     \
    using base_type = type_name;                                                                   \
    static const uint32_t dim = 3;                                                                 \
    static constexpr hostDeviceInline type make(base_type a = zero_value,                          \
                                                base_type b = zero_value,                          \
                                                base_type c = zero_value) {                        \
      return type{a, b, c};                                                                        \
    }                                                                                              \
    static constexpr hostDevice type init(base_type a) { return make(a, a, a); }                   \
    static constexpr hostDevice type min() { return make(min_value, min_value, min_value); }       \
    static constexpr hostDevice type max() { return make(max_value, max_value, max_value); }       \
    static constexpr hostDevice type one() { return make(one_value, one_value, one_value); }       \
    static constexpr hostDevice type zero() { return make(zero_value, zero_value, zero_value); }   \
  };
#define TYPE_WRAP2(type_name, min_value, max_value, zero_value, one_value)                         \
  template <> struct vector_t<type_name, 2> {                                                      \
    using type = type_name##2;                                                                     \
    using base_type = type_name;                                                                   \
    static const uint32_t dim = 2;                                                                 \
    static constexpr hostDevice type make(base_type a = zero_value, base_type b = zero_value) {    \
      return type{a, b};                                                                           \
    }                                                                                              \
    static constexpr hostDevice type init(base_type a) { return make(a, a); }                      \
    static constexpr hostDevice type min() { return make(min_value, min_value); }                  \
    static constexpr hostDevice type max() { return make(max_value, max_value); }                  \
    static constexpr hostDevice type one() { return make(one_value, one_value); }                  \
    static constexpr hostDevice type zero() { return make(zero_value, zero_value); }               \
  };
#define TYPE_WRAP1(type_name, min_value, max_value, zero_value, one_value)                         \
  template <> struct vector_t<type_name, 1> {                                                      \
    using type = type_name##1;                                                                     \
    using base_type = type_name;                                                                   \
    static const uint32_t dim = 1;                                                                 \
    static constexpr hostDevice type make(base_type a = zero_value) { return type{a}; }            \
    static constexpr hostDevice type init(base_type a) { return make(a); }                         \
    static constexpr hostDevice type min() { return make(min_value); }                             \
    static constexpr hostDevice type max() { return make(max_value); }                             \
    static constexpr hostDevice type one() { return make(one_value); }                             \
    static constexpr hostDevice type zero() { return make(zero_value); }                           \
  };
#define TYPE_WRAP0(type_name, min_value, max_value, zero_value, one_value)                         \
  template <> struct vector_t<type_name, 0> {                                                      \
    using type = type_name;                                                                        \
    using base_type = type_name;                                                                   \
    static const uint32_t dim = 0;                                                                 \
    static constexpr hostDevice type make(base_type a = zero_value) { return a; }                  \
    static constexpr hostDevice type init(base_type a) { return make(a); }                         \
    static constexpr hostDevice type min() { return make(min_value); }                             \
    static constexpr hostDevice type max() { return make(max_value); }                             \
    static constexpr hostDevice type one() { return make(one_value); }                             \
    static constexpr hostDevice type zero() { return make(zero_value); }                           \
  };
#define WRAP_TYPE(type_name, min_value, max_value, zero_value, one_value)                          \
  TYPE_WRAP0(type_name, min_value, max_value, zero_value, one_value);                              \
  TYPE_WRAP1(type_name, min_value, max_value, zero_value, one_value);                              \
  TYPE_WRAP2(type_name, min_value, max_value, zero_value, one_value);                              \
  TYPE_WRAP3(type_name, min_value, max_value, zero_value, one_value);                              \
  TYPE_WRAP4(type_name, min_value, max_value, zero_value, one_value);

typedef uint32_t uint;

WRAP_TYPE(float, -FLT_MAX, FLT_MAX, 0.f, 1.f);
WRAP_TYPE(double, -DBL_MAX, DBL_MAX, 0.0, 1.0);
WRAP_TYPE(uint, 0u, UINT_MAX, 0u, 1u);
WRAP_TYPE(int, INT_MIN, INT_MAX, 0, 1);

template <typename T>
using vec = vector_t<decltype(math::weak_get<1>(std::declval<T>())), math::dimension<T>::value>;

namespace math {
template <typename T, typename U, typename = std::enable_if_t<math::dimension<U>::value == 4>, typename = void,
          typename = void, typename = void, typename = void, typename = void>
hostDeviceInline auto castTo(U &&arg) {
  using type = decltype(std::declval<T>().x);
  return T{static_cast<type>(math::weak_get<1>(arg)), static_cast<type>(math::weak_get<2>(arg)),
           static_cast<type>(math::weak_get<3>(arg)), static_cast<type>(math::weak_get<4>(arg))};
}
template <typename T, typename U, typename = std::enable_if_t<math::dimension<U>::value == 3>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto castTo(U &&arg) {
  using type = decltype(std::declval<T>().x);
  return T{static_cast<type>(math::weak_get<1>(arg)), static_cast<type>(math::weak_get<2>(arg)),
           static_cast<type>(math::weak_get<3>(arg))};
}
template <typename T, typename U, typename = std::enable_if_t<math::dimension<U>::value == 2>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto castTo(U &&arg) {
  using type = decltype(std::declval<T>().x);
  return T{static_cast<type>(math::weak_get<1>(arg)), static_cast<type>(math::weak_get<2>(arg))};
}
template <typename T, typename U, typename = std::enable_if_t<math::dimension<U>::value == 1>, typename = void>
hostDeviceInline auto castTo(U &&arg) {
  using type = decltype(std::declval<T>().x);
  return T{static_cast<type>(math::weak_get<1>(arg))};
}
template <typename T, typename U, typename = std::enable_if_t<math::dimension<U>::value == 0>, typename = void, typename = void, typename = void, typename = void, typename = void, typename = void>
hostDeviceInline auto castTo(U &&arg) {
  using type = decltype(math::weak_get<1>(std::declval<T>()));
  return T{static_cast<type>(arg)};
}
template <typename T, typename U,
          typename = typename std::enable_if<math::dimension<U>::value == 0xDEADBEEF, T>::type>
hostDeviceInline auto castTo(U &&val) {
  return castTo<T>(val.val);
}
} // namespace math
