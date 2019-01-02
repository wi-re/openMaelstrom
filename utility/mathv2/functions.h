#pragma once
#include <cmath>
#include <cstddef>
#include <utility/mathv2/get.h>
#include <utility/mathv2/math_constants.h>
#include <utility/mathv2/operators.h>

/**Helper template to extract typenames wrapped in paranthesis due to macro
 * invocation. E.g. calling a macro with macro(ty<T1, T2>) would be interpreted
 * as two seperate macro arguments and calling with macro((ty<T1, T2>)) would
 * expand the argument with the surrounding paranthesis. Types within
 * paranthesis however are not interpreted as types (normally) so they are
 * converted to be a function type void((ty<T1, T2>)) where the language
 * requires the extra pair of paranthesis to be dropped. The function argument
 * type can then be extracted via the function syntax and returned.**/
template <typename T> struct argument_type;
template <typename T, typename U> struct argument_type<T(U)> { typedef U type; };
// Helper macros to help with a less verbose definition of functions using
// vectors and unit vectors. These macros are all UNDEFINED AT THE END OF THIS
// FILE so do not use them outside of this file as they are not accesible.
// The first set of macros are the actual function call wrappers to avoid
// having to write the same code blocks over and over (especially the forward)
#define func_1_i(f) fn_indexed([&](auto I, auto &&a) { f }, getValue(std::forward<T>(lhs)))
#define func_2_i(f)                                                                                  \
  fn_indexed([&](auto I, auto &&a, auto &&b) { f }, getValue(std::forward<T>(lhs)),                \
             getValue(std::forward<U>(rhs)))
#define func_3_i(f)                                                                                  \
  fn_indexed([&](auto I, auto &&a, auto &&b, auto &&c) { f }, getValue(std::forward<T>(lhs)),      \
             getValue(std::forward<U>(rhs)), getValue(std::forward<V>(ths)))
// Wraps macro within type conversion using the argument_type trick
#define fn_wrapper_i(fun, f, Uy)                                                                     \
  std::decay_t<typename argument_type<void Uy>::type> { fun(f) }
// Helper macros for common cases
#define fn_wrapper_1_U_i(f, Uy) fn_wrapper_i(func_1_i, f, Uy)
#define fn_wrapper_2_U_i(f, Uy) fn_wrapper_i(func_2_i, f, Uy)
#define fn_wrapper_3_U_i(f, Uy) fn_wrapper_i(func_3_i, f, Uy)
#define fn_wrapper_1_i(f) fn_wrapper_1_U_i(f, (T))
#define fn_wrapper_2_i(f) fn_wrapper_2_U_i(f, (return_types<T, U>))
#define fn_wrapper_3_i(f) fn_wrapper_3_U_i(f, (return_types<T, U, V>))

#define func_1(f) fn([&](auto &&a) { f }, getValue(std::forward<T>(lhs)))
#define func_2(f)                                                                                  \
  fn([&](auto &&a, auto &&b) { f }, getValue(std::forward<T>(lhs)),                \
             getValue(std::forward<U>(rhs)))
#define func_3(f)                                                                                  \
  fn([&](auto &&a, auto &&b, auto &&c) { f }, getValue(std::forward<T>(lhs)),      \
             getValue(std::forward<U>(rhs)), getValue(std::forward<V>(ths)))
// Wraps macro within type conversion using the argument_type trick
#define fn_wrapper(fun, f, Uy)                                                                     \
  std::decay_t<typename argument_type<void Uy>::type> { fun(f) }
// Helper macros for common cases
#define fn_wrapper_1_U(f, Uy) fn_wrapper(func_1, f, Uy)
#define fn_wrapper_2_U(f, Uy) fn_wrapper(func_2, f, Uy)
#define fn_wrapper_3_U(f, Uy) fn_wrapper(func_3, f, Uy)
#define fn_wrapper_1(f) fn_wrapper_1_U(f, (T))
#define fn_wrapper_2(f) fn_wrapper_2_U(f, (return_types<T, U>))
#define fn_wrapper_3(f) fn_wrapper_3_U(f, (return_types<T, U, V>))

namespace math {
/* Calculates floorf on all components, will always pick floorf regardless of
 * type, e.g. doubles will get converted to float values.*/
template <typename T> hostDeviceInline auto floorf(T &&lhs) {
  return fn_wrapper_1({ return ::floorf(a); });
}
/* Corresponding function for floorf*/
template <typename T> hostDeviceInline auto ceilf(T &&lhs) {
  return fn_wrapper_1({ return ::ceilf(a); });
}
/* Same as floorf but uses floor regardless of argument type */
template <typename T> hostDeviceInline auto floor(T &&value) {
  return fn(value, [](auto a) { return ::floor(a); });
}
/* Same as ceilf but uses ceil regardless of argument type */
template <typename T> hostDeviceInline auto ceil(T &&value) {
  return fn(value, [](auto a) { return ::ceil(a); });
}
/* Calculates log2 for every component of the argument. Requires the argument to
 * be of void unit (so either unit less or empty unit) as log2 cannot be applied
 * SI-Units.*/
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto log2(T &&lhs) {
  return fn_wrapper_1({ return ::log2(a); });
}
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto log2f(T &&lhs) {
  return fn_wrapper_1({ return ::log2f(a); });
}
/* tanf wrapper*/
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto atan(T &&lhs) {
  return fn_wrapper_1({ return ::atan(a); });
}
/* cosf wrapper*/
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto cosf(T &&lhs) {
  return fn_wrapper_1({ return ::cosf(a); });
}
/* sinf wrapper*/
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto sinf(T &&lhs) {
  return fn_wrapper_1({ return ::sinf(a); });
}
template <typename T, enable_ty(SI::void_u<T>)> hostDeviceInline auto expf(T &&lhs) {
  return fn_wrapper_1({ return ::expf(a); });
}
// Helper function to calculate the maximum of two values using ternary operator
template <typename T, typename U> hostDeviceInline auto max_impl(T &&lhs, U &&rhs) {
  return lhs > rhs ? lhs : rhs;
}
// Helper function to calculate the minimum of two values using ternary operator
template <typename T, typename U> hostDeviceInline auto min_impl(T &&lhs, U &&rhs) {
  return lhs < rhs ? lhs : rhs;
}
/* Function to calculate the maximum of two arguments. This function requires
 * either argument to be a scalar or both arguments to have the same dimension
 * as well as all units of all arguments to be compatible, e.g.: max(1.0m,
 * 0.25), max(float4_u<SI::m>{1,2,3,4}, 2.5m) and max(1,2) are all valid calls.
 * Calling the function with a vector will calculate the result for every
 * component of the vector and return a vector again.*/
template <typename T, typename U,
          enable_ty(((dimu_eq<T, 0> || dimu_eq<U, 0> ||
                      dim_u<T> == dim_u<U>)&&SI::compatible_unit<T, U>))>
hostDeviceInline auto max(T &&lhs, U &&rhs) {
  return fn_wrapper_2({ return max_impl(a, b); });
}
/* Analoguous to max but for minimum calculation*/
template <typename T, typename U,
          enable_ty(((dimu_eq<T, 0> || dimu_eq<U, 0> ||
                      dim_u<T> == dim_u<U>)&&SI::compatible_unit<T, U>))>
hostDeviceInline auto min(T &&lhs, U &&rhs) {
  return fn_wrapper_2({ return min_impl(a, b); });
}
/* Calculates the ::powf of the two arguments. The arguments need to follow the
 * same dimension requirements as max but all units need to be void (or empty)
 * as the argument to powf is runtime dependant and cannot change the type at
 * compille time. For non void (or empty) arguments see the other version of pow
 * instead.*/
template <typename T, typename U,
          enable_ty(((dimu_eq<T, 0> || dimu_eq<U, 0> ||
                      dim_u<T> == dim_u<U>)&&SI::void_unit<T, U>))>
hostDeviceInline auto pow(T &&lhs, U &&rhs) {
  return fn_wrapper_2({ return ::powf(a, b); });
}
/* pow function for generic types. For unit based types pow functions are only
 * possible with compile time constant powers (except if the unit is void). Thus
 * this version of pow is required that takes the power as a ratio-like type,
 * e.g. SI::ratio which contains a num(erator) and den(ominator) and calcualtes
 * the power as their fraction value as a float.*/
template <typename R, typename T> hostDeviceInline auto pow(T &&lhs) {
  // rhs and U are required for the fn_wrapper_2_U macro to work.
  auto rhs = ((float)R::num) / ((float)R::den);
  using U = decltype(rhs);
  return fn_wrapper_2_U({ return ::powf(a, b); }, (SI::multiply_ratios_ty<T, R>));
}
template <typename R, typename T> hostDeviceInline auto power(T &&lhs) {
  // rhs and U are required for the fn_wrapper_2_U macro to work.
  auto rhs = ((float)R::num) / ((float)R::den);
  using U = decltype(rhs);
  return fn_wrapper_2_U({ return ::powf(a, b); }, (SI::multiply_ratios_ty<T, R>));
}
template <int32_t R, typename T> hostDeviceInline auto power(T &&lhs) {
  // rhs and U are required for the fn_wrapper_2_U macro to work.
  auto rhs = ((float)R);
  using U = decltype(rhs);
  return fn_wrapper_2_U({ return ::powf(a, b); }, (SI::multiply_ratios_ty<T, SI::ratio<R,1>>));
}
/* Returns the sqrt of the given argument */
template <typename T> hostDeviceInline auto sqrt(T &&lhs) {
  return fn_wrapper_1_U({ return ::sqrtf(a); }, (SI::multiply_ratios_ty<T, SI::ratio<1, 2>>));
}
/* Returns the absolute value of a given argument. uses fabsf for all types*/
template <typename T> hostDeviceInline auto abs(T &&lhs) {
  return fn_wrapper_1({ return fabsf(a); });
}
/* Calculates the linear interpolation between lhs and rhs given alpha=ths. All
 * arguments need to be of compatible units and dimensions.
 * Calculates result = lhs + ths * (rhs - lhs) */
template <typename T, typename U, typename V,
          enable_ty(((dim_u<T> == dim_u<U> || dimu_eq<T, 0> ||
                      dimu_eq<U, 0>)&&SI::compatible_unit<T, U, V>))>
hostDeviceInline auto lerp(T &&lhs, U &&rhs, V &&ths) {
  return fn_wrapper_3({ return a + c * (b - a); });
}
/* Clamps a value between a minimum and maximum value. Calls max and min. All
 * arguments need to be compatible in units and dimensions for this to work. */
template <typename T, typename U, typename V,
          enable_ty(((dim_u<T> == dim_u<U> || dimu_eq<T, 0> ||
                      dimu_eq<U, 0>)&&SI::compatible_unit<T, U, V>))>
hostDeviceInline auto clamp(T &&value, U &&min_value, V &&max_value) {
  return min(max(value, min_value), max_value);
}
/* Returns a unit less value of the same dimension as the argument lhs which
 * contains the result of a sign function applied to every component. The sign
 * function here returns 1 for positive, -1 for negative and 0 for (exact) 0
 * values.*/
template <typename T> hostDeviceInline auto sign(T &&lhs) {
  return std::decay_t<SI::type_of<T>>{func_1({
    using return_t = std::decay_t<decltype(a)>;
    if (a == return_t(0))
      return return_t(0);
    else if (a > return_t(0))
      return return_t(1);
    else
      return return_t(-1);
  })};
}
/* Returns the maximum element in the given argument. Preserves units.*/
template <typename T> hostDeviceInline auto max_elem(T &&lhs) {
  return max(weak_get<1>(lhs), max(weak_get<2>(lhs), max(weak_get<3>(lhs), weak_get<4>(lhs))));
}
/* Returns the minimum element in the given argument. Preserves units.*/
template <typename T> hostDeviceInline auto min_elem(T &&lhs) {
  return min(weak_get<1>(lhs), min(weak_get<2>(lhs), min(weak_get<3>(lhs), weak_get<4>(lhs))));
}
/* Calculates the dot product of two arguments with the same dimensions. Dot
 * product here means a simple euclidean distance measure. */
template <typename T, typename U, enable_ty((dim_u<T> == dim_u<U>))>
hostDeviceInline auto dot(T &&lhs, U &&rhs) {
  auto res = weak_get<1>(lhs) * weak_get<1>(rhs);
  res += weak_get<2>(lhs) * weak_get<2>(rhs);
  res += weak_get<3>(lhs) * weak_get<3>(rhs);
  res += weak_get<4>(lhs) * weak_get<4>(rhs);
  return res;
}
/* Calculates a dot product on up to the first 3 components of the arguments,
 * useful for situations where the fourth dimension contains some value for
 * memory efficiency that is not part of the actual value.*/
template <typename T, typename U, enable_ty((dim_u<T> == dim_u<U>) || (dim_u<T> >= 3 && dim_u<U> >= 3))>
hostDeviceInline auto dot3(T &&lhs, U &&rhs) {
  auto res = weak_get<1>(lhs) * weak_get<1>(rhs);
  res += weak_get<2>(lhs) * weak_get<2>(rhs);
  res += weak_get<3>(lhs) * weak_get<3>(rhs);
  return res;
}
/* Length of a vector as the sqrt of the dot product with itself */
template <typename T> hostDeviceInline auto length(T &&value) { return sqrt(dot(value, value)); }
/* Square Length of a vector as the dot product with itself */
template <typename T> hostDeviceInline auto sqlength(T &&value) { return (dot(value, value)); }
/* Length of a vector as the sqrt of the dot3 product with itself */
template <typename T> hostDeviceInline auto length3(T &&value) { return sqrt(dot3(value, value)); }
/* Square Length of a vector as the dot3 product with itself */
template <typename T> hostDeviceInline auto sqlength3(T &&value) { return (dot3(value, value)); }
/* Normalizes a vector by dividing it with the length of itself.
 * DOES NOT CHECK FOR ZERO LENGTH!*/
template <typename T> hostDeviceInline auto normalize(T &&value) { return value / length(value); }
/* Normalizes a vector by dividing it with the length3 of itself.
 * DOES NOT CHECK FOR ZERO LENGTH! Applies to all dimensions. */
template <typename T> hostDeviceInline auto normalize3(T &&value) { return value / length3(value); }
/* Calculates the dot product of two vectors of atleast 3 dimensions. For
 * dimensions greater than 3 the resullt is simply the dot product as if the
 * arguments were 3 dimensional with all other components being default
 * initialized*/
template <typename T, typename U, enable_ty((dimu_gt<T, 2> && dimu_gt<U, 2>))>
hostDeviceInline auto cross(T &&lhs, U &&rhs) {
  return SI::add_units_ty<T, U>{get<2>(lhs) * get<3>(rhs) - get<3>(lhs) * get<2>(rhs),
                                get<3>(lhs) * get<1>(rhs) - get<1>(lhs) * get<3>(rhs),
                                get<1>(lhs) * get<2>(rhs) - get<2>(lhs) * get<1>(rhs)};
}
/* Calculates the length of the difference between lhs and rhs. */
template <typename T, typename U> hostDeviceInline auto distance(T &&lhs, U &&rhs) {
  return length(rhs - lhs);
}
/* Calculates the length3 of the difference between lhs and rhs. */
template <typename T, typename U> hostDeviceInline auto distance3(T &&lhs, U &&rhs) {
  return length3(rhs - lhs);
}
/* Calculates the square length of the difference between lhs and rhs. */
template <typename T, typename U> hostDeviceInline auto sqdistance(T &&lhs, U &&rhs) {
  return sqlength(rhs - lhs);
}
/* Calculates the square length3 of the difference between lhs and rhs. */
template <typename T, typename U> hostDeviceInline auto sqdistance3(T &&lhs, U &&rhs) {
  return sqlength3(rhs - lhs);
}
/* Calculates the reflected vector of lhs with respect to rhs */
template <typename T, typename U> hostDeviceInline auto reflect(T &&lhs, U &&rhs) {
  return lhs - 2.f * rhs * dot(rhs, lhs);
}
// Cast function, ignores unit compatibility!
template <typename T, typename U> hostDeviceInline auto castTo(U &&rhs, T lhs = T{}) {
  return fn_wrapper_1_i({ return static_cast<std::decay_t<decltype(a)>>(weak_at_s(rhs, I)); });
}
template <typename T, typename U> hostDeviceInline auto to(U &&rhs, T lhs = T{}) {
  return fn_wrapper_1_i({ return static_cast<std::decay_t<decltype(a)>>(weak_at_s(rhs, I)); });
}
template <typename T, typename U>
hostDeviceInline constexpr auto planeDistance(T &&plane, U &&point) {
  return math::dot3(plane, point) + (plane.val.w);
}
template <typename Func, typename T = float>
hostDeviceInline auto brentsMethod(Func f, T lower, T upper, T tol, int32_t max_iter) {
  using R = decltype(f(lower));
  T a = lower;
  T b = upper;
  R fa = f(a);
  R fb = f(b);
  R fs = 0.f;

  if (!(fa * fb < 0.f)) {
    return T{std::numeric_limits<float>::max()};
  }

  if (math::abs(fa) < math::abs(fb)) {
    auto t1 = a;
    a = b;
    b = t1;
    auto t2 = fa;
    fa = fb;
    fb = t2;
  }

  T c = a;
  R fc = fa;
  bool mflag = true;
  T s{std::numeric_limits<float>::max()};
  T d;

  for (int32_t iter = 1; iter < max_iter; ++iter) {

    if (math::abs(b - a) < tol) {
      return s;
    }

    if (fa != fc && fb != fc) {
      s = math::getValue((a * fb * fc / ((fa - fb) * (fa - fc))) + (b * fa * fc / ((fb - fa) * (fb - fc))) +
          (c * fa * fb / ((fc - fa) * (fc - fb))));
    } else {
      s = b - fb * (b - a) / (fb - fa);
    }

    if (((s < (3.f * a + b) * 0.25f) || (s > b)) ||
        (mflag && (math::abs(s - b) >= (math::abs(b - c) * 0.5f))) ||
        (!mflag && (math::abs(s - b) >= (math::abs(c - d) * 0.5f))) ||
        (mflag && (math::abs(b - c) < tol)) || (!mflag && (math::abs(c - d) < tol))) {
      s = (a + b) * 0.5f;
      mflag = true;
    } else {
      mflag = false;
    }

    fs = f(s);
    d = c;
    c = b;
    fc = fb;

    if (fa * fs < 0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }

    if (math::abs(fa) < math::abs(fb)) {
      auto t1 = a;
      a = b;
      b = t1;
      auto t2 = fa;
      fa = fb;
      fb = t2;
    }
  }

  // std::cout << "The solution does not converge or iterations are not sufficient" << std::endl;
  return s;
}

template <typename T> hostDeviceInline constexpr auto square(T a) { return a * a; }
template <typename T> hostDeviceInline constexpr auto cubic(T a) { return a * a * a; }

} // namespace math
#include <utility/mathv2/unit_types.h>
template <typename T, uint32_t d>
hostDeviceInline constexpr typename vector_t<T,d>::type vector_t<T, d>::make(base_type a) {
  return math::castTo<type>(a);
}

#undef func_1
#undef func_2
#undef func_3
#undef fn_wrapper
#undef fn_wrapper_1_U
#undef fn_wrapper_2_U
#undef fn_wrapper_3_U
#undef fn_wrapper_1
#undef fn_wrapper_2
#undef fn_wrapper_3