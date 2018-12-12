#pragma once

#include <utility/mathv2/baseTypes.h>
#include <utility/mathv2/nonesuch.h>
#include <utility/mathv2/ratio.h>

/** Helper macro to generate checks for using definitions within a type. This
 * HAS to be done at global namespace level as intellisense otherwise will not
 * recognize the templates and as such will lead to no proper syntax
 * highlighting. This macro emits:
 * - ty##_t, the actual detection check, is T::ty
 * - ty##_type_template, the detected_or_t idiom
 * - has_def_##ty, a constexpr bool containing the check result**/
#define HAS_DEF(ty)                                                                                \
  template <class T> using ty##_t = typename T::ty;                                                \
  template <class Ptr> using ty##_type_template = SI::detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_def_##ty =                                                                    \
      !std::is_same<ty##_type_template<std::decay_t<T>>, std::ptrdiff_t>::value;

/** Helper macro to generate checks for members within a type. This
 * HAS to be done at global namespace level as intellisense otherwise will not
 * recognize the templates and as such will lead to no proper syntax
 * highlighting. This macro emits:
 * - ty##_t, the actual detection check, is T::ty
 * - ty##_type_template, the detected_or_t idiom
 * - has_def_##ty, a constexpr bool containing the check result **/
#define HAS_MEM(ty)                                                                             \
  template <class T> using ty##_t = decltype(std::declval<T>().ty);                                \
  template <class Ptr> using ty##_type_template = SI::detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_mem_##ty = !std::is_same<ty##_type_template<T>, std::ptrdiff_t>::value;

// Check if a type contains a 'unit' alias
HAS_DEF(unit);
// Check if a type contains '_unit_type_identifier' used in SI_unit
HAS_DEF(_unit_type_identifier);
// Helper template alias to make has_def_unit more readable.
// NEEDS TO BE AT GLOBAL SCOPE FOR INTELLISENSE
template <typename T>
constexpr bool is_unit = !std::is_same<unit_type_template<std::decay_t<T>>, std::ptrdiff_t>::value;
// Helper template alias to make has_def__unit_type_identifier more readable.
// NEEDS TO BE AT GLOBAL SCOPE FOR INTELLISENSE
template <typename T>
constexpr bool is_unit_type =
    !std::is_same<_unit_type_identifier_type_template<std::decay_t<T>>, std::ptrdiff_t>::value;

template <typename T, typename Enable = void> struct val_type { using type = T; };
template <typename T> struct val_type<T, std::enable_if_t<is_unit<T>>> {
  using type = typename T::type;
};

namespace math {
namespace detail {
// Checks for x, y, z and w members for vector detection.
// These can be at non global namespace
HAS_MEM(x);
HAS_MEM(y);
HAS_MEM(z);
HAS_MEM(w);

// Compound checks for x, xy, xyz, xyzw member presence
template <typename T> constexpr bool has_x = has_mem_x<T>;
template <typename T> constexpr bool has_xy = has_mem_x<T> &&has_mem_y<T>;
template <typename T> constexpr bool has_xyz = has_mem_x<T> &&has_mem_y<T> &&has_mem_z<T>;
template <typename T>
constexpr bool has_xyzw = has_mem_x<T> &&has_mem_y<T> &&has_mem_z<T> &&has_mem_w<T>;

// Helper function which returns the dimension of a type. Returns 0 for scalar
// types(those which are either std::is_integral or std::is_floating_point)
// returns 0xDEADBEEF for non vector and non scalar types.
template <typename T> static constexpr uint32_t dimension_fn() {
  using Ty = std::decay_t<T>;
  if (std::is_integral<Ty>::value || std::is_floating_point<Ty>::value)
    return 0;
  else if (has_x<Ty> && !has_xy<Ty>)
    return 1;
  else if (has_xy<Ty> && !has_xyz<Ty>)
    return 2;
  else if (has_xyz<Ty> && !has_xyzw<Ty>)
    return 3;
  else if (has_xyzw<Ty>)
    return 4;
  else
    return 0xDEADBEEF;
}
// Wrapper struct for the dimension check, for backwards compatibility
template <typename T> struct dimension {
  static constexpr uint32_t value = dimension_fn<std::decay_t<T>>();
};
// Wrapper for the struct as calling dimension_fn() directly could be confusing
// as there is no const_eval to indicate that the function should only be
// evaluated at compile time and this might lead to misusage of the dimension
// check.
template <typename T, typename Ty = std::decay_t<T>>
constexpr uint32_t dimension_v = dimension<Ty>::value;
template <typename T, typename Ty = std::decay_t<T>>
constexpr uint32_t unit_dimension_v = dimension<typename val_type<Ty>::type>::value;
// Two types are dimensionally compatible with the given dimension if either
// dimension is equal to d and either the dimensions are equal or one (or both)
// types are scalar types. For example: dimension_compatible<float4,float,4>
// would be valid but dimension_compatible<float3,float,4> would not be.
template <typename T, typename U, uint32_t d>
constexpr bool dimension_compatible = (dimension_v<T> == d || dimension_v<U> == d) &&
                                      (dimension_v<T> == dimension_v<U> || dimension_v<U> == 0 ||
                                       dimension_v<T> == 0);
template <typename T, typename U, uint32_t d>
constexpr bool dim_comp = dimension_compatible<T, U, d>;
template <typename T, typename U, uint32_t d>
constexpr bool unit_dimension_compatible = (unit_dimension_v<T> == d || unit_dimension_v<U> == d) &&
                                           (unit_dimension_v<T> == unit_dimension_v<U> ||
                                            unit_dimension_v<U> == 0 || unit_dimension_v<T> == 0);
template <typename T, typename U, uint32_t d>
constexpr bool unit_dim_comp = unit_dimension_compatible<T, U, d>;
// Checks if the dimension of both arguments is equal to d
template <typename T, typename U, uint32_t d>
constexpr bool dimension_equal = (dimension_v<T> == d && dimension_v<U> == d);
template <typename T, typename U, uint32_t d> constexpr bool dim_equal = dimension_equal<T, U, d>;
// Determines the return type of a function for math. Here this is simply the
// type with the higher dimension. The result is only defined if both types are
// compatible with respect to the higher dimension.
template <typename T, typename U, uint32_t dT = unit_dimension_v<T>,
          uint32_t dU = unit_dimension_v<U>,
          enable_ty((unit_dim_comp < T, U, (dT < dU) ? dT : dU >))>
using return_type = std::conditional_t < dT<dU, std::decay_t<U>, std::decay_t<T>>;
// Determines the return type of a tuple of arguments. Only works if all
// arguments are compatible with the highest dimension.
template <typename T, typename... Us> struct return_types_s;
template <typename T> struct return_types_s<T> { using type = T; };
template <typename T, typename U> struct return_types_s<T, U> { using type = return_type<T, U>; };
template <typename T, typename U, typename... Us> struct return_types_s<T, U, Us...> {
  using type = typename return_types_s<typename return_types_s<T, U>::type, Us...>::type;
};
template <typename... Us> using return_types = std::decay_t<typename return_types_s<Us...>::type>;
}; // namespace detail
// yoink.
using namespace detail;
// Helper expressions for dimension comparisions
template <typename T, uint32_t d> constexpr bool dim_l = dimension_v<T> <= d;
template <typename T, uint32_t d> constexpr bool dim_g = dimension_v<T> >= d;
template <typename T, uint32_t d> constexpr bool dim_lt = dimension_v<T> < d;
template <typename T, uint32_t d> constexpr bool dim_gt = dimension_v<T>> d;
template <typename T, uint32_t d> constexpr bool dim_eq = dimension_v<T> == d;
template <typename T, uint32_t d> constexpr bool dim_neq = unit_dimension_v<T> != d;
template <typename T, uint32_t d> constexpr bool dimu_l = unit_dimension_v<T> <= d;
template <typename T, uint32_t d> constexpr bool dimu_g = unit_dimension_v<T> >= d;
template <typename T, uint32_t d> constexpr bool dimu_lt = unit_dimension_v<T> < d;
template <typename T, uint32_t d> constexpr bool dimu_gt = unit_dimension_v<T>> d;
template <typename T, uint32_t d> constexpr bool dimu_eq = unit_dimension_v<T> == d;
template <typename T, uint32_t d> constexpr bool dimu_neq = unit_dimension_v<T> != d;
template <typename T> constexpr uint32_t dim = dimension_v<T>;
template <typename T> constexpr uint32_t dim_u = unit_dimension_v<T>;
} // namespace math
