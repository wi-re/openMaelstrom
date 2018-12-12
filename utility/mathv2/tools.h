#pragma once
#include "baseTypes.h"
#include "detection.h"
#include "ratio.h"
namespace SI {
namespace detail {
template <typename T, class Enable = void> struct unit_get;
template <typename T>
struct unit_get<
    T, std::enable_if_t<!has_def_unit<T> && has_def__unit_type_identifier<T>>> {
  using unit_t = typename std::decay_t<T>;
};
template <typename T>
struct unit_get<
    T, std::enable_if_t<has_def_unit<T> && !has_def__unit_type_identifier<T>>> {
  using unit_t = typename std::decay_t<T>::unit;
};
template <typename T>
struct unit_get<T, std::enable_if_t<!has_def_unit<T> &&
                                    !has_def__unit_type_identifier<T>>> {
  using unit_t = SI::unit_ty<>;
};
template <typename T, class Enable = void> struct type_of_s;
template <typename T> struct type_of_s<T, std::enable_if_t<is_unit<T>>> {
  using type = typename T::type;
};
template <typename T> struct type_of_s<T, std::enable_if_t<!is_unit<T>>> {
  using type = T;
};
} // namespace detail
template <typename T>
using unit_of = typename detail::unit_get<std::decay_t<T>>::unit_t;

template <typename T>
using type_of = typename detail::type_of_s<std::decay_t<T>>::type;

namespace detail {
template <typename T, typename U> struct is_same_ty {
  constexpr static bool const value =
      (is_unit<T> && is_unit<U> &&
       std::is_same<SI::unit_of<T>, SI::unit_of<U>>::value) ||
      (!is_unit<T> && is_unit<U> &&
       std::is_same<SI::unit_of<U>, SI::unit_ty<>>::value) ||
      (is_unit<T> && !is_unit<U> &&
       std::is_same<SI::unit_of<T>, SI::unit_ty<>>::value);
};
template <typename T, typename U> struct is_compatible_ty {
  constexpr static bool const value = SI::detail::is_same_ty<T, U>::value ||
                                      is_unit<T> == false ||
                                      is_unit<U> == false;
};

template <typename T, typename U> struct is_void_ty {
  constexpr static bool const value =
      std::is_same<SI::unit_of<T>, SI::unit_of<U>>::value;
};

template <template <typename, typename> typename C, typename T, typename... Us>
struct all_binary_fn;
template <template <typename, typename> typename C, typename T>
struct all_binary_fn<C, T> {
  constexpr static bool const value = true;
};
template <template <typename, typename> typename C, typename T, typename U>
struct all_binary_fn<C, T, U> {
  constexpr static bool const value =
      C<std::decay_t<T>, std::decay_t<U>>::value;
};
template <template <typename, typename> typename C, typename T, typename U,
          typename... Us>
struct all_binary_fn<C, T, U, Us...> {
  constexpr static bool const value =
      all_binary_fn<C, T, U>::value && all_binary_fn<C, T, Us...>::value;
};

template <template <typename> typename C, typename T, typename... Us>
struct all_unary_fn {
  constexpr static bool const value =
      C<T>::value && all_unary_fn<C, Us...>::value;
};
template <template <typename> typename C, typename T>
struct all_unary_fn<C, T> {
  constexpr static bool const value = C<T>::value;
};
} // namespace detail
using void_ty = value_unit<float, SI::unit_ty<>>;

template <typename... Us> using combine = typename SI::add_units<Us...>;
template <typename... Us> using derived_unit = typename SI::add_units<Us...>;
template <typename... Ts>
constexpr bool same_unit =
    detail::all_binary_fn<detail::is_same_ty, Ts...>::value;

template <typename... Ts>
constexpr bool void_unit =
    detail::all_binary_fn<detail::is_void_ty, void_ty, Ts...>::value;
template <typename... Ts> constexpr bool is_void = void_unit<Ts...>;

template <typename... Ts>
constexpr bool compatible_unit =
    detail::all_binary_fn<detail::is_compatible_ty, Ts...>::value;

template <typename T> constexpr bool void_u = is_void<unit_of<std::decay_t<T>>>;

template <typename Ty> struct is_not_unit_ty {
  static constexpr bool value = !is_unit<Ty>;
};
template <template <typename...> typename Fn, typename... Ts>
using unit_ops_ty = std::conditional_t<
    detail::all_unary_fn<is_not_unit_ty, Ts...>::value,
    std::decay_t<math::return_types<Ts...>>,
    value_unit<std::decay_t<type_of<math::return_types<Ts...>>>,
              Fn<unit_of<Ts>...>>>;
template <template <typename...> typename Fn, typename T, typename... Ts>
using ratio_ops_ty = std::conditional_t<
    !is_unit<T>, std::decay_t<T>,
                       value_unit<std::decay_t<type_of<T>>, Fn<unit_of<T>, Ts...>>>;

template <typename... Ts>
using add_units_ty = unit_ops_ty<SI::add_units, Ts...>;
template <typename... Ts>
using sub_units_ty = unit_ops_ty<SI::sub_units, Ts...>;
template <typename... Ts>
using multiply_units_ty = unit_ops_ty<SI::multiply_units, Ts...>;
template <typename... Ts>
using divide_units_ty = unit_ops_ty<SI::divide_units, Ts...>;

template <typename T, typename... Rs>
using add_ratios_ty = ratio_ops_ty<SI::add_ratios, T, Rs...>;
template <typename T, typename... Rs>
using sub_ratios_ty = ratio_ops_ty<SI::sub_ratios, T, Rs...>;
template <typename T, typename... Rs>
using multiply_ratios_ty = ratio_ops_ty<SI::multiply_ratios, T, Rs...>;
template <typename T, typename... Rs>
using divide_ratios_ty = ratio_ops_ty<SI::divide_ratios, T, Rs...>;

} // namespace SI

// template <typename T>
// constexpr bool void_unit3 = SI::same_unit<SI::void_ty, T>;