#pragma once
#include <type_traits>
#include "baseTypes.h"
#include <cstdint>

namespace SI {
inline constexpr auto gcd(int a, int b) {
  while (b != 0) {
    auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template <int32_t n, int32_t d> struct ratio {
  static constexpr int32_t num = n;
  static constexpr int32_t den = d;
};

template <template <int32_t, int32_t> typename R, std::intmax_t num,
          std::intmax_t den>
using ratio_reduce = R<num / gcd(num, den), den / gcd(num, den)>;
template <template <int32_t, int32_t> typename R, typename R1, typename R2>
using ratio_add =
    ratio_reduce<R, R1::num * R2::den + R2::num * R1::den, R1::den * R2::den>;
template <template <int32_t, int32_t> typename R, typename R1, typename R2>
using ratio_sub =
    ratio_reduce<R, R1::num * R2::den - R2::num * R1::den, R1::den * R2::den>;
template <template <int32_t, int32_t> typename R, typename R1, typename R2>
using ratio_multiply = ratio_reduce<R, R1::num * R2::num, R1::den * R2::den>;
template <template <int32_t, int32_t> typename R, typename R1, typename R2>
using ratio_divide = ratio_reduce<R, R1::num / R2::num, R1::den / R2::den>;

template <typename R1, typename R2>
struct ratio_equal
    : public std::integral_constant<bool, R1::num == R2::num &&
                                              R1::den == R2::den> {};
template <typename R1, typename R2>
struct ratio_not_equal
    : public std::integral_constant<bool, !ratio_equal<R1, R2>::value> {};
template <typename R1, typename R2>
struct ratio_less
    : public std::integral_constant<bool,
                                    (R1::num * R2::den < R2::num * R1::den)> {};
template <typename R1, typename R2>
struct ratio_greater
    : public std::integral_constant<bool,
                                    (R1::num * R2::den > R2::num * R1::den)> {};
template <typename R1, typename R2>
struct ratio_less_equal
    : public std::integral_constant<bool, ratio_less<R1, R2>::value ||
                                              ratio_equal<R1, R2>::value> {};
template <typename R1, typename R2>
struct ratio_greater_equal
    : public std::integral_constant<bool, ratio_greater<R1, R2>::value ||
                                              ratio_equal<R1, R2>::value> {};

namespace ops {
#define _action_ty                                                             \
  template <template <int32_t, int32_t> typename, typename, typename> typename

template <_action_ty action, typename T, typename U>
using applyUnits =
    SI::unit_ty<action<SI::Base::m, typename T::_m, typename U::_m>,
                action<SI::Base::kg, typename T::_kg, typename U::_kg>,
                action<SI::Base::s, typename T::_s, typename U::_s>,
                action<SI::Base::A, typename T::_A, typename U::_A>,
                action<SI::Base::K, typename T::_K, typename U::_K>,
                action<SI::Base::mol, typename T::_mol, typename U::_mol>,
                action<SI::Base::cd, typename T::_cd, typename U::_cd>>;

template <_action_ty action, typename T, typename U>
using applyRatios = SI::unit_ty<action<SI::Base::m, typename T::_m, U>,
                                action<SI::Base::kg, typename T::_kg, U>,
                                action<SI::Base::s, typename T::_s, U>,
                                action<SI::Base::A, typename T::_A, U>,
                                action<SI::Base::K, typename T::_K, U>,
                                action<SI::Base::mol, typename T::_mol, U>,
                                action<SI::Base::cd, typename T::_cd, U>>;
// Variadic helper for applying multiple units at once
template <_action_ty action, typename T, typename... Us> struct var_applyUnits;
template <_action_ty action, typename T> struct var_applyUnits<action, T> {
  using type = T;
};
template <_action_ty action, typename T, typename U>
struct var_applyUnits<action, T, U> {
  using type = applyUnits<action, T, U>;
};
template <_action_ty action, typename T, typename U, typename... Us>
struct var_applyUnits<action, T, U, Us...> {
  using type =
      typename var_applyUnits<action, applyUnits<action, T, U>,
                              Us...>::type;
};
// Variadic hellper for applying multiple ratios at once
template <_action_ty action, typename T, typename... Us> struct var_applyRatios;
template <_action_ty action, typename T> struct var_applyRatios<action, T> {
  using type = T;
};
template <_action_ty action, typename T, typename U>
struct var_applyRatios<action, T, U> {
  using type = applyRatios<action, T, U>;
};
template <_action_ty action, typename T, typename U, typename... Us>
struct var_applyRatios<action, T, U, Us...> {
  using type =
      typename var_applyRatios<action, applyRatios<action, T, U>,
                               Us...>::type;
};
} // namespace ops
template <typename... Ts>
using add_units = typename ops::var_applyUnits<ratio_add, Ts...>::type;
template <typename... Ts>
using sub_units = typename ops::var_applyUnits<ratio_sub, Ts...>::type;
template <typename... Ts>
using multiply_units =
    typename ops::var_applyUnits<ratio_multiply, Ts...>::type;
template <typename... Ts>
using divide_units = typename ops::var_applyUnits<ratio_divide, Ts...>::type;

template <typename... Ts>
using add_ratios = typename ops::var_applyRatios<ratio_add, Ts...>::type;
template <typename... Ts>
using sub_ratios = typename ops::var_applyRatios<ratio_sub, Ts...>::type;
template <typename... Ts>
using multiply_ratios =
    typename ops::var_applyRatios<ratio_multiply, Ts...>::type;
template <typename... Ts>
using divide_ratios = typename ops::var_applyRatios<ratio_divide, Ts...>::type;

template<typename T, typename R>
using SI_Unit = multiply_ratios<T,R>;

} // namespace SI
template <int32_t n = 1, int32_t d = 1>
using ratio = SI::ratio<n,d>;