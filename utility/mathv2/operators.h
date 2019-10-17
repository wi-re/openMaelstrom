#pragma once
#include <utility/mathv2.h>
#include <utility/mathv2/get.h>
#include <iomanip>

#define OPERATOR_NEWER(op)                                                                         \
  template <typename T, typename U, enable_ty((op_compat<T, U>))>                                  \
  hostDeviceInline auto operator op(T &&a, U &&b) {                                                \
    return fn([](auto &&a, auto &&b) { return a op b; }, std::forward<T>(a), std::forward<U>(b));  \
  }                                                                                                \
  template <typename T, typename U, enable_ty((ass_compat<T, U>))>                                 \
  hostDeviceInline auto &operator op##=(T &a, U &&b) {                                             \
    fn_void([](auto &a, auto &&b) { a op## = b; }, a, std::forward<U>(b));                         \
    return a;                                                                                      \
  }

#define OPERATOR_NEWER2(op)                                                                        \
  template <typename T, typename U, enable_ty((op_dim<T, U, 1>))>                                  \
  hostDeviceInline auto operator op(T &&a, U &&b) {                                                \
    return return_types<T, U>{math::weak_get<1>(a) op math::weak_get<1>(b)};                       \
  }                                                                                                \
  template <typename T, typename U, enable_ty((op_dim<T, U, 2>))>                                  \
  hostDeviceInline auto operator op(T &&a, U &&b) {                                                \
    return return_types<T, U>{math::weak_get<1>(a) op math::weak_get<1>(b),                        \
                              math::weak_get<2>(a) op math::weak_get<2>(b)};                       \
  }                                                                                                \
  template <typename T, typename U, enable_ty((op_dim<T, U, 3>))>                                  \
  hostDeviceInline auto operator op(T &&a, U &&b) {                                                \
    return return_types<T, U>{math::weak_get<1>(a) op math::weak_get<1>(b),                        \
                              math::weak_get<2>(a) op math::weak_get<2>(b),                        \
                              math::weak_get<3>(a) op math::weak_get<3>(b)};                       \
  }                                                                                                \
  template <typename T, typename U, enable_ty((op_dim<T, U, 4>))>                                  \
  hostDeviceInline auto operator op(T &&a, U &&b) {                                                \
    return return_types<T, U>{math::weak_get<1>(a) op math::weak_get<1>(b),                        \
                              math::weak_get<2>(a) op math::weak_get<2>(b),                        \
                              math::weak_get<3>(a) op math::weak_get<3>(b),                        \
                              math::weak_get<4>(a) op math::weak_get<4>(b)};                       \
  }                                                                                                \
  template <typename T, typename U, enable_ty((ass_compat<T, U>))>                                 \
  hostDeviceInline auto &operator op##=(T &a, U &&b) {                                             \
    a = a op b;                                                                                    \
    return a;                                                                                      \
  }

#define enable_compat(T, U)                                                                        \
  std::enable_if_t<(is_unit<std::decay_t<T>> || is_unit<std::decay_t<U>>)&&SI::compatible_unit<    \
      std::decay_t<T>, std::decay_t<U>>> * = nullptr
#define enable(T, U)                                                                               \
  std::enable_if_t<(has_def_unit<std::decay_t<T>> || has_def_unit<std::decay_t<U>>)> * = nullptr

#define OPERATOR_COMPAT(op)                                                                        \
  template <typename T, typename U, enable_compat(T, U)>                                           \
  hostDeviceInline auto operator op(T &&lhs, U &&rhs) {                                            \
    return std::decay_t<math::return_types<T, U>>{math::getValue(std::forward<T>(lhs))             \
                                                      op math::getValue(std::forward<U>(rhs))};    \
  }
#define OPERATOR_COMPAT_ASSIGNMENT(op)                                                             \
  template <typename T, typename U, enable_compat(T, U)>                                           \
  hostDeviceInline auto &&operator op##=(T &&lhs, U &&rhs) {                                       \
    math::getValue(std::forward<T>(lhs)) op## = math::getValue(std::forward<U>(rhs));              \
    return lhs;                                                                                    \
  }
#define OPERATOR_UNIT(op) OPERATOR_COMPAT(op) OPERATOR_COMPAT_ASSIGNMENT(op)
#define OPERATOR_COMPARE(op)                                                                       \
  template <typename T, typename U, enable_compat(T, U)>                                           \
  hostDeviceInline auto operator op(T &&lhs, U &&rhs) {                                            \
    return math::getValue(std::forward<T>(lhs)) op math::getValue(std::forward<U>(rhs));           \
  }

namespace math {
namespace operators {
template <typename T, typename U>
constexpr bool dead_beef = math::dim<T> == 0xDEADBEEF || math::dim<U> == 0xDEADBEEF;
template <typename T, typename U>
constexpr bool op_compat = !dead_beef<T, U> && ((math::dim<T> == math::dim<U>) ||
                                                (math::dim<T> == 0 && math::dim<U> != 0) ||
                                                (math::dim<T> != 0 && math::dim<U> == 0));
template <typename T, typename U, uint32_t d>
constexpr bool op_dim = !dead_beef<T, U> && ((math::dim<T> == d && d == math::dim<U>) ||
                                             (math::dim<T> == 0 && math::dim<U> == d) ||
                                             (math::dim<T> == d && math::dim<U> == 0));
template <typename T, typename U>
constexpr bool ass_compat = !dead_beef<T, U> && ((math::dim<T> == math::dim<U>) ||
                                                 (math::dim<T> != 0 && math::dim<U> == 0));

OPERATOR_NEWER2(+);
OPERATOR_NEWER2(-);
OPERATOR_NEWER2(*);
OPERATOR_NEWER2(/);
OPERATOR_NEWER2(%);

template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator==(T &&a, U &&b) {
  return fn_acc(
      [&](bool x0, bool x1 = true, bool x2 = true, bool x3 = true) { return x0 && x1 && x2 && x3; },
      [&](auto &&a, auto &&b) { return a == b; }, std::forward<T>(a), std::forward<U>(b));
}
template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator<=(T &&a, U &&b) {
  return fn_acc(
      [&](bool x0, bool x1 = true, bool x2 = true, bool x3 = true) { return x0 && x1 && x2 && x3; },
      [&](auto &&a, auto &&b) { return a <= b; }, std::forward<T>(a), std::forward<U>(b));
}
template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator<(T &&a, U &&b) {
  return (a <= b) && !(a == b);
}
template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator>=(T &&a, U &&b) {
  return !(a < b);
}
template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator>(T &&a, U &&b) {
  return (a >= b) && !(a == b);
}
template <typename T, typename U, enable_ty((op_compat<T, U>))>
hostDeviceInline auto operator!=(T &&a, U &&b) {
  return !(a == b);
}
template <typename T, enable_ty((op_compat<T, T>))> hostDeviceInline auto operator-(const T &lhs) {
  return lhs * (-1.f);
}

OPERATOR_UNIT(+);
OPERATOR_UNIT(-);
OPERATOR_UNIT (^);
OPERATOR_UNIT(|);
OPERATOR_UNIT(&);
OPERATOR_UNIT(%);
OPERATOR_UNIT(<<);
OPERATOR_UNIT(>>);

OPERATOR_COMPARE(<);
OPERATOR_COMPARE(>);
OPERATOR_COMPARE(<=);
OPERATOR_COMPARE(>=);
OPERATOR_COMPARE(==);
OPERATOR_COMPARE(!=);

template <typename T, typename U, enable(T, U)> hostDeviceInline auto operator*(T &&lhs, U &&rhs) {
  return std::move(SI::add_units_ty<T, U>{math::getValue(std::forward<T>(lhs)) *
                                          math::getValue(std::forward<U>(rhs))});
}

template <typename T, typename U, enable(T, U)> hostDeviceInline auto operator/(T &&lhs, U &&rhs) {
  return std::move(SI::sub_units_ty<T, U>{math::getValue(std::forward<T>(lhs)) /
                                          math::getValue(std::forward<U>(rhs))});
}

template <typename T, typename U, enable_ty(((is_unit<T> || is_unit<U>)&&SI::void_unit<U>))>
hostDeviceInline auto operator*=(T &&lhs, U &&rhs) {
  math::getValue(std::forward<T>(lhs)) *= math::getValue(std::forward<U>(rhs));
  return lhs;
}

template <typename T, typename U, enable_ty(((is_unit<T> || is_unit<U>)&&SI::void_unit<U>))>
hostDeviceInline auto operator/=(T &&lhs, U &&rhs) {
  math::getValue(std::forward<T>(lhs)) /= math::getValue(std::forward<U>(rhs));
  return lhs;
}

template <typename T, enable_ty((is_unit<T>))> hostDeviceInline auto operator-(const T &lhs) {
  return lhs * (-1.f);
}
#ifndef __CUDA_ARCH__
template <typename Ty, uint32_t dim = math::dim<Ty>, enable_ty((dim != 0 && dim != 0xDEADBEEF))>
std::ostream &operator<<(std::ostream &str, Ty &&lhs) {
  str << "[" << math::weak_get_s<1>(std::forward<Ty>(lhs));
  if (dim > 1)
    str << ", " << math::weak_get_s<2>(std::forward<Ty>(lhs));
  if (dim > 2)
    str << ", " << math::weak_get_s<3>(std::forward<Ty>(lhs));
  if (dim > 3)
    str << ", " << math::weak_get_s<4>(std::forward<Ty>(lhs));
  str << "]";
  return str;
}

template <typename Ty, enable_ty(is_unit<Ty>)>
std::ostream &operator<<(std::ostream &str, Ty &&lhs) {
  str << std::fixed << std::setprecision(6) << math::getValue(std::forward<Ty>(lhs)) << " ";
  auto print_unit = [&](auto u, auto unit) {
    using U = decltype(u);
    if (U::num == 0)
      return;
    str << unit;
    if (U::num != 1 || U::den != 1)
      str << "^[" << U::num;
    if (U::den != 1)
      str << "/" << U::den;
    if (U::num != 1 || U::den != 1)
      str << "]";
  };
  using Uy = SI::unit_of<Ty>;
  print_unit(typename Uy::_m{}, "m");
  print_unit(typename Uy::_s{}, "s");
  print_unit(typename Uy::_kg{}, "kg");
  print_unit(typename Uy::_A{}, "A");
  print_unit(typename Uy::_mol{}, "mol");
  print_unit(typename Uy::_cd{}, "cd");
  return str;
}
#endif

} // namespace operators
} // namespace math


#ifndef NO_OPERATORS
using namespace math::operators;
#endif

#undef OPERATOR_NEWER
#undef enable_compat
#undef enable
#undef OPERATOR_COMPAT
#undef OPERATOR_COMPAT_ASSIGNMENT
#undef OPERATOR_UNIT
#undef OPERATOR_COMPARE