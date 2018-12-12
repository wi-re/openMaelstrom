#pragma once
#include <cstddef>
#include <algorithm>
#include <utility/math.h>

inline constexpr auto gcd(int a, int b) {
  while (b != 0) {
    auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template <std::intmax_t n, std::intmax_t d> struct ratio {
  static constexpr int num = n;
  static constexpr int den = d;
};

template<typename R1>
using ratio_reduce = ratio<R1::num / gcd(R1::num, R1::den), R1::den / gcd(R1::num, R1::den)>;
template<typename R1, typename R2>
using ratio_add = ratio_reduce<ratio<R1::num * R2::den + R2::num * R1::den, R1::den * R2::den>>;
template<typename R1, typename R2>
using ratio_sub = ratio_reduce<ratio<R1::num * R2::den - R2::num * R1::den, R1::den * R2::den>>;
template<typename R1, typename R2>
using ratio_multiply = ratio_reduce<ratio<R1::num * R2::num, R1::den * R2::den>>;
template<typename R1, typename R2>
using ratio_divide = ratio_reduce<ratio<R1::num / R2::num, R1::den / R2::den>>;
template< typename R1, typename R2 >
struct ratio_equal : public std::integral_constant <bool, R1::num == R2::num && R1::den == R2::den> {};
template< typename R1, typename R2 >
struct ratio_not_equal : public std::integral_constant <bool, !ratio_equal<R1, R2>::value> {};
template< typename R1, typename R2 >
struct ratio_less : public std::integral_constant <bool, (R1::num * R2::den < R2::num * R1::den)> {};
template< typename R1, typename R2 >
struct ratio_greater : public std::integral_constant <bool, (R1::num * R2::den > R2::num * R1::den)> {};
template< typename R1, typename R2 >
struct ratio_less_equal : public std::integral_constant <bool, ratio_less<R1, R2>::value || ratio_equal<R1, R2>::value> {};
template< typename R1, typename R2 >
struct ratio_greater_equal : public std::integral_constant <bool, ratio_greater<R1, R2>::value || ratio_equal<R1, R2>::value> {};
