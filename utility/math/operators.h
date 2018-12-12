#pragma once
#include <utility/math/get.h>
#include <utility/unit_math.h>

//#pragma region operators
namespace math {
namespace ops {

OPERATOR_NEW(+);
OPERATOR_NEW(-);
OPERATOR_NEW(*);
OPERATOR_NEW(/);
OPERATOR_NEW(%);

template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto operator==(T&& a, U&& b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b) &&
         math::get<3>(a) == math::get<3>(b) && math::get<4>(a) == math::get<4>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto operator==(T&& a, U&& b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b) &&
         math::get<3>(a) == math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,
          typename = void>
hostDeviceInline auto operator==(T&& a, U&& b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 1>>
hostDeviceInline auto operator==(T&& a, U&& b) -> bool {
  return math::get<1>(a) == math::get<1>(b);
}

template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto operator!=(T&& a, U&& b) -> bool {
  return !(a == b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto operator!=(T&& a, U&& b) -> bool {
  return !(a == b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,
          typename = void>
hostDeviceInline auto operator!=(T&& a, U&& b) -> bool {
  return !(a == b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 1>>
hostDeviceInline auto operator!=(T&& a, U&& b) -> bool {
  return !(a == b);
}
template <typename T, typename std::enable_if<(math::dimension<T>::value > 0), std::nullptr_t>::type
                          * = nullptr>
hostDeviceInline auto operator-(const T &lhs) -> T {
  return lhs * (-1.f);
}
} // namespace ops
using namespace math::ops;
} // namespace math

#ifndef NO_OPERATORS
using namespace math::ops;
#endif

//#pragma endregion
