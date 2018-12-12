#pragma once
#include <utility/math.h>
#include <utility/unitmath/SI_Unit.h>
#include <utility/math/operators.h>

template<typename T, typename U>
using unit_compatible = std::enable_if_t < SI::is_same_unit<typename T::unit, typename U::unit>::value || SI::is_compatible<typename T::unit, typename U::unit>::value>;

template<typename... Conds>
  struct and_
  : std::true_type
  { };

template <bool B>
using bool_constant = std::integral_constant<bool, B>;
	
template<typename Cond, typename... Conds>
  struct and_<Cond, Conds...>
  : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type
  { };

template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...> 
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};
template<class...> struct disjunction : std::false_type { };
template<class B1> struct disjunction<B1> : B1 { };
template<class B1, class... Bn>
struct disjunction<B1, Bn...> 
    : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>>  { };
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

template<typename T, typename U>
using is_compatible = //disjunction< 
			//		conjunction< std::is_same<T, void>, std::is_same<U,void> >, 
					disjunction< SI::is_same_unit<T, U>, SI::is_compatible<T, U>>
			//	 >
;



#define ASSIGNMENT_OP_PASSTHROUGH(op)                                                              \
  template <typename T, typename U, typename V,                                                    \
            typename = std::enable_if_t<math::dimension<T>::value != 0 &&                          \
                                        math::dimension<V>::value == 0>>                           \
  constexpr hostDevice auto operator op(value_unit<T, U> &lhs, V rhs) {            \
    using namespace math::ops;                                                                     \
    lhs.val op rhs;                                                                                \
  }                                                                                                \
  template <typename T>                                                                            \
  constexpr hostDevice auto operator op(T &lhs, typename T::type rhs) {            \
    using namespace math::ops;                                                                     \
    lhs.val op rhs;                                                                                \
  }                                                                                                \
  template <typename T, typename U,                                                                \
            typename =                                                                             \
                std::enable_if_t<is_compatible<typename T::unit, typename U::unit>::value>>     \
  constexpr hostDevice auto operator op(T &lhs, U rhs) {                           \
    using namespace math::ops;                                                                     \
    lhs.val op rhs.val;                                                                            \
  }\
  template <typename T,                                                                \
            typename =                                                                             \
                std::enable_if_t<is_compatible<typename T::unit, void>::value>>     \
  constexpr hostDevice auto operator op(typename T::type &lhs, T rhs) {                           \
    using namespace math::ops;                                                                     \
    lhs op rhs.val;                                                                            \
  }

#define COMP_OP_PASSTHROUGH(op, fn)                                                                \
  template <typename T, typename U,                                                                \
            typename =                                                                             \
                std::enable_if_t<is_compatible<typename T::unit, typename U::unit>::value>>     \
  constexpr hostDevice auto operator op(T lhs, U rhs) {                            \
    return math::fn(lhs.val, rhs.val);                                                             \
  }                                                                                                \
  template <typename T, typename U, typename V,                                                    \
            typename = std::enable_if_t<!std::is_same<U, void>::value &&                           \
                                        math::dimension<V>::value != 0xDEADBEEF>>                  \
  constexpr hostDeviceInline auto operator op(const value_unit<T, U> &lhs,           \
                                                            const V &rhs) {                        \
    return math::fn(lhs.val, rhs);                                                                 \
  }                                                                                                \
  template <typename T, typename U, typename V,                                                    \
            typename = std::enable_if_t<!std::is_same<U, void>::value &&                           \
                                        math::dimension<V>::value != 0xDEADBEEF>>                  \
  constexpr hostDeviceInline auto operator op(const V &rhs,                          \
                                                            const value_unit<T, U> &lhs) {         \
    return math::fn(rhs, lhs.val);                                                                 \
  }

#define VALUE_UNIT_PASSTHROUGH(x)                                                                  \
  template <typename T, typename U, typename V,                                                    \
            typename = std::enable_if_t<!std::is_same<U, void>::value &&                           \
                                        math::dimension<V>::value != 0xDEADBEEF>>                  \
  constexpr hostDeviceInline auto x(V rhs, value_unit<T, U> lhs) {                                \
    return math::x(rhs, lhs.val);                                                                  \
  }                                                                                                \
  template <typename T, typename U, typename V,                                                    \
            typename = std::enable_if_t<!std::is_same<U, void>::value &&                           \
                                        math::dimension<V>::value != 0xDEADBEEF>>                  \
  constexpr hostDeviceInline auto x(value_unit<T, U> lhs,  V rhs) {                                \
    return math::x(lhs.val, rhs);                                                                  \
  }                                                                                                \
  template <typename T, typename U,                                                                \
            typename =                                                                             \
                std::enable_if_t<is_compatible<typename T::unit, typename U::unit>::value>>     \
  constexpr hostDeviceInline auto x(T lhs, U rhs) {                                                      \
    return math::x(lhs.val, rhs.val);                                                              \
  }\
  template <typename T, typename U> \
  constexpr hostDeviceInline auto x(value_unit<T,U> lhs, value_unit<T,U> rhs) {                                                      \
    return true; /*math::x(lhs.val, rhs.val);    */                                                          \
  }

ASSIGNMENT_OP_PASSTHROUGH(+= );
ASSIGNMENT_OP_PASSTHROUGH(-= );
ASSIGNMENT_OP_PASSTHROUGH(*= );
ASSIGNMENT_OP_PASSTHROUGH(/= );

namespace math {
template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto eq3(T a, U b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b) &&
         math::get<3>(a) == math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto eq3(T a, U b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b) &&
         math::get<3>(a) == math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,
          typename = void>
hostDeviceInline auto eq3(T a, U b) -> bool {
  return math::get<1>(a) == math::get<1>(b) && math::get<2>(a) == math::get<2>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 1>>
hostDeviceInline auto eq3(T a, U b) -> bool {
  return math::get<1>(a) == math::get<1>(b);
}
	VALUE_UNIT_PASSTHROUGH(eq);
	VALUE_UNIT_PASSTHROUGH(eq3);

template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,
          typename = void, typename = void, typename = void, typename = void>
hostDeviceInline auto le3(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b) &&
         math::get<3>(a) < math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto le3(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b) &&
         math::get<3>(a) < math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto le3(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 1>, typename = void>
hostDeviceInline auto le3(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 0>>
hostDeviceInline auto le3(T a, U b) -> bool {
  return a < b;
}
	VALUE_UNIT_PASSTHROUGH(le);
	VALUE_UNIT_PASSTHROUGH(le3);

template <typename T, typename U> hostDeviceInline auto leq3(T a, U b) -> bool {
  return eq3(a, b) || le3(a, b);
}
template <typename T, typename U> hostDeviceInline auto ge3(T a, U b) -> bool {
  return !leq3(a, b);
}
template <typename T, typename U> hostDeviceInline auto geq3(T a, U b) -> bool {
  return !le3(a, b);
}
template <typename T, typename U> hostDeviceInline auto neq3(T a, U b) -> bool {
  return !eq3(a, b);
}

template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,
          typename = void, typename = void, typename = void, typename = void>
hostDeviceInline auto le(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b) &&
         math::get<3>(a) < math::get<3>(b) && math::get<4>(a) < math::get<4>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,
          typename = void, typename = void, typename = void>
hostDeviceInline auto le(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b) &&
         math::get<3>(a) < math::get<3>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,
          typename = void, typename = void>
hostDeviceInline auto le(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b) && math::get<2>(a) < math::get<2>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 1>, typename = void>
hostDeviceInline auto le(T a, U b) -> bool {
  return math::get<1>(a) < math::get<1>(b);
}
template <typename T, typename U, typename = math::DimCheck<T, U, 0>>
hostDeviceInline auto le(T a, U b) -> bool {
  return a < b;
}

template <typename T, typename U> hostDeviceInline auto leq(T a, U b) -> bool {
  return (a == b) || le(a, b);
}
template <typename T, typename U> hostDeviceInline auto ge(T a, U b) -> bool {
  return !leq(a, b);
}
template <typename T, typename U> hostDeviceInline auto geq(T a, U b) -> bool {
  return !le(a, b);
}
template <typename T, typename U> hostDeviceInline auto neq(T a, U b) -> bool {
  return !(a == b);
}
template <typename T, typename U> hostDeviceInline auto eq(T a, U b) -> bool {
  return (a == b);
}

}

COMP_OP_PASSTHROUGH(<, le);
COMP_OP_PASSTHROUGH(<= , leq);
COMP_OP_PASSTHROUGH(>, ge);
COMP_OP_PASSTHROUGH(>= , geq);
COMP_OP_PASSTHROUGH(== , eq);
COMP_OP_PASSTHROUGH(!= , neq);

template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator+(V rhs, value_unit<T, U> lhs) {
	using namespace math::ops;
	return value_unit<T, U>{rhs + lhs.val};
}
template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator+(value_unit<T, U> lhs, V rhs) {
	using namespace math::ops;
	return value_unit<T, U>{lhs.val + rhs};
}
template<typename T>
constexpr hostDevice auto operator+(T lhs, typename T::type rhs) {
	using namespace math::ops;
	return T{ lhs.val + rhs };
}
template<typename T>
constexpr hostDevice auto operator+(typename T::type lhs, T rhs) {
	using namespace math::ops;
	return T{ lhs + rhs.val };
}
template<typename T, typename U, typename = std::enable_if_t < is_compatible<typename T::unit, typename U::unit>::value>>
constexpr hostDevice auto operator+(T lhs, U rhs) {
	using namespace math::ops;
	return T{ lhs.val + rhs.val };
}

template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator-(V rhs, value_unit<T, U> lhs) {
	using namespace math::ops;
	return value_unit<T, U>{rhs - lhs.val};
}
template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator-(value_unit<T, U> lhs, const V rhs) {
	using namespace math::ops;
	return value_unit<T, U>{lhs.val - rhs};
}
template<typename T>
constexpr hostDevice auto operator-(T lhs, typename T::type rhs) {
	using namespace math::ops;
	return T{ lhs.val - rhs };
}
template<typename T>
constexpr hostDevice auto operator-(typename T::type lhs, T rhs) {
	using namespace math::ops;
	return T{ lhs - rhs.val };
}
template<typename T, typename U, typename = std::enable_if_t < is_compatible<typename T::unit, typename U::unit>::value>>
constexpr hostDevice auto operator-(T lhs, U rhs) {
	using namespace math::ops;
	return T{ lhs.val - rhs.val };
}

template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator*(V rhs, value_unit<T, U> lhs) {
	using namespace math::ops;
	return value_unit<T, U>{rhs * lhs.val};
}
template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator*(value_unit<T, U> lhs, V rhs) {
	using namespace math::ops;
	return value_unit<T, U>{lhs.val * rhs};
}
template<typename T, typename V>
constexpr hostDevice auto operator*(value_unit<T, V> lhs, T rhs) {
	using namespace math::ops;
	return value_unit<T, V>{ lhs.val * rhs };
}
template<typename T, typename V>
constexpr hostDevice auto operator*(T lhs, value_unit<T, V> rhs) {
	using namespace math::ops;
	return value_unit<T, V>{ lhs * rhs.val };
}
template<typename T, typename V, typename U, typename W>
constexpr hostDevice auto operator*(value_unit<T, V> lhs, value_unit<U, W> rhs) {
	using namespace math::ops;
	return value_unit<decltype(lhs.val * rhs.val), typename SI::add_unit<V, W>>{ lhs.val * rhs.val };
}

template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator/(V rhs, value_unit<T, U> lhs) {
	using namespace math::ops;
	return value_unit<T, SI::multiply_ratio<U, ratio<-1, 1>>>{rhs / lhs.val};
}
template<typename T, typename U, typename V, typename = std::enable_if_t< math::dimension<T>::value != 0 && math::dimension<V>::value == 0>>
constexpr hostDevice auto operator/(value_unit<T, U> lhs, V rhs) {
	using namespace math::ops;
	return value_unit<T, U>{lhs.val / rhs};
}
template<typename T>
constexpr hostDevice auto operator/(T lhs, typename T::type rhs) {
	using namespace math::ops;
	return T{ lhs.val / rhs };
}
template<typename T, typename U>
constexpr hostDevice auto operator/(T lhs, value_unit<T, U> rhs) {
	using namespace math::ops;
	return value_unit<T, SI::multiply_ratio<U, ratio<-1, 1>>>{ lhs / rhs.val };
}
template<typename T1, typename U1, typename T2, typename U2>
constexpr hostDevice auto operator/(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
	using namespace math::ops;
	return value_unit<decltype(lhs.val / rhs.val), typename SI::sub_unit<U1, U2>>{ lhs.val / rhs.val };
}

template<typename T, typename U>
constexpr hostDeviceInline auto operator-(value_unit<T, U> lhs) {
	return value_unit<T, U>{-lhs.val};
}
