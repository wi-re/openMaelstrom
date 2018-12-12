#pragma once
#ifndef __CUDACC__
#include <boost/preprocessor/repetition/repeat.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <utility/template/tuple_for_each.h>

struct any_type {
	template <class T> constexpr operator T();
};
template <class T, std::size_t... I>
decltype((void) void(T{ ((void)I, std::declval<any_type>())... }), std::true_type{})
test_is_braces_constructible_n(std::index_sequence<I...>);
template <class, class...> std::false_type test_is_braces_constructible_n(...);
template <class T, std::size_t N>
using is_braces_constructible_n =
decltype(test_is_braces_constructible_n<T>(std::make_index_sequence<N>{}));

template <class T, std::size_t L = 0u, std::size_t R = sizeof(T) + 1u>
constexpr std::size_t to_tuple_size_f() {
	constexpr std::size_t M = (L + R) / 2u;
	if constexpr (M == 0)
		return std::is_empty<T>{} ? 0u
		: throw "Unable to determine number of elements";
	else if constexpr (L == M)
		return M;
	else if constexpr (is_braces_constructible_n<T, M>{})
		return to_tuple_size_f<T, M, R>();
	else
		return to_tuple_size_f<T, L, M>();
}
template <class T>
using to_tuple_size = std::integral_constant<std::size_t, to_tuple_size_f<T>()>;

#ifndef TO_TUPLE_MAX
#define TO_TUPLE_MAX 64
#endif

template <class T>
auto to_tuple_impl(T &&, std::integral_constant<std::size_t, 0>) noexcept {
	return std::make_tuple();
}

#define TO_TUPLE_P(Z, N, _) , p##N
#define TO_TUPLE_SPECIALIZATION(Z, N, _)                                       \
  template <class T>                                                           \
  auto to_tuple_impl(T &&object,                                               \
					 std::integral_constant<std::size_t, N + 1>) noexcept {    \
	auto && [p BOOST_PP_REPEAT_##Z(N, TO_TUPLE_P, nil)] = object;              \
	return std::make_tuple(p BOOST_PP_REPEAT_##Z(N, TO_TUPLE_P, nil));         \
  }
BOOST_PP_REPEAT(TO_TUPLE_MAX, TO_TUPLE_SPECIALIZATION, nil)
#undef TO_TUPLE_SPECIALIZATION
#undef TO_TUPLE_P

template <class T, class = struct current_value, std::size_t = TO_TUPLE_MAX,
	class = struct required_value, std::size_t N>
	auto to_tuple_impl(T &&, std::integral_constant<std::size_t, N>) noexcept {
	static_assert(N <= TO_TUPLE_MAX, "Please increase TO_TUPLE_MAX");
}

template <class T, class = std::enable_if_t<std::is_class<T>::value &&
	std::is_standard_layout<T>::value>>
	auto to_tuple(T &&object) noexcept {
	return to_tuple_impl(std::forward<T>(object),
		to_tuple_size<std::decay_t<T>>{});
}

template<typename C, typename... Ts>
void fn_all(C fn, Ts&&... args) {
	(fn(std::forward<Ts&&>(args)), ...);
}

#define TO_TUPLE_P(Z, N, _) , p##N
#define TO_TUPLE_SPECIALIZATION(Z, N, _)                                       \
  template <class T, typename C>                                                           \
  auto for_struct_impl(T &&object,                                               \
					 std::integral_constant<std::size_t, N + 1>, C fn) noexcept {    \
	auto && [p BOOST_PP_REPEAT_##Z(N, TO_TUPLE_P, nil)] = object;              \
	fn_all(fn, p BOOST_PP_REPEAT_##Z(N, TO_TUPLE_P, nil));         \
  }
BOOST_PP_REPEAT(TO_TUPLE_MAX, TO_TUPLE_SPECIALIZATION, nil)
#undef TO_TUPLE_SPECIALIZATION
#undef TO_TUPLE_P

template <class T, typename C, class = struct current_value, std::size_t = TO_TUPLE_MAX,
	class = struct required_value, std::size_t N>
	auto for_struct_impl(T &&, std::integral_constant<std::size_t, N>, [[maybe_unused]] C fn) noexcept {
	static_assert(N <= TO_TUPLE_MAX, "Please increase TO_TUPLE_MAX");
}

template <class T, typename C>
auto for_struct_fn(T &&object, C fn) noexcept {
	for_struct_impl(std::forward<T>(object),
		to_tuple_size<std::decay_t<T>>{}, fn);
}

template <class S, std::size_t... Is, class Tup>
S to_struct(std::index_sequence<Is...>, Tup &&tup) {
	using std::get;
	return { get<Is>(std::forward<Tup>(tup))... };
}
template <class S, class Tup> S to_struct(Tup &&tup) {
	using T = std::remove_reference_t<Tup>;

	return to_struct<S>(std::make_index_sequence<std::tuple_size<T>{}>{},
		std::forward<Tup>(tup));
}

template <typename S, typename C> void for_struct(S &&arg, C fun) {
	using T = typename std::remove_reference<S>::type;
	auto tuple = to_tuple(std::forward<T>(arg));
	for_each(tuple, fun);
	arg = to_struct<T>(tuple);
}
#endif
