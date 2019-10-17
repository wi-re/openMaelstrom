#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

template <std::size_t I = 0, typename FuncT, typename... Tp>
constexpr inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each_r(std::tuple<Tp...>, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
constexpr inline typename std::enable_if <
	I<sizeof...(Tp), void>::type for_each_r(std::tuple<Tp...>t, FuncT f) {
	f(std::get<I>(t));
	for_each_r<I + 1, FuncT, Tp...>(t, f);
}

template <std::size_t I = 0, typename FuncT, typename... Tp>
constexpr inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each(std::tuple<Tp...> &, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    constexpr inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t));
  for_each<I + 1, FuncT, Tp...>(t, f);
}
template <std::size_t I = 0, typename FuncT, typename T>
void for_each(std::array<T, I> &t, FuncT f) {
  for (std::size_t II = 0; II < t.size(); ++II)
    f(t[II]);
}

template <std::size_t I = 0, typename FuncT, typename... Tp>
constexpr inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each_i(std::tuple<Tp...> &, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    constexpr inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each_i(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t), I);
  for_each_i<I + 1, FuncT, Tp...>(t, f);
}

template <std::size_t I = 0, typename FuncT, typename T>
void for_each_i(std::array<T, I> &t, FuncT f) {
  for (std::size_t II = 0; II < t.size(); ++II)
    f(t[II], II);
}
#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
template<template<typename> typename C, typename... Ts, typename... Us>
auto callOnTypes(std::tuple<Ts...>, Us&&... args) {
	(C<Ts>()(args...), ...);
}
#endif