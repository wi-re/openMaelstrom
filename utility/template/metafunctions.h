#pragma once
#include <type_traits>
#include <utility/template/nonesuch.h>
#include <utility>
#define HAS_TYPENAME(ty)                                                                           \
  template <class T> using ty##_t = typename T::ty;                                                \
  template <class Ptr> using ty##_type_template = detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_##ty = !std::is_same<ty##_type_template<T>, std::ptrdiff_t>::value;
#define HAS_MEMBER(ty)                                                                             \
  template <class T> using ty##_t = decltype(std::declval<T>().ty);                                \
  template <class Ptr> using ty##_type_template = detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_##ty = !std::is_same<ty##_type_template<T>, std::ptrdiff_t>::value;
#define HAS_STATIC_MEMBER(ty)                                                                      \
  template <class T> using ty##_t = decltype(T::ty);                                               \
  template <class Ptr> using ty##_type_template = detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_##ty = !std::is_same<ty##_type_template<T>, std::ptrdiff_t>::value;
