#pragma once
#include <cstdint>
#include <type_traits>
#include <utility/math/macro.h>
#include <utility/template/metafunctions.h>

namespace math {
namespace detail {
#define HAS_MEM(ty)                                                                             \
  template <class T> using ty##_t = decltype(std::declval<T>().ty);                                \
  template <class Ptr> using ty##_type_template = detected_or_t<std::ptrdiff_t, ty##_t, Ptr>;      \
  template <typename T>                                                                            \
  constexpr bool has_mem_##ty = !std::is_same<ty##_type_template<T>, std::ptrdiff_t>::value;

	HAS_MEM(x);
	HAS_MEM(y);
	HAS_MEM(z);
	HAS_MEM(w);

	template<typename T>
	constexpr bool has_x = has_mem_x<T>;
	template<typename T>
	constexpr bool has_xy = has_mem_x<T> && has_mem_y<T>;
	template<typename T>
	constexpr bool has_xyz = has_mem_x<T> && has_mem_y<T> && has_mem_z<T>;
	template<typename T>
	constexpr bool has_xyzw = has_mem_x<T> && has_mem_y<T> && has_mem_z<T> && has_mem_w<T>;
	template <typename T>
	static constexpr uint32_t dimension_fn() {
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
	template<typename T>
	struct dimension {
		static constexpr uint32_t value = dimension_fn<T>();
	};
	template<typename T>
	constexpr uint32_t dimension_v = dimension<T>::value;
	template<typename T, typename U, uint32_t d>
	constexpr bool dimension_compatible =
		(dimension_v<T> == d || dimension_v<U> == d) &&
		(dimension_v<T> == dimension_v<U> || dimension_v<U> == 0 || dimension_v<T> == 0);


template <typename T>
using base_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <uint32_t idx, uint32_t ref>
using EnableIfLim = typename std::enable_if<idx == ref, std::nullptr_t>::type;

template <typename T, typename U, uint32_t d>
using DimCheck =
    typename std::enable_if<(dimension_v<T> == d || dimension_v<U> == d) &&
                                (dimension_v<T> == dimension_v<U> ||
									dimension_v<U> == 0 || dimension_v<T> == 0),
	std::decay_t<T>>::type;

template <typename T, typename U, uint32_t dim>
using DimRequired =
    typename std::enable_if<(dimension<T>::value == dim && dimension<U>::value == dim), std::decay_t<T>>::type;

template <bool B, class T = void, class U = void> struct enable_else { typedef U type; };

template <class T, class U> struct enable_else<true, T, U> { typedef T type; };

template <typename T, typename U>
using return_type =
    typename enable_else<(dimension<base_type<T>>::value < dimension<base_type<U>>::value),
                         base_type<U>, base_type<T>>::type;
}; // namespace detail
using namespace detail;
template<size_t check_dim, typename T>
using greaterDimension = std::enable_if_t<(math::dimension<T>::value > check_dim), std::nullptr_t>;
template<size_t check_dim, typename T>
using equalDimension = std::enable_if_t<(math::dimension<T>::value == check_dim), std::nullptr_t>;


} // namespace math
