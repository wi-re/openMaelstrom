#pragma once
#include <tuple>
#include <utility/unitmath/ratio.h>
namespace SI {
enum struct Base { m, kg, s, A, K, mol, cd };

template <Base _unit, typename T = ratio<1, 1>> struct SI_Unit {
  constexpr static Base unit = _unit;
  using ratio = T;
};

namespace detail {
template <typename T, typename Tuple> struct has_unit;

template <typename T> struct has_unit<T, std::tuple<>> : std::false_type {};

template <Base B, typename T, Base C, typename U, typename... Ts>
struct has_unit<SI_Unit<B, T>, std::tuple<SI_Unit<C, U>, Ts...>>
    : has_unit<SI_Unit<B, T>, std::tuple<Ts...>> {};

template <Base B, typename T, typename U, typename... Ts>
struct has_unit<SI_Unit<B, T>, std::tuple<SI_Unit<B, U>, Ts...>> : std::true_type {};

template <Base B, typename Tuple> struct get_unit;

template <Base B> struct get_unit<B, std::tuple<>> { using type = SI_Unit<B, ratio<0, 0>>; };

template <Base B, Base C, typename U, typename... Ts>
struct get_unit<B, std::tuple<SI_Unit<C, U>, Ts...>> {
  using type = typename get_unit<B, std::tuple<Ts...>>::type;
};

template <Base B, typename U, typename... Ts> struct get_unit<B, std::tuple<SI_Unit<B, U>, Ts...>> {
  using type = SI_Unit<B, U>;
};

template <Base B, typename Tuple> struct filter_unit;

template <Base B> struct filter_unit<B, std::tuple<>> { using type = std::tuple<>; };
template <Base B, Base C, typename U, typename... Ts>
struct filter_unit<B, std::tuple<SI_Unit<C, U>, Ts...>> {
  using type =
      decltype(std::tuple_cat(std::declval<std::tuple<SI_Unit<C, U>>>(),
                              std::declval<typename filter_unit<B, std::tuple<Ts...>>::type>()));
};

template <Base B, typename U, typename... Ts>
struct filter_unit<B, std::tuple<SI_Unit<B, U>, Ts...>> {
  using type = typename filter_unit<B, std::tuple<Ts...>>::type;
};

template <Base B, typename T, Base C, typename U> struct add_units_impl {
  using type = SI_Unit<B, T>;
};

template <Base B, typename T, typename U> struct add_units_impl<B, T, B, U> {
  using type = SI_Unit<B, ratio_add<T, U>>;
};

template <typename T, typename Tuple> struct append_unit;

template <typename T, typename... Ts> struct append_unit<T, std::tuple<Ts...>> {
  using type = std::tuple<Ts..., T>;
};

template <Base B, typename T> struct tuple_type { using type = std::tuple<SI_Unit<B, T>>; };

template <Base B, int d> struct tuple_type<B, ratio<0, d>> { using type = std::tuple<>; };

template <Base B, typename T> struct void_helper { using type = SI_Unit<B, T>; };
template <Base B, int d> struct void_helper<B, ratio<0, d>> { using type = void; };

namespace add {
template <typename T, typename Tuple, typename enable = void> struct add_unit;
template <> struct add_unit<void, void> { using type = void; };
template <Base B, typename T, Base C, typename U> struct add_unit<SI_Unit<B, T>, SI_Unit<C, U>> {
  using type = typename add_unit<SI_Unit<B, T>, std::tuple<SI_Unit<C, U>>>::type;
};
template <Base B, typename T> struct add_unit<SI_Unit<B, T>, void> {
  using type = typename void_helper<B, T>::type;
};
template <Base B, typename T> struct add_unit<void, SI_Unit<B, T>> {
  using type = typename void_helper<B, T>::type;
};
template <typename... Ts> struct add_unit<std::tuple<Ts...>, void> {
  using type = std::tuple<Ts...>;
};
template <typename... Ts> struct add_unit<void, std::tuple<Ts...>> {
  using type = std::tuple<Ts...>;
};
template <Base B, typename T, typename U> struct add_unit<SI_Unit<B, T>, SI_Unit<B, U>> {
  using type = typename void_helper<B, ratio_add<T, U>>::type;
};
template <typename T>
struct add_unit<T, std::tuple<>, std::enable_if_t<!std::is_same<T, void>::value>> {
  using type = std::tuple<T>;
};
template <Base B, typename T, Base C, typename U, typename... Ts>
struct add_unit<SI_Unit<B, T>, std::tuple<SI_Unit<C, U>, Ts...>> {
  using type = decltype(
      std::tuple_cat(std::declval<std::tuple<SI_Unit<C, U>>>(),
                     std::declval<typename add_unit<SI_Unit<B, T>, std::tuple<Ts...>>::type>()));
};
template <Base B, typename T, typename U, typename... Ts>
struct add_unit<SI_Unit<B, T>, std::tuple<SI_Unit<B, U>, Ts...>> {
  using type = decltype(std::tuple_cat(
      std::declval<typename tuple_type<B, ratio_add<T, U>>::type>(), std::tuple<Ts...>()));
};

template <typename... Ts, Base B, typename T> struct add_unit<std::tuple<Ts...>, SI_Unit<B, T>> {
  using type = typename add_unit<SI_Unit<B, T>, std::tuple<Ts...>>::type;
};

template <typename T, typename... Us> struct add_unit<std::tuple<>, std::tuple<T, Us...>> {
  using type = std::tuple<T, Us...>;
};

template <typename... Ts, typename T, typename... Us>
struct add_unit<std::tuple<T, Ts...>, std::tuple<Us...>> {
  using type =
      typename add_unit<std::tuple<Ts...>, typename add_unit<T, std::tuple<Us...>>::type>::type;
};

} // namespace add
namespace sub {
template <typename T, typename Tuple, typename enable = void> struct sub_unit;
template <Base B, typename T, Base C, typename U> struct sub_unit<SI_Unit<B, T>, SI_Unit<C, U>> {
  using type = typename sub_unit<SI_Unit<B, T>, std::tuple<SI_Unit<C, U>>>::type;
};
template <> struct sub_unit<void, void> { using type = void; };
template <typename T> struct sub_unit<T, T> { using type = void; };
template <Base B, typename T> struct sub_unit<SI_Unit<B, T>, SI_Unit<B, T>> { using type = void; };
template <Base B, typename T> struct sub_unit<SI_Unit<B, T>, void> {
  using type = typename void_helper<B, T>::type;
};
template <Base B, typename T> struct sub_unit<void, SI_Unit<B, T>> {
  using type = typename void_helper<B, ratio_multiply<ratio<-1, 1>, T>>::type;
};
template <> struct sub_unit<std::tuple<>, void> { using type = void; };
template <> struct sub_unit<void, std::tuple<>> { using type = void; };
template <typename... Ts> struct sub_unit<std::tuple<Ts...>, void> {
  using type = std::tuple<Ts...>;
};
template <typename... Ts> struct sub_unit<void, std::tuple<Ts...>> {
  using type = std::tuple<typename sub_unit<void, Ts>::type...>;
};
template <Base B, typename T, typename U> struct sub_unit<SI_Unit<B, T>, SI_Unit<B, U>> {
  using type = typename void_helper<B, ratio_sub<T, U>>::type;
};
template <typename T> struct sub_unit<T, std::tuple<>> { using type = std::tuple<T>; };
template <Base B, typename T> struct sub_unit<std::tuple<>, SI_Unit<B, T>> {
  using type = std::tuple<SI_Unit<B, ratio_multiply<T, ratio<-1, 1>>>>;
};
template <Base B, typename T, Base C, typename U, typename... Ts>
struct sub_unit<SI_Unit<B, T>, std::tuple<SI_Unit<C, U>, Ts...>> {
  using type = decltype(
      std::tuple_cat(std::declval<std::tuple<typename sub_unit<void, SI_Unit<C, U>>::type>>(),
                     std::declval<typename sub_unit<SI_Unit<B, T>, std::tuple<Ts...>>::type>()));
};
template <Base B, typename T, typename U, typename... Ts>
struct sub_unit<SI_Unit<B, T>, std::tuple<SI_Unit<B, U>, Ts...>> {
  using type =
      decltype(std::tuple_cat(std::declval<typename tuple_type<B, ratio_sub<T, U>>::type>(),
                              std::tuple<typename sub_unit<void, Ts>::type...>()));
};

template <Base B, typename T, Base C, typename U, typename... Ts>
struct sub_unit<std::tuple<SI_Unit<C, U>, Ts...>, SI_Unit<B, T>> {
  using type = decltype(
      std::tuple_cat(std::declval<typename tuple_type<C, U>::type>(),
                     std::declval<typename sub_unit<std::tuple<Ts...>, SI_Unit<B, T>>::type>()));
};
template <Base B, typename T, typename U, typename... Ts>
struct sub_unit<std::tuple<SI_Unit<B, U>, Ts...>, SI_Unit<B, T>> {
  using type = decltype(std::tuple_cat(
      std::declval<typename tuple_type<B, ratio_sub<U, T>>::type>(), std::tuple<Ts...>()));
};

template <typename... Us> struct sub_unit<std::tuple<Us...>, std::tuple<>> {
  using type = std::tuple<Us...>;
};

template <typename... Ts, typename T, typename... Us>
struct sub_unit<
    std::tuple<Ts...>, std::tuple<T, Us...>,
    typename std::enable_if<!std::is_same<std::tuple<Ts...>, std::tuple<T, Us...>>::value>::type> {
  using type =
      typename sub_unit<typename sub_unit<std::tuple<Ts...>, T>::type, std::tuple<Us...>>::type;
};
} // namespace sub
template <typename T, typename Ratio> struct multiply_ratio {
  using type = typename void_helper<T::unit, ratio_multiply<typename T::ratio, Ratio>>::type;
};
template <typename Ratio> struct multiply_ratio<void, Ratio> { using type = void; };
template <typename Ratio> struct multiply_ratio<std::tuple<>, Ratio> { using type = std::tuple<>; };

template <typename T, typename... Ts, typename Ratio>
struct multiply_ratio<std::tuple<T, Ts...>, Ratio> {
  using type = decltype(std::tuple_cat(
      std::declval<typename tuple_type<T::unit, ratio_multiply<typename T::ratio, Ratio>>::type>(),
      std::declval<typename multiply_ratio<std::tuple<Ts...>, Ratio>::type>()));
};

template <typename> struct Tag {};

template <typename T> constexpr std::size_t type_counter(const Tag<std::tuple<>> &) { return 0; }

template <typename T, typename V, typename... Ts>
constexpr std::size_t type_counter(const Tag<std::tuple<V, Ts...>> &) {
  return std::is_same<T, V>::value + type_counter<T>(Tag<std::tuple<Ts...>>{});
}

template <typename Tuple1, typename Tuple2, std::size_t... Is>
constexpr bool equivalent_types(const Tag<Tuple1> &, const Tag<Tuple2> &,
                                std::index_sequence<Is...>) {
  constexpr bool count_match[] = {
      true, (type_counter<std::tuple_element_t<Is, Tuple1>>(Tag<Tuple1>{}) ==
             type_counter<std::tuple_element_t<Is, Tuple1>>(Tag<Tuple2>{}))...};

  for (const auto b : count_match) {
    if (!b) {
      return false;
    }
  }
  return true;
}

template <typename Tuple1, typename Tuple2>
constexpr bool equivalent_types(const Tuple1 &, const Tuple2 &) {
  constexpr auto s1 = std::tuple_size<Tuple1>::value;
  constexpr auto s2 = std::tuple_size<Tuple2>::value;

  return s1 == s2 && equivalent_types(Tag<Tuple1>{}, Tag<Tuple2>{},
                                      std::make_index_sequence<std::min(s1, s2)>());
}

template <typename T, typename U>
struct is_same_unit : std::integral_constant<bool, std::is_same<T, U>::value> {};
template <typename T> struct is_same_unit<T, std::tuple<T>> : std::true_type {};
template <typename T> struct is_same_unit<std::tuple<T>, T> : std::true_type {};

template <typename... Ts, typename... Us>
struct is_same_unit<std::tuple<Ts...>, std::tuple<Us...>>
    : std::integral_constant<bool, equivalent_types(std::tuple<Ts...>(), std::tuple<Us...>())> {};

template <typename T, typename U> struct is_compatible_unit : std::false_type {};
template <typename T> struct is_compatible_unit<T, T> : std::true_type {};
template <typename T> struct is_compatible_unit<T, std::tuple<T>> : std::true_type {};
template <typename T> struct is_compatible_unit<std::tuple<T>, T> : std::true_type {};
template <typename T> struct is_compatible_unit<T, void> : std::true_type {};
template <typename T> struct is_compatible_unit<void, T> : std::true_type {};
template <> struct is_compatible_unit<void, void> : std::true_type {};

template <typename T> struct flatten { using type = T; };
template <typename T> struct flatten<std::tuple<T>> { using type = T; };

template <typename T, typename Enable = void> struct void_test { using type = T; };
template <typename T>
struct void_test<T, typename std::enable_if<is_same_unit<T, void>::value>::type> {
  using type = void;
};

} // namespace detail

template <typename... Ts> struct list{ using type = std::tuple<Ts...>;};
template <typename T> struct list<T>{ using type = T;};
template <> struct list<void>{ using type = void;};

template <typename, typename> struct list_append_impl;

template <typename... Ts, typename... Us>
struct list_append_impl<list<Ts...>, list<Us...>> {
    using type = list<Ts..., Us...>;
};

template <template <typename> class, typename...>
struct filter_impl;

template <template <typename> class Predicate>
struct filter_impl<Predicate> {
    using type = list<>;
};

template <template <typename> class Predicate, typename T, typename... Rest>
struct filter_impl<Predicate, T, Rest...> {
    using type = typename list_append_impl<
                    std::conditional_t<
                        Predicate<T>::value,
                        list<T>,
                        list<>
                    >,
                    typename filter_impl<Predicate, Rest...>::type
                 >::type;
};

template <template <typename> class Predicate, typename... Ts>
using filter = typename filter_impl<Predicate, Ts...>::type;

template<typename T>
using unit_ratio = typename T::ratio;

template<typename T>
using filter_condition = std::integral_constant<bool, unit_ratio<T>::num != 0>;

template <typename T> struct flatten2 { using type = T; };
template <typename T> struct flatten2<std::tuple<T>> { using type = T; };
template <typename... Ts> struct flatten2<std::tuple<Ts...>>{ using type = typename detail::flatten<typename filter<filter_condition, Ts...>::type>::type; };

template <typename T, typename U>
using add_units =
    typename detail::add_units_impl<T::unit, typename T::ratio, U::unit, typename U::ratio>::type;

template <typename T, typename U>
using add_unit = typename detail::void_test<
    typename flatten2<typename detail::add::add_unit<T, U>::type>::type>::type;
template <typename T, typename U>
using sub_unit = typename detail::void_test<
    typename flatten2<typename detail::sub::sub_unit<T, U>::type>::type>::type;
template <typename T, typename U>
using multiply_ratio = typename flatten2<typename detail::multiply_ratio<T, U>::type>::type;
template <typename T, typename U> using is_same_unit = typename detail::is_same_unit<T, U>;
template <typename T, typename U> using is_equiv_unit = typename detail::is_same_unit<typename flatten2<typename flatten2<T>::type>::type, typename flatten2<typename flatten2<U>::type>::type>;
template <typename T, typename U> using is_compatible = typename detail::is_compatible_unit<T, U>;

template <typename T, typename U> using has_unit = detail::has_unit<T, U>;

template <Base B, typename U> using get_unit_by_base = detail::get_unit<B, U>;

template <typename T, typename U> using get_unit = detail::get_unit<T::unit, U>;

template <Base T, typename U> using filter_unit_by_base = detail::filter_unit<T, U>;

template <typename T, typename U> using filter_unit = detail::filter_unit<T::unit, U>;

template <typename T> struct derived_unit_impl;
template <> struct derived_unit_impl<std::tuple<>> { using type = void; };
template <typename T, typename... Ts> struct derived_unit_impl<std::tuple<T, Ts...>> {
  using type = add_unit<T, typename derived_unit_impl<std::tuple<Ts...>>::type>;
};

template <typename... Ts> using derived_unit = typename derived_unit_impl<std::tuple<Ts...>>::type;

template <typename T> using recip = multiply_ratio<T, ratio<-1, 1>>;
template <typename T> using recip_2 = multiply_ratio<T, ratio<-2, 1>>;
template <typename T> using recip_3 = multiply_ratio<T, ratio<-3, 1>>;
template <typename T> using id = multiply_ratio<T, ratio<1, 1>>;
template <typename T> using square = multiply_ratio<T, ratio<2, 1>>;
template <typename T> using cubic = multiply_ratio<T, ratio<3, 1>>;
} // namespace SI

template <typename Value, typename Unit> struct value_unit {
  Value val;
  using unit = Unit;
  using type = Value;
  constexpr hostDevice value_unit() { val = vec<Value>::zero(); };

  template <typename... Ts>
  constexpr explicit hostDevice value_unit(Ts... args) : val{args...} {}

  template <typename T, typename U>
  constexpr hostDevice value_unit<Value, Unit>(
      const value_unit<T, U> &rhs, std::enable_if_t<SI::is_same_unit<Unit, U>::value, int *> = 0) {
    val = rhs.val;
  }

  

  constexpr explicit hostDevice operator Value() const { return val; }
};

template <typename Value> struct value_unit<Value, void> {
  Value val;
  using unit = void;
  using type = Value;
  constexpr hostDevice value_unit() { val = vec<Value>::zero(); };

  template <typename... Ts> constexpr hostDevice value_unit(Ts... args) : val{args...} {}

  // template <typename T, typename U>
  // constexpr hostDevice value_unit<Value, void>(
  //     const value_unit<T, U> &rhs, std::enable_if_t<!SI::is_equiv_unit<void, U>::value, int *> = 0) = delete;

  template <typename T, typename U>
  constexpr hostDevice value_unit<Value, void>(
      const value_unit<T, U> &rhs, std::enable_if_t<SI::is_equiv_unit<void, U>::value, int *> = 0) {
    val = rhs.val;
  }

  constexpr hostDevice operator Value() const { return val; }
};