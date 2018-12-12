#pragma once
#define WINDOWS_DUMMY_TOKEN

//#pragma region MACROS
#define GENERATE_HAS_MEMBER(member)                                                                \
  template <typename T> struct Has##member {                                                       \
    struct Fallback {                                                                              \
      int member;                                                                                  \
    };                                                                                             \
    struct Derived : T, Fallback {};                                                               \
                                                                                                   \
    template <typename C, C> struct ChT;                                                           \
                                                                                                   \
    template <typename C> static char (&f(ChT<int Fallback::*, &C::member> *))[1];                 \
    template <typename C> static char (&f(...))[2];                                                \
                                                                                                   \
    static bool const value = sizeof(f<Derived>(nullptr)) == 2;                                    \
  };                                                                                               \
  template <typename T, typename Enable = void> struct has_member_##member {};                     \
  template <class T>                                                                               \
  struct has_member_##member<T, typename std::enable_if<std::is_class<T>::value>::type>            \
      : public std::integral_constant<bool, Has##member<T>::value> {};                             \
  template <class T>                                                                               \
  struct has_member_##member<T, typename std::enable_if<!std::is_class<T>::value>::type>           \
      : public std::integral_constant<bool, false> {};

#define OPERATOR_NEW(op)                                                                           \
  template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,           \
            typename = void, typename = void, typename = void>                                     \
  hostDeviceInline auto operator op##=(T &a, U&& b)->math::return_type<T, U> {                \
    math::get<1>(a) op## = math::get<1>(b);                                                        \
    math::get<2>(a) op## = math::get<2>(b);                                                        \
    math::get<3>(a) op## = math::get<3>(b);                                                        \
    math::get<4>(a) op## = math::get<4>(b);                                                        \
    return a;                                                                                      \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,           \
            typename = void, typename = void>                                                      \
  hostDeviceInline auto operator op##=(T &a, U&& b)->math::return_type<T, U> {                \
    math::get<1>(a) op## = math::get<1>(b);                                                        \
    math::get<2>(a) op## = math::get<2>(b);                                                        \
    math::get<3>(a) op## = math::get<3>(b);                                                        \
    return a;                                                                                      \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,           \
            typename = void>                                                                       \
  hostDeviceInline auto operator op##=(T &a, U&& b)->math::return_type<T, U> {                \
    math::get<1>(a) op## = math::get<1>(b);                                                        \
    math::get<2>(a) op## = math::get<2>(b);                                                        \
    return a;                                                                                      \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 1>>                            \
  hostDeviceInline auto operator op##=(T &a, U&& b)->math::return_type<T, U> {                \
    math::get<1>(a) op## = math::get<1>(b);                                                        \
    return a;                                                                                      \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 4>, typename = void,           \
            typename = void, typename = void, typename = void>                                     \
  hostDeviceInline auto operator op(T&& a, U&& b)->math::return_type<T, U> {                    \
    return math::return_type<T, U>{                                                                \
        math::get<1>(a) op math::get<1>(b), math::get<2>(a) op math::get<2>(b),                    \
        math::get<3>(a) op math::get<3>(b), math::get<4>(a) op math::get<4>(b)};                   \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 3>, typename = void,           \
            typename = void, typename = void>                                                      \
  hostDeviceInline auto operator op(T&& a, U&& b)->math::return_type<T, U> {                    \
    return math::return_type<T, U>{math::get<1>(a) op math::get<1>(b),                             \
                                   math::get<2>(a) op math::get<2>(b),                             \
                                   math::get<3>(a) op math::get<3>(b)};                            \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 2>, typename = void,           \
            typename = void>                                                                       \
  hostDeviceInline auto operator op(T&& a, U&& b)->math::return_type<T, U> {                    \
    return math::return_type<T, U>{math::get<1>(a) op math::get<1>(b),                             \
                                   math::get<2>(a) op math::get<2>(b)};                            \
  }                                                                                                \
  template <typename T, typename U, typename = math::DimCheck<T, U, 1>>                            \
  hostDeviceInline auto operator op(T&& a, U&& b)->math::return_type<T, U> {                    \
    return math::return_type<T, U>{math::get<1>(a) op math::get<1>(b)};                            \
  }

//#define greaterDimension(check_dim, T)                                                               
//  typename std::enable_if<(math::dimension<T>::value > check_dim), std::nullptr_t>::type

//#pragma endregion