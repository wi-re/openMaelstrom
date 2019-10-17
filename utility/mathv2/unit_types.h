#pragma once
#include <cstdint>
#include <utility/macro.h>
#include <utility/mathv2/detection.h>
#include <utility/mathv2/tools.h>
#include <utility/mathv2/vector_types.h>
#include <utility/mathv2/get.h>
#ifndef WIN32
#include <limits>
#endif
template <typename T, uint32_t dim> struct vec_ty;
template <> struct vec_ty<float, 0> { using type = float; };
template <> struct vec_ty<float, 2> { using type = float2; };
template <> struct vec_ty<float, 3> { using type = float3; };
template <> struct vec_ty<float, 4> { using type = float4; };
template <> struct vec_ty<double, 0> { using type = double; };
template <> struct vec_ty<double, 2> { using type = double2; };
template <> struct vec_ty<double, 3> { using type = double3; };
template <> struct vec_ty<double, 4> { using type = double4; };
template <> struct vec_ty<int32_t, 0> { using type = int32_t; };
template <> struct vec_ty<int32_t, 2> { using type = int2; };
template <> struct vec_ty<int32_t, 3> { using type = int3; };
template <> struct vec_ty<int32_t, 4> { using type = int4; };
template <> struct vec_ty<uint32_t, 0> { using type = uint32_t; };
template <> struct vec_ty<uint32_t, 2> { using type = uint2; };
template <> struct vec_ty<uint32_t, 3> { using type = uint3; };
template <> struct vec_ty<uint32_t, 4> { using type = uint4; };

template <typename T, uint32_t d> struct vector_t {
  using type = typename vec_ty<T, d>::type;
  using base_type = T;
  static constexpr const uint32_t dim = d;

  hostDeviceInline static constexpr type make(base_type a = base_type{});
  hostDeviceInline static constexpr auto min() {
    return make(std::numeric_limits<base_type>::min());
  }
  hostDeviceInline static constexpr auto max() {
    return make(std::numeric_limits<base_type>::max());
  }
  hostDeviceInline static constexpr auto one() { return make(base_type{1}); }
  hostDeviceInline static constexpr auto zero() { return make(base_type{0}); }
};

template <typename T>
using vec = vector_t<decltype(math::weak_get_s<1>(std::declval<T>())),
                     math::dimension<T>::value>;

/* This type represents a normal type Ty associated with some SI_unit type. This
 * type allows the system to work with the units as compile time constants and
 * as such allows only the compilation of "correct" functions, e.g. prohibiting
 * calls like float_u<SI::m> a = 0.5s; In order to achieve this properly a
 * significant number of constructors are required which are chosen such that no
 * invalid configuration should be possible. The internally stored value is zero
 * initialized by default and the value_unit can also wrap around references by
 * using value_unit<Ty&,unit> which binds to another reference to an original
 * unit less type which can be useful in generic methods.*/
template <typename Ty, typename _unit> struct value_unit {
  // The only member allowed in these types can be one that is exactly the size
  // of the original type. This is done to preserve byte size compatibility to
  // easily cast arrays from Ty to value_unit<Ty,_unit> without issues. Do not
  // however that allocation of arrays should still be done on the original Ty
  // to respect any potential alignment requirements that might not be properly
  // respected transitively by some compilers.
  Ty val;
  // alias definitions for writing meta code
  using unit = _unit;
  using type = Ty;
  // alias definitions to reduce verbosity on constructors
  using elem_t = std::decay_t<decltype(
      math::weak_get_s<1>(std::declval<std::decay_t<Ty>>()))>;
  using unit_t = value_unit<elem_t, unit>;
  // as this is a static constexpr value the size of the type is not increased.
  static constexpr const auto dimension =
      math::unit_dimension_v<std::decay_t<Ty>>;

  // Constructor for reference semantics. The version required for unit based
  // references, e.g. to value_unit<U&,_unit> in a transitive way is found
  // further on. However, all other constructors need to be disaabled via
  // checking Ty for being a lvalue reference. The template <typename U = Ty> is
  // required to make this constructor, and the other constructors, eligible for
  // SFINAE deletion
  template <typename U = Ty>
  constexpr hostDevice value_unit(std::decay_t<Ty> &v,
                       enable_ty((std::is_lvalue_reference<U>::value)))
      : val(v) {}
  // Copy constructor
  template <typename U = Ty>
 constexpr  hostDevice value_unit(const typename vector_t<elem_t, dimension>::type &lhs =
                           vector_t<elem_t, dimension>::zero(),
                       enable_ty((!std::is_lvalue_reference<U>::value)))
      : val(lhs) {}
  // Set of constructors for 1 2 3 and 4 dimensional arguments. This could be
  // done via parameter packs, however in that case it becomes very difficult to
  // ensure type compatibiity to not allow for undesired conversions between
  // incompatible units. Additionally the constructor taking 1 argument is
  // disabled for scalar types (dimension == 0) as that specific case is
  // already handled by the copy constructor taking basic Ty.
  template <typename U = Ty,
            enable_ty((dimension == 1 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(elem_t a0) : val{a0} {}
  template <typename U = Ty,
            enable_ty((dimension == 2 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(elem_t a0, elem_t a1 = elem_t{}) : val{a0, a1} {}
  template <typename U = Ty,
            enable_ty((dimension == 3 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(elem_t a0, elem_t a1 = elem_t{}, elem_t a2 = elem_t{}) : val{a0, a1, a2} {}
  template <typename U = Ty,
            enable_ty((dimension == 4 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(elem_t a0, elem_t a1 = elem_t{}, elem_t a2 = elem_t{}, elem_t a3 = elem_t{})
      : val{a0, a1, a2, a3} {}

  // Constructors for unit based arguments
  // Constructor for transitive reference semantics
  template <typename U,
            enable_ty((is_unit<U> &&
                       (std::is_lvalue_reference<U>::value ||
                        std::is_lvalue_reference<SI::type_of<U>>::value)))>
  constexpr hostDevice value_unit(U &&v) : val(v.val) {}

  // Element based constructors similar to before. The version for 1 argument is
  // completly removed now as this case is handled by the default copy
  // constructor.
  template <typename U = Ty,
            enable_ty((dimension == 2 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(unit_t a0, unit_t a1) : val{a0.val, a1.val} {}
  template <typename U = Ty,
            enable_ty((dimension == 3 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(unit_t a0, unit_t a1, unit_t a2 = unit_t{})
      : val{a0.val, a1.val, a2.val} {}
  template <typename U = Ty,
            enable_ty((dimension == 4 && !std::is_lvalue_reference<U>::value))>
  constexpr hostDevice value_unit(unit_t a0, unit_t a1, unit_t a2 = unit_t{}, unit_t a3 = unit_t{})
      : val{a0.val, a1.val, a2.val, a3.val} {}

  // Assignment operator which is only allowed if either U is unit free or U has
  // the same unit as this type to avoid undesired vonersions.
  template <typename U, enable_ty((SI::same_unit<U, value_unit<Ty, _unit>> ||
                                   !is_unit<U>))>
  constexpr hostDevice value_unit &operator=(U &&other) {
    val = math::getValue(std::forward<U>(other));
    return *this;
  }
  template <typename U,
            enable_ty((!SI::same_unit<U, value_unit<Ty, _unit>> && is_unit<U>))>
  constexpr  hostDevice value_unit &operator=(U &&other) = delete;
};
