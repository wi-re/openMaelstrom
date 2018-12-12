#pragma once
// required for int32_t and similar types
#include <cstdint>
// required for float4 etc, will emulate types if cuda is not available
#include <utility/mathv2/vector_types.h>

// Helper macro to replace std::enable_if_t<x> * = nullptr due to verbosity
#define enable_ty(x) std::enable_if_t<x> * = nullptr
// Helper macro for CUDA to create ", typename = void" dummies
#define enable_dummy , typename = void

/** Base types for unit system to create shorter names in compile warnings these
 * are defined at global namespace level. These types offer a representation of
 * fractions Q = N/N which are used to determine the units of a type using
 * operators defined in mathv2/ratio.h **/
#define RATIO_TYPE(x)                                                          \
  template <int32_t n = 0, int32_t d = 1> struct x {                           \
    static constexpr int32_t num = n;                                          \
    static constexpr int32_t den = d;                                          \
  };
RATIO_TYPE(SI_m);
RATIO_TYPE(SI_kg);
RATIO_TYPE(SI_s);
RATIO_TYPE(SI_A);
RATIO_TYPE(SI_K);
RATIO_TYPE(SI_mol);
RATIO_TYPE(SI_cd);

/** This type is the basis for all unit calculations and contains one template
 * parameter per SI unit. This for example would repsent velocity [m/s] as the
 * instantiation:
 * SI_value_unit<SI_m<1,1>, SI_kg<0,1>, SI_s<-1,1>, SI_A<0,1>, SI_K<0,1>,
 * SI_mol<0,1>, SI_cd<0,1>>
 *
 * This listing of unused units is not very elegant but a more elegant solution,
 * e.g. using tuples to store the lists of actual templates, can lead to
 * problems on certain compilers which do not allow for the necessary
 * simplifications in the type system to make this system fully useable. **/
template <typename m = SI_m<>, typename kg = SI_kg<>, typename s = SI_s<>,
          typename A = SI_A<>, typename K = SI_K<>, typename mol = SI_mol<>,
          typename cd = SI_cd<>>
struct SI_value_unit {
  using _m = m;
  using _kg = kg;
  using _s = s;
  using _A = A;
  using _K = K;
  using _mol = mol;
  using _cd = cd;

  using _value_unit_identifier = void;
};

/** Wrappers to make the global namespace templates available within the SI
 * namespace for clarity in other parts. **/
namespace SI {
namespace Base {
template <int32_t n = 0, int32_t d = 1> using m = ::SI_m<n, d>;
template <int32_t n = 0, int32_t d = 1> using kg = ::SI_kg<n, d>;
template <int32_t n = 0, int32_t d = 1> using s = ::SI_s<n, d>;
template <int32_t n = 0, int32_t d = 1> using A = ::SI_A<n, d>;
template <int32_t n = 0, int32_t d = 1> using K = ::SI_K<n, d>;
template <int32_t n = 0, int32_t d = 1> using mol = ::SI_mol<n, d>;
template <int32_t n = 0, int32_t d = 1> using cd = ::SI_cd<n, d>;
} // namespace Base

template <typename m = SI_m<>, typename kg = SI_kg<>, typename s = SI_s<>,
          typename A = SI_A<>, typename K = SI_K<>, typename mol = SI_mol<>,
          typename cd = SI_cd<>>
using unit_ty = SI_value_unit<m, kg, s, A, K, mol, cd>;
} // namespace SI

template <typename Ty = float, typename _unit = SI::unit_ty<>>
struct value_unit;

