#pragma once
#include "baseTypes.h"
#include "detection.h"
#include "ratio.h"
#include "tools.h"

template<typename T, typename U>
struct pair {
	T first;
	U second;
};


/* Helpful unit definitions for easier access to things */
namespace SI {
template <typename T> using recip = multiply_ratios<T, SI::ratio<-1, 1>>;
template <typename T> using recip_2 = multiply_ratios<T, SI::ratio<-2, 1>>;
template <typename T> using recip_3 = multiply_ratios<T, SI::ratio<-3, 1>>;
template <typename T> using square = multiply_ratios<T, SI::ratio<2, 1>>;
template <typename T> using cubic = multiply_ratios<T, SI::ratio<3, 1>>;

using m =
    SI::unit_ty<SI::Base::m<1>, SI::Base::kg<0>, SI::Base::s<0>, SI::Base::A<0>,
                SI::Base::K<0>, SI::Base::mol<0>, SI::Base::cd<0>>;
using kg =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<1>, SI::Base::s<0>, SI::Base::A<0>,
                SI::Base::K<0>, SI::Base::mol<0>, SI::Base::cd<0>>;
using s =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<0>, SI::Base::s<1>, SI::Base::A<0>,
                SI::Base::K<0>, SI::Base::mol<0>, SI::Base::cd<0>>;
using A =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<0>, SI::Base::s<0>, SI::Base::A<1>,
                SI::Base::K<0>, SI::Base::mol<0>, SI::Base::cd<0>>;
using K =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<0>, SI::Base::s<0>, SI::Base::A<0>,
                SI::Base::K<1>, SI::Base::mol<0>, SI::Base::cd<0>>;
using mol =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<0>, SI::Base::s<0>, SI::Base::A<0>,
                SI::Base::K<0>, SI::Base::mol<1>, SI::Base::cd<0>>;
using cd =
    SI::unit_ty<SI::Base::m<0>, SI::Base::kg<0>, SI::Base::s<0>, SI::Base::A<0>,
                SI::Base::K<0>, SI::Base::mol<0>, SI::Base::cd<1>>;

// named units except for lx (eqv to cd unit wise), rad (unitless) and sr
// (unitless)
using Hz = SI::combine<recip<s>>;
using N = SI::combine<kg, m, recip_2<s>>;
using Pa = SI::combine<N, recip_2<m>>;
using J = SI::combine<N, m>;
using W = SI::combine<J, recip<s>>;
using C = SI::combine<s, A>;
using V = SI::combine<W, recip<A>>;
using F = SI::combine<C, recip<V>>;
using Ohm = SI::combine<V, recip<A>>;
using S = SI::combine<A, recip<V>>;
using Wb = SI::combine<J, recip<A>>;
using T = SI::combine<Wb, recip_2<m>>;
using H = SI::combine<Ohm, s>;
using Bq = SI::combine<recip<s>>;
using Sv = SI::combine<J, recip<kg>>;
using Gy = SI::combine<J, recip<kg>>;
using kat = SI::combine<mol, recip<s>>;

// useful derived units
using area = SI::combine<square<m>>;
using volume = SI::combine<cubic<m>>;
using density = SI::combine<kg, recip_3<m>>;
using velocity = SI::combine<m, Hz>;
using acceleration = SI::combine<velocity, Hz>;

// helper base units for ease of use
using m1 = multiply_ratios<m, ratio<1, 1>>;
using m2 = multiply_ratios<m, ratio<2, 1>>;
using m3 = multiply_ratios<m, ratio<3, 1>>;
using m4 = multiply_ratios<m, ratio<4, 1>>;
using m5 = multiply_ratios<m, ratio<5, 1>>;
using m_1 = multiply_ratios<m, ratio<-1, 1>>;
using m_2 = multiply_ratios<m, ratio<-2, 1>>;
using m_3 = multiply_ratios<m, ratio<-3, 1>>;
using m_4 = multiply_ratios<m, ratio<-4, 1>>;
using m_5 = multiply_ratios<m, ratio<-5, 1>>;

using kg1 = multiply_ratios<kg, ratio<1, 1>>;
using kg2 = multiply_ratios<kg, ratio<2, 1>>;
using kg3 = multiply_ratios<kg, ratio<3, 1>>;
using kg4 = multiply_ratios<kg, ratio<4, 1>>;
using kg5 = multiply_ratios<kg, ratio<5, 1>>;
using kg_1 = multiply_ratios<kg, ratio<-1, 1>>;
using kg_2 = multiply_ratios<kg, ratio<-2, 1>>;
using kg_3 = multiply_ratios<kg, ratio<-3, 1>>;
using kg_4 = multiply_ratios<kg, ratio<-4, 1>>;
using kg_5 = multiply_ratios<kg, ratio<-5, 1>>;

using s1 = multiply_ratios<s, ratio<1, 1>>;
using s2 = multiply_ratios<s, ratio<2, 1>>;
using s3 = multiply_ratios<s, ratio<3, 1>>;
using s4 = multiply_ratios<s, ratio<4, 1>>;
using s5 = multiply_ratios<s, ratio<5, 1>>;
using s_1 = multiply_ratios<s, ratio<-1, 1>>;
using s_2 = multiply_ratios<s, ratio<-2, 1>>;
using s_3 = multiply_ratios<s, ratio<-3, 1>>;
using s_4 = multiply_ratios<s, ratio<-4, 1>>;
using s_5 = multiply_ratios<s, ratio<-5, 1>>;

template<typename Ty, int32_t N = 1, int32_t M = 1>
using exp = multiply_ratios<Ty, ratio<N, M>>;
template<typename... Ts>
using combine = derived_unit<Ts...>;

} // namespace SI

template <typename _unit = SI::unit_ty<>>
using float_u = value_unit<float, _unit>;
template <typename _unit = SI::unit_ty<>>
using float2_u = value_unit<float2, _unit>;
template <typename _unit = SI::unit_ty<>>
using float3_u = value_unit<float3, _unit>;
template <typename _unit = SI::unit_ty<>>
using float4_u = value_unit<float4, _unit>;

template <typename... _unit>
using uFloat = value_unit<float, SI::combine<_unit...>>;
template <typename... _unit>
using uFloat2 = value_unit<float2, SI::combine<_unit...>>;
template <typename... _unit>
using uFloat3 = value_unit<float3, SI::combine<_unit...>>;
template <typename... _unit>
using uFloat4 = value_unit<float4, SI::combine<_unit...>>;

template <typename _unit = SI::unit_ty<>>
using double_u = value_unit<double, _unit>;
template <typename _unit = SI::unit_ty<>>
using double2_u = value_unit<double2, _unit>;
template <typename _unit = SI::unit_ty<>>
using double3_u = value_unit<double3, _unit>;
template <typename _unit = SI::unit_ty<>>
using double4_u = value_unit<double4, _unit>;

template <typename... _unit>
using uDouble = value_unit<double, SI::combine<_unit...>>;
template <typename... _unit>
using uDouble2 = value_unit<double2, SI::combine<_unit...>>;
template <typename... _unit>
using uDouble3 = value_unit<double3, SI::combine<_unit...>>;
template <typename... _unit>
using uDouble4 = value_unit<double4, SI::combine<_unit...>>;

template <typename _unit = SI::unit_ty<>> using int_u = value_unit<int, _unit>;
template <typename _unit = SI::unit_ty<>> using int2_u = value_unit<int2, _unit>;
template <typename _unit = SI::unit_ty<>> using int3_u = value_unit<int3, _unit>;
template <typename _unit = SI::unit_ty<>> using int4_u = value_unit<int4, _unit>;

typedef uint32_t uint;

#define UNIT_UDL(u) constexpr value_unit<float, SI::u> operator "" _##u(long double wgt) {return value_unit<float, SI::u>{static_cast<float>(wgt)};}
UNIT_UDL(m);
UNIT_UDL(s);
UNIT_UDL(kg);
UNIT_UDL(A);
UNIT_UDL(K);
UNIT_UDL(mol);
UNIT_UDL(cd);

UNIT_UDL(Hz);
UNIT_UDL(N);
UNIT_UDL(Pa);
UNIT_UDL(J);
UNIT_UDL(W);
UNIT_UDL(C);
UNIT_UDL(V);
UNIT_UDL(F);
UNIT_UDL(Ohm);
UNIT_UDL(S);
UNIT_UDL(Wb);
UNIT_UDL(T);
UNIT_UDL(H);
UNIT_UDL(Bq);
UNIT_UDL(Sv);
UNIT_UDL(Gy);
UNIT_UDL(kat);
#undef UNIT_UDL
#define UNIT_UDL_E(u, e)constexpr value_unit<float, SI::e> operator "" _##u(long double wgt) {return value_unit<float, SI::e>{static_cast<float>(wgt)};}
// useful derived units
UNIT_UDL_E(m2, area);
UNIT_UDL_E(m3, volume);
UNIT_UDL_E(m_s, velocity);
UNIT_UDL_E(m_s2, acceleration);
#undef UNIT_UDL_E