#pragma once
#include <utility/unitmath/SI_Unit.h>

template<typename T, typename U>
struct pair {
	T first;
	U second;
};

namespace SI {
	// Basic SI-Units
	using m = SI_Unit<Base::m>;
	using kg = SI_Unit<Base::kg>;
	using s = SI_Unit<Base::s>;
	using A = SI_Unit<Base::A>;
	using K = SI_Unit<Base::K>;
	using mol = SI_Unit<Base::mol>;
	using cd = SI_Unit<Base::cd>;

	// named units except for lx (eqv to cd unit wise), rad (unitless) and sr (unitless)
	using Hz = derived_unit<recip<s>>;
	using N = derived_unit<kg, m, recip_2<s>>;
	using Pa = derived_unit< N, recip_2<m>>;
	using J = derived_unit< N, m>;
	using W = derived_unit<J, recip<s>>;
	using C = derived_unit<s, A>;
	using V = derived_unit<W, recip<A>>;
	using F = derived_unit<C, recip<V>>;
	using Ohm = derived_unit<V, recip<A>>;
	using S = derived_unit<A, recip<V>>;
	using Wb = derived_unit<J, recip<A>>;
	using T = derived_unit<Wb, recip_2<m>>;
	using H = derived_unit<Ohm, s>;
	using Bq = derived_unit<recip<s>>;
	using Sv = derived_unit<J, recip<kg>>;
	using Gy = derived_unit<J, recip<kg>>;
	using kat = derived_unit<mol, recip<s>>;

	// useful derived units
	using area = derived_unit<square<m>>;
	using volume = derived_unit<cubic<m>>;
	using density = derived_unit<kg, recip_3<m>>;
	using velocity = derived_unit<m, Hz>;
	using acceleration = derived_unit<velocity, Hz>;
}

template<typename... Ts>
using float_u = value_unit<float, SI::derived_unit<Ts...>>;
template<typename... Ts>
using float2_u = value_unit<float2, SI::derived_unit<Ts...>>;
template<typename... Ts>
using float3_u = value_unit<float3, SI::derived_unit<Ts...>>;
template<typename... Ts>
using float4_u = value_unit<float4, SI::derived_unit<Ts...>>;
template<typename... Ts>
using double_u = value_unit<double, SI::derived_unit<Ts...>>;
template<typename... Ts>
using double2_u = value_unit<double2, SI::derived_unit<Ts...>>;
template<typename... Ts>
using double3_u = value_unit<double3, SI::derived_unit<Ts...>>;
template<typename... Ts>
using double4_u = value_unit<double4, SI::derived_unit<Ts...>>;


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