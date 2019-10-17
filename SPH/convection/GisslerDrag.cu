#define NEW_STYLE
#include <SPH/convection/GisslerDrag.cuh>
#include <utility/include_all.h>

// This function calculates not just the deformation but the complete air drag force. Contrary to
// the way the paper is structured this all can be done within a single method as most of this
// method can be calculated on the fly on a per particle basis.
neighFunctionType computeDeformation(SPH::GisslerDrag::Memory arrays) {
	checkedParticleIdx(i);
	alias_arrays((pos, position));
	const float_u<SI::density> rho_a{ 1.2041f };
	const float_u<SI::derived_unit<SI::N, SI::recip<SI::m>>> sigma{ 0.0724f };
	const float_u<SI::derived_unit<SI::Pa, SI::s>> mu_l{ 0.00102f };
	const auto C_F = 1.0f / 3.0f;
	const auto C_k = 8.0f;
	const auto C_d = 5.0f;
	const auto C_b = 0.5f;
	const float_u<SI::derived_unit<SI::Pa, SI::s>> mu_a{ 0.00001845f };

	const auto radius = math::power<ratio<1,3>>(arrays.volume[i] / PI4O3);
	const auto diam = 2.f * radius;
	static const float pi = CUDART_PI_F;
	const value_unit<float,SI::density> rho_l = arrays.rest_density;

	// Air velocity.
	auto v3_a = arrays.air_velocity;
	auto va = float4_u<SI::velocity>{ v3_a.val.x,v3_a.val.y, v3_a.val.z,0.f };

	const float_u<SI::m> L = math::pow(PI4O3_1, 1.f/3.f) * diam;

	const float_u<SI::Hz> inv_td = 0.5f*C_d * mu_l / (rho_l * L*L);
	const float_u<SI::s> td = 1.0f / inv_td;
	const float_u<SI::square<SI::Hz>> omegaSquare = C_k * sigma / (rho_l * L*L*L) - inv_td*inv_td;
	
	const float_u<SI::Hz> omega = math::power<ratio<1,2>>(omegaSquare);

	// Equation (6)
	float_u<> val = td*td*omegaSquare;
	val = math::sqrt(val + 1.f) + td*omega;
	val = math::max(val, -0.5f * pi);
	val = math::min(val, 0.5f * pi);
	const float_u<SI::s> t_max = -2.f * (math::atan(val) - pi) / omega;

	// Equation (7)
	const float_u<> c_def = 1.f - math::expf((-t_max / td)) * (math::cosf((omega * t_max)) + 1.f / (omega*td) * math::sinf((omega * t_max)));

	// Weber number without velocity

	const float_u<SI::recip_2<SI::velocity>> We_i_wo_v = rho_a * L / sigma;

	// Equation (8)
	const auto y_coeff = (C_F * We_i_wo_v * c_def) / (C_k * C_b);

	const auto n_full = Kernel<kernel_kind::spline4>::neighbor_number;
	const auto n_full_23 = n_full * 2.f / 3.f;

	auto vi = arrays.velocity[i];
	auto v_i_rel = va - vi;
	const float_u<SI::square<SI::velocity>> vi_rel_square = math::length3(v_i_rel) * math::length3(v_i_rel);
	const float_u<SI::velocity> vi_rel_norm = math::sqrt(vi_rel_square);
	//const float_u<> We_i = We_i_wo_v * vi_rel_square;

	if (vi_rel_norm <= 1.0e-6f)
		return;
	// Else.
	float4_u<> v_i_rel_n = v_i_rel / vi_rel_norm;
	// Equation (8)
	const auto y_i_max = math::min(vi_rel_square * y_coeff, 1.f);
	 // 1 / ( m^-1 s^-1)
	const float_u<> Re_i = 2.f*math::max((rho_a * vi_rel_norm * L) / mu_a, 0.1f);
	float_u<> C_Di_sphere;
	if (Re_i <= 1000.0f)
		C_Di_sphere = 24.0f / Re_i * (1.0f + 1.0f / 6.0f * math::power<ratio<2,3>>(Re_i));
	else
		C_Di_sphere = 0.424f;

	// Equation (9)
	const float_u<> C_Di_Liu = (C_Di_sphere * (1.0f + 2.632f * y_i_max));

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	float_u<> max_v_x{ 0.f };
	auto xi = arrays.position[i];
	int32_t numNeighbors = 0;
	iterateNeighbors(j) {
		if (W_ij == 0.f) continue;
		if (i == j) continue;
		auto xj = pos[j];	
		auto xixj = xi - xj;
		auto xixj_n = math::normalize(xixj);
		const auto x_v = math::dot3(v_i_rel_n, xixj_n);
		max_v_x = math::max(max_v_x, x_v.val);

		numNeighbors++;
	}
	
	// Equation (10)
	float_u<> C_Di;
	const auto factor_n = math::min(n_full_23, (float)numNeighbors) / n_full_23;
	if (numNeighbors == 1)
		C_Di = C_Di_Liu;
	else
		C_Di = (1.f - factor_n) * C_Di_Liu + factor_n;

	// Equation (12)
	const auto h1 = (L + C_b*L*y_i_max);
	const auto A_i_droplet = pi * h1*h1;

	// Equation (13)
	const auto A_i_unoccluded = (1.f - factor_n) * A_i_droplet + factor_n * diam*diam;

	// Equation (15)
	auto w_i = math::max(0.f, math::min(1.f, 1.f - max_v_x));
	auto A_i = w_i * A_i_unoccluded;
	auto accel = 0.5f / (arrays.volume[i] * arrays.rest_density) * rho_a * (v_i_rel * vi_rel_norm) * C_Di * A_i;

	// Equation (14)
	
#ifdef DEBUG_INVALID_PARITLCES
	auto a_old = arrays.acceleration[i].val;
	auto a_add = accel.val;
	if (a_add.x != a_add.x)
		printf("%s: Invalid particle %d: " _VECSTR " + " _VECSTR " : %f %f %f %f %f %f " _VECSTR  _VECSTR "\n",
			__FUNCTION__, i,
			_VEC(a_old), _VEC(a_add), arrays.volume[i].val, arrays.rest_density.val, rho_a.val, vi_rel_norm.val, C_Di.val, A_i.val,
		_VEC(va.val), _VEC(vi.val));
#endif
	arrays.acceleration[i] += accel;
}
 
neighFunction(dragForce, computeDeformation, "Gissler Drag Force", caches<float4, float>{});

void SPH::GisslerDrag::drag(Memory mem){
	launch<dragForce>(mem.num_ptcls, mem);
}
