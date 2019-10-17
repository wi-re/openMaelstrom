#define NEW_STYLE
#include <SPH/convection/monaghanViscosity.cuh>
#include <utility/include_all.h>

//basicFunctionType BoundaryDragKernel(SPH::Viscosity::Memory arrays) {
//	checkedParticleIdx(i);
//	auto x_i = arrays.position[i];
//	auto a_i = arrays.acceleration[i];
//	auto v_i = arrays.velocity[i] + a_i * arrays.timestep;
//	auto H_i = support_H(arrays.position[i]);
//
//	auto cs = 300.f;
//	auto eps = 0.01f;
//	auto boundaryViscosity = arrays.boundaryViscosity / cs;
//	auto nu = 2.f * boundaryViscosity * H_i * cs / (arrays.density[i] * arrays.rest_density + arrays.rest_density);
//
//	auto volumeDistance = pDistance.val;
//	if (volumeDistance.w >= 1e20f)
//		return;
//	auto x_b = x_i + volumeDistance * (volumeDistance.w - 1.f * H_i);
//	auto v_b = float4_u < SI::velocity>{ 0.f,0.f,0.f,0.f };
//
//	auto v_ib = v_i - v_b;
//	auto x_ib = x_i - x_b;
//
//	auto dot = math::dot3(v_ib, x_ib);
//	//if (dot >= 0.f) 
//	{
//		auto PI = -nu * math::min(dot, float_u<SI::derived_unit<SI::area, SI::Hz>>(0.f)) / (math::sqlength3(x_ib) + eps * math::square(H_i));
//
//		a_i += -arrays.rest_density * PI * volumeDistance * pGW_ib;
//		arrays.acceleration[i] = a_i;
//	}
//}
//basicFunctionType VolumeDragKernel(SPH::Viscosity::Memory arrays) {
//	checkedParticleIdx(i);
//	auto x_i = arrays.position[i];
//	auto a_i = arrays.acceleration[i];
//	auto v_i = arrays.velocity[i] + a_i * arrays.timestep;
//	auto H_i = support_H(arrays.position[i]);
//
//	auto cs = 300.f;
//	auto eps = 0.01f;
//	auto boundaryViscosity = arrays.boundaryViscosity / cs;
//	auto nu = 2.f * boundaryViscosity * H_i * cs / (arrays.density[i] * arrays.rest_density + arrays.rest_density);
//	for (int32_t b = 0; b < arrays.boundaryCounter; ++b) {
//		auto volumeDistance = vDistance(b).val;
//		if (volumeDistance.w >= 1e20f)
//			continue;
//		auto x_b = x_i + volumeDistance * (volumeDistance.w - 1.f * H_i);
//		auto v_b = float4_u < SI::velocity>{ 0.f,0.f,0.f,0.f };
//
//		auto v_ib = v_i - v_b;
//		auto x_ib = x_i - x_b;
//		auto dot = math::dot3(v_ib, x_ib);
//		//if (dot >= 0.f) 
//		{
//			auto PI = -nu * math::min(dot, float_u<SI::derived_unit<SI::area, SI::Hz>>(0.f)) / (math::sqlength3(x_ib) + eps * math::square(H_i));
//
//			a_i += -arrays.rest_density * PI * vGW_ib(b);
//		}
//	}
//	arrays.acceleration[i] = a_i;
//}
// This function implements a classical artificial viscosity based on the works of Monaghan. However
// this function only causes an actual force if a particles lifetime is negative. This is only the
// case for particles created with splitting which results in an additional layer of blending on top
// of the normal one. Contrary to the normal velocity blending this method basically blends the
// velocity of split particles with all surrounding particles which helps dampen the effects caused
// by density errors on splitting.
neighFunctionType ViscosityForce(SPH::Viscosity::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume), (vel, velocity), (dens, density));
	auto beta_i = getBlendWeight(arrays, i);
	auto support = support_H(pos[i]);

	auto accel = arrays.acceleration[i];

	auto cs = 300.f;

	iterateNeighbors(j) {
		if (i == j)
			continue;
		auto support = (support_H(pos[i]) + support_H(pos[j])) * 0.5f;

		auto beta_j = getBlendWeight(arrays, j);
		auto beta_ij = 0.5f * (beta_i + beta_j);
		auto viscosity_constant = arrays.monaghan_viscosity / cs  * beta_ij; 

		auto dist_vec = pos[i] - pos[j];
		auto dot = math::dot3(vel[i] + arrays.acceleration[i] * arrays.timestep - vel[j] + arrays.acceleration[j] * arrays.timestep, pos[i] - pos[j]);
		if (dot < 0.f) {
			auto nu = 2.f * viscosity_constant * support * cs / (dens[i] * arrays.rest_density + dens[j] * arrays.rest_density);
			auto PI = -nu * (dot / (math::sqlength3(dist_vec) + support * support * 0.01f));

			accel += -arrays.volume[j] * arrays.rest_density * PI * GW_ij;
		}
	}
#ifdef DEBUG_INVALID_PARITLCES
	auto a_old = arrays.acceleration[i].val;
	auto a_add = accel.val;
	if (a_add.x != a_add.x)
		printf("%s: Invalid particle %d: " _VECSTR " : " _VECSTR "\n",
			__FUNCTION__, i,
			_VEC(a_old), _VEC(a_add));
#endif
	arrays.acceleration[i] = accel;
}

neighFunction(monaghanViscosity, ViscosityForce, "Monaghan Viscosity", caches<float4, float4, float, float>{});
basicFunction(boundaryDrag, BoundaryDragKernel, "Boundary Drag Force");
basicFunction(volumeDrag, VolumeDragKernel, "Volume Drag Force");

void SPH::Viscosity::drag(Memory mem) {
	//launch<boundaryDrag>(mem.num_ptcls, mem);
	//launch<volumeDrag>(mem.num_ptcls, mem);
}
void SPH::Viscosity::viscosity(Memory mem) {
	static bool once = true;
	if (once) {
		std::cout << "Viscosity module built " << __TIMESTAMP__ << std::endl;
		once = false;
	}
	if(get<parameters::modules::adaptive>() == true)
		launch<monaghanViscosity>(mem.num_ptcls, mem);
}
