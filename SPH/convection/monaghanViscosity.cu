#define NEW_STYLE
#include <SPH/convection/monaghanViscosity.cuh>
#include <utility/include_all.h>

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

	iterateNeighbors(j) {
		auto beta_j = getBlendWeight(arrays, j);
		auto beta_ij = 0.5f * (beta_i + beta_j);
		auto viscosity_constant = arrays.monaghan_viscosity * beta_ij;

		auto dist_vec = pos[i] - pos[j];
		auto velocity_term = math::dot3(vel[j] - vel[i], pos[i] - pos[j]) /
			(math::sqlength(dist_vec) + support * support * 0.01f);

		auto viscosity_term = (2.0f * support * viscosity_constant) / (dens[j] + dens[i]);

		accel += -arrays.volume[j] * viscosity_term * velocity_term * GW_ij;
	}
	arrays.acceleration[i] = accel;
}

neighFunction(monaghanViscosity, ViscosityForce, "Monaghan Viscosity", caches<float4, float4, float, float>{});

void SPH::Viscosity::viscosity(Memory mem) { launch<monaghanViscosity>(mem.num_ptcls, mem); }
