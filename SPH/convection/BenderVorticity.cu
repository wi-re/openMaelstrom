#define NEW_STYLE
#include <SPH/convection/BenderVorticity.cuh>
#include <utility/include_all.h>

// This function implements the Micropolar SPH Model to improve the vorticity of the simulation at
// no significant cost.
neighFunctionType computeVorticity(SPH::BenderVorticity::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));
  alias_arrays((omega, angularVelocity.first), (rho, density), (vel, velocity));
  float4_u<SI::acceleration> angular, linear = arrays.acceleration[i];
  auto density_i2 = 1.f / (rho[i] * rho[i]);
  float_u<SI::velocity> nu_t{arrays.vorticity_nu_t};

  iterateNeighbors(j) {
    //auto kernel = W_ij;
    auto omegaij = omega[i] - omega[j];
    auto density_j2 = 1.f / (rho[j] * rho[j]);

    angular -= 1.f / arrays.timestep * arrays.vorticity_viscosity * vol[j] / rho[j] * omegaij * W_ij;
    angular -= math::cross(
		nu_t * rho[i] * vol[j] * (vel[i] * density_i2 + vel[j] * density_j2) * arrays.intertia_inverse,
                           GW_ij);
    linear -= math::cross(
		nu_t * rho[i] * vol[j] * (omega[i] * density_i2 + omega[j] * density_j2),
                          GW_ij);
  }
  angular -= (2.f * arrays.intertia_inverse * arrays.vorticity_nu_t * omega[i]).val;
  arrays.angularVelocity.second[i] = omega[i] + arrays.timestep * angular;
  arrays.acceleration[i] += linear;
}

neighFunction(Vorticity, computeVorticity, "Bender Vorticity", caches<float4, float>{});

void SPH::BenderVorticity::vorticity(Memory mem) {
  launch<Vorticity>(mem.num_ptcls, mem);
  cuda::memcpy(arrays::angularVelocity::ptr, arrays::angularVelocity::rear_ptr, sizeof(arrays::angularVelocity::type) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
}
