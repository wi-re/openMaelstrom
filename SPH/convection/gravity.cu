#define NEW_STYLE
#include <SPH/convection/gravity.cuh>
#include <utility/include_all.h>

// This function adds a simple consant acceleration to all particles in the simulation.
basicFunctionType gravityForce(SPH::External::Memory arrays) {
  checkedParticleIdx(i);
  arrays.acceleration[i] += math::castTo<float4>(arrays.external_force);
}

basicFunction(gravity, gravityForce, "External Force");

void SPH::External::gravity_force(Memory mem) { launch<gravity>(mem.num_ptcls, mem); }
