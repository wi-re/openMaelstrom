#define NEW_STYLE
#include <SPH/convection/gravity.cuh>
#include <utility/include_all.h>

// This function adds a simple consant acceleration to all particles in the simulation.
basicFunctionType gravityForce(SPH::External::Memory arrays) {
  checkedParticleIdx(i);
#ifdef DEBUG_INVALID_PARITLCES
  auto a_old = arrays.acceleration[i].val;
  auto a_add = math::castTo<float4>(arrays.external_force);
  if (a_add.x != a_add.x)
	  printf("%s: Invalid particle %d: " _VECSTR " + " _VECSTR "\n",
		  __FUNCTION__, i,
		  _VEC(a_old), _VEC(a_add));
#endif
  arrays.acceleration[i] += math::castTo<float4>(arrays.external_force);
}

basicFunction(gravity, gravityForce, "External Force");

void SPH::External::gravity_force(Memory mem) { launch<gravity>(mem.num_ptcls, mem); }
