#include <SPH/generation/particleCleanup.cuh>
#include <utility/include_all.h>

basicFunctionType boundary_remove_particles(SPH::cleanup::Memory arrays){
    checkedParticleIdx(i);
    auto p = arrays.position[i];
    auto r = 0.35f * arrays.radius;
    bool invalid = false;
  auto POS = pDistance;
  if (POS.val.w <= r) {
      invalid = true;
  }
  for (int32_t b = 0; b < arrays.volumeBoundaryCounter; ++b) {
	  auto VOS = volumeDistance(b);
	  if (VOS.val.w <= r) {
		  invalid = true;
	  }
  }
  if(invalid)
    math::get<4>(arrays.position[i]) = FLT_MAX;
}

basicFunction(cleanupParticles, boundary_remove_particles, "Cleanup particles", caches<>{})

void SPH::cleanup::cleanup_particles(Memory mem){
    launch<cleanupParticles>(mem.num_ptcls, mem);
    get<parameters::particleCleanUp>() = false;
}