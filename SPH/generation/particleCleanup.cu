#include <SPH/generation/particleCleanup.cuh>
#include <utility/include_all.h>

basicFunctionType boundary_remove_particles(SPH::cleanup::Memory arrays){
    checkedParticleIdx(i);
    auto p = arrays.position[i];
    auto r = arrays.radius * 0.f;
    bool invalid = false;
  auto POS = boundary::POSfunction(p, arrays);
  if (POS.val.w <= r) {
      invalid = true;
  }
  auto VOS = volumeBoundary::volumeDistance(p, arrays);
  if (VOS.val.w <= r) {
      invalid = true;
  }
  if(invalid)
    math::get<4>(arrays.position[i]) = FLT_MAX;
}

basicFunction(cleanupParticles, boundary_remove_particles, "Cleanup particles", caches<>{})

void SPH::cleanup::cleanup_particles(Memory mem){
    launch<cleanupParticles>(mem.num_ptcls, mem);
    get<parameters::particleCleanUp>() = false;
}