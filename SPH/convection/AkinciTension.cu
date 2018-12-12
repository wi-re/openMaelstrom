#define NEW_STYLE
#include <SPH/convection/AkinciTension.cuh>
#include <utility/include_all.h>

// This function estimates the normal of a particle according to the methods described in the source
// paper. This needs to be done in an initial step.
neighFunctionType akinciFirst(SPH::AkinciTension::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (dens, density));

  float4_u<SI::recip<SI::m>> kernelSum;
  iterateNeighbors(j) kernelSum += vol[j] / dens[j] * GW_ij;

  arrays.particleNormal[i] = (kernelSum * support_H(pos[i])).val;
}

// This function calculates the actual surface tension, however it does not exactly replicate the
// source paper as the source has some issues with the units of the scaling parameter that need to
// manually fixed.
neighFunctionType akinciSecond(SPH::AkinciTension::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (normal, particleNormal), (vol, volume),
               (dens, density));

  float4_u<SI::N> cohesionForce, curvatureForce;

  iterateNeighbors(j) {
    auto kernel = Kernel<cohesion>::value<kernel_kind::spline4>(pos[i], pos[j]);
    if (kernel == 0.f)
      return;
    auto scaling = 2.f * arrays.rest_density /
                   (dens[i] * arrays.rest_density + dens[j] * arrays.rest_density + 1e-6f);

    // cohesion should be SI::N is kg�m^-3 -> factor must be m^4 kg^-1 s^-2
    auto cohesion = -arrays.tension_akinci * vol[i] * arrays.rest_density * vol[j] *
                    arrays.rest_density * kernel;
    //auto cohesion_unit_fixed = cohesion / (dens[i] * arrays.rest_density);
    // cohesion should be SI::N is kg -> factor must be m s^-2
    auto curvature =
        -arrays.tension_akinci * vol[i] * arrays.rest_density * (normal[i] - normal[j]);
    curvatureForce += (scaling * curvature);
    cohesionForce += (scaling * (cohesion)).val;
  }
  arrays.acceleration[i] += (curvatureForce + cohesionForce) / (vol[i] * arrays.rest_density);
}

neighFunction(calculateNormals, akinciFirst, "Akinci Tension: normals", caches<float4, float, float>{});
neighFunction(calculateSurfaceTension, akinciSecond, "Akinci Tension: force", caches<float4, float4, float, float>{});

void SPH::AkinciTension::tension(SPH::AkinciTension::Memory mem) {
  launch<calculateNormals>(mem.num_ptcls, mem);
  launch<calculateSurfaceTension>(mem.num_ptcls, mem);
}
