#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

/** Blends the density of a particle that has recently been created from splitting processes with
 * the density of an estimated parent particle based on the tracked velocitiy and position of the
 * parent particle. This function requires a full loop over all particles in all neighboring cells
 * as the neighborhood of the parent particle is based on a larger support radius than the split
 * particles. The blend process ignores particles with the same parent particle.**/
cellFunctionType blendDensities_device(SPH::adaptive::Memory arrays, int32_t threads) {
  checkedThreadIdx(idx);
  int32_t i = arrays.particleIndexCompact[idx];
  if (arrays.splitIndicator[i] != 1)
    return; 
  cache_arrays((pos, position.first), (vol, volume.first));

  auto x_p = arrays.parentPosition.first[i];
  //x_p = pos[i];
  x_p.val.w = support_from_volume(arrays.parentVolume[i]).val;
  auto rho_p = boundary::spline(x_p, arrays.parentVolume[i], 0.5f, arrays);
  rho_p += arrays.parentVolume[i] * kernel(x_p, x_p);

  iterateCells(x_p, j) {
	  if (arrays.parentIndex.first[i] != arrays.parentIndex.first[j])
		  rho_p += vol[j] * kernel(x_p, pos[j]);
  }
  if (rho_p != rho_p)
	  rho_p = arrays.density[i];

  auto beta = getBlendWeight(arrays, i);

#ifdef DEBUG_INVALID_PARITLCES
  if(beta != beta || rho_p != rho_p || arrays.density[i] != arrays.density[i] || arrays.density[i] == 0.f)
	  printf("%s: Invalid particle %d: %f %f %f %f %f %f " _VECSTR "\n",
		  __FUNCTION__, i, arrays.lifetime[i].val, arrays.parentVolume[i].val, support_from_volume(arrays.parentVolume[i]).val, beta.val, rho_p.val, arrays.density[i].val, _VEC(x_p.val));
#endif
  rho_p = math::clamp(rho_p, 0.1f, 1.f);

  arrays.density[i] = arrays.density[i] * (1.f - beta) + rho_p * beta;
}

/** Blends the velocity of a particle that has recently been created from splitting processes with
 * the velocities of particles with the same parent particle. Additionally updates the tracked
 * position and velocity of the original particle using the blended velocities. This function
 * requires only a normal neighborhood search as we only consider particles with the same parent
 * that are close by to avoid weird behaviour due to divergence. **/
neighFunctionType blendVelocity_device(SPH::adaptive::Memory arrays, int32_t threads) {
  checkedThreadIdx(idx);
  int32_t i = arrays.particleIndexCompact[idx];
  if (arrays.splitIndicator[i] != 1)
    return;
  //cache_arrays((vel, velocity));

  float counter = 0.f;
  float4_u<SI::velocity> v_p{0.f, 0.f, 0.f, 0.f};

  iterateNeighbors(j) {
    if (arrays.parentIndex.first[i] == arrays.parentIndex.first[j]) {
      counter += 1.f;
      v_p += arrays.velocity[j];
    }
  }
  v_p /= counter;
  if (counter == 0.f) {
	  v_p = arrays.velocity[i];
  }

  auto beta = getBlendWeight(arrays, i);
#ifdef DEBUG_INVALID_PARITLCES
  if (beta != beta || v_p != v_p)
	  printf("%s: Invalid particle %d: %f %d " _VECSTR _VECSTR _VECSTR"\n",
		  __FUNCTION__, i, beta.val, counter, _VEC(v_p.val), _VEC(arrays.velocity[i].val), _VEC(arrays.parentPosition.first[i].val));
#endif
  arrays.parentPosition.first[idx] += arrays.timestep * v_p;

  float threshold = support_from_volume(arrays.parentVolume[i]).val * kernelSize();
  auto pDist = planeBoundary::distance(arrays.parentPosition.first[idx], arrays.parentVolume[i], arrays);
  if (pDist.val.w < threshold)
	  arrays.parentPosition.first[idx] -= (pDist.val) * (pDist.val.w - threshold);

  arrays.velocity[i] = arrays.velocity[i] * (1.f - beta) + v_p * beta;
}

cellFunction(blendDensities, blendDensities_device, "Adaptive: blending density", caches<float4, float>{});
neighFunction(blendVelocities, blendVelocity_device, "Adaptive: blending velocity", caches<float4>{});