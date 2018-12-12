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
  if (arrays.adaptiveSplitIndicator[i] != 1)
    return; 
  cache_arrays((pos, position), (vol, volume));

  auto x_p = arrays.adaptiveParentPosition.first[i];
  auto rho_p = arrays.adaptiveParentVolume[i] * kernel(x_p, x_p);

  iterateCells(x_p, j) {
	  if (arrays.adaptiveParentIndex.first[i] != arrays.adaptiveParentIndex.first[j])
		  rho_p += vol[j] * kernel(x_p, pos[j]);
  }
  rho_p = math::clamp(rho_p, 0.1f, 1.f);

  auto beta = getBlendWeight(arrays, i);
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
  if (arrays.adaptiveSplitIndicator[i] != 1)
    return;
  //cache_arrays((vel, velocity));

  float counter = 0.f;
  float4_u<SI::velocity> v_p{0.f, 0.f, 0.f, 0.f};

  iterateNeighbors(j) {
    if (arrays.adaptiveParentIndex.first[i] == arrays.adaptiveParentIndex.first[j]) {
      counter += 1.f;
      v_p += arrays.velocity[j];
    }
  }
  v_p /= counter;

  arrays.adaptiveParentPosition.first[idx] += arrays.timestep * v_p;

  auto beta = getBlendWeight(arrays, i);
  arrays.velocity[i] = arrays.velocity[i] * (1.f - beta) + v_p * beta;
}

cellFunction(blendDensities, blendDensities_device, "Adaptive: blending density", caches<float4, float>{});
neighFunction(blendVelocities, blendVelocity_device, "Adaptive: blending velocity", caches<float4>{});