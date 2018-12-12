#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

/**Function used to estimate the target resolution (as volume) for a particle based on the distance
 * to the surface (as absolute distance not signed)**/
hostDeviceInline auto level_estimate(SPH::adaptive::Memory &arrays, value_unit<float, SI::m> val) {
  auto clampedDistance = math::clamp(val - 2.f * math::abs(arrays.surface_phiMin), 0.0_m, math::abs(arrays.surface_levelLimit));
  auto scaledDistance  = clampedDistance / math::abs(arrays.surface_levelLimit);
  auto resolution      = arrays.adaptive_resolution;
  auto scalingFactor   = 1.f / resolution + scaledDistance * (1.f - 1.f / resolution);
  return PI4O3 * math::power<3>(arrays.radius) * scalingFactor;
}

/** Function used to classify particles into sSlLo categories based on the difference to the ideal
  * particle size and their lifetime. **/
basicFunctionType decisionFunction(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  auto targetVolume  = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
  auto currentVolume = arrays.volume[i];

  if      ( arrays.simulationTime < arrays.adaptive_delay) 
	  arrays.adaptiveClassification[i] = 0;
  else if (currentVolume < 0.51f * targetVolume) // S particle
	  arrays.adaptiveClassification[i] = -2;
  else if (currentVolume < 0.9f * targetVolume) // s particle
	  arrays.adaptiveClassification[i] = -1;
  else if (currentVolume > 1.99f * targetVolume)  // L particle
	  arrays.adaptiveClassification[i] = ceilf(currentVolume.val / targetVolume.val);
  else if (currentVolume > 1.1f * targetVolume) // l particle
	  arrays.adaptiveClassification[i] = 1;
  else                                      // 0 particle
	  arrays.adaptiveClassification[i] = 0;
}

basicFunction(decide, decisionFunction, "Adaptive: classify Particles");
