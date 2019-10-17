#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

/**Function used to estimate the target resolution (as volume) for a particle based on the distance
 * to the surface (as absolute distance not signed)**/
hostDeviceInline auto level_estimate(SPH::adaptive::Memory &arrays, value_unit<float, SI::m> val) {
  auto clampedDistance = math::clamp(val - 2.f * math::abs(arrays.surface_phiMin), 0.0_m, math::abs(arrays.surface_levelLimit));
  auto scaledDistance  = clampedDistance / math::abs(arrays.surface_levelLimit);
  auto resolution      = (float) arrays.resolution;
  if (arrays.useVolume == 1) {
	   auto scalingFactor   = 1.f / resolution + scaledDistance * (1.f - 1.f / resolution);
	   return PI4O3 * math::power<3>(arrays.radius) * scalingFactor;
  }
  else {
	  auto scalingFactor = 1.f / powf(resolution, 1.f / 3.f) + scaledDistance * (1.f - 1.f / powf(resolution, 1.f / 3.f));
	  auto volume = PI4O3 * math::power<3>(arrays.radius * scalingFactor);
	  return math::clamp(volume, PI4O3 * math::power<3>(arrays.radius) / ((float)resolution), PI4O3 * math::power<3>(arrays.radius));
  }
}

/** Function used to classify particles into sSlLo categories based on the difference to the ideal
  * particle size and their lifetime. **/
basicFunctionType decisionFunction(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  auto targetVolume  = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
  auto currentVolume = arrays.volume.first[i];

  auto ratio = ceilf(currentVolume.val / targetVolume.val);

  if      ( arrays.simulationTime < arrays.delay  || arrays.lifetime[i] < 0.f) 
	  arrays.adaptiveClassification[i] = 0;
  else if (currentVolume < 0.51f * targetVolume) // S particle
	  arrays.adaptiveClassification[i] = -2;
  else if (currentVolume < 0.9f * targetVolume) // s particle
	  arrays.adaptiveClassification[i] = -1;
  else if (currentVolume > 1.99f * targetVolume)  // L particle
	  arrays.adaptiveClassification[i] = ratio != ratio ? 0 : ratio;
  else if (currentVolume > 1.1f * targetVolume) // l particle
	  arrays.adaptiveClassification[i] = 1;
  else                                      // 0 particle
	  arrays.adaptiveClassification[i] = 0;
  //arrays.debugArray[i] = float4{
	 // (float)arrays.adaptiveClassification[i],
	 // currentVolume.val,
	 // targetVolume.val,
	 // ratio
  //};
}

basicFunction(decide, decisionFunction, "Adaptive: classify Particles");
