#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

// Function that markes all particles based on their classification as a particle that should be
// merged (-1) or any other particle (-3)
basicFunctionType mergeDetect(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  arrays.adaptiveMergeable[i] = -3;
  if (arrays.adaptiveClassification[i] <= -0.5f)
    arrays.adaptiveMergeable[i] = -1;
  else
    return;
}

// Finds merging partners with even indices for particles with odd indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes.
neighFunctionType mergeGrabEven(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  if (i % 2 == 0 || arrays.adaptiveMergeable[i] != -1)
    return;
  cache_arrays((pos, position), (vol, volume), (classification, adaptiveClassification));
  float counter = arrays.adaptiveMergeCounter[i] + 1.f;
   
  iterateNeighbors(j) {
    if (j != i) {
      if (math::distance3(pos[i], pos[j]) < support_h(pos[i])) {
        if (j % 2 == 0) {
          if (classification[j] > -0.5f)
            continue;
          cuda_atomic<int32_t> neighbor_mergeable(arrays.adaptiveMergeable + j);
          int32_t cas_val = neighbor_mergeable.CAS(-1, i);
		  if (cas_val != -1) 
			continue;
		  auto sum_volume = vol[j] + vol[i] / counter;;
          auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
		  auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
		  if (sum_radius > arrays.radius || sum_volume > targetVolume * 1.8f)
            neighbor_mergeable.CAS(i, -1);
		  else {
			  arrays.adaptiveMergeable[i] = -2;
			  arrays.adaptiveMergeCounter[i]++;
			  counter++;
		  }
        }
      }
    }
  }
}

// Finds merging partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes. This function is an almost duplicate of mergeGrabEven.
neighFunctionType mergeGrabOdd(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (i % 2 != 0 || arrays.adaptiveMergeable[i] != -1)
		return;
	cache_arrays((pos, position), (vol, volume), (classification, adaptiveClassification));
	float counter = arrays.adaptiveMergeCounter[i] + 1.f;
	iterateNeighbors(j) {
		if (j != i) {
			if (math::distance3(pos[i], pos[j]) < support_h(pos[i])) {
				if (j % 2 != 0) {
					if (classification[j] > -0.5f)
						continue;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.adaptiveMergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-1, i);
					if (cas_val != -1)
						continue;
					auto sum_volume = vol[j] + vol[i] / counter;
					auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
					auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
					if (sum_radius > arrays.radius || sum_volume > targetVolume * 1.8f)
						neighbor_mergeable.CAS(i, -1);
					else {
						arrays.adaptiveMergeable[i] = -2;
						arrays.adaptiveMergeCounter[i]++;
						counter++;
					}
				}
			}
		}
	}
}

// Variadic recursion base
hostDeviceInline void mergeValue(uint32_t, uint32_t, float_u<SI::volume>, float_u<SI::volume>) {}
#ifdef _WIN32
#pragma warning( push )  
#pragma warning( disable : 4244 )  
#endif
// Caclulates the mass weighed average of a pointer pair stored in arg based on the particle masses.
// This function is recursive with respect to it's variadic argument list.
template <typename T, typename... Ts>
hostDeviceInline void mergeValue(uint32_t particle_idx, uint32_t neighbor_idx, float_u<SI::volume> particle_mass,
                                    float_u<SI::volume> partner_mass, T arg, Ts... ref) {
  if (arg != nullptr){
    auto a_i = math::getValue(arg[particle_idx]);
    auto m_i = particle_mass.val;
    auto a_j = math::getValue(arg[neighbor_idx]);
    auto m_j = partner_mass.val;
    using Ty = typename std::decay_t<decltype(*arg)>;
    using Ay = std::decay_t<decltype(a_i)>;
    using Uy = decltype(math::weak_get<1>(std::declval<Ay>()));
    using Fy = typename vector_t<float, math::dimension<Ay>::value>::type;
    using Gy = typename vector_t<Uy, math::dimension<Ay>::value>::type;
    arg[neighbor_idx] = Ty{math::castTo<Gy>((math::castTo<Fy>(a_i) * m_i + math::castTo<Fy>(a_j) * m_j) / (m_i + m_j))};
        }
  mergeValue(particle_idx, neighbor_idx, particle_mass, partner_mass, ref...);
}
#ifdef _WIN32
#pragma warning(pop)
#endif
// This function merges all particles that have found merging partners with their appropriate
// merging partners. The particle that is being merged is removed from the simulation by being
// marked invalid.
neighFunctionType mergeGrabbed(SPH::adaptive::Memory arrays, Ts... tup) {
  checkedParticleIdx(i);
  if (arrays.adaptiveMergeable[i] != -2)
    return;

  cuda_atomic<int32_t> num_ptcls(arrays.adaptiveNumPtcls);
  ++num_ptcls;

  float counter = static_cast<float>(arrays.adaptiveMergeCounter[i]);
  auto V_i = arrays.volume[i] / counter;

  iterateNeighbors(j) {
    if (arrays.adaptiveMergeable[j] != i)
      continue;

    auto V_j = arrays.volume[j];
    auto V_m = V_i + V_j;

    mergeValue(i, j, V_i, V_j, tup...);

    arrays.adaptiveSplitIndicator[j] = 0;
    arrays.lifetime[j] = -arrays.blendsteps * arrays.timestep *0.25f;
    arrays.volume[j] = V_m;

    math::unit_assign<4>(arrays.position[j], support_from_volume(V_m));
  }
  math::unit_assign<4>(arrays.position[i], float_u<SI::m>(FLT_MAX));
}

basicFunction(detectMergingParticles, mergeDetect, "Adaptive: init (merge)");
neighFunction(grabEvenMergingParticles, mergeGrabEven, "Adaptive: find even partners (merge)", caches<float4, float, float>{});
neighFunction(grabOddMergingParticles, mergeGrabOdd, "Adaptive: find odd partners (merge)", caches<float4, float, float>{});
neighFunction(mergeParticles, mergeGrabbed, "Adaptive: merging particles");

// Helper function to call the merging function properly by transforming the arguments from a tuple to a variadic list
template <typename... Ts> auto MergeGrabbed(std::tuple<Ts...>, SPH::adaptive::Memory arrays) {
  launch<mergeParticles, decltype((typename Ts::unit_type *)Ts::ptr)...>(
		arrays.num_ptcls, arrays, (typename Ts::unit_type *)Ts::ptr...);
}
