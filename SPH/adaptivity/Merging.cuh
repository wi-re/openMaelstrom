#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

// Function that markes all particles based on their classification as a particle that should be
// merged (-1) or any other particle (-3)
basicFunctionType mergeDetect(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	arrays.mergeCounter[i] = 0;
	arrays.mergeable[i] = -4;
	//arrays.debugArray[i] = float4{ 0.f,0.f,0.f,0.f };
	if (arrays.adaptiveClassification[i] <= -1.5f && arrays.lifetime[i] >= 0.f)
		arrays.mergeable[i] = -1;
	if (arrays.lifetime[i].val < 10.f * arrays.timestep.val)
		arrays.mergeable[i] = -4;
}

// Finds merging partners with even indices for particles with odd indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes.
neighFunctionType mergeGrabEven(SPH::adaptive::Memory arrays, int32_t* idxArray) {
	checkedParticleIdx(i);
	i = idxArray[i];
	if (i % 2 == 0 || arrays.mergeable[i] != -1 || arrays.adaptiveClassification[i] > -1.5f)
		return;
	cache_arrays((pos, position.first), (vol, volume.first), (classification, adaptiveClassification));
	float counter = arrays.mergeCounter[i] + 1.f;
	//float4 dbgVal = float4{ 0.f,0.f,0.f,0.f };
	int32_t c = 0;
	iterateNeighbors(j) {
		if (++c > 5) break;
		if (j != i) {
			auto sum_volume = vol[j] + vol[i] / counter;
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			if (sum_volume > targetVolume *1.1f) continue;
			//dbgVal.x++;
			//if (sum_radius <= arrays.radius) dbgVal.y++;
			if (sum_radius <= arrays.radius &&
				math::distance3(pos[i], pos[j]) < (support_from_volume(sum_volume) * kernelSize() * 0.5f)) {
				//dbgVal.z++;
				auto pDist = planeBoundary::distance((arrays.position.first[i] * vol[i] / counter + arrays.position.first[j] * vol[j]) / (vol[i] / counter + vol[j]), vol[i] / counter + vol[j], arrays);
				if (pDist.val.w < OFFSET(sum_volume.val))
					continue;
				if (j % 2 == 0) {
					//if (!(classification[j] < -1.5f || (classification[j] > 0.1f && classification[j] < 1.1f)))
					if (!(classification[j] < -1.5f))
						continue;
					//dbgVal.w++;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-1, i);
					if (cas_val != -1)
						continue;
					//dbgVal.w = -1;
					auto sum_volume = vol[j] + vol[i] / counter;;
					auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
					auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
					//if (sum_radius > arrays.radius /*|| sum_volume > targetVolume * 1.8f*/)
					//	neighbor_mergeable.CAS(i, cas_val);
					//else {
						arrays.mergeable[i] = -2;
						arrays.mergeCounter[i]++;
						counter++;
						//break;
					//}
				}
			}
		}
	}
	//arrays.debugArray[i] = dbgVal;
}

// Finds merging partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes. This function is an almost duplicate of mergeGrabEven.
neighFunctionType mergeGrabOdd(SPH::adaptive::Memory arrays, int32_t* idxArray) {
	checkedParticleIdx(i);
	i = idxArray[i];
	if (i % 2 != 0 || arrays.mergeable[i] != -1 || arrays.adaptiveClassification[i] > -1.5f)
		return;
	cache_arrays((pos, position.first), (vol, volume.first), (classification, adaptiveClassification));
	float counter = arrays.mergeCounter[i] + 1.f;
	//float4 dbgVal = arrays.debugArray[i];
	int32_t c = 0;
	iterateNeighbors(j) {
		if (++c > 5) break;
		if (j != i) {
			auto sum_volume = vol[j] + vol[i] / counter;
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			if (sum_volume > targetVolume * 1.1f) continue;
			//dbgVal.x++;
			//if (sum_radius <= arrays.radius) dbgVal.y++;
			if (sum_radius <= arrays.radius && 
				math::distance3(pos[i], pos[j]) < (support_from_volume(sum_volume) * kernelSize() * 0.5f)) {
				//dbgVal.z++;

				auto pDist = planeBoundary::distance((arrays.position.first[i] * vol[i] / counter + arrays.position.first[j] * vol[j]) / (vol[i] / counter + vol[j]), vol[i] / counter + vol[j], arrays);
				if (pDist.val.w < OFFSET(sum_volume.val))
					continue;

				if (j % 2 != 0) {
					//if (!(classification[j] < -1.5f || (classification[j] > 0.1f && classification[j] < 1.1f)))
					if (!(classification[j] < -1.5f))
						continue;
					//dbgVal.w++;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-1, i);
					if (cas_val != -1)
						continue;
					//dbgVal.w = -1;
					//else {
						arrays.mergeable[i] = -2;
						arrays.mergeCounter[i]++;
						counter++;
						//break;
					//}
				}
			}
		}
	}
	//arrays.debugArray[i] = dbgVal;
}

// Variadic recursion base
hostDeviceInline void mergeValue(uint32_t, uint32_t, float_u<SI::volume>, float_u<SI::volume>) {}

// Caclulates the mass weighed average of a pointer pair stored in arg based on the particle masses.
// This function is recursive with respect to it's variadic argument list.
template <typename T, typename... Ts>
hostDeviceInline void mergeValue(uint32_t particle_idx, uint32_t neighbor_idx, float_u<SI::volume> particle_mass,
	float_u<SI::volume> partner_mass, T arg, Ts... ref) {
	if (arg != nullptr)
		arg[neighbor_idx] = static_cast<typename std::decay<decltype(*arg)>::type>(((arg[particle_idx] * particle_mass.val + arg[neighbor_idx] * partner_mass.val) / (particle_mass.val + partner_mass.val)));
	mergeValue(particle_idx, neighbor_idx, particle_mass, partner_mass, ref...);
}

// This function merges all particles that have found merging partners with their appropriate
// merging partners. The particle that is being merged is removed from the simulation by being
// marked invalid.
neighFunctionDeviceType mergeGrabbed(SPH::adaptive::Memory arrays, Ts... tup) {
	checkedParticleIdx(i);
	if (arrays.mergeable[i] != -2)
		return;
	if (arrays.adaptiveClassification[i] > -1.5f)
		return;
	cuda_atomic<int32_t> num_ptcls(arrays.ptclCounter);
	++num_ptcls;

	float counter = static_cast<float>(arrays.mergeCounter[i]);
	auto V_i = arrays.volume.first[i] / counter;

	atomicAdd(arrays.adaptivityCounter + (math::clamp(arrays.mergeCounter[i], 1, 16) - 1), 1);
	iterateNeighbors(j) {
		if (arrays.mergeable[j] != i)
			continue;

		auto V_j = arrays.volume.first[j];
		auto V_m = V_i + V_j;

		mergeValue(i, j, V_i, V_j, tup...);

		arrays.splitIndicator[j] = 2;
		arrays.lifetime[j] = -0.f * arrays.blendSteps * arrays.timestep *0.25f;
		arrays.volume.first[j] = V_m;

		auto h = support_from_volume(V_m).val;
		float threshold = 0.24509788f * h * kernelSize() * 1.f;
		auto pDist = planeBoundary::distance(arrays.position.first[j], V_m, arrays);
		//if (pDist.val.w < threshold)
		//	arrays.position.first[j] -= (pDist.val) * (pDist.val.w - threshold);
		math::unit_assign<4>(arrays.position.first[j], support_from_volume(V_m));
	}
	math::unit_assign<4>(arrays.position.first[i], float_u<SI::m>(FLT_MAX));
}

basicFunction(detectMergingParticles, mergeDetect, "Adaptive: init (merge)");
neighFunction(grabEvenMergingParticles, mergeGrabEven, "Adaptive: find even partners (merge)", caches<float4, float, float>{});
neighFunction(grabOddMergingParticles, mergeGrabOdd, "Adaptive: find odd partners (merge)", caches<float4, float, float>{});
neighFunctionDevice(mergeParticles, mergeGrabbed, "Adaptive: merging particles");

// Helper function to call the merging function properly by transforming the arguments from a tuple to a variadic list
template <typename... Ts> auto MergeGrabbed(std::tuple<Ts...>, SPH::adaptive::Memory arrays) {
	launchDevice<mergeParticles>(
		arrays.num_ptcls, arrays, (typename Ts::unit_type *)Ts::ptr...);
}
