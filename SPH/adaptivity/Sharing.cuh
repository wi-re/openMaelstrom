#pragma once 
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>
#include <SPH/adaptivity/AdaptivityDevice.cuh>

// This function currently serves no purpose.
basicFunctionType shareDetect(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	arrays.mergeCounter[i] = 0;
	arrays.mergeable[i] = -3;
	if (arrays.adaptiveClassification[i] >= 0.5f && arrays.adaptiveClassification[i] < 2.f)
		arrays.mergeable[i] = -1;
	else if (arrays.adaptiveClassification[i] < 0.f)
		arrays.mergeable[i] = -2;
	auto pDist = planeBoundary::distance(arrays.position.first[i], arrays.volume.first[i], arrays);
	if (pDist.val.w < support_from_volume(1.f) * kernelSize() * 1.5f)
		arrays.mergeable[i] = -3;
	if (arrays.lifetime[i].val < 5.f * arrays.timestep.val)
		arrays.mergeable[i] = -3;
	return;
}

// Finds sharing partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be sharing with and sharing candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the sharing result would not violate the systemic limit to particle
// sizes.
neighFunctionType shareGrabEven(SPH::adaptive::Memory arrays, int32_t* randomIdx) {
	checkedParticleIdx(i);
	i = randomIdx[i];
	if (i % 2 == 0 || arrays.mergeable[i] != -1)
		return;
	cache_arrays((pos, position.first), (vol, volume.first), (classification, adaptiveClassification));

	auto V_t = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
	auto V_i = arrays.volume.first[i];
	auto V_d = (V_i - V_t) * 0.5f;
	if (V_d < 0.f || V_t <= 0.f)
		return;

	auto ctr = arrays.mergeCounter[i] + 1;

	iterateNeighbors(j) {
		if (j != i) {
			auto sum_volume = vol[j] + V_d / (ctr);
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			if (sum_radius > arrays.radius)
				continue;
			auto V_jt = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			if (sum_volume >= V_jt * 1.f)
				continue;

			auto x_i = pos[i];
			auto x_j = (pos[i] * V_d / ctr + pos[j] * vol[j]) / (sum_volume);
			auto pDist = planeBoundary::distance(x_i, V_t, arrays);
			if (pDist.val.w < OFFSET(V_t.val))
				continue;
			pDist = planeBoundary::distance(x_j, sum_volume, arrays);
			if (pDist.val.w < OFFSET(sum_volume.val))
				continue;
			if (math::distance3(x_i, x_j) < OFFSET(V_t.val) * 2.f ||
				math::distance3(x_i, x_j) < OFFSET(sum_volume.val) * 2.f) continue;
			if (math::distance3(pos[i], pos[j]) < support_from_volume(sum_volume) * kernelSize() * 0.45f) {
				if (j % 2 == 0) {
					if (classification[j] >= -0.5f)
						continue;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-2, i);
					if (cas_val != -2)
						continue;
					arrays.mergeable[i] = -4;
					ctr++;
					arrays.mergeCounter[i]++;
					//return;
				}
			}
		}
	}
}

// Finds sharing partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be sharing with and sharing candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the sharing result would not violate the systemic limit to particle
// sizes. This function is an almost duplicate of shareGrabEven.
neighFunctionType shareGrabOdd(SPH::adaptive::Memory arrays, int32_t* randomIdx) {
	checkedParticleIdx(i);
	i = randomIdx[i];
	if (i % 2 != 0 || arrays.mergeable[i] != -1)
		return;
	cache_arrays((pos, position.first), (vol, volume.first), (classification, adaptiveClassification));

	auto V_t = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
	auto V_i = arrays.volume.first[i];
	auto V_d = (V_i - V_t)*0.5f;

	auto ctr = arrays.mergeCounter[i] + 1;

	if (V_d < 0.f || V_t <= 0.f)
		return;

	iterateNeighbors(j) {
		if (j != i) {
			auto sum_volume = vol[j] + V_d / (ctr);
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			if (sum_radius > arrays.radius)
				continue;
			auto V_jt = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			if (sum_volume >= V_jt * 1.f)
				continue;

			auto x_i = pos[i];
			auto x_j = (pos[i] * V_d / ctr + pos[j] * vol[j]) / (sum_volume);
			auto pDist = planeBoundary::distance(x_i, V_t, arrays);
			if (pDist.val.w < OFFSET(V_t.val))
				continue;
			pDist = planeBoundary::distance(x_j, sum_volume, arrays);
			if (pDist.val.w < OFFSET(sum_volume.val))
				continue;
			if (math::distance3(x_i, x_j) < OFFSET(V_t.val) * 2.f ||
				math::distance3(x_i, x_j) < OFFSET(sum_volume.val) * 2.f) continue;
			if (math::distance3(pos[i], pos[j]) < support_from_volume(sum_volume) * kernelSize() * 0.45f) {
				if (j % 2 != 0) {
					if (classification[j] >= -0.5f)
						continue;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-2, i);
					if (cas_val != -2)
						continue;

						arrays.mergeable[i] = -4;
						ctr++;
						arrays.mergeCounter[i]++;
						//return;
				}
			}
		}
	}
}

// Variadic recursion base
hostDeviceInline void shareValue(uint32_t, uint32_t, float_u<SI::volume>, float_u<SI::volume>) {}

// Caclulates the mass weighed average of a pointer pair stored in arg based on the particle masses.
// This function is recursive with respect to it's variadic argument list.
template <typename T, typename... Ts>
hostDeviceInline void shareValue(uint32_t i, uint32_t j, float_u<SI::volume> m_s,
	float_u<SI::volume> m_p, T arg, Ts... ref) {
	if (arg.first != nullptr) {
		auto a = arg.first[i];
		auto b = arg.first[j];
		arg.first[j] = (m_s.val * a + m_p.val * b) / (m_p.val + m_s.val);
		arg.first[i] = a;
	}
	shareValue(i, j, m_s, m_p, ref...);
}


// This function shares all particles that have found sharing partners with their appropriate
// sharing partners.
neighFunctionDeviceType shareGrabbed(SPH::adaptive::Memory arrays, Ts... tup) {
  checkedParticleIdx(i);
  if (arrays.mergeable[i] != -4)
    return;

  float counter = static_cast<float>(arrays.mergeCounter[i]);
  atomicAdd(arrays.adaptivityCounter + (math::clamp(arrays.mergeCounter[i], 1, 16) - 1), 1);

  cuda_atomic<int32_t> num_ptcls(arrays.ptclCounter);
  ++num_ptcls;

  auto V_s = (arrays.volume.first[i] - level_estimate(arrays, -arrays.distanceBuffer.first[i]))*0.5f / counter;

  iterateNeighbors(j) {
    if (arrays.mergeable[j] != i)
      continue;
    auto V_j = arrays.volume.first[j];
    auto V_m = V_j + V_s;

    shareValue(i, j, V_s, V_j, tup...);

    arrays.splitIndicator[j] = 0;
    arrays.lifetime[j] = 0.0_s;
    arrays.volume.first[j] = V_m;

	auto h = support_from_volume(V_m).val;
	float threshold = 0.f * 0.24509788f * h * kernelSize();
	auto pDist = planeBoundary::distance(arrays.position.first[j], V_m, arrays);
	//if (pDist.val.w < 0.f * h * kernelSize())
	//	arrays.position.first[j] -= (pDist.val) * (pDist.val.w - threshold);

	math::unit_assign<4>(arrays.position.first[j], support_from_volume(V_m));
    arrays.volume.first[i] = arrays.volume.first[i] - V_s;
  }

  auto h = support_from_volume(arrays.volume.first[i]).val;
  float threshold = 0.f * 0.24509788f * h * kernelSize();
  auto pDist = planeBoundary::distance(arrays.position.first[i], arrays.volume.first[i], arrays);
  //if (pDist.val.w < 0.f * h * kernelSize())
//	  arrays.position.first[i] -= (pDist.val) * (pDist.val.w - threshold);
  math::unit_assign<4>(arrays.position.first[i], support_from_volume(arrays.volume.first[i]));

  arrays.lifetime[i] = 0.0_s;
}

basicFunction(detectSharingParticles, shareDetect, "Adaptive: init (share)");
neighFunction(grabEvenSharingParticles, shareGrabEven, "Adaptive: find even partners (share)", caches<float4, float, float>{});
neighFunction(grabOddSharingParticles, shareGrabOdd, "Adaptive: find odd partners (share)", caches<float4, float, float>{});
neighFunctionDevice(shareParticles, shareGrabbed, "Adaptive: sharing particles");

// Helper function to call the sharing function properly by transforming the arguments from a tuple to a variadic list
template<typename... Ts>
auto ShareGrabbed(std::tuple<Ts...>, SPH::adaptive::Memory mem) {
	launchDevice<shareParticles, decltype(std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr))...>(
		mem.num_ptcls, mem, std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr)...);
}
