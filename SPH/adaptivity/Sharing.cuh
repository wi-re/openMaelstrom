#pragma once 
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>
#include <SPH/adaptivity/AdaptivityDevice.cuh>

// This function currently serves no purpose.
basicFunctionType shareDetect(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	arrays.adaptiveMergeable[i] = -3;
	if (arrays.adaptiveClassification[i] >= 0.5f && arrays.adaptiveClassification[i] < 2.f)
		arrays.adaptiveMergeable[i] = -1;
	else if (arrays.adaptiveClassification[i] <= -0.5f)
		arrays.adaptiveMergeable[i] = -2;
	return;
}

// Finds sharing partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be sharing with and sharing candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the sharing result would not violate the systemic limit to particle
// sizes.
neighFunctionType shareGrabEven(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (i % 2 == 0 || arrays.adaptiveMergeable[i] != -1)
		return;
	cache_arrays((pos, position), (vol, volume), (classification, adaptiveClassification));

	auto V_t = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
	auto V_i = arrays.volume[i];
	auto V_d = 0.5f * (V_i - V_t);

	auto ctr = arrays.adaptiveMergeCounter[i];

	iterateNeighbors(j) {
		if (j != i) {
			if (math::distance3(pos[i], pos[j]) < support_h(pos[i])) {
				if (j % 2 == 0) {
					if (classification[j] >= 0.f)
						continue;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.adaptiveMergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-2, i);
					if (cas_val != -2)
						continue;

					auto sum_volume = vol[j] + V_d / (ctr + 1.f);
					auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
					if (sum_radius > arrays.radius)
						neighbor_mergeable.CAS(i, -2);
					else {
						arrays.adaptiveMergeable[i] = -4;
						ctr++;
						arrays.adaptiveMergeCounter[i]++;
					}
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
neighFunctionType shareGrabOdd(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (i % 2 != 0 || arrays.adaptiveMergeable[i] != -1)
		return;
	cache_arrays((pos, position), (vol, volume), (classification, adaptiveClassification));

	auto V_t = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
	auto V_i = arrays.volume[i];
	auto V_d = V_i - V_t;

	auto ctr = arrays.adaptiveMergeCounter[i];

	iterateNeighbors(j) {
		if (j != i) {
			if (math::distance3(pos[i], pos[j]) < support_h(pos[i])) {
				if (j % 2 != 0) {
					if (classification[j] >= 0.f)
						continue;
					cuda_atomic<int32_t> neighbor_mergeable(arrays.adaptiveMergeable + j);
					int32_t cas_val = neighbor_mergeable.CAS(-2, i);
					if (cas_val != -2)
						continue;

					auto sum_volume = vol[j] + V_d / (ctr + 1.f);
					auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
					if (sum_radius > arrays.radius)
						neighbor_mergeable.CAS(i, -2);
					else {
						arrays.adaptiveMergeable[i] = -4;
						ctr++;
						arrays.adaptiveMergeCounter[i]++;
					}
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
    auto a_i = math::getValue(arg.first[i]);
    auto m_i = m_s.val;
    auto a_j = math::getValue(arg.first[j]);
    auto m_j = m_p.val;
    using Ty = typename std::decay_t<decltype(*(arg.first))>;
    using Ay = std::decay_t<decltype(a_i)>;
    using Uy = decltype(math::weak_get<1>(std::declval<Ay>()));
    using Fy = typename vector_t<float, math::dimension<Ay>::value>::type;
    using Gy = typename vector_t<Uy, math::dimension<Ay>::value>::type;
    arg.first[i] = Ty{a_i};
    arg.first[j] = Ty{math::castTo<Gy>((math::castTo<Fy>(a_i) * m_i + math::castTo<Fy>(a_j) * m_j) / (m_i + m_j))};

    // auto a = arg.first[i];
    // auto b = arg.first[j];
    // arg.first[j] = (m_s * a + m_p * b) / (m_p + m_s);
    // arg.first[i] = a;
  }
  shareValue(i, j, m_s, m_p, ref...);
}

// This function shares all particles that have found sharing partners with their appropriate
// sharing partners.
neighFunctionType shareGrabbed(SPH::adaptive::Memory arrays, Ts... tup) {
  checkedParticleIdx(i);
  if (arrays.adaptiveMergeable[i] != -4)
    return;

  float counter = static_cast<float>(arrays.adaptiveMergeCounter[i]);

  cuda_atomic<int32_t> num_ptcls(arrays.adaptiveNumPtcls);
  ++num_ptcls;

  auto V_s = 0.5f * (arrays.volume[i] - level_estimate(arrays, -arrays.distanceBuffer.first[i])) / counter;

  iterateNeighbors(j) {
    if (arrays.adaptiveMergeable[j] != i)
      continue;
    auto V_j = arrays.volume[j];
    auto V_m = V_j + V_s;

    shareValue(i, j, V_s, V_j, tup...);

    arrays.adaptiveSplitIndicator[j] = 0;
    arrays.lifetime[j] = 0.0_s;
    arrays.volume[j] = V_m;
    math::unit_assign<4>(arrays.position[j], support_from_volume(V_m));
    arrays.volume[i] = arrays.volume[i] - V_s;
  }
  math::unit_assign<4>(arrays.position[i], support_from_volume(arrays.volume[i]));
  arrays.lifetime[i] = 0.0_s;
}

basicFunction(detectSharingParticles, shareDetect, "Adaptive: init (share)");
neighFunction(grabEvenSharingParticles, shareGrabEven, "Adaptive: find even partners (share)", caches<float4, float, float>{});
neighFunction(grabOddSharingParticles, shareGrabOdd, "Adaptive: find odd partners (share)", caches<float4, float, float>{});
neighFunction(shareParticles, shareGrabbed, "Adaptive: sharing particles");

// Helper function to call the sharing function properly by transforming the arguments from a tuple to a variadic list
template<typename... Ts>
auto ShareGrabbed(std::tuple<Ts...>, SPH::adaptive::Memory mem) {
	launch<shareParticles, decltype(std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr))...>(
		mem.num_ptcls, mem, std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr)...);
}
