#pragma once 
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>
#include <SPH/adaptivity/Patterns.cuh>

// Variadic recursion base
hostDeviceInline void copyValue(int32_t, int32_t) {
}

// Copies the properties of the parent particle (srcIdx) to a child particle (trgIdx).
// This function is recursive with respect to it's variadic argument list.
template<typename T, typename... Ts>
hostDeviceInline void copyValue(int32_t trgIdx, int32_t srcIdx, T arg, Ts... ref) {
	if (arg.first != nullptr)
		arg.first[trgIdx] = arg.first[srcIdx];
	copyValue(trgIdx, srcIdx, ref...);
}

// This function executes the actual particle splitting process. This is done by first increment the
// global particle count, and if the new count would violate the maximum number of particles nothing
// is done. New particles will be created on the splitting patterns in Patterns.cuh and all
// properties will initially be copied to the new particles and adjusted if needed for quantities
// like volume. The initial particlle  is replaced with the last particle of the splitting pattern.
template <int32_t split_count, typename... Ts>
hostDeviceInline void initialize_particles(SPH::adaptive::Memory &arrays, Ts... tup) {
  checkedParticleIdx(i);

  cuda_atomic<int32_t> num_ptcls(arrays.adaptiveNumPtcls);
  int32_t split_idx = num_ptcls.add(split_count - 1);
  if (split_idx > arrays.max_numptcls - split_count - 1) {
    num_ptcls.sub(split_count);
    return;
  }

  float w = 1.f / ((float)split_count);
  auto x_i = arrays.position[i];
  auto V_i = arrays.volume[i];
  auto V_s = arrays.volume[i] * w;
  auto r = math::power<ratio<1, 3>>(V_i * PI4O3_1);
  auto h = support_from_volume(V_s);
  auto t_0 = -arrays.blendsteps * arrays.timestep;
  int32_t parent_idx = split_idx + i;

  for (int j = 0; j < split_count - 1; ++j) {
    auto x_j = x_i + get_position<split_count>(j) * r;
    math::unit_assign<4>(x_j, h);

    uint32_t new_idx = split_idx + j;
    copyValue(new_idx, i, tup...);

    arrays.position[new_idx] = x_j;
    arrays.volume[new_idx] = V_s;
    arrays.lifetime[new_idx] = t_0;
    arrays.adaptiveSplitIndicator[new_idx] = 1;
    arrays.adaptiveParentPosition.first[new_idx] = x_i;
    arrays.adaptiveParentIndex.first[new_idx] = parent_idx;
    arrays.adaptiveParentVolume[new_idx] = V_i;
  }

  auto x_j = x_i + get_position<split_count>(split_count - 1) * r;
  math::unit_assign<4>(x_j, h);
  arrays.adaptiveSplitIndicator[i] = 1;
  arrays.adaptiveParentPosition.first[i] = x_i;
  arrays.adaptiveParentIndex.first[i] = parent_idx;
  arrays.adaptiveParentVolume[i] = arrays.volume[i];
  arrays.position[i] = x_j;
  arrays.volume[i] = V_s;
  arrays.lifetime[i] = t_0;
}

// This function decides if a particle should actually be split based on
//	-	it's classification
//	-	it's lifetime being greater than 0s
//	-	if both previous conditions have been met for atleast 6 timesteps
// If the last condition is not met the counter is reset to 0 to avoid temporal instabilities
// (especially in the surface detection) to create false regions of adaptivity which can cause
// instabilities overall as instabilities in the surface detection often are a sympton of an
// underlying instability which gets amplified when splitting.
templateFunctionType splitParticles(SPH::adaptive::Memory arrays, Ts... tup) {
	checkedParticleIdx(i);
	if (arrays.lifetime[i] < 0.0_s)
		return;
	int32_t decision = static_cast<int32_t>(arrays.adaptiveClassification[i]);
	
	if (decision > 1) {
//		uint indicator = arrays.adaptiveSplitIndicator[i];
			arrays.adaptiveSplitIndicator[i] += 1;
		if (arrays.adaptiveSplitIndicator[i]  < 13) {
			return;
		}
		switch (decision) {
		case 2:  initialize_particles<2>( arrays, tup...); break;
		case 3:  initialize_particles<3>( arrays, tup...); break;
		case 4:  initialize_particles<4>( arrays, tup...); break;
		case 5:  initialize_particles<5>( arrays, tup...); break;
		case 6:  initialize_particles<6>( arrays, tup...); break;
		case 7:  initialize_particles<7>( arrays, tup...); break;
		case 8:  initialize_particles<8>( arrays, tup...); break;
		case 9:  initialize_particles<9>( arrays, tup...); break;
		case 10: initialize_particles<10>(arrays, tup...); break;
		case 11: initialize_particles<11>(arrays, tup...); break;
		case 12: initialize_particles<12>(arrays, tup...); break;
		case 13: initialize_particles<13>(arrays, tup...); break;
		case 14: initialize_particles<14>(arrays, tup...); break;
		case 15: initialize_particles<15>(arrays, tup...); break;
		case 16: initialize_particles<16>(arrays, tup...); break;
		default: initialize_particles<16>(arrays, tup...); break;
		}
	}
	else {
			arrays.adaptiveSplitIndicator[i] = 2;
	}
}

templateFunction(particleSplitting, splitParticles, "Adaptive: splitting particles");

template<typename... Ts>
auto callSplit(std::tuple<Ts...>, SPH::adaptive::Memory mem) {
	launch<particleSplitting, decltype(std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr))...>(
		mem.num_ptcls, mem, std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr)...);
}
