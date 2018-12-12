#define NEW_STYLE
#include <SPH/adaptivity/AdaptivityDevice.cuh>
#include <SPH/adaptivity/Blending.cuh>
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <SPH/adaptivity/Merging.cuh>
#include <SPH/adaptivity/Sharing.cuh>
#include <SPH/adaptivity/Splitting.cuh>
#include <utility/include_all.h>

/** This function is used to adjust the resolution of particles in the simulation it does splitting,
 * merging and mass sharing closely following the reference paper**/
void SPH::adaptive::adapt(Memory mem) {
  uint32_t split_ptcls;
  /* To avoid certain extra checks in the code we can restrict the function to either merge
   particles (decreasing resolution) or to split particles (increasing resolution). As this is
   done on a 2 frame period this should have no appreciable effect on the adaptation rate.*/
  if (parameters::frame{} % 2 == 0) {
    launch<decide>(mem.num_ptcls, mem);
    cuda::Memset(mem.adaptiveMergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
    cuda::Memset(mem.adaptiveNumPtcls, 0x00, sizeof(float));
    cuda::Memset(mem.adaptiveMergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
    launch<detectMergingParticles>(mem.num_ptcls, mem);
    launch<grabEvenMergingParticles>(mem.num_ptcls, mem);
    launch<grabOddMergingParticles>(mem.num_ptcls, mem);
    MergeGrabbed(sorting_list, mem);
    cuda::memcpy(&split_ptcls, mem.adaptiveNumPtcls, sizeof(uint), cudaMemcpyDeviceToHost);
    get<parameters::merged_ptcls>() = split_ptcls;
  } else {
    // Share particles
    launch<decide>(mem.num_ptcls, mem);
    cuda::Memset(mem.adaptiveNumPtcls, 0x00, sizeof(float));
    cuda::Memset(mem.adaptiveMergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
    cuda::Memset(mem.adaptiveMergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
    launch<detectSharingParticles>(mem.num_ptcls, mem);
    launch<grabEvenSharingParticles>(mem.num_ptcls, mem);
    ShareGrabbed(sorting_list, mem);
    cuda::memcpy(&split_ptcls, mem.adaptiveNumPtcls, sizeof(uint), cudaMemcpyDeviceToHost);
    get<parameters::shared_ptcls>() = split_ptcls;
    launch<decide>(mem.num_ptcls, mem);
    // Split particles, if the old particle count is close to the maximum particle count of the
    // simulation do nothing.
    int32_t old = mem.num_ptcls;
    cuda::memcpy(mem.adaptiveNumPtcls, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
    callSplit(sorting_list, mem);
    cuda::memcpy(&mem.num_ptcls, mem.adaptiveNumPtcls, sizeof(int32_t), cudaMemcpyDeviceToHost);
    get<parameters::num_ptcls>() = mem.num_ptcls;
    get<parameters::split_ptcls>() = mem.num_ptcls - old;
  }

  auto min = algorithm::reduce_min(mem.volume, mem.num_ptcls);
  auto max = PI4O3 * math::power<3>(mem.radius);
  auto ratio = max / min;
  *parameters::adaptiveRatio::ptr = ratio.val;
}

basicFunctionType genparticleIndex(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.adaptiveSplitIndicator[i] != 1)
		arrays.particleIndex[i] = INT_MAX;
	else
		arrays.particleIndex[i] = i;
}
basicFunction(indexBlendingParticles, genparticleIndex, "Adaptive: indexing blending particles");

struct is_valid {
	hostDeviceInline bool operator()(const int x) { return x != INT_MAX; }
};
// Main function to call the density blending funciton
void SPH::adaptive::blendDensity(Memory mem) {
	launch<indexBlendingParticles>(mem.num_ptcls, mem);
	*parameters::blend_ptcls::ptr = (int32_t) algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendDensities>(*parameters::blend_ptcls::ptr, mem, *parameters::blend_ptcls::ptr);
}

// Main function to call the velocity blending funciton
void SPH::adaptive::blendVelocity(Memory mem) {
	launch<indexBlendingParticles>(mem.num_ptcls, mem);
	*parameters::blend_ptcls::ptr = (int32_t) algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendVelocities>(*parameters::blend_ptcls::ptr, mem, *parameters::blend_ptcls::ptr);
}
