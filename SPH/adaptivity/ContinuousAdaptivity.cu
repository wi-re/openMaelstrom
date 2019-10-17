//#define OLD_STYLE
#define EIGENADAPTIVE
//#define OFFSET(v) (0.24509788f * support_from_volume(v) * kernelSize())
#define OFFSET(v) ( 0.2f * powf(v, 1.f / 3.f)) // 0.6203504908f
//#define INDIVIDUAL_PCA
#include "dummy.h"
#include <SPH/adaptivity/AdaptivityDevice.cuh>
#include <SPH/adaptivity/Blending.cuh>
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <SPH/adaptivity/Merging.cuh>
#include <SPH/adaptivity/Sharing.cuh>
#include <SPH/adaptivity/Splitting.cuh>
#include <utility/include_all.h>

basicFunctionType genparticleIndex(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.splitIndicator[i] != 1)
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
	*parameters::blendedPtcls::ptr = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendDensities>(*parameters::blendedPtcls::ptr, mem, *parameters::blendedPtcls::ptr);
}

// Main function to call the velocity blending funciton
void SPH::adaptive::blendVelocity(Memory mem) {
	launch<indexBlendingParticles>(mem.num_ptcls, mem);
	*parameters::blendedPtcls::ptr = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendVelocities>(*parameters::blendedPtcls::ptr, mem, *parameters::blendedPtcls::ptr);
}

#include <curand_kernel.h>
/** In general for our code basicFunctionType denotes a function that is called on the GPU with no neighborhood
 *information, which are configured via basicFunction to be easier to launch. Similarly neighFunctionType describes a
 *function which requires information about the neighborhood of a particle. deviceInline is just a define to __device__
 *__inline__.
 **/
 // This struct is used to enable the compact operation by creating a thrust compatible functor
struct is_valid_split {
	hostDeviceInline bool operator()(const int x) { return x != INT_MAX; }
};
// this enum is used to make the optimization code more readable by avoiding weakly typed constants.
enum struct threadAssignment : int8_t {
	split, neighbor, none
};
// this function takes a volume as an argument and returns the support radius h based on this value. This value is not
// multiplied by the kernel scale H/h.
deviceInline auto hFromV(float volume) {
	auto target_neighbors = kernelNeighbors();
	auto kernel_epsilon =
		(1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	auto h = kernel_epsilon * powf(volume, 1.f / 3.f);
	return h;
}
// this function implements an atomic min function for CUDA using CAS as the builtin atomicmin function only works for
// integers and not floats.
__device__ static float atomicMinFloat(float* address, float val) {
	int32_t* address_as_i = (int32_t*)address;
	int32_t old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
/**
  *This function is intended to classify the particles w.r.t. the splitting operation. Theoretically it should be
  *possible to immedieatly split a particle as soon as it becomes too large, however in practice this does not make
  *sense due to instabilities in the sizing function causing unnecessary fluctuations in spatial resolution. This time
  *delay of 11 timesteps used here is similar to the prior work but not as necessary as there due to the inherently more
  *stable splitting process.
 **/
basicFunctionType indicateParticles(SPH::adaptive::Memory arrays) {
	// This macro returns the index assigned to this thread for this function call for either GPU or CPU execution
	// and returns out of the function if i is larger or equal to the number of threads being run.
	checkedParticleIdx(i);
	// INT_MAX is used to indicate for the compact operation that no splitting is taking place
	arrays.particleIndex[i] = INT_MAX;
	// Particles that do not yet fully exist in the simulation, e.g. those recently split, are exempt from classification
	if (arrays.lifetime[i] < 0.0_s)
		return;
	// adaptiveClassification contains the classification of the particle according to prior work where a positive value
	// indicates that the particle is too large and should be split into n points if the value is n. This value is not
	// clamped to certain ranges
	int32_t decision = static_cast<int32_t>(arrays.adaptiveClassification[i]);
	// if the particle can be split into atleast 2 particles
	if (decision > 1) {
		// an indicator of 0 is the initial starting value for a particle (zero initialized) and a value of 1 is used to
		// indicate a particle that is being blended. This means that we need to "skip" the 1 value by incrementing twice
		// if the current indicator is 0
		if (arrays.splitIndicator[i] == 0)
			arrays.splitIndicator[i] += 1;
		// the normal increment operation 
		arrays.splitIndicator[i] += 1;
		// which makes sure that a certain number of timesteps have passed since the particle was classified as too large
		// which avoids the aforementioned resolution changes. The threshold of 13 seems to work nicely, although lower 
		// and higher thresholds are a possibility.
		if (arrays.splitIndicator[i] < 13) {
			return;
		}
		// the current implementation supports only splitting into up to 16 particles so the value needs to be clamped
		decision = math::clamp(decision, 1, 16);
		// increment the global counter of particles by the number of particles to insert - 1 as the storage of the 
		// original particle is repurposed
		cuda_atomic<int32_t> num_ptcls(arrays.ptclCounter);
		int32_t split_idx = num_ptcls.add(decision - 1);
		// if the split_idx is too large we revert the global increment atomically and return
		if (split_idx >= arrays.max_numptcls - decision - 100) {
			num_ptcls.sub(decision - 1);
			return;
		}
		// we store the split_idx in the parent index for now
		arrays.parentIndex.first[i] = split_idx;
		// the particleIndex array is used in compact function to make the overall process significantly faster by 
		// reducing the divergence significantly (of the code). A value != INT_MAX is seen as valid.
		arrays.particleIndex[i] = i;
	}
	else {
		// if the particle is not classified as too large we reset the splitIndicator to 2 instantly.
		arrays.splitIndicator[i] = 2;
	}
}
/**This function is used to optimize the particles positions using gradient descent and the mass ratios using
 * evolutionary optimization. This function is not called directly but used by another kernel function defined further
 * down. The arguments are:
 * - split_count: the number of particles this particle is split into
 * - X: a shared memory array to store the current positions for GD
 * - gradX: a shared memory array to store the current gradients for GD
 * - tau_s: a shared memory array to store the tau_s value for all split particles
 * - tau_h: a shared memory array to store the tau_h value for all (to an extent) neighboring particles
 * - tau: a shared memory value to store the global minimum refinement error
 * - pIdx: the index of the parent particle, in equations this would be o
 * - sIdx: the index at which the newly refined particles are being inserted into memory
 * - cIdx: the index assigned to a thread from a neighboring particle (H)
 * - sArray: a structure containing pointers to all persistent arrays within the simulation
 * - seed: a random initial value, per block, for the evolutionary optimization
 **/
hostDeviceInline uint32_t xorshift32(uint32_t& state){
	uint32_t x = state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return state = x;
}

deviceInline auto getScalingFactor(SPH::adaptive::Memory& arrays, int32_t i) {
	float pVol = arrays.volume.second[i].val;
	float hl = support_from_volume(pVol);
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	//factor = arrays.adaptivityScaling;
	return hl * kernelSize() * arrays.adaptivityScaling;
}

deviceInline auto getThreshold(SPH::adaptive::Memory& arrays, float volume) {
	return OFFSET(volume) * 1.f * arrays.adaptivityThreshold;
}
#define DBG_STMNT if(false)

template<neighbor_list neigh>
deviceInline void generateInitial(SPH::adaptive::Memory &arrays, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	constexpr auto neighborhood = neigh;
	__syncthreads();
	if (threadIdx.x == 0) {
		uint32_t state = (uint32_t)((pIdx + sIdx + cIdx) * split_count);
		float u1 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u2 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u3 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float4 q = {
			sqrtf(1.f - u1) * sinf(2.f * CUDART_PI_F * u2),
			sqrtf(1.f - u1) * cosf(2.f * CUDART_PI_F * u2),
			sqrtf(u1) * sinf(2.f * CUDART_PI_F * u3),
			sqrtf(u1) * cosf(2.f * CUDART_PI_F * u3)
		};
		gradX[0] = q;
	}
	// for this function we need to make sure that many of these steps are taken in unity to avoid potential
	// race conditions which would lead to the particle de-synchronizing which breaks their global state that is not
	// explicitly stored but only stored within each local thread which leads to significant problems. As such the usage
	// of __syncthreads() is overly conservative but avoids all potential issues, especially w.r.t. future HW changes
	__syncthreads();
	float thresholdFactor = 1.f * arrays.adaptivityThreshold;// 2.f / 3.f * sqrtf(6.f);
	float pVol = arrays.volume.second[pIdx].val;
	// initially we assume that the ratios are all lambda = 1/n as we split the optimization problems in order to avoid
	// the problems of having to solve large non linear constrained optimization problems.
	float V_s = pVol / (float)split_count;
	float hl = support_from_volume(pVol);
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	factor = getScalingFactor(arrays, pIdx);
	//factor = arrays.adaptivityScaling;
	//factor = hl * kernelSize() / 0.5477225150103355f;
	//factor = 1.f / factor;
	// the factor is used to scale the statically calculated patterns to the actual particle size by calculating
	// the scaling of the support radii which for the static case was 1 with a radius of 0.271...
	auto h = hFromV(V_s);
	// this is the previously described assignment.
	auto tas = threadAssignment::none;
	float4 assignedPosition;
	//float assignedVolume;
	if (threadIdx.x < split_count) {
		float4 offset = getshape(split_count, threadIdx.x) * factor;
		float4 v = offset;
		v.w = 0.f;
		float4 u = gradX[0];
		float s = u.w;
		u.w = 0.f;
		// Do the math
		if (false)
			offset = 2.0f * math::dot3(u, v) * u
			+ (s*s - math::dot3(u, u)) * v
			+ 2.0f * s * math::cross(u, v);

		assignedPosition = arrays.position.second[pIdx].val + offset;
		assignedPosition.w = h;

		float threshold = OFFSET(V_s) * thresholdFactor;
		auto pDist = planeBoundary::distance(uFloat4<SI::m>(assignedPosition), float_u<SI::volume>(V_s), arrays);
		if (pDist.val.w < threshold)
			assignedPosition -= (pDist.val) * (pDist.val.w - threshold);



		X[threadIdx.x] = assignedPosition;
		X[threadIdx.x].w = h;
		//assignedVolume = V_s;
	}
	else if (cIdx != -1) {
		assignedPosition = arrays.position.second[cIdx].val;
	}
	__syncthreads();
	//return assignedPosition;
}
template<neighbor_list neigh>
deviceInline void gradientDescent(SPH::adaptive::Memory &arrays, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t cIdx, threadAssignment tas, float pVol, float V_s, float hl, float h) {
	// configuration parameters for GD
	float4 assignedPosition = getAssignedPosition(arrays, X, cIdx, tas);
	if (tas == threadAssignment::split)
		tau_s[threadIdx.x] = 0.f;
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0) printf("Entering gradient descent for [%d : %d] @ %d\n", blockIdx.x, threadIdx.x, pIdx);
	__syncthreads();
	constexpr auto neighborhood = neigh;
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	factor = getScalingFactor(arrays, pIdx);
	//factor = arrays.adaptivityScaling;
	auto gamma0 = arrays.adaptivityGamma;
	float thresholdFactor = 1.f * arrays.adaptivityThreshold;// 2.f / 3.f * sqrtf(6.f);
	constexpr auto beta = 0.5f;
	float prevError = FLT_MAX;
	int32_t i = pIdx;
	float gamma = gamma0;
	// these factors can be used to weigh the errors but setting them to 1 works in practice
	constexpr auto hexWeight = 1.f;
	constexpr auto splitWeight = 1.f;
	*tau = 0.f;
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && threadIdx.x == 0)
		printf("%d - [%d : %d] -> factor %e gamma %e beta %e prev %e tau %e\n", __LINE__, blockIdx.x, threadIdx.x, factor, gamma, beta, prevError, *tau);
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && tas == threadAssignment::split) {
		printf("%d - [%d : %d] -> [%f %f %f %f]\n", __LINE__, blockIdx.x, threadIdx.x, X[threadIdx.x].x, X[threadIdx.x].y, X[threadIdx.x].z, X[threadIdx.x].w);
	}
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && tas == threadAssignment::neighbor) {
		printf("%d - [%d : %d] -> [%f %f %f %f] @ %d\n", __LINE__, blockIdx.x, threadIdx.x, assignedPosition.x, assignedPosition.y, assignedPosition.z, assignedPosition.w, cIdx);
	}
	__syncthreads();
	// the code for GD is blocked with an if(true) to easily allow turning this part off.
	if (true) {
		// initially the gradients are set to 0.
		int32_t attempt = 0;
		if (tas == threadAssignment::split) {
			gradX[threadIdx.x] = float4{ 0.f,0.f,0.f,0.f };
		}
		// the actual gradient descent method
		for (int32_t gradientIt = 0; gradientIt < 32 && attempt < 32;) {
			// the following section calculates the error terms tau_s and tau_h in parallel with the equations as they
			// were in the submission.
			if (tas == threadAssignment::neighbor) {
				float tauH = -pVol * kernel(assignedPosition, arrays.position.second[i].val);
				for (int32_t i = 0; i < split_count; ++i) {
					tauH += V_s * kernel(assignedPosition, X[i]);
				}
				tau_h[threadIdx.x - split_count] = tauH * hexWeight;
				// DBG_STMNT if(blockIdx.x == 0) printf("%d - [%d : %d] -> tau_h = %e\n", __LINE__, blockIdx.x, threadIdx.x, tauH);
				atomicAdd(tau, tauH * tauH * hexWeight * hexWeight);
			}
			__syncthreads();
			//float t = *tau;
			__syncthreads();
			if (tas == threadAssignment::split) {
				float tauS = -arrays.density[pIdx].val + 
					boundary::spline(float4_u<SI::m>{assignedPosition}, float_u<SI::volume>{V_s}, 0.5f, arrays);
				//SWH::spline(float4_u<SI::m>(assignedPosition), arrays).val;
				for (int32_t ia = 0; ia < split_count; ++ia) {
					tauS += V_s * kernel(assignedPosition, X[ia]);
				}
				iterateNeighbors(j) {
					if (j != pIdx) {
						tauS += arrays.volume.second[j].val * kernel(assignedPosition, arrays.position.second[j].val);
					}
				} 
				// DBG_STMNT if (blockIdx.x == 0) printf("%d - [%d : %d] -> tau_s = %e\n", __LINE__, blockIdx.x, threadIdx.x, tauS);
				//tauS += SWH::spline4(float4_u<SI::m>(assignedPosition), arrays);
				tau_s[threadIdx.x] = tauS * splitWeight;
				atomicAdd(tau, tauS * tauS * splitWeight * splitWeight);
			}
			__syncthreads();
			DBG_STMNT if (blockIdx.x == 0 && threadIdx.x == 0) printf("%d - [%d : %d] -> %e : %e @ [%d : %d] x %e\n", __LINE__, blockIdx.x, threadIdx.x, *tau, prevError, gradientIt, attempt, gamma);
			__syncthreads();
			// we use a very simple condition, which works in practice, to avoid the additional computational complexity
			if (*tau < prevError) {
				// synchronize to make sure all threads entered this point then update the global state
				__syncthreads();
				prevError = *tau;
				*tau = 0.f;
				gradientIt++;
				attempt = 0;
				gamma = gamma0;
				__syncthreads();
				// the next section calculates the gradient in parallel as per the submission. This update is only required
				// when a new position is found
				float4 gradH{ 0.f,0.f,0.f,0.f };
				if (tas == threadAssignment::split) {
					float4 grad{ 0.f,0.f,0.f,0.f };
					int32_t ctr = 0;
					iterateNeighbors(j) {
						if (j != pIdx) {
							if (ctr++ >= 96 - split_count) break;
							auto g = arrays.volume.second[j].val * gradient(X[threadIdx.x], arrays.position.second[j].val);
							grad += tau_h[ctr - 1] * g;
							gradH += g;
						}
					}
					gradX[threadIdx.x] = hexWeight * (2.f * grad);
				}
				if (tas == threadAssignment::split) {
					float4 gradA{ 0.f,0.f,0.f,0.f };
					for (int32_t i = 0; i < split_count; ++i) {
						auto g = V_s * gradient(X[threadIdx.x], X[i]);
						gradA += g;
					}
					gradA += V_s * boundary::splineGradient(float4_u<SI::m>{X[threadIdx.x]}, arrays.volume.second[i] * (1.f / (float)split_count), 0.5f, arrays, boundary::kind::plane);

					float4 firstTerm{ 0.f,0.f,0.f,0.f };
					firstTerm = 2.f * tau_s[threadIdx.x] * (gradA + gradH);

					float4 secondTerm{ 0.f,0.f,0.f,0.f };
					for (int32_t i = 0; i < split_count; ++i) {
						if (i != threadIdx.x) {
							secondTerm += -2.f * tau_s[i] * V_s * gradient(X[i], X[threadIdx.x]);
						}
					}
					gradX[threadIdx.x] += firstTerm + secondTerm;
				}
			}
			else {
				// increment the attempt counter after synchronizing and move the global position state back to where it
				// was before and then update gamma. This avoids having to store the actual positions and the potentially
				// updated positions.
				__syncthreads();
				*tau = 0.f;
				attempt++;
				if (tas == threadAssignment::split) {
					X[threadIdx.x] += gradX[threadIdx.x] * gamma;
					X[threadIdx.x].w = assignedPosition.w;
				}
				gamma *= beta;
			}
			__syncthreads();
			// update the positions based on the gradient and step distance
			//float4 previous;
			if (tas == threadAssignment::split) {
				gradX[threadIdx.x].w = 0.f;
				//previous = X[threadIdx.x];
				float threshold = OFFSET(V_s) * thresholdFactor;
				//auto pd = math::distance3(X[threadIdx.x], arrays.position.second[pIdx].val);
				//if (pd > h * kernelSize())
				//	X[threadIdx.x] -= (pd - h * kernelSize()) * math::normalize3(X[threadIdx.x] - arrays.position.second[pIdx].val);

				X[threadIdx.x] = X[threadIdx.x] - gradX[threadIdx.x] * gamma;
				auto pd = math::distance3(X[threadIdx.x], arrays.position.second[pIdx].val);
				auto t = math::length3(getshape(split_count, threadIdx.x)) * factor * 1.5f;
				if (pd > t)
					X[threadIdx.x] = X[threadIdx.x] - (pd - t) * math::normalize3(X[threadIdx.x] - arrays.position.second[pIdx].val);
				auto pDist = planeBoundary::distance(uFloat4<SI::m>(X[threadIdx.x]), float_u<SI::volume>(V_s), arrays);
				if (pDist.val.w < threshold)
					X[threadIdx.x] -= (pDist.val) * (pDist.val.w - threshold);
				X[threadIdx.x].w = assignedPosition.w;
				assignedPosition = X[threadIdx.x];
			}
			__syncthreads();
		}
	}
	// This check is required to avoid potential numerical issues with the optimization method.
	if (tas == threadAssignment::split)
		if (X[threadIdx.x].x != X[threadIdx.x].x || X[threadIdx.x].y != X[threadIdx.x].y || X[threadIdx.x].z != X[threadIdx.x].z || X[threadIdx.x].w != X[threadIdx.x].w)
			X[threadIdx.x] = arrays.position.second[pIdx].val + getshape(split_count, threadIdx.x) * factor;

	float threshold = OFFSET(V_s) * thresholdFactor *arrays.adaptivityThreshold;
	if (tas == threadAssignment::split) {
		auto pDist = planeBoundary::distance(uFloat4<SI::m>(X[threadIdx.x]), float_u<SI::volume>(V_s), arrays);
		if (pDist.val.w < threshold)
			X[threadIdx.x] -= (pDist.val) * (pDist.val.w - threshold);
	}
	__syncthreads();

}
deviceInline int32_t emit_particles(SPH::adaptive::Memory &arrays, threadAssignment tas, float pVol, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	int32_t new_idx = 0;
	__syncthreads();
	if (tas == threadAssignment::split) {
		//int32_t split_idx = sIdx;
		int32_t parent_idx = sIdx + pIdx;
		auto t_0 = -arrays.blendSteps * arrays.timestep;
		auto x_j = X[threadIdx.x];
		new_idx = threadIdx.x == split_count - 1 ? pIdx : sIdx + threadIdx.x;
		// simply copy all properties by default
		sArray->callOnArray([&](auto* first, auto* second) {
			if (first != nullptr)
				first[new_idx] = first[pIdx];
			});
	}
	__syncthreads();
	if (tas == threadAssignment::split) {
		int32_t parent_idx = sIdx + pIdx;
		auto t_0 = -arrays.blendSteps * arrays.timestep;
		auto x_j = X[threadIdx.x];
		arrays.lifetime[new_idx] = t_0;
		arrays.splitIndicator[new_idx] = 1;
		arrays.parentPosition.first[new_idx] = arrays.position.second[pIdx];
		arrays.parentIndex.first[new_idx] = parent_idx;
		arrays.parentVolume[new_idx] = float_u<SI::volume>(pVol);
		arrays.position.first[new_idx] = float4_u<SI::m>{ x_j.x, x_j.y, x_j.z, hFromV(pVol *  1.f / (float)split_count) };
		arrays.volume.first[new_idx] = float_u<SI::volume>(pVol * 1.f / (float)split_count);
	}
	__syncthreads();
	return new_idx;
}

template<neighbor_list neigh>
deviceInline float optimizeMassDistribution(SPH::adaptive::Memory &arrays, threadAssignment tas, float pVol, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	__syncthreads();
	//return  1.f / (float)split_count;
	int32_t i = pIdx;
	constexpr auto neighborhood = neigh;
	for (int32_t ii = 0; ii < split_count; ++ii)
		tau_s[ii] = 1.f / (float)split_count;
	__syncthreads();
	// The next section executes the actual mass ratio optimization as per the submission
	//if (split_count > 3) {
	//	__syncthreads();
		// initially we assume that the weights are simply lambda = 1/n as the static optimizations do not change the
		// weights significantly.
	for (int32_t ii = 0; ii < split_count; ++ii)
		tau_s[ii] = 1.f / (float)split_count;
	// initialize the curand state for the normal distributions
	curandState localState;
	curand_init(seed + blockIdx.x, threadIdx.x, 0, &localState);
	// configuration parameters
	auto variance = 2.f;
	constexpr auto varianceMultiplier = 0.5f;
	for (int32_t gi = 0; gi < 8; ++gi) {
		// in contrast to before we cannot parallelize the calculation of a single error term but instead
		// calculate the exact same error term on all threads which still works very efficiently due to ideal
		// memory access patterns
		float volumes[16];
		float sum = 0.f;
		// Update the evolution
		for (int32_t ii = 0; ii < split_count; ++ii) {
			// the actual evolution step
			float elem = math::clamp(tau_s[ii] * (float)split_count + curand_normal(&localState) * variance, 0.25f, 2.f);
			if (threadIdx.x == 0)
				elem = 1.f;
			volumes[ii] = elem;
			sum += elem;
		}
		for (int32_t ii = 0; ii < split_count; ++ii) {
			volumes[ii] /= sum;
		}
		float error = 0.f;
		// calculate the error terms
		iterateNeighbors(j) {
			auto Xs = arrays.position.second[j].val;
			float tauH = -pVol * kernel(Xs, arrays.position.second[i].val);
			for (int32_t ii = 0; ii < split_count; ++ii) {
				auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
				tauH += pVol * volumes[ii] * kernel(Xs, Xi);
			}
			//tauH /= (float)arrays.neighborListLength[i];
			error += tauH * tauH;
		}
		//float tauB = -pVol * boundary::spline(arrays.position.second[i], arrays.volume.second[i], arrays, boundary::kind::plane, -1).val;
		//for (int32_t ii = 0; ii < split_count; ++ii) {
		//	auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
		//	tauB += pVol * volumes[ii] * boundary::spline(Xi, arrays.volume.second[i], arrays, boundary::kind::plane, -1).val;
		//}
		//error += tauB * tauB;

		//float tauS0 = -arrays.density[i].val +
		//	 boundary::spline(arrays.position.first[i], pVol * volumes[ii], arrays);
			//SWH::spline(arrays.position.first[i], arrays).val;
		for (int32_t ii = 0; ii < split_count; ++ii) {
			auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
			float tauS = -arrays.density[i].val + boundary::spline(Xi, pVol * volumes[ii], 0.5f, arrays).val;
			for (int32_t ia = 0; ia < split_count; ++ia) {
				auto Xs = float4{ X[ia].x, X[ia].y, X[ia].z, hFromV(pVol * volumes[ia]) };
				tauS += pVol * volumes[ia] * kernel(Xs, Xi);
			}
			iterateNeighbors(j) {
				if (j != pIdx) {
					tauS += arrays.volume.second[j].val * kernel(Xi, arrays.position.second[j].val);
				}
			}
			//tauS /= (float)split_count;
			error += tauS * tauS;
		}
		// we now pick the term with the lowest error using an atomic operation and use that threads properties
		// to update the global state stored in tau_s
		*tau = FLT_MAX;
		__syncthreads();
		atomicMinFloat(tau, error);
		__syncthreads();
		//if(threadIdx.x == 0){
		if (atomicCAS((int32_t*)tau, __float_as_int(error), __float_as_int(-error)) == __float_as_int(error)) {
			for (int32_t ii = 0; ii < split_count; ++ii)
				tau_s[ii] = volumes[ii];
		}
		//if(tas == threadAssignment::split) arrays.debugArray[new_idx].x = error;
		// update the optimizer
		//__syncthreads(); 
		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	printf("##################################################\n%f : %f @ %f[%d] -> \n[%f %f %f %f |\n %f %f %f %f |\n %f %f %f %f |\n %f %f %f %f]\n",
		//		error, *tau, variance, gi,
		//		split_count < 1 ? 0.f : tau_s[0], split_count < 2 ? 0.f : tau_s[1], split_count < 3 ? 0.f : tau_s[2], split_count < 4 ? 0.f : tau_s[3],
		//		split_count < 5 ? 0.f : tau_s[4], split_count < 6 ? 0.f : tau_s[5], split_count < 7 ? 0.f : tau_s[6], split_count < 8 ? 0.f : tau_s[7],
		//		split_count < 9 ? 0.f : tau_s[8], split_count < 10 ? 0.f : tau_s[9], split_count < 11 ? 0.f : tau_s[10], split_count < 12 ? 0.f : tau_s[11],
		//		split_count < 13 ? 0.f : tau_s[12], split_count < 14 ? 0.f : tau_s[13], split_count < 15 ? 0.f : tau_s[14], split_count < 16 ? 0.f : tau_s[15]);
		//}
		__syncthreads();
		variance *= varianceMultiplier;
	}
	__syncthreads();
	//}
	if (tas != threadAssignment::split)
		return 0.f;
	// all non split assigned particles are now done and all that is left to do is to create the actual refined particles
	if (threadIdx.x == 0) {
		float sum = 0.f;
		bool flag = false;
		for (int32_t ii = 0; ii < split_count; ++ii) {
			sum += tau_s[ii];
			flag = tau_s[ii] <= 0.f || (tau_s[ii] != tau_s[ii]) || flag;
		}
		if (flag) {
			for (int32_t ii = 0; ii < split_count; ++ii)
				tau_s[ii] = 1.f;
			sum = split_count;

			for (int32_t ii = 0; ii < split_count; ++ii) {
				tau_s[ii] /= sum;
			}
		}
	}
	//__syncthreads();
	// 0.469850
	// 0.24509788044034622
	return tau_s[threadIdx.x];

}
deviceInline auto getAssignment(int32_t split_count, int32_t cIdx) {
	if (threadIdx.x < split_count) {
		return threadAssignment::split;
	}
	else if (cIdx != -1) {
		return threadAssignment::neighbor;
	}
	return threadAssignment::none;
}
deviceInline auto getAssignedPosition(SPH::adaptive::Memory& arrays, float4* X, int32_t cIdx, threadAssignment tas) {
	if (tas == threadAssignment::split) {
		return X[threadIdx.x];
	}
	else if (tas == threadAssignment::neighbor) {
		return arrays.position.second[cIdx].val;
	}
	return float4{ 0.f,0.f,0.f,0.f };
}

deviceInline bool parentCloseToBoundary(SPH::adaptive::Memory& arrays, int32_t i) {
	auto pDist = planeBoundary::distance(arrays.position.second[i], arrays.volume.second[i], arrays);
	if (pDist.val.w < support_from_volume(1.f) * kernelSize() * 1.5f)
		return true;
	return false;
}

template<neighbor_list neighborhood>
deviceInline void randomizePattern(SPH::adaptive::Memory& arrays, int32_t split_count, float4* X, float4* refX, float* result, int32_t i, uint32_t state) {
	float pVol = arrays.volume.second[i].val;
	float4 pPos = arrays.position.second[i].val;
	float V_s = pVol / (float)split_count;
	if (threadIdx.x < split_count) {
		refX[threadIdx.x] = getshape(split_count, threadIdx.x) * getScalingFactor(arrays, i);
		refX[threadIdx.x].w = 0.f;
	}
	__syncthreads();
	state += threadIdx.x;
	*result = -FLT_MAX;
	__syncthreads();
	for (int32_t j = 0; j < 8; ++j) {
		if (threadIdx.x == 0)
			*result = -*result;
		__syncthreads();

		float u1 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u2 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u3 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float4 q = {
			sqrtf(1.f - u1) * sinf(2.f * CUDART_PI_F * u2),
			sqrtf(1.f - u1) * cosf(2.f * CUDART_PI_F * u2),
			sqrtf(u1) * sinf(2.f * CUDART_PI_F * u3),
			sqrtf(u1) * cosf(2.f * CUDART_PI_F * u3)
		};
		auto getNew = [&](int32_t idx) {
			float4 v = refX[idx];
			v.w = 0.f;
			float4 u = q;
			float s = u.w;
			u.w = 0.f;
			float4 offset = 2.0f * math::dot3(u, v) * u + (s*s - math::dot3(u, u)) * v + 2.0f * s * math::cross(u, v);
			float4 result = pPos + offset;
			result.w = support_from_volume(V_s);
			return result;
		};
		float densityS[17];
		for (int32_t s = 0; s < split_count + 1; ++s)
			densityS[s] = 0.f;
		densityS[16] = pVol * kernel(pPos, pPos) + boundary::spline(pPos, pVol, 0.5f, arrays);
		__syncthreads();
		float tau = 0.f;
		iterateNeighbors(j) {
			auto x_j = arrays.position.second[j].val;
			auto V_j = arrays.volume.second[j].val;
			auto kp = kernel(x_j, pPos);
			float tauj = -pVol * kp;
			densityS[16] += V_j * kp;
			for (int32_t s = 0; s < split_count; ++s) {
				auto ks = kernel(x_j, getNew(s));
				tauj += V_s * ks;
				densityS[s] += V_j * ks;
			}
			tau += tauj * tauj;
		}
		for (int32_t s = 0; s < split_count; ++s) {
			auto xs = getNew(s);
			for (int32_t ss = 0; ss < split_count; ++ss)
				densityS[s] += V_s * kernel(xs, getNew(ss));
		}
		float tauS = 0.f;
		for (int32_t s = 0; s < split_count; ++s) {
			densityS[s] = densityS[16] - densityS[s] - boundary::spline(getNew(s), V_s, 0.5f, arrays);
			tauS += densityS[s] * densityS[s];
		}
		tau += tauS;
		__syncthreads();
		atomicMinFloat(result, tau);
		__syncthreads();
		if (atomicCAS((int32_t*)result, __float_as_int(tau), __float_as_int(-tau)) == __float_as_int(tau))
			for (int32_t s = 0; s < split_count; ++s)
				X[s] = getNew(s);
		__syncthreads();
	}
}

template<neighbor_list neigh>
deviceInline void gradient_particles(SPH::adaptive::Memory &arrays, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	*tau = 0.f;
	constexpr auto neighborhood = neigh;
	int32_t i = pIdx;
	float pVol = arrays.volume.second[pIdx].val;
	float hl = support_from_volume(pVol);	
	float V_s = pVol / (float)split_count;
	auto h = hFromV(V_s);

	auto tas = getAssignment(split_count, cIdx);
	generateInitial<neigh>(arrays, split_count, X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
	//randomizePattern<neigh>(arrays, split_count, X, gradX, tau, pIdx, (uint32_t)((pIdx + sIdx + cIdx + threadIdx.x) * split_count));
	gradientDescent<neigh>(arrays, split_count, X, gradX, tau_s, tau_h, tau, pIdx, cIdx, tas, pVol, V_s, hl, h);
	int32_t new_idx = emit_particles(arrays,tas, pVol, split_count,X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
	if (parentCloseToBoundary(arrays,i) || (split_count < 5) )
		return;
	auto ratio = optimizeMassDistribution<neigh>(arrays, tas, pVol, split_count, X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
	if (tas == threadAssignment::split) {
		math::unit_assign<4>(arrays.position.first[new_idx], float_u<SI::m>(hFromV(pVol * ratio)));
		arrays.volume.first[new_idx] = float_u<SI::volume>(pVol * ratio);
	}

}
/** This function is used to initialize the state for the dynamic optimization by declaring
 * all reuqired shared memory arrays, as well as assigning neighboring particles to threads of a block. This
 * implementation in general works by assigning n = 96 threads to a particle and then splitting the thread assginments.
 * The first s threads, with s being the number of particles a particle is split into, are assigned to handle the split
 * particles for the error function evaluation, wheras the other threads arae  initially assigned to a separate
 * neighboring particle. If the number of neighbors is larger than the number of threads we ignore this point. In order
 * to avoid potential kernel time outs, especially on windows, this function is inteded to be called with a limited
 * number of blocks, where each block is still indexed from 1 to this number based on the way blockIdx works. As such we
 * pass the offset of the current call to this function to be able to calculate the actual index. It is then
 * straight-forward to calculate the actual index using the result of the compact operation. ptcls is the number of
 * particles that are being split, seed is used as the seed for the evolutionary optimization. sArray is just passed.**/
 /* The state word must be initialized to non-zero */
template<neighbor_list neighborhood>
__global__ void gradientSplit(SPH::adaptive::Memory arrays, int32_t offset, int32_t ptcls, int32_t seed, sortingArray* sArray) {
	// the global shared memory state. In total 32 float4 elements and 113 float elements are used for a total of
	// 964 Byte of shared memory per block. With most gpu architectures 64KB of SM are available per SMX which means
	// that the limit of blocks based on SM with a block size of 96 is 67 which is larger than the theoretical HW limit
	// of 21 so no limit to occupancy exists here.
	__shared__ float4 xn[16];
	__shared__ float4 gradXn[16];
	__shared__ float tau_s[16];
	__shared__ float tau_h[96];
	__shared__ float tau[1];
	// first we get the actual index of the particle by looking at the result of the compaction
	DBG_STMNT if (blockIdx.x != 0) return;
	DBG_STMNT	printf("%d - [%d : %d] -> %d : %d @ %d\n", __LINE__, blockIdx.x, threadIdx.x, offset, offset + blockIdx.x, ptcls);
	int32_t idx = blockIdx.x + offset;
	if (idx >= ptcls) return;
	int32_t i = arrays.particleIndexCompact[idx];
	// and load all required information, as well as reclamping the decision for safety
	int32_t decision = arrays.adaptiveClassification[i];
	int32_t splitIdx = arrays.parentIndex.first[i];
	int32_t careTakerIdx = -1;
	decision = math::clamp(decision, 2, 16);
	//if (decision == 3)
	//	decision = 4;
	// this synchthreads is not strictly required but helps conceptually make the code more clear
	__syncthreads();
	// for all threads > s, as described before, we assign one neighboring particle. This works by
	// iterating over all neighbors of o and assigning the c-th neighbor to the c + s-th thread. This
	// could be significantly simplified when using certain neighbor list algorithms.
	if (threadIdx.x >= decision) {
		int32_t ctr = 0;
		// we always use neighbor_list::constrained for adaptive simulations.
		//constexpr auto neighborhood = neighbor_list::cell_based;
		iterateNeighbors(j) {
			// we do not assign the original particle o to any thread
			if (j == i) continue;
			auto k = kernel(arrays.position.second[i], arrays.position.second[j]);
			if (k > 0.f) {
				if (ctr + decision == threadIdx.x)
					careTakerIdx = j;
				ctr++;
			}
		}
	}
	if (threadIdx.x == 0)
		atomicAdd(arrays.adaptivityCounter + (math::clamp(decision, 2, 16) - 1), 1);
	__syncthreads();
	//return;
	// call the actual function that does the dynamic optimizations.
	gradient_particles<neighborhood>(arrays, decision, xn, gradXn, tau_s, tau_h, tau, i, splitIdx, careTakerIdx, sArray, seed);
}
// This wrapper is there to simplify calling certain functions and wraps indicateParticles into splitIndices
basicFunction(splitIndices, indicateParticles, "Adaptive: splitting particles");

// The main function of the splitting process on host side
void splittingParticles(SPH::adaptive::Memory mem) {
	// decide is a function that classifies the particles using some user defined method, e.g. using surface distance
	// which is use-case specific. The results are stored in adaptiveClassification as floats.
	launch<decide>(mem.num_ptcls, mem);
#ifndef EIGENADAPTIVE
	int32_t old = mem.num_ptcls;
	cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
	callSplit(sorting_list, mem);
	cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	get<parameters::num_ptcls>() = mem.num_ptcls;
	get<parameters::splitPtcls>() = mem.num_ptcls - old;
#else
	// we use a simple static set of variables to calculate the seeds using C++11 random functions.
	// store the old number of particles to calculate the number of changed entries for visualization
	int32_t old = mem.num_ptcls;
	// store the number of particles into a global to be atomically incremented/decremented
	cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
	// indicate the particles by either storing INT_MAX (invalid) or their index(valid) in particleIndex
	launch<splitIndices>(mem.num_ptcls, mem);
	// we compact particleIndex which gives the number of particles that are being split.
	int32_t ptcls = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid_split());
	if (ptcls > 0) {
		// we store the old positions in the rear pointers of the arrays as the actual values change in the split process
		// which also allows us to not have to recalculate particle neighbors after every split operation. However, this
		// is similar in concept to assuming that all split operations are independently optimized which leads to the 
		// errors as described in the submission.
		cuda::memcpy(arrays::position::rear_ptr, arrays::position::ptr, mem.num_ptcls * sizeof(float4), cudaMemcpyDeviceToDevice);
		cuda::memcpy(arrays::volume::rear_ptr, arrays::volume::ptr, mem.num_ptcls * sizeof(float), cudaMemcpyDeviceToDevice);
		// sortingArray contains an entry per persistent array that needs to be touched by the splitting process. The 
		// fillArray function assigns the correct pointers to all entries.
		sortingArray sArray{};
		static sortingArray* sArrayPtr;
		static bool once = true;
		if (once) {
			cudaMalloc(&sArrayPtr, sizeof(sortingArray));
			once = false;
		}
		sArray.fillArray();
		cudaMemcpy(sArrayPtr, &sArray, sizeof(sortingArray), cudaMemcpyHostToDevice);
		cuda::sync("Before split");
		int32_t stepSize = 768*2;
		// this incremental call is intended to avoid kernel launch time outs on windows.
		for (int32_t i = 0; i <= ptcls / stepSize; ++i) {
			switch (*parameters::neighborhood_kind::ptr) {
			case neighbor_list::cell_based:
				gradientSplit<neighbor_list::cell_based> << <stepSize, 96 >> > (mem, i * stepSize, ptcls, get<parameters::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::compactCell:
				gradientSplit<neighbor_list::compactCell> << <stepSize, 96 >> > (mem, i * stepSize, ptcls, get<parameters::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::constrained:
				gradientSplit<neighbor_list::constrained> << <stepSize, 96 >> > (mem, i * stepSize, ptcls, get<parameters::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::compactMLM:
				gradientSplit<neighbor_list::compactMLM> << <stepSize, 96 >> > (mem, i * stepSize, ptcls, get<parameters::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::masked:
				gradientSplit<neighbor_list::masked> << <stepSize, 96 >> > (mem, i * stepSize, ptcls, get<parameters::frame>() + i, sArrayPtr);
				break;
			}
			cuda::sync();
			//break;
		}
		cuda::sync("After split");
		// we read back the current state of the atomic counter in order to update the number of particles.
		cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	}
	cuda::sync();
	// update global parameters based on the result of the splitting process.
	get<parameters::num_ptcls>() = mem.num_ptcls;
	//get<parameters::splitPtcls>() = mem.num_ptcls - old;
#endif
}
#include <random>

/** This function is used to adjust the resolution of particles in the simulation it does splitting,
 * merging and mass sharing closely following the reference paper**/
void SPH::adaptive::adapt(Memory mem) {
	static bool once = true;
	if (once) {
		std::cout << "Continuous Adaptivity module built " << __TIMESTAMP__ << std::endl;

		get<parameters::splitPtcls>().resize(16);
		get<parameters::mergedPtcls>().resize(16);
		get<parameters::sharedPtcls>().resize(16);
		once = false;
	}

	uint32_t split_ptcls;
	/* To avoid certain extra checks in the code we can restrict the function to either merge
	 particles (decreasing resolution) or to split particles (increasing resolution). As this is
	 done on a 2 frame period this should have no appreciable effect on the adaptation rate.*/
	if (parameters::frame{} % 2 == 0) {
		launch<decide>(mem.num_ptcls, mem);
		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::Memset(mem.mergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
		cuda::Memset(mem.ptclCounter, 0x00, sizeof(float));
		cuda::Memset(mem.mergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
		std::random_device rngd;
		auto rng = std::default_random_engine{rngd()};
		std::vector<int32_t> idxs(mem.num_ptcls);
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::shuffle(std::begin(idxs), std::end(idxs), rng);
		int32_t* randomIdx = nullptr;
		cudaMalloc(&randomIdx, sizeof(int32_t) * mem.num_ptcls);
		cudaMemcpy(randomIdx, idxs.data(), sizeof(int32_t) * mem.num_ptcls, cudaMemcpyHostToDevice);

		launch<detectMergingParticles>(mem.num_ptcls, mem);
		cuda::sync();
		launch<grabEvenMergingParticles>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		launch<grabOddMergingParticles>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		cudaFree(randomIdx);
		MergeGrabbed(sorting_list, mem);
		cuda::memcpy(get<parameters::mergedPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		cuda::memcpy(&split_ptcls, mem.ptclCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		//get<parameters::mergedPtcls>() = split_ptcls;
	}
	else {
		//std::cout << "..." << std::endl;
		// Share particles
		launch<decide>(mem.num_ptcls, mem);
		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::Memset(mem.ptclCounter, 0x00, sizeof(float));
		cuda::Memset(mem.mergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
		cuda::Memset(mem.mergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
		std::random_device rngd;
		auto rng = std::default_random_engine{ rngd() };
		std::vector<int32_t> idxs(mem.num_ptcls);
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::shuffle(std::begin(idxs), std::end(idxs), rng);
		int32_t* randomIdx = nullptr;
		cudaMalloc(&randomIdx, sizeof(int32_t) * mem.num_ptcls);
		cudaMemcpy(randomIdx, idxs.data(), sizeof(int32_t) * mem.num_ptcls, cudaMemcpyHostToDevice);
		launch<detectSharingParticles>(mem.num_ptcls, mem);
		launch<grabEvenSharingParticles>(mem.num_ptcls, mem, randomIdx);
		launch<grabOddSharingParticles>(mem.num_ptcls, mem, randomIdx);		
		cudaFree(randomIdx);
		ShareGrabbed(sorting_list, mem);
		cuda::memcpy(&split_ptcls, mem.ptclCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		cuda::memcpy(get<parameters::sharedPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		//get<parameters::sharedPtcls>() = split_ptcls;
		// Split particles, if the old particle count is close to the maximum particle count of the
		// simulation do nothing.

		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::sync();
		splittingParticles(mem);
		cuda::sync();
		cuda::memcpy(get<parameters::splitPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		//int32_t old = mem.num_ptcls;
		//cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
		//callSplit(sorting_list, mem);
		//cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
		//get<parameters::num_ptcls>() = mem.num_ptcls;
		//get<parameters::splitPtcls>() = mem.num_ptcls - old;
	}

	auto min = algorithm::reduce_min(mem.volume.first, mem.num_ptcls);
	auto max = PI4O3 * math::power<3>(mem.radius);
	auto ratio = max / min;
	get<parameters::minVolume>() = min.val;
	*parameters::adaptive::ratio::ptr = ratio.val;
}
