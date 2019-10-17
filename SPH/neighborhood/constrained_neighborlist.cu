#include <SPH/neighborhood/constrained_neighborlist.cuh>
#include <SPH/neighborhood/ordered_list.cuh>
#include <utility/include_all.h>
#include <queue>

// This function updates the support radius of alll particles by calculating a new one based on the
// current support radius and the rest support radius based on the rest volume. This value is stored
// in temporary arrays to avoid some race conditions.
basicFunctionType initializeSupport(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i); 
	if (h_i == FLT_MAX)
		return;
	auto V_i = arrays.volume[i];
	float targetNeighbors = kernelNeighbors() * 0.95f;
	auto h_c = support_from_volume(V_i); 
	h_i = arrays.omega * h_i + (1.f - arrays.omega) * h_c;
	h_i = math::clamp(h_i, 0.0_m, h_c);

	arrays.closestNeighbor[i] = 0;
	arrays.supportEstimate[i] = h_i;
	arrays.support[i] = h_i;
}

// This function tries to create a neighborlist using the constrained neighborlist and if it fails
// it will first try temporarily storing neighborhood data in the overflow array and subsequently
// reducing the support radius of the particle with too many neighbors.
cellFunctionType try_neighborhood(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);
#ifdef __CUDA_ARCH__
  extern __shared__ float_u<SI::m> supports[];
  ordered_list_highest<float_u<SI::m>, int16_t> distance_list(supports, float_u < SI::m>{FLT_MAX}, blockDim.x, ELEM_NUM);
#else
	std::priority_queue<float_u<SI::m>, std::vector<float_u<SI::m>>, std::function<bool(float_u<SI::m>, float_u<SI::m>)>> distance_list([](auto lhs, auto rhs){return lhs < rhs;});
#endif
	auto x_i = arrays.position[i];
	if (math::unit_get<4>(x_i) == FLT_MAX)
		return;
	auto h_i = arrays.support[i];
	auto idx = [=](auto id) {return arrays.max_numptcls * id + i; };
  auto overHeadSize = arrays.overhead_size;
  auto allPtcls = arrays.max_numptcls;
  auto overheadRoom = allPtcls / overHeadSize;
  auto capacity = arrays.target_neighbors + arrays.support_leeway;
  int32_t curNeighborIdx = 0;
  int32_t overhead_idx = static_cast<int32_t>(UINT_MAX);

  cuda_atomic<int32_t> overheads(arrays.closestNeighbor);

  iterateCells(x_i, j) {
	  auto x_j = arrays.position[j];
	  if (math::unit_get<4>(x_i) == FLT_MAX)
		  continue;
	  auto h_j = arrays.support[j];
	  auto x_ij = math::distance3(x_i, x_j);
	  auto h_ij = 0.5f * (h_i + h_j) * kernelSize();
	  if (x_ij <= h_ij) {
		  auto k_ij = x_ij / kernelSize() * 2.f - h_j;
		  distance_list.push(k_ij);
		  if (arrays.neighborListSwap[0] != 1) {
			  if (curNeighborIdx < capacity)
				  arrays.neighborList[idx(curNeighborIdx++)] = j;
			  else {
				  if (overhead_idx == static_cast<int32_t>(UINT_MAX)) {
					  overhead_idx = overheads.inc();
					  arrays.neighborList[idx(curNeighborIdx)] = static_cast<int32_t>(UINT_MAX - 1u);
					  if (overhead_idx >= overHeadSize) {
						  arrays.neighborListSwap[0] = 1;
						  continue;
					  }
				  }
				  if (curNeighborIdx - capacity < overheadRoom) {
					  arrays.neighborOverhead[overHeadSize * (curNeighborIdx++ - capacity) + overhead_idx] = j;
				  }
				  else {
					  arrays.neighborListSwap[0] = 1;
					  continue;
				  }
			  }
		  }
	  }
  };
  if (curNeighborIdx >= arrays.target_neighbors + arrays.support_leeway - 1) {
#ifdef __CUDA_ARCH__
    int32_t diff = curNeighborIdx - (arrays.target_neighbors) + arrays.error_factor;
    int32_t clamped = math::clamp(diff, 0, ELEM_NUM - 1);
    distance_list.sort_values_descendeding();
    auto new_support = SM_ID(supports, clamped);
    if (SM_ID(supports, clamped) == SM_ID(supports, math::clamp(clamped - 1, 0, ELEM_NUM))) {
      while (SM_ID(supports, clamped) == SM_ID(supports, math::clamp(clamped - 1, 0, ELEM_NUM - 1)) && clamped != ELEM_NUM - 1) {
        ++clamped;
      }
      new_support = SM_ID(supports, math::clamp(clamped, 0, ELEM_NUM - 1));
    }
#else
	  for (int32_t it = curNeighborIdx; it >= arrays.target_neighbors + arrays.support_leeway - 2; it--)
		  distance_list.pop();
	  auto new_support = distance_list.top();
#endif
	  arrays.supportEstimate[i] = new_support;
      arrays.supportMarker[i] = i;
  }
  arrays.neighborListLength[i] = curNeighborIdx;
  arrays.neighborOverheadCount[i] = overhead_idx;
}

// This function merges the normal neighborlist with the temporary overhead list in case it was
// needed and finalizes the neighborlist. If the lists could not be merged an error counter will be
// increased to cause another iteration.
basicFunctionType clean_neighborhood(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);

  auto capacity = arrays.target_neighbors + arrays.support_leeway;
  auto curNeighborIdx = 0;
  auto neighborCount = arrays.neighborListLength[i];
  auto overhead_idx = arrays.neighborOverheadCount[i];
  auto overheadSize = arrays.overhead_size;
  auto allPtcls = arrays.max_numptcls;
  auto overheadRoom = allPtcls / overheadSize;

  auto x_i = arrays.position[i];
  if (math::unit_get<4>(x_i) == FLT_MAX)
	  return;
  auto h_i = arrays.supportEstimate[i];

  if (neighborCount >= capacity) {
    if (arrays.neighborListSwap[0] == 0) {
      for (int32_t it = 0; it < capacity; ++it) {
		  auto j = arrays.neighborList[allPtcls * it + i];
        if (j == static_cast<int32_t>(UINT_MAX - 1u))
          break;
		auto x_j = arrays.position[j];
		auto h_j = arrays.supportEstimate[j];
		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = 0.5f * (h_i + h_j) * kernelSize();

        if (x_ij <= h_ij) 
          arrays.neighborList[allPtcls * curNeighborIdx++ + i] = j;
      }
	  for (int32_t it = capacity; it < capacity + overheadRoom; ++it) {
        if (overhead_idx >= overheadSize || overhead_idx < 0)
          break;
		auto j = arrays.neighborOverhead[overheadSize * (it - capacity) + overhead_idx];
		if (j == static_cast<int32_t>(UINT_MAX))
			break;
		auto x_j = arrays.position[j];
		auto h_j = arrays.supportEstimate[j];
		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = 0.5f * (h_i + h_j) * kernelSize();

        if (x_ij <= h_ij) {
          if (curNeighborIdx >= capacity) {
            arrays.supportMarker[i] = i;
            break;
          }
          arrays.neighborList[allPtcls * curNeighborIdx++ + i] = j;
        }
      }
    }
	arrays.neighborList[allPtcls * curNeighborIdx + i] = static_cast<int32_t>(UINT_MAX);
	arrays.neighborListLength[i] = curNeighborIdx;
  }
  arrays.support[i] = arrays.supportEstimate[i];
  arrays.supportMarker[i] += arrays.neighborListSwap[0];
}

// This function updates the support radius of every particle to the constrained support radius.
basicFunctionType finalize_neighborhood(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);

	auto x_i = arrays.position[i];
	if (math::unit_get<4>(x_i) == FLT_MAX)
		return;
	auto h_i = arrays.supportEstimate[i];
	auto h_s = arrays.support[i];
	math::unit_assign<4>(arrays.position[i], h_s);
	arrays.supportEstimate[i] = arrays.support[i];
	arrays.support[i] -= h_i;
#ifdef DEBUG_INVALID_PTCLS
	if (arrays.neighborListLength[i] == 0) {
		printf("%d has 0 neighbors [%f %f %f %f] -> %f : %f : %f\n", i, x_i.val.x, x_i.val.y, x_i.val.z, x_i.val.w, arrays.support[i].val, arrays.supportEstimate[i].val, arrays.volume[i].val);
	}
#endif
}

// This function updates the support radius of all particles by calculating a new one based on the
// current support radius and the rest support radius based on the rest volume. This value is stored
// in temporary arrays to avoid some race conditions.
basicFunctionType initSupport(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);
	if (h_i == FLT_MAX)
		return;
	arrays.closestNeighbor[i] = 0;
	arrays.supportEstimate[i] = h_i;
	arrays.support[i] = h_i;
}


cellFunctionType basic_neighborhood(SPH::ConstrainedNeighborList::Memory arrays) {
	checkedParticleIdx(i);
	//cache_arrays((pos, position));
	//auto idx = [=](auto id) {return arrays.max_numptcls * id + i; };
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);
	if (h_i == FLT_MAX) return;

	int32_t curNeighborIdx = 0;
	auto capacity = arrays.target_neighbors + arrays.support_leeway; 
	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX) continue;
		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = 0.5f * (h_i + h_j) * kernelSize();
		if (x_ij <= h_ij) {
			if(curNeighborIdx < capacity)
				arrays.neighborList[arrays.max_numptcls * curNeighborIdx + i] = j;
			else
				arrays.neighborListSwap[0] = 1;
			curNeighborIdx++;
		}
	}
	arrays.neighborListLength[i] = curNeighborIdx;
#ifdef DEBUG_INVALID_PTCLS
	if (arrays.neighborListLength[i] == 0) {
		printf("%d has 0 neighbors [%f %f %f %f] -> %f : %f : %f\n", i, x_i.val.x, x_i.val.y, x_i.val.z, x_i.val.w, arrays.support[i].val, arrays.supportEstimate[i].val, arrays.volume[i].val);
	}
#endif
}

basicFunction(initBasic, initSupport, "Neighborhood: C init");
cellFunction(basicNeighborlist, basic_neighborhood, "Neighborhood: Basic", caches<float4>{});

basicFunction(initConstrained, initializeSupport, "Neighborhood: C init");
cellFunction(constrainedNeighborlist, try_neighborhood, "Neighborhood: C trying", caches<float4, float4, float4>{});
basicFunction(cleanConstrained, clean_neighborhood, "Neighborhood: C clean");
basicFunction(finalizeConstrained, finalize_neighborhood, "Neighborhood: C finalize");

struct is_valids {
	hostDeviceInline bool operator()(const uint32_t x) { return x != UINT_MAX; }
};

// This is the main function of the module and creates a constrained neighborlist. This method is
// usually but especially for adaptive simulations the method might require multiple iterations
// which can become very expensive as every iteration requires an expensive cell iteration. For
// adaptive simulations using a cell based approach might be more beneficial but this method works
// well combined with an external support constrainer that allows for a very small overhead in
// memory consumption.
void SPH::ConstrainedNeighborList::calculate_neighborlist(Memory mem) {
  get<parameters::adjusted_particles>() = 0;
  auto resetArrays = []() {
    cuda::arrayMemset<arrays::supportMarker>(0xFFFFFFFF);
    cuda::arrayMemset<arrays::neighborListSwap>(0x0000);
    cuda::arrayMemset<arrays::closestNeighbor>(0x0000);
    cuda::arrayMemset<arrays::neighborOverheadCount>(0x0000);
    cuda::arrayMemset<arrays::neighborOverhead>(0xFFFFFFFF);
  };
  if (*parameters::modules::support::ptr == "constrained") {
    launch<initBasic>(mem.num_ptcls, mem);
    resetArrays();
	launch<basicNeighborlist>(mem.num_ptcls, mem);
	int32_t finished;
	cuda::memcpy(&finished, mem.neighborListSwap, sizeof(int32_t), cudaMemcpyDeviceToHost);

	if (finished == 0)
		return;
  } else
	  launch<initConstrained>(mem.num_ptcls, mem);

  uint32_t it = 0;
  do {
    resetArrays();
	launch<constrainedNeighborlist>(mem.num_ptcls, mem);
    get<parameters::adjusted_particles>() += (int32_t) algorithm::copy_if(
        mem.supportMarker, mem.supportMarkerCompacted, mem.num_ptcls, is_valids{});
    cuda::arrayMemset<arrays::supportMarker>(0xFFFFFFFF);
	launch<cleanConstrained>(mem.num_ptcls, mem);
    ++it;
  } while (algorithm::copy_if(mem.supportMarker, mem.supportMarkerCompacted, mem.num_ptcls,
                              is_valids{}) > 0);
  launch<finalizeConstrained>(mem.num_ptcls, mem);

  get<parameters::support_current_iteration>() = it;
}
