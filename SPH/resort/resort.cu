#include <SPH/resort/resort.cuh>
#include <utility/include_all.h>

basicFunctionType cudaHashParticles(SPH::Resort::Memory arrays) {
	checkedParticleIdx(i);
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);
	
	if (h_i != FLT_MAX && math::unit_get<1>(x_i) == math::unit_get<1>(x_i)) {
		arrays.resortIndex[i] = position_to_linear(x_i,arrays);
		arrays.particleparticleIndex[i] = i;
	}
	else {
		arrays.resortIndex[i] = arrays.grid_size.x * arrays.grid_size.y * arrays.grid_size.z - 1;
		arrays.particleparticleIndex[i] = i;
	}
}

hostDeviceInline void resort_arrays(uint32_t, uint32_t) {}

template <typename T, typename... Ts> hostDeviceInline void resort_arrays(uint32_t trgIdx, uint32_t srcIdx, T arg, Ts... ref) {
	if (arg.second != nullptr && arg.first != nullptr)
		arg.second[trgIdx] = arg.first[srcIdx];
	resort_arrays(trgIdx, srcIdx, ref...);
}

templateFunctionType cudaSort(SPH::Resort::Memory arrays, Ts... tup) {
	checkedParticleIdx(i);
#ifdef __CUDA_ARCH__
	extern __shared__ uint sharedHash[]; // blockSize + 1 elements

	int32_t num_cells =(int32_t)( arrays.grid_size.x * arrays.grid_size.y * arrays.grid_size.z );

	auto ptclHash = arrays.resortIndex[i];
	sharedHash[threadIdx.x + 1] = ptclHash;
	if (i > 0 && threadIdx.x == 0)
		sharedHash[0] = arrays.resortIndex[i - 1];
	__syncthreads();
	if (i == 0 || ptclHash != sharedHash[threadIdx.x]) {
		if (ptclHash < num_cells)
			arrays.cellBegin[ptclHash] = i;
			int32_t sHash = sharedHash[threadIdx.x];
		if (i > 0 && sHash < num_cells)
			arrays.cellEnd[sHash] = i;
	}
	if (i == arrays.num_ptcls - 1 && ptclHash < num_cells)
		arrays.cellEnd[ptclHash] = i + 1;
#else
	int32_t num_cells =(int32_t)( arrays.grid_size.x * arrays.grid_size.y * arrays.grid_size.z );
	auto ptclHash = arrays.resortIndex[i];
	if (i == 0 || ptclHash != arrays.resortIndex[i - 1]) {
		if (ptclHash < num_cells)
			arrays.cellBegin[ptclHash] = i;
		if (i > 0) {
			int32_t sHash = arrays.resortIndex[i - 1];
			if (sHash < num_cells)
				arrays.cellEnd[sHash] = i;
		}
	}
	if (i == arrays.num_ptcls - 1 && ptclHash < num_cells)
		arrays.cellEnd[ptclHash] = i + 1;
#endif
	resort_arrays(i, arrays.particleparticleIndex[i], tup...);
}

basicFunction(hashParticles, cudaHashParticles, "Linear sorting: Hashing");
templateFunction(sort, cudaSort, "Linear sorting : sort", caches<int32_t, int32_t>{} )

template <typename... Ts> auto callSort2(SPH::Resort::Memory mem, int32_t threads, Ts... tup) {
	launch<sort>(threads, mem, tup...);
}
template <typename... Ts> auto callSort(std::tuple<Ts...>, SPH::Resort::Memory mem) {
	callSort2(mem, mem.num_ptcls, std::make_pair(Ts::ptr, Ts::rear_ptr)...);
}

void SPH::Resort::resortParticles(Memory mem) {
	if (mem.num_ptcls == 0)
		return;
	launch<hashParticles>(mem.num_ptcls, mem);
	algorithm::stable_sort_by_key(mem.num_ptcls, mem.resortIndex, mem.particleparticleIndex);
	cuda::Memset(get<arrays::cellBegin>(), 0x0, info<arrays::cellBegin>().alloc_size);
	cuda::Memset(get<arrays::cellEnd>(), 0x0, info<arrays::cellEnd>().alloc_size);
	callSort(sorting_list, mem);

	uint lastcelloffset = mem.grid_size.x * mem.grid_size.y * mem.grid_size.z - 1;
	uint end = 0;
	uint begin = 0;
	cuda::memcpy(&end, mem.cellEnd + lastcelloffset, sizeof(uint), cudaMemcpyDeviceToHost);
	cuda::memcpy(&begin, mem.cellBegin + lastcelloffset, sizeof(uint), cudaMemcpyDeviceToHost);
	uint numInvalidPtcls = end - begin;
	if (numInvalidPtcls > 0) {
		get<parameters::num_ptcls>() -= numInvalidPtcls;
	}

	for_each(sorting_list, [](auto x) { x.swap(); });
}
