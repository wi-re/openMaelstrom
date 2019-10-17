#include <SPH/support/enforce_symmetry.cuh>
#include <utility/include_all.h>

basicFunctionType cudaMLMResolution(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
	auto V_i = arrays.volume[i];
	auto H_0 = support_from_volume(V_i) * kernelSize();
	int32_t r_i = (int32_t) math::clamp(math::floorf(math::abs(math::log2f(math::unit_get<1>(arrays.cell_size) / H_0))) - 0, 0, arrays.mlm_schemes - 1).val;
	arrays.MLMResolution[i] = r_i;
}

#define CORRECT_SYMMETRY

cellFunctionType enforceSymmetry(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position));
	alias_arrays((res, particleIndex));
	auto x_i = pos[i];
	auto h_i = support_h(x_i);
	auto r_i = res[i];
	if (h_i == FLT_MAX) return;
	h_i *= kernelSize();

	iterateCells(x_i, j) {
		auto x_j = pos[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX)
			continue;
		h_j *= kernelSize();

		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = (h_i + h_j) / 2.f;
		if (h_ij <= x_ij)
			continue;

		auto r_j = res[j];

		if (r_j > r_i) {
#ifdef CORRECT_SYMMETRY
			auto f_j = powf(0.5f, (float)r_j);
			auto hatX_i = integerCoords(x_i, arrays, f_j);
			auto hatX_j = integerCoords(x_j, arrays, f_j);
			auto difference = hatX_j - hatX_i;
			difference = math::fn([](auto a) {return abs(a); }, difference);
			if (math::max_elem(difference) > 1)
#endif
			{
				cuda_atomic<int32_t> res_atomic(res + j);
				int32_t snapped_mlm, old = res_atomic.val();
				do {
					snapped_mlm = old;
					if (snapped_mlm <= r_i)
						break;
					old = res_atomic.CAS(snapped_mlm, r_i);
				} while (old != snapped_mlm);
			}
		}
	}

}

cellFunctionType constrainSupport(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
#ifdef __CUDA_ARCH__
	//auto IDX = [](auto i) { return threadIdx.x * 4 + i % 4 + (i / 4) * blockDim.x * 4; };
	auto IDX = [](auto i) { return threadIdx.x * 32 + i; };
	extern __shared__ int8_t histogram[];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
	__syncthreads();
#else
	auto IDX = [](auto i) { return i; };
	int8_t histogram[32];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
#endif
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);

	if (h_i == FLT_MAX)
		return;
	h_i *= kernelSize();

	const float lower_edge = 0.5f;
	const int32_t limit = arrays.neighborlimit - 15;
	const auto bin_size = (h_i * (1.f - lower_edge)) / 32.f;

	int32_t neighs = 0;

	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX)
			continue;
		h_j *= kernelSize();

		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = (h_i + h_j) / 2.f;
		if (h_ij > x_ij)
			neighs++;
		else
			continue;

		auto constrained = x_ij * 2.f - h_j;
		auto actual = constrained - lower_edge * h_i;
		int32_t bin = math::castTo<int32_t>(math::floorf(actual / bin_size));
		bin = math::clamp(bin, 0, 31);
		histogram[IDX(bin)]++;
	}

	auto h_i_old = h_i;
	int32_t sum = 0;
	if (neighs >= limit)
		for (int32_t i = 0; i < 32; ++i) {
			sum += histogram[IDX(i)];
			if (sum >= limit) {
				h_i = h_i * lower_edge + ((float)i - 1) * bin_size;
				break;
			}
		}

	h_i /= kernelSize();
	if (h_i != h_i || h_i > h_i_old)
		h_i = h_i_old;
	h_i = math::clamp(h_i, 0.75f * support_h(x_i), support_from_volume(arrays.volume[i]));
	math::unit_assign<4>(arrays.position[i], h_i);
}

cellFunctionType constrainSupportBase(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
#ifdef __CUDA_ARCH__
	//auto IDX = [](auto i) { return threadIdx.x * 4 + i % 4 + (i / 4) * blockDim.x * 4; };
	auto IDX = [](auto idx) { return threadIdx.x * 32 + idx; };
	extern __shared__ int8_t histogram[];
	for (int32_t idx = 0; idx < 32; ++idx)
		histogram[IDX(idx)] = 0;
	__syncthreads();
#else
	auto IDX = [](auto i) { return i; };
	int8_t histogram[32];
	for (int32_t idx = 0; idx < 32; ++idx)
		histogram[IDX(idx)] = 0;
#endif
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);

	if (h_i == FLT_MAX)
		return;	
	h_i *= kernelSize();

	const float lower_edge = 0.5f;
	int32_t limit = arrays.neighborlimit - 10;
	const auto bin_size = (h_i * (1.f - lower_edge)) / 32.f;
	int32_t neighs = 0;

	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX)
			continue;
		h_j *= kernelSize();

		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = (h_i + h_j) / 2.f;
		if (h_ij > x_ij)
			neighs++;
		else
			continue;

		auto constrained = x_ij * 2.f - h_j;
		auto actual = constrained - lower_edge * h_i;
		int32_t bin = static_cast<int32_t>(floorf(actual / bin_size));
		if (bin <= 0) bin = 0;
		if (bin >= 31) bin = 31;
		//bin = math::clamp(bin, 0, 31);
		histogram[IDX(bin)]++;
	}

	auto h_i_old = h_i;
	int32_t sum = 0;
	//if (neighs >= limit && neighs - limit > 16) limit = limit + 16;

	if (neighs >= limit)
		for (int32_t idx = 0; idx < 32; ++idx) {
			sum += histogram[IDX(idx)];
			if (sum >= limit) {
				h_i = h_i * lower_edge + ((float)idx - 1) * bin_size;
				arrays.particleIndex[i] = idx;
				//if (idx  == 0)
				//	arrays.particleIndex[i] = i;
				break;
			}
		}
	else arrays.particleIndex[i] = -1;

	h_i /= kernelSize();
	if (h_i != h_i || h_i > h_i_old)
		h_i = h_i_old;
	//h_i = math::clamp(h_i, 0.75f * support_h(x_i), support_from_volume(arrays.volume[i]));
	math::unit_assign<4>(arrays.position[i], h_i);
}

cellFunctionType constrainSupportCompact(SPH::enforceSymmetry::Memory arrays, int32_t threads) {
	checkedThreadIdx(idx);
	int32_t i = arrays.particleIndexCompact[idx];

#ifdef __CUDA_ARCH__
	//auto IDX = [](auto i) { return threadIdx.x * 4 + i % 4 + (i / 4) * blockDim.x * 4; };
	auto IDX = [](auto i) { return threadIdx.x * 32 + i; };
	extern __shared__ int8_t histogram[];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
	__syncthreads();
#else
	auto IDX = [](auto i) { return i; };
	int8_t histogram[32];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
#endif
	auto x_i = arrays.position[i];
	auto h_i = support_h(x_i);

	if (h_i == FLT_MAX)
		return;
	h_i *= kernelSize();

	const float lower_edge = 0.5f;
	const int32_t limit = arrays.neighborlimit - 15;
	const auto bin_size = (h_i * (1.f - lower_edge)) / 32.f;

	int32_t neighs = 0;

	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX)
			continue;
		h_j *= kernelSize();

		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = (h_i + h_j) / 2.f;
		if (h_ij > x_ij)
			neighs++;
		else
			continue;

		auto constrained = x_ij * 2.f - h_j;
		auto actual = constrained - lower_edge * h_i;
		int32_t bin = static_cast<int32_t>(floorf(actual / bin_size));
		bin = math::clamp(bin, 0, 31);
		histogram[IDX(bin)]++;
	}

	auto h_i_old = h_i;
	int32_t sum = 0;
	if (neighs >= limit)
		for (int32_t i = 0; i < 32; ++i) {
			sum += histogram[IDX(i)];
			if (sum >= limit) {
				h_i = h_i * lower_edge + ((float)i - 1) * bin_size;
				break;
			}
		}

	h_i /= kernelSize();
	if (h_i != h_i || h_i > h_i_old)
		h_i = h_i_old;
	math::unit_assign<4>(arrays.position[i], h_i);

}


cellFunctionType constrainSupportAndSymmetry(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
#ifdef __CUDA_ARCH__
	auto IDX = [](auto i) { return threadIdx.x * 4 + i % 4 + (i / 4) * blockDim.x * 4; };
	extern __shared__ int8_t histogram[];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
#else
	auto IDX = [](auto i) { return i; };
	int8_t histogram[32];
	for (int32_t i = 0; i < 32; ++i)
		histogram[IDX(i)] = 0;
#endif
	alias_arrays((res, particleIndex), (pos, position));

	auto x_i = pos[i];
	auto h_i = support_h(x_i);
	auto r_i = res[i];

	if (h_i == FLT_MAX)
		return;
	h_i *= kernelSize();

	const float lower_edge = 0.5f;
	const int32_t limit = arrays.neighborlimit - 2;
	const auto bin_size = (h_i * (1.f - lower_edge)) / 32.f;

	int32_t neighs = 0;

	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_h(x_j);
		if (h_j == FLT_MAX)
			continue;
		h_j *= kernelSize();

		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = (h_i + h_j) / 2.f;
		if (h_ij > x_ij)
			neighs++;
		else
			continue;

		auto constrained = x_ij * 2.f - h_j - lower_edge * h_i;
		auto bin = math::clamp(constrained / bin_size, 0.f, 31.f);
		int32_t bin_i = math::castTo<int32_t>(math::floorf(bin));
		histogram[IDX(bin_i)]++;

		auto r_j = res[j];

		if (r_j > r_i) {
#ifdef CORRECT_SYMMETRY
			auto f_j = powf(0.5f, (float)r_j);
			auto hatX_i = integerCoords(x_i, arrays, f_j);
			auto hatX_j = integerCoords(x_j, arrays, f_j);
			auto difference = hatX_j - hatX_i;
			difference = math::fn([](auto a) {return abs(a); }, difference);
			if (math::max_elem(difference) > 1)
#endif
			{
				cuda_atomic<int32_t> res_atomic(res + j);
				int32_t snapped_mlm, old = res_atomic.val();
				do {
					snapped_mlm = old;
					if (snapped_mlm <= r_i)
						break;
					old = res_atomic.CAS(snapped_mlm, r_i);
				} while (old != snapped_mlm);
			}
		}
	}

	auto h_i_old = h_i;
	int32_t sum = 0;
	if (neighs >= limit)
		for (int32_t i = 0; i < 32; ++i) {
			sum += histogram[IDX(i)];
			if (sum >= limit) {
				h_i = h_i * lower_edge + ((float)i - 1) * bin_size;
				break;
			}
		}

	h_i /= kernelSize();
	if (h_i != h_i || h_i > h_i_old)
		h_i = h_i_old;
	math::unit_assign<4>(arrays.position[i], h_i);
}


basicFunctionType initializeSupport(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
	auto h_c = support_h(arrays.position[i]);
	auto h_0 = support_from_volume(arrays.volume[i]);
	if (h_c == FLT_MAX)
		return;

	math::unit_assign<4>(arrays.position[i], 0.5f * (h_c + h_0));
}

basicFunctionType setSupport(SPH::enforceSymmetry::Memory arrays) {
	checkedParticleIdx(i);
	auto h = support_h(arrays.position[i]);
	auto h_max = math::min(support_from_volume(arrays.volume[i]), math::unit_get<1>(arrays.cell_size) / kernelSize());

	if (h.val == FLT_MAX)
		return;

	h = math::min(h, h_max);

	math::unit_assign<4>(arrays.position[i], h);
}

basicFunction(initialize, initializeSupport, "Initializing Support");
basicFunction(finalize, setSupport, "Finalizing Support");
basicFunction(Resolution, cudaMLMResolution, "Calcualting Resolution");


cellFunction(constrainAndSymmetry, constrainSupportAndSymmetry, "Constraining support and enforcing symmetry", caches<float4, float4>{});
cellFunction(symmetry, enforceSymmetry, "Enforce symmetry", caches<float4>{});
cellFunction(constrain, constrainSupport, "Constrain support", caches<float4, float4>{});
cellFunction(constrainBase, constrainSupportBase, "Constrain support", caches<float4, float4>{});
cellFunction(constrainCompact, constrainSupportCompact, "Constrain support", caches<float4, float4>{});

#ifdef __CUDA_ARCH__
#define IDX(id) threadIdx.x * 4 + id % 4  + (id / 4) * blockDim.x * 4
#else
#define IDX(id) id
#endif


struct is_valid {
	hostDeviceInline bool operator()(const int x) { return x != UINT_MAX; }
};

void SPH::enforceSymmetry::constrain_support(Memory mem) {

	if (*parameters::modules::support::ptr != "constrained") {
		//if (get<parameters::adaptive>() != true) return;
		if (parameters::cell_structure{} == cell_structuring::MLM
			|| parameters::cell_structure{} == cell_structuring::compactMLM) {
			cuda::memcpy(mem.particleIndex, mem.MLMResolution, sizeof(int32_t) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
			launch<symmetry>(mem.num_ptcls, mem);
			cuda::memcpy(mem.MLMResolution, mem.particleIndex, sizeof(int32_t) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
		}
		return;
	}
	launch<initialize>(mem.num_ptcls, mem);

	if (parameters::cell_structure{} == cell_structuring::MLM
		|| parameters::cell_structure{} == cell_structuring::compactMLM) {
		//launch<Resolution>(mem.num_ptcls, mem);
		cuda::memcpy(mem.particleIndex, mem.MLMResolution, sizeof(int32_t) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
		launch<constrainAndSymmetry>(mem.num_ptcls, mem);
		//launch<symmetry>(mem.num_ptcls, mem);
		cuda::memcpy(mem.MLMResolution, mem.particleIndex, sizeof(int32_t) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
		//cuda::arrayMemset<arrays::particleIndex>(0xFFFFFFFF);
		//launch<constrainBase>(mem.num_ptcls, mem);
		//launch<constrain>(mem.num_ptcls, mem);
	}
	else {
		//cuda::arrayMemset<arrays::particleIndex>(0xFFFFFFFF); 
		//cuda::arrayMemset<arrays::particleIndex>(0x00);
		launch<constrain>(mem.num_ptcls, mem);
		//*parameters::adjusted_particles::ptr = algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
		//launch <constrainCompact>(*parameters::adjusted_particles::ptr, mem, *parameters::adjusted_particles::ptr);
		//std::cout << "Adjusted particle: " << *parameters::adjusted_particles::ptr << std::endl;
	}
	launch<finalize>(mem.num_ptcls, mem);
}
 