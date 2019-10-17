#include <SPH/neighborhood/compactNeighborMask.cuh>
#include <utility/include_all.h>

basicFunctionType createCompactList(SPH::compactNeighborMask::Memory arrays) {
	checkedParticleIdx(i);
	auto x_i = arrays.position[i];
	auto h_i = x_i.w * kernelSize();
	auto factor = powf(0.5f, (float)arrays.MLMResolution[i]);
	auto resolution = arrays.MLMResolution[i];
	int32_t counter = 0;
	int3 idx = pToIdx3(x_i, arrays.min_coord, arrays.cell_size.x * factor);
	int32_t cellCounter = 0;
	for (int32_t xi = -1; xi <= 1; ++xi) {
		for (int32_t yi = -1; yi <= 1; ++yi) {
			for (int32_t zi = -1; zi <= 1; ++zi) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + xi), static_cast<uint32_t>(idx.y + yi), static_cast<uint32_t>(idx.z + zi) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
					continue;
				}
				auto morton = idx3D_to_morton(cell);
				auto s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries) +
					arrays.hash_entries * resolution];
				if (s.beginning == UINT31_MAX) continue;
				cell_span cs;
				bool found = false;
				if (s.compacted) {
					cs = cell_span{ (int32_t)s.beginning, s.length };
					if (position_to_morton(arrays.position[cs.beginning], arrays, factor) != morton)
						continue;
					found = true;
				}
				else {
					for (int32_t si = (int32_t)s.beginning; si < (int32_t) s.beginning + s.length; ++si) {
						cs = arrays.compactCellSpan[si + arrays.max_numptcls * resolution];
						if (position_to_morton(arrays.position[cs.beginning], arrays, factor) != morton)
							continue;
						found = true;
						break;
					}
				}
				if (!found) continue;
				compactCellNeighbors neighs;
				neighs.xi = xi + 2;
				neighs.yi = yi + 2;
				neighs.zi = zi + 2;
				neighs.mask = 0;
				for (int32_t j = (int32_t)cs.beginning; j < (int32_t)cs.beginning + cs.length; ++j) {
					auto x_j = arrays.position[j];
					auto h_j = x_j.w * kernelSize();
					auto x_ij = math::distance3(x_i, x_j);
					auto h_ij = 0.5f * (h_i + h_j);
					if (x_ij <= h_ij) {
						counter++;
						if (j - cs.beginning >= 25) {
							neighs.mask = MAX_VAL_26BIT;
							continue;
						}
						neighs.mask = neighs.mask | ( 1 << (j - cs.beginning));
					}
				}
				if (neighs.mask != 0x00) {
					arrays.neighborMask[i + arrays.max_numptcls * cellCounter] = neighs;
					cellCounter++;
				}
			}
		}
	}

	arrays.neighborListLength[i] = counter;
}

basicFunction(compactList, createCompactList, "Neighborhood: Cell Based");

template<typename Func, typename T>
hostDeviceInline auto iterateMasked(int32_t i, T& arrays, Func fn) {
	auto x_i = arrays.position[i];
	auto factor = powf(0.5f, (float)arrays.MLMResolution[i]);
	auto resolution = arrays.MLMResolution[i];
	int3 idx = pToIdx3(x_i, arrays.min_coord, arrays.cell_size.x * factor);
				for (int32_t ci = 0; ci < 27; ++ci) {
					auto c = arrays.neighborMask[i + ci * arrays.max_numptcls];
					if (c.mask == 0) break;
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + (int32_t)c.xi - 2), static_cast<uint32_t>(idx.y + (int32_t)c.yi - 2), static_cast<uint32_t>(idx.z + (int32_t)c.zi - 2) };
				auto morton = idx3D_to_morton(cell);
				auto s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries) + arrays.hash_entries * resolution];
				cell_span cs;
				//bool found = false;
				if (s.compacted) {
					cs = cell_span{ (int32_t)s.beginning, s.length };
					if (position_to_morton(arrays.position[cs.beginning], arrays, factor) != morton)
						continue;
					//found = true;
				}
				else {
					for (int32_t si = (int32_t)s.beginning; si < (int32_t)s.beginning + s.length; ++si) {
						cs = arrays.compactCellSpan[si + arrays.max_numptcls * resolution];
						if (position_to_morton(arrays.position[cs.beginning], arrays, factor) != morton)
							continue;
						//found = true;
						break;
					}
				}
				//if(found)
				if (c.mask == MAX_VAL_26BIT) {
				for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
					fn(j);
				}
				}
				else {
					while (c.mask != 0) {
						int32_t j = cs.beginning + ffs(c.mask) - 1;
						c.mask = c.mask ^ (1 << (ffs(c.mask) - 1));
						fn(j);
					}
				}
			}
	//	}
	//}
}


neighFunctionType testDensity(SPH::compactNeighborMask::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));
	auto unit_density = 0.f;
	iterateNeighbors(j) 
		unit_density += vol[j] * W_ij;

	float test_density = 0.f;
	iterateMasked(i, arrays, [&](int32_t j) {
		test_density += vol[j] * W_ij;
	});
	float test_density_it = 0.f;
	constexpr auto neighborhood = neighbor_list::masked;
	iterateNeighbors(j){
		test_density_it += vol[j] * W_ij;
	}
	if (arrays.neighborListLength[i] == 0 || unit_density == 0.f) {
		printf("%d @ %d -> %f : %f - [%f]\n", i, arrays.neighborListLength[i], test_density, test_density_it, test_density / test_density_it);
	}
}

neighFunction(testList, testDensity, "Neighborhood: Test Neighborhood", caches<float4, float>{});


void SPH::compactNeighborMask::calculate_neighborlist(Memory arrays){
	cudaMemset(arrays.neighborMask, 0x00, sizeof(compactCellNeighbors) * arrays.max_numptcls * 28);
	cuda::sync();
	launch<compactList>(arrays.num_ptcls, arrays);
	cuda::sync();
	launch<testList>(arrays.num_ptcls, arrays);
}