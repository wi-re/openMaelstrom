#include <SPH/neighborhood/compactCellList.cuh>
#include <utility/include_all.h>

cellFunctionType createCompactList(SPH::compactCellList::Memory arrays) {
	checkedParticleIdx(i);
	auto x_i = arrays.position[i];
	auto h_i = support_H(x_i);
	auto f_i = structure == cell_structuring::MLM ? powf(0.5f, (float)arrays.MLMResolution[i]) : 1.f;
	//f_i = 1.f;
	auto code_equal = [=](auto x_i, auto x_j) {
		if (structure == cell_structuring::MLM || (structure == cell_structuring::hashed && hash_width == hash_length::bit_64))
			return position_to_morton(x_i, arrays, f_i) == position_to_morton(x_j, arrays, f_i);
		else
			return position_to_morton_32(x_i, arrays, f_i) == position_to_morton_32(x_j, arrays, f_i);
	};
	uint8_t scale = 1;
	auto x_p = float4_u<SI::m>{ FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	uint32_t currentLength = 0;
	uint32_t currentBegin = COMPACT_IDX_MAX;
	uint32_t valid_idx = 0;
	uint32_t counter = 0;
	iterateCells(x_i, j) {
		auto x_j = arrays.position[j];
		auto h_j = support_H(x_j);
		auto x_ij = math::distance3(x_i, x_j);
		auto h_ij = 0.5f * (h_i + h_j);
		if (x_ij <= h_ij) {
			counter++;
			if (!code_equal(x_p, x_j)) {
				if (currentLength > COMPACT_LEN_MAX)
					scale = math::max(currentLength / COMPACT_LEN_MAX + 1, scale);

				if (currentBegin != COMPACT_IDX_MAX)
					arrays.compactCellList[i + (valid_idx++) * arrays.num_ptcls] = 
					compactSpan{ currentBegin, currentLength / scale };
				currentLength = 0;
				currentBegin = (uint32_t) j;
				x_p = x_j;
			}
			else
				currentLength = (uint32_t) j - currentBegin;
		}
	}
	if (currentLength > COMPACT_LEN_MAX)
		scale = math::max(currentLength / COMPACT_LEN_MAX + 1, scale);

	if (currentBegin != COMPACT_IDX_MAX)
		arrays.compactCellList[i + (valid_idx++) * arrays.num_ptcls] = compactSpan{ currentBegin, currentLength / scale };

	arrays.compactCellList[i + (valid_idx)*arrays.num_ptcls] = compactSpan{ COMPACT_IDX_MAX, 0 };

	arrays.compactCellScale[i] = scale;
	arrays.neighborListLength[i] = counter;
}

cellFunction(compactList, createCompactList, "Neighborhood: Cell Based");

void SPH::compactCellList::calculate_neighborlist(Memory arrays) { 
	launch<compactList>(arrays.num_ptcls, arrays);
}
