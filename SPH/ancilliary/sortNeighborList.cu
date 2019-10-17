#include <SPH/ancilliary/sortNeighborList.cuh>
#include <utility/include_all.h>

neighFunctionType sortNeighbors(SPH::sortedNeighborList::Memory arrays) {
	checkedParticleIdx(i);
	int32_t n = arrays.neighborListLength[i];
	int32_t p = arrays.max_numptcls;
	int32_t* arr = arrays.neighborList;
	float4 x_i = arrays.position[i];

	auto d = [&](int32_t j) {
		return math::distance3(x_i, arrays.position[j]);
	};

	for (int32_t k = 1; k < n; k++) {
		int32_t key = arr[i + k * p];
		float dist = d(key);
		int32_t ptr = k - 1;
		while (ptr >= 0 && d(arr[i + ptr * p]) > dist) {
			arr[i + (ptr + 1) * p] = arr[i + ptr * p];
			ptr = ptr - 1;
		}
		arr[i + (ptr + 1) * p] = key;
	}
}

neighFunction(sortNeighs, sortNeighbors, "Adaptive: indexing blending particles");

void SPH::sortedNeighborList::sort(Memory arrays){
	//std::cout << "Sorting neighbors!" << std::endl;
	launch< sortNeighs>(arrays.num_ptcls, arrays);
}