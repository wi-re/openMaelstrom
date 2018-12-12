#include <SPH/resort/cluster.cuh>
#include <utility/include_all.h>

void SPH::cluster::clusterParticles(Memory) { 
	cudaDeviceSynchronize();  
	// for (int32_t i = 0; i < arrays.num_ptcls; ++i) {
    //       arrays.classification[i] = -1;
	// }
	// std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    // std::uniform_int_distribution<> dis(0, arrays.num_ptcls);

	// constexpr hash_length hash_width = hash_length::bit_64;
	// constexpr cell_ordering order = cell_ordering::z_order;
	// constexpr cell_structuring structure = cell_structuring::MLM;

	// int32_t n = 1024;
	// for(int32_t i = 0; i < n; ++i){
	// 	int32_t idx = dis(gen);
	// 	iterateCells(arrays.position[idx],j){
	// 		arrays.classification[j] = i;
	// 	}
	// }

	cudaDeviceSynchronize();
}