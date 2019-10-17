/** This file contains the implementation of the embedded version of our multi level memory model. The purpose of this
 * code is to start with an input of simulation data, e.g. from a prior timestep and to re-sort this data first into
 * cells and then to construct the according hash maps for this data. Significant amounts of this code are external,
 * e.g. the sorting method itself as this was not part of our contribution.*/
#include <SPH/compactMLMResort/compactMLM.cuh>
#include <utility/include_all.h>
#include "dummy.h"  
     
// The SPH namespace contains all of our simulation methods to avoid global namespace collisions
namespace SPH {        
	// compactMLM refers to the embedded version of our MLM data structure where the cell table is embedded in the has table
	namespace compactMLM {
		// This function is the first step of the hash table construction
		// It calculates the first cell in every hash map entry
		basicFunctionType cudaHashTablea(SPH::compactMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == 0 || h != arrays.resortIndex[i - 1]) {
				arrays.compactHashMap[h].beginning = i;
			}
		}
		// This function is the second step of the hash map construction
		// It calculates the length of every hash map entry, where a length > 1 means a collision occured
		basicFunctionType cudaHashTableb(SPH::compactMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == threads - 1 || arrays.resortIndex[i + 1] != arrays.resortIndex[i])
				arrays.compactHashMap[h].length = i - arrays.compactHashMap[h].beginning + 1;
		}
		// This function is the first step of the cell table construction
		// It calculates the first entry and the cell length simultaneously
		basicFunctionType cudaCellTablea(SPH::compactMLM::Memory arrays, int32_t threads, int32_t *compact, float) {
			checkedThreadIdx(i);
			arrays.compactCellSpan[i] = cell_span{ compact[i], compact[i + 1] - compact[i] };
		}
		// This function is the second step of the cell table construction
		// It prepares the re-sort index by storing H_c in the index arrays by using the first particle in the cell
		basicFunctionType cudaCellTableb(SPH::compactMLM::Memory arrays, int32_t threads, int32_t *compact, float ratio) {
			checkedThreadIdx(i);
			auto x_i = arrays.position[compact[i]];
			arrays.resortIndex[i] = position_to_hash(x_i, arrays.min_coord, (arrays.cell_size.x) * ratio, arrays.hash_entries);
			arrays.particleparticleIndex[i] = i;
		}
		// This function determines the ideal resolution r for each particle according to the submission
		basicFunctionType cudaMLMResolution(SPH::compactMLM::Memory arrays, float* volume) {
			checkedParticleIdx(idx);
			float target_neighbors = Kernel<kernel_kind::spline4>::neighbor_number * 0.95f;
			float kernel_epsilon =
				(1.f) * powf((target_neighbors) / ((4.f / 3.f) * CUDART_PI_F), 1.f / 3.f) / Kernel<kernel_kind::spline4>::kernel_size();
			auto particle_volume = volume[idx];
			auto actual_support = kernel_epsilon * powf(particle_volume, 1.f / 3.f);

			float h_i = actual_support * Kernel<kernel_kind::spline4>::kernel_size();
			int32_t r_i = (int32_t)math::clamp(math::floorf(math::abs(log2f(arrays.cell_size.x / h_i))) - 0, 0, arrays.mlm_schemes - 1);
			float f_i = powf(0.5f, ((float)r_i));
			arrays.MLMResolution[idx] = r_i;
		}
		// This function is used to set the re-sorting indices for particles by calculating both the Morton Code for 32 and 64 bit
		// Keys as well as the hash index. Invalid particles (marked with a .w entry of FLT_MAX or a volume of 0 or with NaN
		// quantities are set to a re-sorting index that puts them at the end for removal).
		cellFunctionType cudaHashParticles(SPH::compactMLM::Memory arrays, float ratio, float ratio2, float* volumes, float4* velocity) {
			checkedParticleIdx(i);
			auto x_i = arrays.position[i];
			auto x = x_i;
			auto h_i = (x_i.w);
			if (!(x.w == FLT_MAX || x.x != x.x || x.y != x.y || x.z != x.z || x.w != x.w) && 
				!(volumes[i] != volumes[i] || volumes[i] == 0.f || volumes[i] > 1e21f)&&
				!(velocity[i].x != velocity[i].x || velocity[i].y != velocity[i].y || velocity[i].z != velocity[i].z)){
				arrays.ZOrder_64[i] = position_to_morton(x_i, arrays, ratio);
				arrays.ZOrder_32[i] = static_cast<int32_t>(position_to_morton_32(x_i, arrays, ratio));
				arrays.resortIndex[i] = position_to_hash(x_i, arrays, ratio2);
				arrays.particleparticleIndex[i] = i;
			}
			else {
				arrays.ZOrder_64[i] = INT64_MAX;
				arrays.ZOrder_32[i] = INT_MAX;
				arrays.resortIndex[i] = INT_MAX;
				arrays.particleparticleIndex[i] = i;
			}
		}
		// This function is used as the pre processing step of sorting the cells by calculating Z_i for every particle
		// if the particle is in a different cell than the prior cell and inserts an element marking the last cell.
		// This is done as described in the submission.
		cellFunctionType cudaIndexCells(SPH::compactMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
			checkedThreadIdx(i);
			if (i == 0)
				cell_indices[0] = 0;
			i++;
			if (hash_width == hash_length::bit_64)
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_64[i - 1] != arrays.ZOrder_64[i] ? i : -1;
			else
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
		}
		// This function is used to re-ort an input array into an output array based on the sorted indices in parallel
		template <typename T> hostDeviceInline void cudaSortCompactmlm(SPH::compactMLM::Memory arrays, int32_t threads,
			T* input, T* output) {
			checkedThreadIdx(i);
			output[i] = input[arrays.particleparticleIndex[i]];
		}
		// This function embeds the cell information into the hash map if there is no collision for some performance gains
		basicFunctionType compactHashMap(SPH::compactMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			auto hashEntry = arrays.compactHashMap[h];
			auto cell = arrays.compactCellSpan[i];
			if (hashEntry.length == 1) 
				arrays.compactHashMap[h] = compactHashSpan{ 1, (uint32_t) cell.beginning, cell.length };
			else
				arrays.compactHashMap[h].compacted = 0;
		}
		// A set of wrapper functions that allow us to call the functions on either the GPU or CPU.
		cellFunction(hashParticles, cudaHashParticles, "hashing particles", caches<float4>{});
		basicFunction(calculateResolution, cudaMLMResolution, "generating cell table");
		basicFunction(buildCellTable1, cudaCellTablea, "creating cell table I");
		basicFunction(buildCellTable2, cudaCellTableb, "creating cell table II");
		basicFunction(buildHashTable1, cudaHashTablea, "hashing cell table I");
		basicFunction(buildHashTable2, cudaHashTableb, "hashing cell table II");
		cellFunction(indexCells, cudaIndexCells, "indexing cells");
		basicFunction(sort, cudaSortCompactmlm, "compact resorting cells");
		basicFunction(compact, compactHashMap, "compact hashmap");
		// A set of predicate functions used for array compaction and counting
		struct is_valid {
			hostDeviceInline bool operator()(const int x) { return x != -1; }
		};
		struct count_if {
			hostDeviceInline bool operator()(const compactHashSpan x) { return x.beginning != UINT31_MAX && x.length > 1; }
		};
		struct invalid_position {
			hostDeviceInline bool operator()(float4 x) { return x.w == FLT_MAX || x.x != x.x || x.y != x.y || x.z != x.z || x.w != x.w; }
		};
		struct invalid_volume {
			hostDeviceInline bool operator()(float x) { return x == FLT_MAX || x == 0.f || x !=x; }
		};
		struct hash_spans {
			hostDeviceInline compactHashSpan operator()() { return compactHashSpan{0, UINT31_MAX,0 }; }
		};
	} // namespace resort_mlm
} // namespace SPH

// This is the main function of the module and is called externally with a Memory structure pre filled
// The paper uses a reduction to determine the largest support radius, however for our simulation framework we instead
// rely on a user provided upper bound of the cell size stored in mem.cell_size. This could be replaced for a different
// implementation, but our adaptive method also has a lower bound of resolution that is fixed so we based the cell size
// on that resolution instead of recalcuating it every step.
void SPH::compactMLM::resortParticles(Memory mem) {
	if (mem.num_ptcls > 0) {
		// We initially start by calculating D_min and D_max as described in the submission by using reduction Operations
		// Instead of directly using this we instead add a buffer on the outside of the bounding box of 3 cell widths
		// as this can improve re-sorting performance slightly.
		auto min_coord = math::to<float3>(algorithm::reduce_min(arrays::position::ptr, mem.num_ptcls));
		min_coord -= 3.f * mem.cell_size;
		get<parameters::min_coord>() = min_coord;
		auto max_coord = math::to<float3>(algorithm::reduce_max(arrays::position::ptr, mem.num_ptcls));
		max_coord += 3.f * mem.cell_size;
		get<parameters::max_coord>() = max_coord;
		// Next we determine the longest dimension of the simulation domain
		float max_length = math::max_elem(max_coord - get<parameters::min_coord>());
		cuda::sync();
		// This step determines D
		get<parameters::grid_size>() = math::to<int3>((max_coord - min_coord) / get<parameters::cell_size>().x);
		// In order to calculate P2 we divide the max length by the level 0 cell size
		int32_t cells = static_cast<int32_t>(max_length / get<parameters::cell_size>().x);
		int32_t v = cells;
		// This set of operations determines the result of rounding v to the next largest power of two
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		// Update entries in the Memory struct
		mem.min_coord = get<parameters::min_coord>();
		mem.grid_size = get<parameters::grid_size>();

		// hashParticles determines the Z order indices of all particles
		// sort_by_key could be based on thrusts sort index by key method.
		if (parameters::hash_size{} == hash_length::bit_32) {
			// This is the factor described in the paper to determine C_fine 
			float factor_morton = 1.f / ((float)(1024 / v));
			get<parameters::zOrderScale>() = factor_morton;
			cuda::sync();
			launch<hashParticles>(mem.num_ptcls, mem, factor_morton, 1.f, arrays::volume::ptr, arrays::velocity::ptr);
			cuda::sync();
			algorithm::sort_by_key(mem.num_ptcls, mem.ZOrder_32, mem.particleparticleIndex);
		}
		else {
			// This is the factor described in the paper to determine C_fine 
			float factor_morton = 1.f / ((float)(1048576 / v));
			get<parameters::zOrderScale>() = factor_morton;
			cuda::sync();
			launch<hashParticles>(mem.num_ptcls, mem, factor_morton, 1.f, arrays::volume::ptr, arrays::velocity::ptr);
			cuda::sync();
			algorithm::sort_by_key(mem.num_ptcls, mem.ZOrder_64, mem.particleparticleIndex);
		}

		/// This block of code is done to call the actual sorting functions by using a temporary array for resizing 4B
		/// and 16B values. This is highly implementation based and as such not important to our overall contribution
		/// This marks the start of the actual re-sorting block for particles
		{
			void* original = arrays::resortArray::ptr;
			void* original4 = arrays::resortArray4::ptr;
			for_each(sorting_list, [&mem](auto x) {
				using P = std::decay_t<decltype(x)>;
				if (!P::valid()) return;
				using T = typename P::type;
				if (sizeof(T) == sizeof(float)) {
					launch<sort>(mem.num_ptcls, mem, mem.num_ptcls, (float*)P::ptr, arrays::resortArray::ptr);
					cuda::sync();
					void* tmp = P::ptr;
					P::ptr = (T*)arrays::resortArray::ptr;
					arrays::resortArray::ptr = (float*)tmp;
				}
				else if (sizeof(T) == sizeof(float4)) {
					launch<sort>(mem.num_ptcls, mem, mem.num_ptcls, (float4*)P::ptr, arrays::resortArray4::ptr);
					cuda::sync();
					void* tmp = P::ptr;
					P::ptr = (T*)arrays::resortArray4::ptr;
					arrays::resortArray4::ptr = (float4*)tmp;
				}
				else
					LOG_ERROR << "Cannot sort array of data size " << sizeof(T) << std::endl;
			});
			for_each(sorting_list, [&mem, original, original4](auto x) {
				using P = std::decay_t<decltype(x)>;
				if (!P::valid()) return;
				using T = typename P::type;
				if (sizeof(T) == sizeof(float)) {
					if (P::ptr == original) {
						cuda::memcpy(arrays::resortArray::ptr, P::ptr, sizeof(T) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
						cuda::sync();
						P::ptr = (T*)arrays::resortArray::ptr;
						arrays::resortArray::ptr = (float*)original;
					}
				}
				else if (sizeof(T) == sizeof(float4)) {
					if (P::ptr == original4) {
						cuda::memcpy(arrays::resortArray4::ptr, P::ptr, sizeof(T) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
						cuda::sync();
						P::ptr = (T*)arrays::resortArray4::ptr;
						arrays::resortArray4::ptr = (float4*)original4;
					}
				}
			});
		}
		/// This marks the end of the actual re-sorting block for particles

		// update the position pointer to the resorted pointer
		mem.position = arrays::position::ptr;

		// This step uses a predicate to remove potentially invalid particles from the simulation.
		// This is done for particles with invalid positions, velocities or volumes in three steps.
		// The check for invalid velocities and volumes is mainly for stability when developing new methods
		// and could be removed for a normal implementation to avoid the slight performance cost
		cuda::sync();
		auto iter = algorithm::count_if(arrays::position::ptr, mem.num_ptcls, invalid_position());
		cuda::sync();
		cuda::sync();
		if (iter != 0) {
			auto diff = get<parameters::num_ptcls>() - iter;
			get<parameters::num_ptcls>() = static_cast<int32_t>(diff);
			mem.num_ptcls = static_cast<int32_t>(diff);
		}
		cuda::sync();
		iter = algorithm::count_if(arrays::velocity::ptr, mem.num_ptcls, invalid_position());
		cuda::sync();
		if (iter != 0) {
			auto diff = get<parameters::num_ptcls>() - iter;
			get<parameters::num_ptcls>() = static_cast<int32_t>(diff);
			mem.num_ptcls = static_cast<int32_t>(diff);
		}
		cuda::sync();
		iter = algorithm::count_if(arrays::volume::ptr, mem.num_ptcls, invalid_volume());
		cuda::sync();
		if (iter != 0) {
			auto diff = get<parameters::num_ptcls>() - iter;
			get<parameters::num_ptcls>() = static_cast<int32_t>(diff);
			mem.num_ptcls = static_cast<int32_t>(diff);
		}
		cuda::sync();
		// After particle removal we determine L_i for all particles
		launch<calculateResolution>(mem.num_ptcls, mem, arrays::volume::ptr);
		cuda::sync();
		cuda::sync();
		// Initialize the set of values keeping track of occupied valid and collided hash and cell entries
		get<parameters::valid_cells>() = 0;
		get<parameters::collision_cells>() = 0;
		get<parameters::occupiedCells>().clear();
		// For every level
		for (int32_t i = 0; i < (int32_t)parameters::mlm_schemes{}; ++i) {
			// Determine the resolution factor 0.5^L iteratively
			float factor = 1.f;
			for (int ii = 0; ii < i; ++ii)
				factor *= 0.5f;
			// As described in the submission, we store all levels in a single continuous array
			compactHashSpan *hashMap = arrays::compactHashMap::ptr + parameters::hash_entries{} *i;
			compact_cellSpan *cellSpan = arrays::compactCellSpan::ptr + parameters::max_numptcls{} *i;
			mem.compactCellSpan = cellSpan;
			mem.compactHashMap = hashMap;
			// Start by resetting all indices
			cuda::arrayMemset<arrays::cellparticleIndex>(0xFFFFFFFF);
			cuda::arrayMemset<arrays::ZOrder_64>(0xFFFFFFFF);
			cuda::sync();
			// Hash particles also calculates the hash index of all particles not just the Z order index.
			// This is done again here as the hash index varies per resolution level
			launch<hashParticles>(mem.num_ptcls, mem, factor, factor,arrays::volume::ptr, arrays::velocity::ptr);
			cuda::sync();
			// Reinitialize the hash map with all invalid entries
			algorithm::generate(hashMap, mem.hash_entries, hash_spans());
			cuda::sync();
			// Index the particles into their cells
			launch<indexCells>(mem.num_ptcls, mem, mem.num_ptcls, arrays::cellparticleIndex::ptr);
			cuda::sync();
			// And compact the array
			int32_t diff = static_cast<int32_t>(algorithm::copy_if(arrays::cellparticleIndex::ptr, arrays::compactparticleIndex::ptr, mem.num_ptcls + 1, is_valid()));
			cuda::sync();
			// Determine C_length based on C_begin^compact
			launch<buildCellTable1>(diff, mem, diff, arrays::compactparticleIndex::ptr, factor);
			cuda::sync();
			// Calculate the hash indices of every cell
			launch<buildCellTable2>(diff, mem, diff, arrays::compactparticleIndex::ptr, factor);
			cuda::sync();
			// The compact operation before gave is the number of occupied cells + 1 thus we remove 1 entry before proceeding
			diff--;
			get<parameters::occupiedCells>().push_back(diff);
			// Resort an index array based on the hashed indices
			algorithm::sort_by_key(diff, mem.resortIndex, mem.particleparticleIndex);
			cuda::sync();
			// Resort the actual cells into a temporary array and copy the results back into the cell list
			launch<sort>(diff, mem, diff, cellSpan, mem.compactCellSpanSwap);
			cuda::sync(); 
			cuda::memcpy(cellSpan, mem.compactCellSpanSwap, sizeof(cell_span) * diff, cudaMemcpyDeviceToDevice);
			cuda::sync();
			// Calculate H_begin for every occupied hash cell
			launch<buildHashTable1>(diff, mem, diff);
			cuda::sync();
			// Calculate H_length for every occupid hash cell
			launch<buildHashTable2>(diff, mem, diff);
			cuda::sync();
			// Embedd C into H if there is no collision
			launch<compact>(diff, mem, diff);
			cuda::sync();
			//int32_t collisionsg = algorithm::count_if((mem.hashMap), parameters::hash_entries{}, count_if());
			//get<parameters::collision_cells>() += collisionsg;
			//if(i == 0)
			//	get<parameters::valid_cells>() += diff;
		}
	}
} 
