#include <SPH/render/auxilliaryMLM.cuh>
#include <utility/include_all.h>
#include <utility/include_all.h>
#include "dummy.h"  

namespace SPH {
	hostDeviceInline auto idx3D_to_hash_local(int3 idx, uint32_t hash_entries) {
		int32_t morton = idx3D_to_morton_32(idx);
		return morton % hash_entries;
	}
	template<typename R, typename S, typename T>
	hostDeviceInline auto position_to_hash_local(R p, S min, T d, uint32_t hash_entries) {
		return idx3D_to_hash_local(position_to_idx3D_i(p, min, d), hash_entries);
	}


	template<typename T, typename U>
	hostDeviceInline auto position_to_hash_local(T p, U& arrays, float factor = 1.f) {
		return idx3D_to_hash_local(position_to_idx3D_i(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor), arrays.hash_entries);
	}

	namespace auxilliaryMLM {
		basicFunctionType cudaHashTablea(SPH::auxilliaryMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == 0 || h != arrays.resortIndex[i - 1]) {
				arrays.auxHashMap[h].hash.beginning = i;
			}
		}

		basicFunctionType cudaHashTableb(SPH::auxilliaryMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == threads - 1 || arrays.resortIndex[i + 1] != arrays.resortIndex[i])
				arrays.auxHashMap[h].hash.length = i - arrays.auxHashMap[h].hash.beginning + 1;
		}

		cellFunctionType cuCompactCellTable(SPH::auxilliaryMLM::Memory arrays, int32_t threads, int32_t *compact, float ratio) {
			checkedThreadIdx(i);
			auto x_i = arrays.position[compact[i]];
			auto voxelPosition = position_to_idx3D_i(x_i, arrays.min_coord, arrays.cell_size.x);
			auto m = position_to_morton_32_i(x_i, arrays);
			auto s = arrays.compactHashMap[idx3D_to_hash(voxelPosition, arrays.hash_entries)];
			auto cs = compact_cellSpan{ 0,0 };
			if (s.compacted == 1) {
				cs.beginning = (int32_t)s.beginning;
				cs.length = s.length;
			}
			if (s.compacted == 0) {
				for (int32_t ii = (int32_t)s.beginning; ii < ((int32_t)s.beginning) + s.length; ++ii) {
					if (position_to_morton_32_i(arrays.position[arrays.compactCellSpan[ii].beginning], arrays) == m) {
						cs = arrays.compactCellSpan[ii];
					}
				}
			}
			float min_test = -FLT_MAX;
			float min_distance = -FLT_MAX;
			int32_t min_neighs = INT_MIN;
			for (int32_t jj = cs.beginning; jj < cs.beginning + cs.length; ++jj) {
				min_test = math::max(arrays.auxTest[jj], min_test);
				min_distance = math::max(arrays.auxDistance[jj], min_distance);
				min_neighs = math::max(min_neighs, arrays.neighborListLength[jj]);
			} 
			//arrays.auxCellSpan[i].cell = compactListEntry::cellEntry{ 1, min_neighs > arrays.internalLimit, position_to_morton_32(x_i, arrays, ratio) };
			uint32_t pred = LIST_COMPACT;
			//printf("%d : %d\n", i, min_neighs);
			//if (x_i.z < 0.f) pred = LIST_ALWAYS_FALSE;
			//if (min_distance < -5.f/* || min_neighs < arrays.vrtxNeighborLimit*/) pred = LIST_ALWAYS_FALSE;
			if (min_test < 0.5f) pred = LIST_ALWAYS_FALSE;
			if (arrays.pruneVoxel == 1) pred = LIST_COMPACT;
			arrays.auxCellSpan[i].cell = compactListEntry::cellEntry{ pred, position_to_hash(x_i, arrays, ratio) };
			arrays.resortIndex[i] = (pred ? 0 : arrays.hash_entries) + position_to_hash_local(x_i, arrays.min_coord, (arrays.cell_size.x) * ratio, arrays.hash_entries);
			arrays.particleparticleIndex[i] = i;
		}
		
		cellFunctionType cudaHashParticles(SPH::auxilliaryMLM::Memory arrays, float ratio, float ratio2) {
			checkedParticleIdx(i);
			auto x_i = arrays.position[i];
			auto h_i = (x_i.w);
			if (h_i != FLT_MAX) {
				arrays.ZOrder_64[i] = position_to_morton(x_i, arrays, ratio);
				arrays.ZOrder_32[i] = static_cast<int32_t>(position_to_morton_32(x_i, arrays, ratio));
				arrays.resortIndex[i] = position_to_hash_local(x_i, arrays, ratio2);
				arrays.particleparticleIndex[i] = i;
			}
			else {
				arrays.ZOrder_64[i] = INT64_MAX;
				arrays.ZOrder_32[i] = INT_MAX;
				arrays.resortIndex[i] = INT_MAX;
				arrays.particleparticleIndex[i] = i;
			}
		}

		cellFunctionType cudaIndexCells(SPH::auxilliaryMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
			checkedThreadIdx(i);
			auto x_i = arrays.position[i];
			auto voxelPosition = position_to_idx3D_i(x_i, arrays.min_coord, arrays.cell_size.x);
			auto m = idx3D_to_morton_32(voxelPosition);
			auto s = arrays.compactHashMap[idx3D_to_hash(voxelPosition, arrays.hash_entries)];
			auto cs = compact_cellSpan{ 0,0 };
			if (s.compacted == 1) {
				cs.beginning = (int32_t)s.beginning;
				cs.length = s.length;
			}
			if (s.compacted == 0) {
				for (int32_t ii = (int32_t)s.beginning; ii < ((int32_t)s.beginning) + s.length; ++ii) {
					if (position_to_morton_32_i(arrays.position[arrays.compactCellSpan[ii].beginning], arrays) == m) {
						cs = arrays.compactCellSpan[ii];
					}
				}
			}
			float min_distance = -FLT_MAX;
			//int32_t min_neighs = INT_MAX;
			for (int32_t jj = cs.beginning; jj < cs.beginning + cs.length; ++jj) {
				min_distance = math::max(arrays.auxDistance[jj], min_distance);
				//min_neighs = math::min(min_neighs, arrays.neighborListLength[jj]);
			}
			if (i == 0)
				cell_indices[0] = -1;
			cell_indices[i + 1] = -1;

			//if (min_distance < -7.5f)
			//	return; 
			//if (x_i.y < 0.f)
			//	return;

			if (i == 0)
				cell_indices[0] = 0;
			i++;
			if (hash_width == hash_length::bit_64)
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_64[i - 1] != arrays.ZOrder_64[i] ? i : -1;
			else
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
		}

		cellFunctionType cudaSpreadCells(SPH::auxilliaryMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
			checkedThreadIdx(i);

			if (i == 0)
				cell_indices[0] = 0;
			i++;
			if (hash_width == hash_length::bit_64)
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_64[i - 1] != arrays.ZOrder_64[i] ? i : -1;
			else
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
		}

		template <typename T> hostDeviceInline void cudaSortCompactmlm(SPH::auxilliaryMLM::Memory arrays, int32_t threads,
			T* input, T* output) {
			checkedThreadIdx(i);
			output[i] = input[arrays.particleparticleIndex[i]];
		}


		basicFunctionType compactHashMap(SPH::auxilliaryMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			auto hashEntry = arrays.auxHashMap[h];
			hashEntry.hash.kind = LIST_ITERATE;
			auto cell = arrays.auxCellSpan[i];
			if (hashEntry.hash.length == 1)
				arrays.auxHashMap[h] = cell;
			else
				arrays.auxHashMap[h] = hashEntry;
		}
		template<typename T>
		__device__ __host__ __inline__ auto square(T&& x) {
			return x * x;
		}
		template<typename T>
		__device__ __host__ __inline__ auto cube(T&& x) {
			return x * x * x;
		}
		__device__ __host__ __inline__ float k(float4 x_i, float4 x_j) {
			auto h = (x_i.w + x_j.w) * 0.5f * kernelSize();
			auto d = math::distance3(x_i, x_j);
			auto s = d / h;
			return math::max(0.f, cube(1.f - square(s)));
		}
		neighFunctionType estimateSurface(SPH::auxilliaryMLM::Memory arrays) {
			checkedParticleIdx(i);
			cache_arrays((pos, position), (vol, volume));
			float4 normal{ 0.f,0.f,0.f,0.f };
			auto numNeighs = 0;
			iterateNeighbors(j) {
				if (i == j) continue;
				auto distance = pos[i] - pos[j];
				normal += math::normalize3(distance);
				auto w_ij = W_ij;
				if (w_ij > 0.f)
					numNeighs += 1;
			}
			normal = math::normalize3(normal);
			bool state = false;
			iterateNeighbors(j) {
				if (i == j) continue;
				auto distance = arrays.position[j] - arrays.position[i];
				auto angle = acosf(math::dot3(normal, math::normalize3(distance)));
				state = state || angle <= CUDART_PI_F / 6;
			}
			auto phi = state ? -20.f : 0.f;
/*
			auto xBar_t = float4{ 0.f,0.f,0.f,0.f };
			auto rBar_t = 0.f;
			auto kernelSum = 0.f;
			auto numNeighs = 0;
			float isoDensity = 0.f;

			iterateCells(pos[i], j){
				auto w_ij = W_ij;
				if (w_ij > 0.f){
					isoDensity += kernel(pos[i], pos[j]);
					kernelSum += w_ij;
					xBar_t += pos[j] * w_ij;
					rBar_t += math::power<ratio<1, 3>>(vol[j] * PI4O3_1) * w_ij;
					numNeighs += 1;
				}
			}

			auto xBar = xBar_t / kernelSum;
			auto rBar = rBar_t / kernelSum;

			auto phi = math::length3(pos[i] - xBar) - rBar;
			auto r = math::power<ratio<1, 3>>(vol[i] * PI4O3_1);
			
			phi = math::min(phi, -0.85f * r);
*/
			if (numNeighs < 15)
				phi = 0.f;
			if (numNeighs > arrays.internalLimit)
				phi = -20.f;
			if (numNeighs > kernelNeighbors() * 1.2f)
				phi = -20.f;
			
			arrays.auxDistance[i] = phi;
			//arrays.auxIsoDensity[i] = isoDensity;
		}
		cellFunctionType levelSetTest(SPH::auxilliaryMLM::Memory arrays, float wmin, float wmax) {
			checkedParticleIdx(i);
			cache_arrays((pos, position), (vol, volume));
			float4 position = pos[i];
			float4 C1{ 0.f,0.f,0.f,0.f };
			float C2 = 0.f;
			iterateCells(pos[i], j) {
					float4 p = arrays.position[j];
					float w = arrays.auxIsoDensity[j];
					C1 += kernel(p, position) / w * p;
					C2 += kernel(p, position) / w;
			}
			auto Cp = C1 / C2;
			Cp.w = position.w;
			float w_c = 0.f;
			iterateCells(pos[i], j) {
				float4 p = arrays.position[j];
				float w = arrays.auxIsoDensity[j];
				w_c += kernel(p, Cp) / w;
			}
			float decay = square(1.f - square(w_c - wmax) / square(wmax - wmin));
			float levelSet = math::distance3(pos[i], Cp) - arrays.radius * decay;
			arrays.auxTest[i] = levelSet;
			//arrays.auxTest[i] = math::distance3(pos[i], Cp);
			//arrays.auxTest[i] = powf(w_c - wmax, 2.f);
			//arrays.auxTest[i] = w_c;
			//if (getThreadIdx_x() == 0) {
			//	printf("%d : [%f %f %f %f] -> [%f %f %f %f] / %f -> [%f %f %f %f] : %f {%f, %f} - %f => %f\n",
			//		i, position.x, position.y, position.z, position.w,
			//		C1.x, C1.y, C1.z, C1.w, C2,
			//		Cp.x, Cp.y, Cp.z, Cp.w,
			//		w_c, wmin, wmax, decay, levelSet);
			//}
		}

		__device__ __host__ __inline__  float squared(float v) { return v * v; }
		__device__ __host__ __inline__ bool doesCubeIntersectSphere(float3 C1, float3 C2, float3 S, float R)
		{
			float dist_squared = R * R;
			/* assume C1 and C2 are element-wise sorted, if not, do that now */
			if (S.x < C1.x) dist_squared -= squared(S.x - C1.x);
			else if (S.x > C2.x) dist_squared -= squared(S.x - C2.x);
			if (S.y < C1.y) dist_squared -= squared(S.y - C1.y);
			else if (S.y > C2.y) dist_squared -= squared(S.y - C2.y);
			if (S.z < C1.z) dist_squared -= squared(S.z - C1.z);
			else if (S.z > C2.z) dist_squared -= squared(S.z - C2.z);
			return dist_squared > 0;
		}
		__device__ __host__ uint32_t cLEasuint(compactListEntry cle) {
			return *reinterpret_cast<uint32_t*>(&cle);
		}
		basicFunctionDeviceType spreadCells(SPH::auxilliaryMLM::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto cs = arrays.compactCellSpan[i];
			if (cs.length == 0) return;
			float min_distance = -FLT_MAX;
			int32_t min_neighs = INT_MIN;
			float max_flag = -FLT_MAX;
			for (int32_t jj = cs.beginning; jj < cs.beginning + cs.length; ++jj) {
				max_flag = math::max(arrays.auxTest[jj], max_flag);
				min_distance = math::max(arrays.auxDistance[jj], min_distance);
				min_neighs = math::max(min_neighs, arrays.neighborListLength[jj]);
			}
			//arrays.auxCellSpan[i].cell = compactListEntry::cellEntry{ 1, min_neighs > arrays.internalLimit, position_to_morton_32(x_i, arrays, ratio) };
			//uint32_t pred = LIST_COMPACT;
			//printf("%d : %d\n", i, min_neighs);
			//if (x_i.z < 0.f) pred = LIST_ALWAYS_FALSE;
			//if (min_distance < -5.f /*|| min_neighs < arrays.vrtxNeighborLimit*/) return ;
			if (max_flag < 0.5f /*|| min_neighs < arrays.vrtxNeighborLimit*/) return;
			

			auto x_i = arrays.position[cs.beginning];
			auto voxel = position_to_idx3D_i(x_i, arrays.min_coord, arrays.cell_size.x);
			auto center = math::castTo<float3>(voxel) * arrays.cell_size + arrays.min_coord;
			for (int32_t i = -1; i <= 1; ++i) {
				for (int32_t j = -1; j <= 1; ++j) {
					for (int32_t k = -1; k <= 1; ++k) {
						int3 localVoxel = voxel + int3{ i,j,k };
						auto m = idx3D_to_morton_32(localVoxel);
						auto h1 = idx3D_to_hash_local(localVoxel, arrays.hash_entries);
						auto h2 = (uint32_t) idx3D_to_hash(localVoxel, arrays.hash_entries);
						auto s = arrays.compactHashMap[h2];
						bool found = false;
							if (s.compacted == 0 && s.beginning != UINT31_MAX) {
								for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
									auto cs = arrays.compactCellSpan[i];
									if (position_to_morton_32(arrays.position[cs.beginning], arrays) == m) {
										break;
									}
								}

							}
							else if (s.compacted == 1) {
								if (position_to_morton_32(arrays.position[s.beginning], arrays) == m) {
								}
						}
							if (found) continue;

						

						auto neigh_center = center + float3{ (float)i, (float)j, (float)k } *arrays.cell_size;
						auto min = math::castTo<float3>(localVoxel) * arrays.cell_size + arrays.min_coord;
						auto max = min + arrays.cell_size;

						for (int32_t ii = cs.beginning; ii < cs.beginning + cs.length; ++ii) {
							auto x_ii = arrays.position[ii];
							auto flag = arrays.auxTest[ii];
							if (flag < 0.5f) continue;
							if (doesCubeIntersectSphere(min, max, math::castTo<float3>(x_ii), support_from_volume(PI4O3 * arrays.radius * arrays.radius*arrays.radius) * kernelSize())) {
								uint32_t* address_as_ul = (uint32_t*)(arrays.auxHashMap + h1);
								uint32_t old = cLEasuint(arrays.auxHashMap[h1]), assumed;
								do{
									assumed = old;
									compactListEntry new_value = arrays.auxHashMap[h1];
									if (new_value.cell.kind == LIST_ITERATE) {
										new_value.cell = compactListEntry::cellEntry{ LIST_ALWAYS_TRUE, h2 };
									}
									else if (new_value.cell.kind == LIST_COMPACT) {
										if (new_value.cell.hash == h2) break;
										new_value.cell = compactListEntry::cellEntry{ LIST_ALWAYS_TRUE, h2 };
									}
									else
										new_value.cell = compactListEntry::cellEntry{ LIST_COMPACT, h2 };
									old = atomicCAS(address_as_ul, assumed, cLEasuint(new_value));
								} while (old != assumed);
							}
							break;
						}
					}
				}
			}

		}
		basicFunctionDevice(spread, spreadCells, "spread cells");

		cellFunction(hashParticles, cudaHashParticles, "hashing particles", caches<float4>{});
		cellFunction(buildCellTable, cuCompactCellTable, "creating cell table I");
		basicFunction(buildHashTable1, cudaHashTablea, "hashing cell table I");
		basicFunction(buildHashTable2, cudaHashTableb, "hashing cell table II");
		cellFunction(indexCells, cudaIndexCells, "indexing cells");
		basicFunction(sort, cudaSortCompactmlm, "compact resorting cells");
		basicFunction(compact, compactHashMap, "compact hashmap");
		neighFunction(surfaceEstimate, estimateSurface, "estimating surface", caches<float4, float>{});
		cellFunction(levelSetTesting, levelSetTest, "estimating surface", caches<float4, float>{});


		neighFunctionType predicateParticles(SPH::auxilliaryMLM::Memory arrays) {
			checkedParticleIdx(i);
			int32_t max_neighs = INT_MIN;
			iterateNeighbors(j) {
				max_neighs = math::max(max_neighs, arrays.neighborListLength[j]);
			}
			if(max_neighs >= arrays.vrtxNeighborLimit && arrays.auxDistance[i] > -5.f)
				arrays.auxTest[i] = 1.f;
			else if(max_neighs < arrays.vrtxNeighborLimit)
				arrays.auxTest[i] = 0.f;
			else
				arrays.auxTest[i] = -1.f;
		}
		neighFunction(predication, predicateParticles, "estimating surface");

		neighFunctionDeviceType spreadSurfaceInformation(SPH::auxilliaryMLM::Memory arrays, uint32_t* atomicCtr) {
			checkedParticleIdx(i);
			float auxTi = arrays.auxTest[i];
			if (auxTi > 1.f || auxTi < FLT_MIN)
				return;
			auto x_i = arrays.position[i];
			iterateNeighbors(j) {
				float auxTj = arrays.auxTest[j];
				if (auxTj == 0.f) {
					auto x_j = arrays.position[j];
					if (math::distance3(x_i, x_j) < support(x_i, x_j)) {
						arrays.auxTest[j] = 0.5f;
						atomicInc(atomicCtr, UINT32_MAX);
					}
				}
			}
			arrays.auxTest[i] = 2.f;
		}
		neighFunctionDevice(spreadSurface, spreadSurfaceInformation, "spreading surface");


		template <typename... Ts> auto callSort2(Memory mem, int32_t threads, Ts... tup) {
			launch<sort, Ts...>(threads, mem, threads, tup...);
		}
		template <typename... Ts> auto callSort3(Memory mem, std::tuple<Ts...> tup) {
			callSort2(mem, mem.num_ptcls, std::make_pair(Ts::ptr, Ts::rear_ptr)...);
		}

		struct count_if {
			hostDeviceInline bool operator()(const compactListEntry x) { return x.cell.kind == LIST_ITERATE && x.hash.beginning != MAX_VAL_26BIT; }
		};
		struct is_valid {
			hostDeviceInline bool operator()(const int x) { return x != -1; }
		};
		struct is_valid_cell {
			hostDeviceInline bool operator()(const compactListEntry x) { return x.hash.kind != LIST_ALWAYS_FALSE; }
		};
		struct invalid_position {
			hostDeviceInline bool operator()(float4 x) { return x.w == FLT_MAX; }
		};
		struct hash_spans {
			hostDeviceInline compactListEntry operator()() { return compactListEntry{ LIST_ALWAYS_FALSE, MAX_VAL_26BIT,0 }; }
		};

	} // namespace resort_mlm

} // namespace SPH


void SPH::auxilliaryMLM::generateAuxilliaryGrid(Memory mem) {
	auto thrust_ptr = [](auto ptr) { return thrust::device_pointer_cast(ptr); };
	static uint32_t* atomicCtr = nullptr;
	static uint32_t hostCtr;
	static bool once = true;
	if (once) {
		cudaMalloc(&atomicCtr, sizeof(uint32_t));
		once = false;
	}

	if (mem.num_ptcls > 0) {
		launch<surfaceEstimate>(mem.num_ptcls, mem);
		//launch<levelSetTesting>(mem.num_ptcls, mem, get<parameters::vrtxWMin>(), get<parameters::vrtxWMax>());
		launch< predication>(mem.num_ptcls, mem);
		//int32_t i = 0;
		do {
			cudaMemset(atomicCtr, 0x00, sizeof(uint32_t));
			launchDevice<spreadSurface>(mem.num_ptcls, mem, atomicCtr);
			cudaMemcpy(&hostCtr, atomicCtr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::cout << "Iteration " << ++i << " affected " << hostCtr << " cells." << std::endl;
		} while (hostCtr > 0);

		float factor = get<parameters::auxScale>();
		launch<hashParticles>(mem.num_ptcls, mem, factor, 1.f);
		get<parameters::auxCellCount>() = 0;

		cuda::arrayMemset<arrays::cellparticleIndex>(0xFFFFFFFF);
		cuda::arrayMemset<arrays::ZOrder_64>(0xFFFFFFFF);

		launch<hashParticles>(mem.num_ptcls, mem, factor, factor);
		algorithm::generate(mem.auxHashMap, mem.hash_entries, hash_spans());

		launch<indexCells>(mem.num_ptcls, mem, mem.num_ptcls, arrays::cellparticleIndex::ptr);
		cuda::sync();
		int32_t diff = static_cast<int32_t>(algorithm::copy_if(arrays::cellparticleIndex::ptr, arrays::compactparticleIndex::ptr, mem.num_ptcls + 1, is_valid()));
		cuda::sync();
		//std::cout << "before build cell table: " << diff << std::endl;
		launch<buildCellTable>(diff, mem, diff, arrays::compactparticleIndex::ptr, factor);
		diff--;
		algorithm::sort_by_key(diff, mem.resortIndex, mem.particleparticleIndex);
		cuda::sync();
		launch<sort>(diff, mem, diff,mem.auxCellSpan, (compactListEntry*)mem.compactCellSpanSwap);
		diff = static_cast<int32_t>(algorithm::copy_if((compactListEntry*)mem.compactCellSpanSwap, mem.auxCellSpan, diff, is_valid_cell()));
		//std::cout << "after build cell table: " << diff << std::endl;
		cuda::sync();
		get<parameters::auxCells>() = diff;
		get<parameters::auxCellCount>() = diff;
		cuda::sync();
		cuda::memcpy(mem.auxCellSpan, (compactListEntry*) mem.compactCellSpanSwap, sizeof(compactListEntry) * diff, cudaMemcpyDeviceToDevice);
		cuda::sync();
		launch<buildHashTable1>(diff, mem, diff);
		cuda::sync();
		launch<buildHashTable2>(diff, mem, diff);
		cuda::sync();
		launch<compact>(diff, mem, diff);
		auto l0_cells = get<parameters::occupiedCells>()[0];
		launchDevice<spread>(l0_cells, mem, l0_cells);
		int32_t collisionsg = (int32_t)thrust::count_if(thrust::device, thrust_ptr(mem.auxHashMap), thrust_ptr(mem.auxHashMap) + parameters::hash_entries{}, count_if());
		get<parameters::auxCollisions>() = collisionsg;

	}
}
