#include <SPH/kernelSupport/anisotropy.cuh>
#include <utility/include_all.h>
#include <cuda.h>
#include "math.h" // CUDA math library
#include <sstream>
#include <algorithm>
#include <utility/SVD.h> 
#define INTERPOLATOR_SCALE 2
namespace SPH {
	namespace anisotropy {
		struct interpolateMLM {
			int3 central_idx;
			bool full_loop;
			Memory &arrays;

			hostDevice interpolateMLM(float4 position, Memory &memory) : arrays(memory) {
				central_idx = position_to_idx3D_i(position, memory.min_coord, memory.cell_size.x);
			}

			struct cell_iterator {
				int3 idx;
				int32_t i = -INTERPOLATOR_SCALE, j = -INTERPOLATOR_SCALE, k = -INTERPOLATOR_SCALE;
				uint32_t ii = 0;
				int32_t jj = 0;

				int32_t neighbor_idx;

				Memory &arrays;
				compactHashSpan s{ 0, UINT31_MAX, -1 };
				compact_cellSpan cs{ -1, -1 };

				hostDevice int32_t cs_loop() {
					if (cs.beginning != -1 && jj < cs.beginning + cs.length) {
						neighbor_idx = jj;
						++jj;
						return neighbor_idx;
					}
					return -1;
				}

				hostDevice int32_t s_loop() {
					if (s.beginning != UINT31_MAX) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
							return -1;
						auto morton = idx3D_to_morton(cell);
						if (s.compacted && ii < s.beginning + s.length) {
							cs = cell_span{ (int32_t)s.beginning, s.length };
							jj = cs.beginning;
							ii = s.beginning + s.length;
							if (position_to_morton(arrays.position[cs.beginning], arrays, 1.f) == morton) {
								if (cs_loop() != -1) {
									return neighbor_idx;
								}
							}
						}
						for (; ii < s.beginning + s.length;) {
							cs = arrays.compactCellSpan[ii];
							++ii;
							jj = cs.beginning;
							if (position_to_morton(arrays.position[cs.beginning], arrays, 1.f) == morton) {
								if (cs_loop() != -1) {
									return neighbor_idx;
								}
							}
						}
						++k;
					}
					return -1;
				}

				hostDevice void increment() {
					if (cs_loop() != -1)
						return;
					if (s_loop() != -1)
						return;

					for (; i <= INTERPOLATOR_SCALE; ++i) {
						for (; j <= INTERPOLATOR_SCALE; ++j) {
							for (; k <= INTERPOLATOR_SCALE;) {
								uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
									static_cast<uint32_t>(idx.z + k) };
								if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
									++k;
									continue;
								}
								auto morton = idx3D_to_morton(cell);

								s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries)];
								ii = s.beginning;
								if (s.beginning == UINT31_MAX) {
									++k;
									continue;
								}
								if (s_loop() != -1)
									return;
							}
							k = -1;
						}
						j = -1;
					}
				}

				hostDevice cell_iterator(int3 c_idx, Memory &memory, int32_t _i = -1, int32_t _j = -1, int32_t _k = -1)
					: idx(c_idx), i(_i), j(_j), k(_k), arrays(memory) {
					increment();
				}

				hostDeviceInline int32_t operator*() { return neighbor_idx; };
				hostDeviceInline bool operator==(const cell_iterator &rawIterator) const {
					return (i == rawIterator.i);
				}
				hostDeviceInline bool operator!=(const cell_iterator &rawIterator) const {
					return (i != rawIterator.i);
				}

				hostDeviceInline cell_iterator &operator++() {
					increment();
					return (*this);
				}
				hostDeviceInline cell_iterator operator++(int) {
					auto temp(*this);
					increment();
					return temp;
				}
			};

			hostDeviceInline cell_iterator begin() const {
				return cell_iterator(central_idx, arrays);
			}
			hostDeviceInline cell_iterator end() const {
				return cell_iterator(central_idx, arrays, INTERPOLATOR_SCALE + 1, INTERPOLATOR_SCALE + 1, INTERPOLATOR_SCALE + 1);
			}
			hostDeviceInline cell_iterator cbegin() const {
				return cell_iterator(central_idx, arrays);
			}
			hostDeviceInline cell_iterator cend() const {
				return cell_iterator(central_idx, arrays, INTERPOLATOR_SCALE + 1, INTERPOLATOR_SCALE + 1, INTERPOLATOR_SCALE + 1);
			}
		};
		template<typename T>
		__device__ __host__ __inline__ auto square(T&& x) {
			return x * x;
		}
		template<typename T>
		__device__ __host__ __inline__ auto cube(T&& x) {
			return x * x * x;
		}
		__device__ __host__ __inline__ float k(float4 x_i, float4 x_j, float scale = 1.f) {
			auto h = (x_i.w + x_j.w) * 0.5f * kernelSize() * scale;
			auto d = math::distance3(x_i, x_j);
			auto s = d / h;
			return math::max(0.f, cube(1.f - square(s)));
		}

		neighFunctionType calculateCenterPosition(Memory arrays) {
			checkedParticleIdx(i);
			cache_arrays((pos, position), (vol, volume));
			float4 positionSum{ 0.f,0.f,0.f,0.f };
			float kernelSum = 0.f;
			for(int32_t j : interpolateMLM(pos[i], arrays)) {
			//iterateNeighbors(j){
				float4 x_j = pos[j];
				float w_ij = k(pos[i], x_j, INTERPOLATOR_SCALE);
				positionSum += pos[j] * w_ij;
				kernelSum += w_ij;
			}
			positionSum /= kernelSum;
			positionSum.w = kernelSum;
			arrays.centerPosition[i] = positionSum;
		}


#define G(i, x, y) (arrays.anisotropicMatrices[i * 9 + (x * 3 + y)])
		neighFunctionType calculateCovarianceMatrix(Memory arrays) {
			checkedParticleIdx(i);
			cache_arrays((pos, position), (vol, volume));
			float4 x_iW = arrays.centerPosition[i];
			float c_00 = 0.f, c_01 = 0.f, c_02 = 0.f;
			float c_10 = 0.f, c_11 = 0.f, c_12 = 0.f;
			float c_20 = 0.f, c_21 = 0.f, c_22 = 0.f;
			float ctr = 0.f;
			float kernelSum = 0.f;
			for(int32_t j : interpolateMLM(pos[i], arrays)) {
			//iterateNeighbors(j) {
				float4 x_j = pos[j];
				float w_ij = k(pos[i], x_j, INTERPOLATOR_SCALE);
				kernelSum += w_ij;
				if (w_ij > 0.f) {
					c_00 += w_ij * (x_j.x - x_iW.x) * (x_j.x - x_iW.x);
					c_01 += w_ij * (x_j.x - x_iW.x) * (x_j.y - x_iW.y);
					c_02 += w_ij * (x_j.x - x_iW.x) * (x_j.z - x_iW.z);
					
					c_10 += w_ij * (x_j.y - x_iW.y) * (x_j.x - x_iW.x);
					c_11 += w_ij * (x_j.y - x_iW.y) * (x_j.y - x_iW.y);
					c_12 += w_ij * (x_j.y - x_iW.y) * (x_j.z - x_iW.z);

					c_20 += w_ij * (x_j.z - x_iW.z) * (x_j.x - x_iW.x);
					c_21 += w_ij * (x_j.z - x_iW.z) * (x_j.y - x_iW.y);
					c_22 += w_ij * (x_j.z - x_iW.z) * (x_j.z - x_iW.z);
					ctr += 1.f;
				}
			}
			auto ks = 1.f / kernelSum;
			if (ks == ks && ks < HUGE_VALF && ks > -HUGE_VALF) {
				G(i, 0, 0) = ks * c_00;
				G(i, 0, 1) = ks * c_01;
				G(i, 0, 2) = ks * c_02;

				G(i, 1, 0) = ks * c_10;
				G(i, 1, 1) = ks * c_11;
				G(i, 1, 2) = ks * c_12;

				G(i, 2, 0) = ks * c_20;
				G(i, 2, 1) = ks * c_21;
				G(i, 2, 2) = ks * c_22;
			}else{
				float e = sqrt(1.f / 3.f);
				G(i, 0, 0) = e;
				G(i, 0, 1) = 0.f;
				G(i, 0, 2) = 0.f;

				G(i, 1, 0) = 0.f;
				G(i, 1, 1) = e;
				G(i, 1, 2) = 0.f;

				G(i, 2, 0) = 0.f;
				G(i, 2, 1) = 0.f;
				G(i, 2, 2) = e;
			}
			arrays.auxDistance[i] = ctr;
		}

		basicFunctionType calculateAnisotropicMatrix(Memory arrays) {
			checkedParticleIdx(i);
			auto A = SVD::Mat3x3::fromPtr(arrays.anisotropicMatrices, i, arrays.max_numptcls);
			auto usv = SVD::svd(A);
			auto& U = usv.U; auto& S = usv.S; auto& V = usv.V;
			auto r = powf(arrays.volume[i] * PI4O3_1, 1.f / 3.f);
			//auto ratio = (PI4O3 * arrays.radius * arrays.radius * arrays.radius) / arrays.volume[i];
			auto ratio = (arrays.radius) / r;
			auto ks = arrays.anisotropicKs * ratio * ratio;
			int32_t numNeighs = (int32_t) arrays.auxDistance[i];
			if (numNeighs > arrays.anisotropicNepsilon) {
				S.m_11 = ks * math::max(S.m_11, S.m_00 / arrays.anisotropicKr);
				S.m_22 = ks * math::max(S.m_22, S.m_00 / arrays.anisotropicKr);
				S.m_00 = ks * S.m_00;
				S.m_00 = 1.f / (S.m_00);
				S.m_11 = 1.f / (S.m_11);
				S.m_22 = 1.f / (S.m_22);
			}
			else {
				S.m_00 = 1.f / (arrays.anisotropicKn);
				S.m_11 = 1.f / (arrays.anisotropicKn);
				S.m_22 = 1.f / (arrays.anisotropicKn);
			}
			auto G = V * S * U.transpose();
			//G = A;

			float h_i = arrays.position[i].w * kernelSize();
			G *= 1.f / h_i;
			G.toPtr(arrays.anisotropicMatrices, i, arrays.max_numptcls);
			constexpr float sigma = 16.f / CUDART_PI_F;			

			arrays.centerPosition[i].w = arrays.volume[i] / arrays.density[i] * sigma * G.norm2();

			//if(arrays.auxTest[i] == 0.f)
			//	arrays.centerPosition[i].w = 0.f;
			//arrays.centerPosition[i].w = arrays.auxTest[i];
		}
		basicFunctionType calcualteSmoothPositions(Memory arrays) {
			checkedParticleIdx(i);
			float4 position = arrays.position[i];
			float4 centerPosition = arrays.centerPosition[i];
			float l = arrays.anisotropicLambda;
			float4 smoothedPosition = (1.f - l) * position + l * centerPosition;
			smoothedPosition.w = centerPosition.w;
			arrays.centerPosition[i] = smoothedPosition;
		}
		neighFunction(centerPositions, calculateCenterPosition, "calculating center positions", caches<float4, float>{});
		neighFunction(covarianceMatrices, calculateCovarianceMatrix, "calculating covariance matrices", caches<float4, float>{});
		basicFunction(anisotropicMatrices, calculateAnisotropicMatrix, "calculating anisotropy matrices");
		basicFunction(smoothPositions, calcualteSmoothPositions, "calculating smooth center positions");
		
		void generateAnisotropicMatrices(Memory mem) {
			cudaDeviceSynchronize();
			launch<centerPositions>(mem.num_ptcls, mem);
			launch<covarianceMatrices>(mem.num_ptcls, mem);
			launch<anisotropicMatrices>(mem.num_ptcls, mem);
			launch<smoothPositions>(mem.num_ptcls, mem);
		}
	}
}