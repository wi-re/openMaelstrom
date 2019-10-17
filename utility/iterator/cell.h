#pragma once
#include <utility/iterator.h>
#include <utility/SPH/indication.h>
#include <utility/math.h>

#include <utility/identifier/resource_helper.h>
#include <utility/SPH.h>

template<typename T>
hostDeviceInline auto pairHelper(T* ptr) {
	return ptr;
}
template<typename T>
hostDeviceInline auto pairHelper(std::pair<T, T> ptr) {
	return ptr.first;
}

template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename T>
struct cell_interpolator;

template <typename Mem>
struct cell_interpolator<hash_length::bit_64, cell_ordering::z_order, cell_structuring::hashed,
	Mem> {
	int3 central_idx;
	bool full_loop;
	Mem &arrays;

	template <typename T>
	hostDevice cell_interpolator(T &&position, Mem &memory, bool all = false)
		: central_idx(
			position_to_idx3D_i(position, memory.min_coord, math::unit_get<1>(memory.cell_size))),
		full_loop(all), arrays(memory) {}

	struct cell_iterator {
		int3 idx;
		bool full_loop;
		int32_t i = -1, j = -1, k = -1;
		int32_t ii = 0, jj = 0;

		int32_t neighbor_idx;

		Mem &arrays;
		hash_span s{ -1, -1 };
		cell_span cs{ -1, -1 };

		hostDevice int32_t cs_loop() {
			if (cs.beginning != -1 && jj < cs.beginning + cs.length) {
				neighbor_idx = jj;
				++jj;
				return neighbor_idx;
			}
			return -1;
		}

		hostDevice int32_t s_loop() {
			if (s.beginning != -1) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					return -1;
				auto morton = idx3D_to_morton(cell);

				for (; ii < s.beginning + s.length;) {
					cs = arrays.cellSpan[ii];
					++ii;
					jj = cs.beginning;
					if (full_loop ||
						position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, 1.f) == morton) {
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

			for (; i <= 1; ++i) {
				for (; j <= 1; ++j) {
					for (; k <= 1;) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
							++k;
							continue;
						}
						auto morton = idx3D_to_morton(cell);

						s = arrays.hashMap[idx3D_to_hash(cell, arrays.hash_entries)];
						ii = s.beginning;
						if (s.beginning == -1) {
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

		hostDevice cell_iterator(int3 c_idx, bool floop, Mem &memory, int32_t _i = -1,
			int32_t _j = -1, int32_t _k = -1)
			: idx(c_idx), full_loop(floop), i(_i), j(_j), k(_k), arrays(memory) {
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
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator end() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
	hostDeviceInline cell_iterator cbegin() const {
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator cend() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
};

template <typename Mem>
struct cell_interpolator<hash_length::bit_32, cell_ordering::z_order, cell_structuring::hashed,
	Mem> {
	int3 central_idx;
	bool full_loop;
	Mem &arrays;

	template <typename T>
	hostDevice cell_interpolator(T &&position, Mem &memory, bool all = false)
		: central_idx(
			position_to_idx3D_i(position, memory.min_coord, math::unit_get<1>(memory.cell_size))),
		full_loop(all), arrays(memory) {}

	struct cell_iterator {
		int3 idx;
		bool full_loop;
		int32_t i = -1, j = -1, k = -1;
		int32_t ii = 0, jj = 0;

		int32_t neighbor_idx;

		Mem &arrays;
		hash_span s{ -1, -1 };
		cell_span cs{ -1, -1 };

		hostDevice int32_t cs_loop() {
			if (cs.beginning != -1 && jj < cs.beginning + cs.length) {
				neighbor_idx = jj;
				++jj;
				return neighbor_idx;
			}
			return -1;
		}

		hostDevice int32_t s_loop() {
			if (s.beginning != -1) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					return -1;
				auto morton = idx3D_to_morton_32(cell);

				for (; ii < s.beginning + s.length;) {
					cs = arrays.cellSpan[ii];
					++ii;
					jj = cs.beginning;
					if (full_loop ||
						position_to_morton_32(pairHelper(arrays.position)[cs.beginning], arrays, 1.f) == morton) {
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

			for (; i <= 1; ++i) {
				for (; j <= 1; ++j) {
					for (; k <= 1;) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
							++k;
							continue;
						}
						auto morton = idx3D_to_morton(cell);

						s = arrays.hashMap[idx3D_to_hash(cell, arrays.hash_entries)];
						ii = s.beginning;
						if (s.beginning == -1) {
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

		hostDevice cell_iterator(int3 c_idx, bool floop, Mem &memory, int32_t _i = -1,
			int32_t _j = -1, int32_t _k = -1)
			: idx(c_idx), full_loop(floop), i(_i), j(_j), k(_k), arrays(memory) {
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
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator end() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
	hostDeviceInline cell_iterator cbegin() const {
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator cend() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
};

template <typename Mem>
struct cell_interpolator<hash_length::bit_32, cell_ordering::linear_order,
	cell_structuring::complete, Mem> {
	int3 central_idx;
	bool full_loop;
	Mem &arrays;

	template <typename T>
	hostDevice cell_interpolator(T &&position, Mem &memory, bool all = false)
		: central_idx(
			position_to_idx3D_i(std::forward<T>(position), memory.min_coord, math::unit_get<1>(memory.cell_size))),
		full_loop(all), arrays(memory) {}

	struct cell_iterator {
		int3 blockIdx;
		bool full_loop;
		int32_t i = -1, j = -1, k = -1;
		int32_t ii = 0;
		uint32_t jj = 0;

		int32_t neighbor_idx;
		uint2 cellspan = { 0, 0 };

		Mem &arrays;
		hostDevice int32_t c_loop() {
			for (; jj < cellspan.y;) {
				neighbor_idx = jj;
				++jj;
				return neighbor_idx;
			}
			return -1;
		}

		hostDevice void increment() {
			if (c_loop() != -1)
				return;
			uint32_t num_cells = math::unit_get<1>(arrays.grid_size) *
				math::unit_get<2>(arrays.grid_size) *
				math::unit_get<3>(arrays.grid_size);
			int3 cellIdx;
			for (; i <= 1; ++i) {
				cellIdx.z = blockIdx.z + i;
				for (; j <= 1; ++j) {
					cellIdx.y = blockIdx.y + j;
					for (; k <= 1;) {
						cellIdx.x = blockIdx.x + k;
						++k;
						if (cellIdx.x > -1 && cellIdx.x < (int)math::unit_get<1>(arrays.grid_size) &&
							cellIdx.y > -1 && cellIdx.y < (int)math::unit_get<2>(arrays.grid_size) &&
							cellIdx.z > -1 && cellIdx.z < (int)math::unit_get<3>(arrays.grid_size)) {
							uint idx = idx3D_to_linear(
								uint3{ static_cast<uint32_t>(cellIdx.x), static_cast<uint32_t>(cellIdx.y),
								static_cast<uint32_t>(cellIdx.z) },
								uint3{ static_cast<uint32_t>(math::unit_get<1>(arrays.grid_size)),
								static_cast<uint32_t>(math::unit_get<2>(arrays.grid_size)),
								static_cast<uint32_t>(math::unit_get<3>(arrays.grid_size)) });
							if (idx < num_cells) {
								cellspan.x = arrays.cellBegin[idx];
								cellspan.y = arrays.cellEnd[idx];
								jj = cellspan.x;
								if (c_loop() != -1)
									return;
							}
						}
					}
					k = -1;
				}
				j = -1;
			}
		}

		hostDevice cell_iterator(int3 c_idx, bool floop, Mem &memory, int32_t _i = -1,
			int32_t _j = -1, int32_t _k = -1)
			: blockIdx(c_idx), full_loop(floop), i(_i), j(_j), k(_k), arrays(memory) {
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
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator end() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
	hostDeviceInline cell_iterator cbegin() const {
		return cell_iterator{ central_idx, full_loop, arrays };
	}
	hostDeviceInline cell_iterator cend() const {
		return cell_iterator{ central_idx, full_loop, arrays, 2, 2, 2 };
	}
};

template <typename Mem>
struct cell_interpolator<hash_length::bit_64, cell_ordering::z_order, cell_structuring::MLM, Mem> {
	int3 central_idx;
	bool full_loop;
	Mem &arrays;

	template <typename T>
	hostDevice cell_interpolator(T &&position, Mem &memory, bool all = false)
		: full_loop(all), arrays(memory) {
		int32_t thread_idx = getThreadIdx();
		int32_t resolution = memory.MLMResolution[thread_idx];
		float factor = powf(0.5f, ((float)resolution));
		central_idx = position_to_idx3D_i(std::forward<T>(position), memory.min_coord,
			math::unit_get<1>(memory.cell_size) * factor);
	}

	struct cell_iterator {
		int3 idx;
		bool full_loop;
		float factor = 1.f;
		int32_t resolution = 1;
		int32_t i = -1, j = -1, k = -1;
		int32_t ii = 0, jj = 0;

		int32_t neighbor_idx;

		Mem &arrays;
		hash_span s{ -1, -1 };
		cell_span cs{ -1, -1 };

		hostDevice int32_t cs_loop() {
			if (cs.beginning != -1 && jj < cs.beginning + cs.length) {
				neighbor_idx = jj;
				++jj;
				return neighbor_idx;
			}
			return -1;
		}

		hostDevice int32_t s_loop() {
			if (s.beginning != -1) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					return -1;
				auto morton = idx3D_to_morton(cell);

				for (; ii < s.beginning + s.length;) {
					cs = arrays.cellSpan[ii + arrays.max_numptcls * resolution];
					++ii;
					jj = cs.beginning;
					if (full_loop ||
						position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, factor) == morton) {
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

			for (; i <= 1; ++i) {
				for (; j <= 1; ++j) {
					for (; k <= 1;) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
							++k;
							continue;
						}
						auto morton = idx3D_to_morton(cell);

						s = arrays.hashMap[idx3D_to_hash(cell, arrays.hash_entries) +
							arrays.hash_entries * resolution];
						ii = s.beginning;
						if (s.beginning == -1) {
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

		hostDevice cell_iterator(int3 c_idx, bool floop, Mem &memory, int32_t _i = -1,
			int32_t _j = -1, int32_t _k = -1)
			: idx(c_idx), full_loop(floop), i(_i), j(_j), k(_k), arrays(memory) {
			int32_t thread_idx = getThreadIdx();
			resolution = memory.MLMResolution[thread_idx];
			factor = powf(0.5f, ((float)resolution));
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
		return cell_iterator(central_idx, full_loop, arrays);
	}
	hostDeviceInline cell_iterator end() const {
		return cell_iterator(central_idx, full_loop, arrays, 2, 2, 2);
	}
	hostDeviceInline cell_iterator cbegin() const {
		return cell_iterator(central_idx, full_loop, arrays);
	}
	hostDeviceInline cell_iterator cend() const {
		return cell_iterator(central_idx, full_loop, arrays, 2, 2, 2);
	}
};

template <typename Mem>
struct cell_interpolator<hash_length::bit_64, cell_ordering::z_order, cell_structuring::compactMLM, Mem> {
	int3 central_idx;
	bool full_loop;
	Mem &arrays;

	template <typename T>
	hostDevice cell_interpolator(T &&position, Mem &memory, bool all = false)
		: full_loop(all), arrays(memory) {
		int32_t thread_idx = getThreadIdx();
		int32_t resolution = memory.MLMResolution[thread_idx];
		float factor = powf(0.5f, ((float)resolution));
		central_idx = position_to_idx3D_i(std::forward<T>(position), memory.min_coord,
			math::unit_get<1>(memory.cell_size) * factor);
	}

	struct cell_iterator {
		int3 idx;
		bool full_loop;
		float factor = 1.f;
		int32_t resolution = 1;
		int32_t i = -1, j = -1, k = -1;
		uint32_t ii = 0;
		int32_t jj = 0;

		int32_t neighbor_idx;

		Mem &arrays;
		compactHashSpan s{0, UINT31_MAX, -1 };
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
					cs = cell_span{ (int32_t) s.beginning, s.length };
					jj = cs.beginning;
					ii = s.beginning + s.length;
					if (full_loop ||
						position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, factor) == morton) {
						if (cs_loop() != -1) {
							return neighbor_idx;
						}
					}
				}
				for (; ii < s.beginning + s.length;) {
					cs = arrays.compactCellSpan[ii + arrays.max_numptcls * resolution];
					++ii;
					jj = cs.beginning;
					if (full_loop ||
						position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, factor) == morton) {
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

			for (; i <= 1; ++i) {
				for (; j <= 1; ++j) {
					for (; k <= 1;) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
							++k;
							continue;
						}
						auto morton = idx3D_to_morton(cell);

						s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries) +
							arrays.hash_entries * resolution];
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

		hostDevice cell_iterator(int3 c_idx, bool floop, Mem &memory, int32_t _i = -1,
			int32_t _j = -1, int32_t _k = -1)
			: idx(c_idx), full_loop(floop), i(_i), j(_j), k(_k), arrays(memory) {
			int32_t thread_idx = getThreadIdx();
			resolution = memory.MLMResolution[thread_idx];
			factor = powf(0.5f, ((float)resolution));
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
		return cell_iterator(central_idx, full_loop, arrays);
	}
	hostDeviceInline cell_iterator end() const {
		return cell_iterator(central_idx, full_loop, arrays, 2, 2, 2);
	}
	hostDeviceInline cell_iterator cbegin() const {
		return cell_iterator(central_idx, full_loop, arrays);
	}
	hostDeviceInline cell_iterator cend() const {
		return cell_iterator(central_idx, full_loop, arrays, 2, 2, 2);
	}
};