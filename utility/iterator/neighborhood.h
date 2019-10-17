#pragma once
#include <utility/iterator.h>
#include <utility/SPH/indication.h>
#include <utility/math.h>

#include <utility/identifier/resource_helper.h>
#include <utility/SPH.h>

template <neighbor_list neighbor> struct neighbor_span;

template <> struct neighbor_span<neighbor_list::constrained> {
	const int32_t *pData;
	const int32_t offset;
	const int32_t elements;
	template <typename T>
	hostDeviceInline neighbor_span(const int32_t idx, const T &arrays)
		: offset(arrays.max_numptcls), pData((const int32_t *)arrays.neighborList + idx),
		elements(arrays.neighborListLength[idx]) {}

	struct neighbor_it {
		const int32_t *pData;
		const int32_t offset;
		hostDeviceInline int32_t operator*() {
#ifdef __CUDA_ARCH__
			return *pData;
			// int32_t value;
			// asm("ld.global.cs.s32 %0, [%1];" : "=r"(value) : "l"(pData));
			// return value;
#else
			return *pData;
#endif
		}
		hostDeviceInline bool operator==(const neighbor_it &rawIterator) const {
			return (pData == rawIterator.pData);
		}
		hostDeviceInline bool operator!=(const neighbor_it &rawIterator) const {
			return (pData != rawIterator.pData);
		}

		hostDeviceInline neighbor_it &operator++() {
			pData += offset;
			return (*this);
		}
		hostDeviceInline neighbor_it &operator--() {
			pData -= offset;
			return (*this);
		}
		hostDeviceInline neighbor_it operator++(int) {
			auto temp(*this);
			pData += offset;
			return temp;
		}
		hostDeviceInline neighbor_it operator--(int) {
			auto temp(*this);
			pData += offset;
			return temp;
		}
	};
	hostDeviceInline neighbor_it begin() const { return neighbor_it{ pData, offset }; }
	hostDeviceInline neighbor_it end() const {
		return neighbor_it{ pData + offset * elements, offset };
	}
	hostDeviceInline neighbor_it cbegin() const { return neighbor_it{ pData, offset }; }
	hostDeviceInline neighbor_it cend() const {
		return neighbor_it{ pData + offset * elements, offset };
	}
};

template <> struct neighbor_span<neighbor_list::cell_based> {
	const neigh_span *pData;
	const int32_t offset;
	template <typename T>
	hostDeviceInline neighbor_span(const int32_t idx, const T &arrays)
		: offset(arrays.num_ptcls), pData((const neigh_span *)arrays.spanNeighborList + idx) {}

	struct neighbor_it {
		const neigh_span *pData;
		const int32_t offset;
		int32_t j = 0;
		int32_t n = 0;

		hostDevice neighbor_it(const neigh_span *data, int32_t _offset, int32_t _j = 0)
			: pData(data), offset(_offset), j(_j) {
			const neigh_span &pSpan = *pData;
			if (_j == 0)
				j = pSpan.beginning;
			n = pSpan.beginning + pSpan.length;
		}

		hostDeviceInline int32_t operator*() { return j; }
		hostDeviceInline bool operator==(const neighbor_it &rawIterator) const {
			return (j == rawIterator.j);
		}
		hostDeviceInline bool operator!=(const neighbor_it &rawIterator) const {
			return (j != rawIterator.j);
		}

		hostDeviceInline void increment() {
#ifdef __CUDA_ARCH__
			if (++j == n + 1) {
				pData += offset;
				const neigh_span &pSpan = *pData;
				j = pSpan.beginning;
				n = pSpan.beginning + pSpan.length;
			}
			// if (++j == n + 1) {
			//	pData += offset;
			//	neigh_span pSpan;
			//	asm("ld.global.cs.s32 %0, [%1];" : "=r"(*(reinterpret_cast<int32_t*>(&pSpan))) :
			//"l"(reinterpret_cast<const int32_t*>(pData))); 	j = pSpan.beginning; 	n = pSpan.beginning +
			//pSpan.length;
			//}
#else
			if (++j == n + 1) {
				pData += offset;
				const neigh_span &pSpan = *pData;
				j = pSpan.beginning;
				n = pSpan.beginning + pSpan.length;
			}

#endif
		}

		hostDeviceInline neighbor_it &operator++() {
			increment();
			return (*this);
		}
		hostDeviceInline neighbor_it operator++(int) {
			auto temp(*this);
			increment();
			return temp;
		}
	};
	hostDeviceInline neighbor_it begin() const { return neighbor_it{ pData, offset }; }
	hostDeviceInline neighbor_it end() const { return neighbor_it{ pData, offset, BITFIELD_MAX_VALUE }; }
	hostDeviceInline neighbor_it cbegin() const { return neighbor_it{ pData, offset }; }
	hostDeviceInline neighbor_it cend() const { return neighbor_it{ pData, offset, BITFIELD_MAX_VALUE }; }
};

template <> struct neighbor_span<neighbor_list::compactCell> {
	const compactSpan *pData;
	const int32_t offset;
	const int8_t scaling;

	template <typename T>
	hostDeviceInline neighbor_span(const int32_t idx, const T &arrays)
		: offset(arrays.num_ptcls), pData((const compactSpan *)arrays.compactCellList + idx), scaling(arrays.compactCellScale[idx]){}

	struct neighbor_it {
		const compactSpan *pData;
		const int32_t offset;
		int32_t j = 0;
		int32_t n = 0;
		int8_t scale = 0;

		hostDevice neighbor_it(const compactSpan *data, int8_t scaling, int32_t _offset, int32_t _j = 0)
			: pData(data), offset(_offset), j(_j), scale(scaling) {
			const compactSpan &pSpan = *pData;
			if (_j == 0)
				j = pSpan.idx;
			n = pSpan.idx + pSpan.len * scale;
		}

		hostDeviceInline int32_t operator*() { return j; }
		hostDeviceInline bool operator==(const neighbor_it &rawIterator) const {
			return (j == rawIterator.j);
		}
		hostDeviceInline bool operator!=(const neighbor_it &rawIterator) const {
			return (j != rawIterator.j);
		}

		hostDeviceInline void increment() {
			#ifdef __CUDA_ARCH__
			if (++j == n + 1) {
				pData += offset;
				compactSpan pSpan;
				asm("ld.global.cg.s32 %0, [%1];" : "=r"(*(reinterpret_cast<int32_t*>(&pSpan))) :
			"l"(reinterpret_cast<const int32_t*>(pData))); 	
				j = pSpan.idx; 	
				n = pSpan.idx + pSpan.len * scale;
			}
			#else
			if (++j == n + 1) {
				pData += offset;
				const compactSpan &pSpan = *pData;
				j = pSpan.idx;
				n = pSpan.idx + pSpan.len * scale;
			}
			#endif
		}

		hostDeviceInline neighbor_it &operator++() {
			increment();
			return (*this);
		}
		hostDeviceInline neighbor_it operator++(int) {
			auto temp(*this);
			increment();
			return temp;
		}
	};
	hostDeviceInline neighbor_it begin() const { return neighbor_it{ pData, scaling, offset }; }
	hostDeviceInline neighbor_it end() const { return neighbor_it{ pData, scaling, offset, COMPACT_IDX_MAX }; }
	hostDeviceInline neighbor_it cbegin() const { return neighbor_it{ pData, scaling, offset }; }
	hostDeviceInline neighbor_it cend() const { return neighbor_it{ pData, scaling, offset, COMPACT_IDX_MAX }; }
};


hostDeviceInline uint32_t ffs(uint32_t v) {
#ifdef __CUDA_ARCH__
	return __ffs(v);
#else
#ifdef _MSC_VER
#pragma warning(disable:4146)
#endif
	return static_cast<uint32_t>(log2(v & -v) + 1);
#ifdef _MSC_VER
#pragma warning(default:4146)
#endif
#endif
}

hostDeviceInline auto pToIdx3(float4 p, float3 min, float3 d) {
	int32_t i = static_cast<int32_t>(floorf((p.x - min.x) / d.x));
	int32_t j = static_cast<int32_t>(floorf((p.y - min.y) / d.x));
	int32_t k = static_cast<int32_t>(floorf((p.z - min.z) / d.x));
	return int3{ i, j, k };
}

hostDeviceInline auto pToIdx3(float4_u<SI::m> p, float3_u<SI::m> min, float3_u<SI::m> d) {
	int32_t i = static_cast<int32_t>(floorf((p.val.x - min.val.x) / d.val.x));
	int32_t j = static_cast<int32_t>(floorf((p.val.y - min.val.y) / d.val.x));
	int32_t k = static_cast<int32_t>(floorf((p.val.z - min.val.z) / d.val.x));
	return int3{ i, j, k };
}

template<typename T>
struct maskIterator {
	int32_t resolution;
	float factor;
	cell_span cs{ 0,0 };
	int32_t ci = 0;
	int32_t i;
	compactCellNeighbors c{ 0u,0u,0u,0u };
	int3 idx;
	T& arrays;
	hostDeviceInline maskIterator(int32_t i_, T& arrays_) :i(i_), arrays(arrays_) {		
		auto x_i = pairHelper(arrays.position)[i];
		resolution = arrays.MLMResolution[i];
		factor = resolution == 0 ? 1.f : powf(0.5f, (float)arrays.MLMResolution[i]);
		idx = pToIdx3(x_i, arrays.min_coord, arrays.cell_size * factor);
	}
	struct iterator {
		maskIterator& mIt;
		bool done = false;
		int32_t j = -1;
		hostDeviceInline  void increment() {
			if (mIt.c.mask == MAX_VAL_26BIT) {
				if (j >= mIt.cs.beginning && j + 1 < mIt.cs.beginning + mIt.cs.length) {
					++j;
					return;
				}
			}
			else {
				if (mIt.c.mask != 0) {
					j = mIt.cs.beginning + ffs(mIt.c.mask) - 1;
					mIt.c.mask = mIt.c.mask ^ (1 << (ffs(mIt.c.mask) - 1));
					return;
				}
			}
			mIt.c = mIt.arrays.neighborMask[mIt.i + mIt.ci * mIt.arrays.max_numptcls];
			mIt.ci++;
			if (mIt.c.mask == 0) {
				j = -1;
				done = true;
				return;
			};
			uint3 cell = uint3{ static_cast<uint32_t>(mIt.idx.x + (int32_t)mIt.c.xi - 2), static_cast<uint32_t>(mIt.idx.y + (int32_t)mIt.c.yi - 2), static_cast<uint32_t>(mIt.idx.z + (int32_t)mIt.c.zi - 2) };
			auto morton = idx3D_to_morton(cell);
			auto s = mIt.arrays.compactHashMap[idx3D_to_hash(cell, mIt.arrays.hash_entries) + mIt.arrays.hash_entries * mIt.resolution];
			mIt.cs = cell_span{ 0,0 };
			if (s.compacted) {
				mIt.cs = cell_span{ (int32_t)s.beginning, s.length };
			}
			else {
				for (int32_t si = s.beginning; si < (int32_t) s.beginning + s.length; ++si) {
					mIt.cs = mIt.arrays.compactCellSpan[si + mIt.arrays.max_numptcls * mIt.resolution];
					if (position_to_morton(pairHelper(mIt.arrays.position)[mIt.cs.beginning], mIt.arrays, mIt.factor) != morton)
						continue;
					break;
				}
			}
			if (mIt.c.mask == MAX_VAL_26BIT) {
				j = mIt.cs.beginning;
			}
			else {
				j = mIt.cs.beginning + ffs(mIt.c.mask) - 1;
				mIt.c.mask = mIt.c.mask ^ (1 << (ffs(mIt.c.mask) - 1));
			}
		}
		hostDeviceInline int32_t operator*() {
			return j;
		};

		hostDeviceInline iterator(maskIterator& ref, bool end = false) :done(end), mIt(ref) {
			if (!end) increment();
		}
		hostDeviceInline bool operator==(const iterator &rawIterator) const {
			return (done == rawIterator.done);
		}
		hostDeviceInline bool operator!=(const iterator &rawIterator) const {
			return (done != rawIterator.done);
		}
		hostDeviceInline iterator &operator++() {
			increment();
			return (*this);
		}
	};
	hostDeviceInline iterator begin() {
		return iterator(*this);
	}
	hostDeviceInline iterator end() {
		return iterator(*this, true);
	}
	hostDeviceInline iterator cbegin() {
		return iterator(*this);
	}
	hostDeviceInline iterator cend() {
		return iterator(*this, true);
	}
};


template <neighbor_list neighbor, typename T> hostDevice auto neighbor_iterate(T &memory) {
	return neighbor_span<neighbor>(getThreadIdx(), memory);
}
#include <utility/iterator/cell.h>
#include <utility/iterator/helper.h>

template<neighbor_list neighbor>
struct neigh_tag_ty {};

template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::basic> = {}) {
	return neighbor_span<neighbor_list::constrained>(idx, memory);
}
template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::cell_based> = {}) {
	return neighbor_span<neighbor_list::cell_based>(idx, memory);
}
template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::constrained> = {}) {
	return neighbor_span<neighbor_list::constrained>(idx, memory);
}
template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::compactCell> = {}) {
	return neighbor_span<neighbor_list::compactCell>(idx, memory);
}
template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::compactMLM> = {}) {
	return cell_iterate<hash_length::bit_64, cell_ordering::z_order, cell_structuring::compactMLM>(pairHelper(memory.position)[idx], memory);
}

template <typename T>
hostDevice auto neighbor_iterate_wrapper(int32_t idx, T &memory, bool all = false, neigh_tag_ty<neighbor_list::masked> = {}) {
	return maskIterator<T>(idx, memory);
}
