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

template <neighbor_list neighbor, typename T> hostDevice auto neighbor_iterate(T &memory) {
	return neighbor_span<neighbor>(getThreadIdx(), memory);
}
