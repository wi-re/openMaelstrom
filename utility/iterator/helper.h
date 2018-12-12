#pragma once
#include <utility/iterator.h>
#include <utility/SPH/indication.h>
#include <utility/math.h>
#include <utility/unit_math.h>

#include <utility/identifier/resource_helper.h>
#include <utility/SPH.h>

template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename T,
	typename U>
	hostDevice auto cell_iterate(U &&position, T &memory, bool all = false) {
	return cell_interpolator<hash_width, order, structure, T>(std::forward<U>(position), memory, all);
}
template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename T>
hostDevice auto cell_iterate(T &memory, bool all = false) {
	return cell_interpolator<hash_width, order, structure, T>(memory.position[getThreadIdx()],
		memory, all);
}

struct boundaryPlanes {
	const float4_u<> *pData;
	const int32_t elements;
	template <typename T>
	hostDeviceInline boundaryPlanes(const T &arrays)
		: pData(arrays.boundaryPlanes), elements(arrays.boundaryCounter) {}

	struct plane_it {
		const float4_u<> * pData;
		hostDeviceInline auto operator*() {
			return *pData;
		}
		hostDeviceInline bool operator==(const plane_it &rawIterator) const {
			return (pData == rawIterator.pData);
		}
		hostDeviceInline bool operator!=(const plane_it &rawIterator) const {
			return (pData != rawIterator.pData);
		}

		hostDeviceInline plane_it &operator++() {
			pData++;
			return (*this);
		}
		hostDeviceInline plane_it &operator--() {
			pData--;
			return (*this);
		}
		hostDeviceInline plane_it operator++(int) {
			auto temp(*this);
			pData++;
			return temp;
		}
		hostDeviceInline plane_it operator--(int) {
			auto temp(*this);
			pData--;
			return temp;
		}
	};
	hostDeviceInline plane_it begin() const { return plane_it{ pData }; }
	hostDeviceInline plane_it end() const {
		return plane_it{ pData + elements };
	}
	hostDeviceInline plane_it cbegin() const { return plane_it{ pData }; }
	hostDeviceInline plane_it cend() const {
		return plane_it{ pData + elements };
	}
};
