#pragma once
#include <utility/identifier/resource_helper.h>
#include <utility/math.h>
template<typename T> hostDeviceInline auto getBlendWeight(T& arrays, int32_t i) {
	if (arrays.lifetime[i] >= 0.f) return float_u<>{0.f};
	return 0.5f * (arrays.lifetime[i]) / math::min(-arrays.blendSteps * arrays.timestep, arrays.lifetime[i]);
}


hostDeviceInline  uint64_t splitBy3(uint32_t a) {
	uint64_t x = a & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

hostDeviceInline uint64_t mortonEncode_magicbits(uint32_t x, uint32_t y, uint32_t z) {
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}

hostDeviceInline auto idx3D_to_morton(uint3 idx) {
	return mortonEncode_magicbits(idx.x, idx.y, idx.z);
}

hostDeviceInline  uint64_t splitBy3(int32_t a) {
	int64_t x = a & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

hostDeviceInline int64_t mortonEncode_magicbits(int32_t x, int32_t y, int32_t z) {
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}

hostDeviceInline auto idx3D_to_morton(int3 idx) {
	return mortonEncode_magicbits(idx.x, idx.y, idx.z);
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
hostDeviceInline int32_t splitBy3_32(int32_t a){
	int32_t x = a;
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

hostDeviceInline int32_t mortonEncode_magicbits_32(int32_t x, int32_t y, int32_t z) {
	int32_t answer = 0;
	answer |= splitBy3_32(x) | splitBy3_32(y) << 1 | splitBy3_32(z) << 2;
	return answer;
}

hostDeviceInline auto idx3D_to_morton_32(int3 idx) {
	return mortonEncode_magicbits_32(idx.x, idx.y, idx.z);
}

hostDeviceInline  uint32_t splitBy3_32(uint32_t a) {
	uint32_t x = a;
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

hostDeviceInline uint32_t mortonEncode_magicbits_32(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t answer = 0;
	answer |= splitBy3_32(x) | splitBy3_32(y) << 1 | splitBy3_32(z) << 2;
	return answer;
}


hostDeviceInline auto idx3D_to_morton_32(uint3 idx) {
	return mortonEncode_magicbits_32(idx.x, idx.y, idx.z);
}


hostDeviceInline auto idx3D_to_hash(uint3 idx, uint32_t hash_entries) {
	return (idx.x * 73856093 + idx.y * 19349663 + idx.z * 83492791) % hash_entries;
}

hostDeviceInline auto idx3D_to_hash(int3 idx, uint32_t hash_entries) {
	return static_cast<int32_t>((static_cast<uint32_t>(idx.x) * 73856093u + static_cast<uint32_t>(idx.y) * 19349663u + static_cast<uint32_t>(idx.z) * 83492791u) % hash_entries);
}

template <typename T, typename U>
hostDeviceInline auto idx3D_to_linear(T dimIdx, U dim) {
	return static_cast<uint32_t>((dimIdx.z * dim.y + dimIdx.y) * dim.x + dimIdx.x);
}


template<typename T, typename U, typename V>
struct position_to_idx3D_helper {
	hostDeviceInline static auto fn(T p, U min, V d) {
		uint32_t i = static_cast<uint32_t>(floorf((math::get<1>(p) - math::get<1>(min)) / d));
		uint32_t j = static_cast<uint32_t>(floorf((math::get<2>(p) - math::get<2>(min)) / d));
		uint32_t k = static_cast<uint32_t>(floorf((math::get<3>(p) - math::get<3>(min)) / d));
		return uint3{ i, j, k };
	}
};

template<typename T1, typename U1, typename T2, typename T3, typename U2>
struct position_to_idx3D_helper<value_unit<T1, U1>, value_unit<T2, U1>, value_unit<T3, U2>> {
	hostDeviceInline static auto fn(value_unit<T1, U1> p, value_unit<T2, U1> min, value_unit<T3, U2> d) {
		uint32_t i = static_cast<uint32_t>(floorf((p.val.x - min.val.x) / d.val));
		uint32_t j = static_cast<uint32_t>(floorf((p.val.y - min.val.y) / d.val));
		uint32_t k = static_cast<uint32_t>(floorf((p.val.z - min.val.z) / d.val));
		return uint3{ i, j, k };
	}
};

template<typename T, typename U, typename V>
constexpr hostDeviceInline auto position_to_idx3D(T a, U b, V c) {
	return position_to_idx3D_helper<T, U, V>::fn(a, b, c);
}

template<typename T, typename U, typename V>
struct position_to_idx3D_i_helper {
	hostDeviceInline static auto fn(T p, U min, V d) {
		int32_t i = static_cast<int32_t>(floorf((math::get<1>(p) - math::get<1>(min)) / d));
		int32_t j = static_cast<int32_t>(floorf((math::get<2>(p) - math::get<2>(min)) / d));
		int32_t k = static_cast<int32_t>(floorf((math::get<3>(p) - math::get<3>(min)) / d));
		return int3{ i, j, k };
	}
};

template<typename T1, typename U1, typename T2, typename T3, typename U2>
struct position_to_idx3D_i_helper<value_unit<T1, U1>, value_unit<T2, U1>, value_unit<T3, U2>> {
	hostDeviceInline static auto fn(value_unit<T1, U1> p, value_unit<T2, U1> min, value_unit<T3, U2> d) {
		int32_t i = static_cast<int32_t>(floorf((p.val.x - min.val.x) / d.val));
		int32_t j = static_cast<int32_t>(floorf((p.val.y - min.val.y) / d.val));
		int32_t k = static_cast<int32_t>(floorf((p.val.z - min.val.z) / d.val));
		return int3{ i, j, k };
	}
};

template<typename T, typename U, typename V>
constexpr hostDeviceInline auto position_to_idx3D_i(T a, U b, V c) {
	return position_to_idx3D_i_helper<T, U, V>::fn(a, b, c);
}

template<typename T, typename U>
constexpr hostDeviceInline auto integerCoords(T a, U& arrays, float factor = 1.f) {
	return position_to_idx3D_i(a, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor);
}

template<typename R, typename S, typename T>
hostDeviceInline auto position_to_hash(R p, S min, T d, uint32_t hash_entries) {
	return idx3D_to_hash(position_to_idx3D(p, min, d), hash_entries);
}

template<typename T, typename U>
hostDeviceInline auto position_to_hash(T p, U& arrays, float factor = 1.f) {
	return idx3D_to_hash(position_to_idx3D(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor), arrays.hash_entries);
}

template<typename T, typename U>
hostDeviceInline auto position_to_linear(T p, U& arrays, float factor = 1.f) {
	return idx3D_to_linear(position_to_idx3D(p, arrays.min_domain, math::unit_get<1>(arrays.cell_size) * factor), arrays.grid_size);
}

template<typename T, typename U>
hostDeviceInline auto position_to_morton_32(T p, U& arrays, float factor = 1.f) {
	return idx3D_to_morton_32(position_to_idx3D(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor));
}

template<typename T, typename U>
hostDeviceInline auto position_to_morton_32_i(T p, U& arrays, float factor = 1.f) {
	return idx3D_to_morton_32(position_to_idx3D_i(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor));
}

template<typename T, typename U>
hostDeviceInline auto position_to_morton(T p, U& arrays, float factor = 1.f) {
	return idx3D_to_morton(position_to_idx3D(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor));
}

template<hash_length h>
hostDeviceInline float cell_factor(int32_t v);


template<>
hostDeviceInline float cell_factor<hash_length::bit_64>(int32_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return 1.f / ((float)(1048576 / v));
}


template<>
hostDeviceInline float cell_factor<hash_length::bit_32>(int32_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return 1.f / ((float)(1024 / v));
}