#pragma once
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <limits.h>
#include <float.h>
#include <sstream>
#include <tuple>
#include <experimental/filesystem>
#include <utility/math.h>
#include <utility/unit_math.h>
#include <utility/mathv2/matrix.h>

namespace stdfs = std::experimental::filesystem;

#if !defined(_WIN32) || defined(__clang__)
#define TEMPLATE_TOKEN template
#else
#define TEMPLATE_TOKEN
#endif

#define FLUID_PARTICLE 0
#define RIGID_PARTICLE 1

enum class resource_t : int {
	array_type,
	uniform_type,
	swap_type,
	aggregate_uniform_type
};

enum struct memory_kind {
	particleData, cellData, customData, diffuseData, singleData, rigidData, spareData, individualData
};

template <typename T> struct complex_type {
	using type = T;
	std::string jsonName;
	T value;
};

struct complex_uniform {};

enum struct launch_config { device, host, debug, pure_host, _used_for_template_specializations };
enum struct hash_length { bit_64, bit_32 };
enum struct cell_ordering { z_order, linear_order };
enum struct cell_structuring { hashed, MLM, complete, compactMLM};
enum struct neighbor_list { basic, constrained, cell_based, compactCell, compactMLM, masked };

//#define BASIC_NEIGHBORLIST_COMPILED
#define CONSTRAINED_NEIGHBORLIST_COMPILED
#define CELLBASED_NEIGHBORLIST_COMPILED
//#define COMPACTCELL_NEIGHBORLIST_COMPILED
#define COMPACTMLM_NEIGHBORLIST_COMPILED
//#define MASKED_NEIGHBORLIST_COMPILED

//#define MLM32_CELL_ALGORITHM
//#define HASHED64_CELL_ALGORITHM
//#define HASHED32_CELL_ALGORITHM
//#define MLM64_CELL_ALGORITHM
#define COMPACT_MLM_CELL_ALGORITHM
//#define LINEAR_CELL_ALGORITHM

//#define BITFIELD_STRUCTURES
#define BITFIELD_WIDTH 25
#define BITFIELD_MAX_VALUE ((1 << BITFIELD_WIDTH)-1) 

#ifndef BITFIELD_STRUCTURES
struct neigh_span {
	uint32_t beginning;
	uint32_t length;
};
#else
struct neigh_span {
	uint32_t beginning : BITFIELD_WIDTH;
	uint32_t length : (32 - BITFIELD_WIDTH);
};
#endif
struct hash_span {
	int32_t beginning;
	int32_t length;
};
struct cell_span {
	int32_t beginning;
	int32_t length;
};

#define OFFSET_INVALID 0x00
struct compactCellNeighbors {
	uint32_t xi : 2;
	uint32_t yi : 2;
	uint32_t zi : 2;
	uint32_t mask : 26;
};

#define COMPACT_IDX 25
#define COMPACT_LEN (32 - COMPACT_IDX)
#define COMPACT_LEN_MAX ((1 << COMPACT_LEN)-1) 
#define COMPACT_IDX_MAX ((1 << COMPACT_IDX)-1) 

struct compactSpan {
	uint32_t idx : COMPACT_IDX;
	uint32_t len : COMPACT_LEN;
};

struct cellInformation {
	cellInformation() : occupied (0), inside(0), xPos(0), xNeg(0), yPos(0), yNeg(0), zPos(0), zNeg(0) {}
	uint8_t occupied : 1;
	uint8_t inside : 1;

	uint8_t xPos : 1;
	uint8_t xNeg : 1;
	uint8_t yPos : 1;
	uint8_t yNeg : 1;
	uint8_t zPos : 1;
	uint8_t zNeg : 1;
};

struct cellSurface {
	float dummy;
};

#define UINT31_MAX 2147483647u

struct compactHashSpan {
	uint32_t compacted : 1;
	uint32_t beginning : 31;
	int32_t length;
};
using compact_cellSpan = cell_span;

#define LIST_ALWAYS_FALSE 0b00
#define LIST_ALWAYS_TRUE 0b01
#define LIST_COMPACT 0b10
#define LIST_ITERATE 0b11

struct compactListEntry {
	struct hashEntry {
		uint32_t kind : 2;
		uint32_t beginning : 25;
		uint32_t length : 5;
	};
	struct cellEntry {
		uint32_t kind : 2;
		uint32_t hash : 30;
	};
	union {
		hashEntry hash;
		cellEntry cell;
	};
};

#define MAX_VAL_00BIT 0
#define MAX_VAL_01BIT 1
#define MAX_VAL_02BIT 3
#define MAX_VAL_03BIT 7
#define MAX_VAL_04BIT 15
#define MAX_VAL_05BIT 31
#define MAX_VAL_06BIT 63
#define MAX_VAL_07BIT 127
#define MAX_VAL_08BIT 255
#define MAX_VAL_09BIT 511
#define MAX_VAL_10BIT 1023
#define MAX_VAL_11BIT 2047
#define MAX_VAL_12BIT 4095
#define MAX_VAL_13BIT 8191
#define MAX_VAL_14BIT 16383
#define MAX_VAL_15BIT 32767
#define MAX_VAL_16BIT 65535
#define MAX_VAL_17BIT 131071
#define MAX_VAL_18BIT 262143
#define MAX_VAL_19BIT 524287
#define MAX_VAL_20BIT 1048575
#define MAX_VAL_21BIT 2097151
#define MAX_VAL_22BIT 4194303
#define MAX_VAL_23BIT 8388607
#define MAX_VAL_24BIT 16777215
#define MAX_VAL_25BIT 33554431
#define MAX_VAL_26BIT 67108863
#define MAX_VAL_27BIT 134217727
#define MAX_VAL_28BIT 268435455
#define MAX_VAL_29BIT 536870911
#define MAX_VAL_30BIT 1073741823
#define MAX_VAL_31BIT 2147483647
#define MAX_VAL_32BIT 4294967295


#define cudaAllocateMemory cudaMallocManaged


//#define DEBUG_INVALID_PARITLCES
#define _VEC(vec) vec.x, vec.y, vec.z, vec.w
#define _VECSTR "[%+.8e %+.8e %+.8e %+.8e]"

#ifndef __CUDACC__
template <typename T> std::string type_name() {
	if constexpr (std::is_same<T, float4>::value)
		return "float4";
	if constexpr (std::is_same<T, float3>::value)
		return "float3";
	if constexpr (std::is_same<T, float2>::value)
		return "float2";
	if constexpr (std::is_same<T, uint4>::value)
		return "uint4";
	if constexpr (std::is_same<T, uint3>::value)
		return "uint3";
	if constexpr (std::is_same<T, uint2>::value)
		return "uint2";
	if constexpr (std::is_same<T, float>::value)
		return "float";
	if constexpr (std::is_same<T, uint32_t>::value)
		return "uint";
	if constexpr (std::is_same<T, int32_t>::value)
		return "int";
	return typeid(T).name();
}
#endif

template <class T> using rear_ptr_t = decltype(T::rear_ptr);                                          
template <class Ptr> using rear_ptr_type_template = detected_or_t<std::ptrdiff_t, rear_ptr_t, Ptr>; 
template <typename T>                                                                       
constexpr bool has_rear_ptr = !std::is_same<rear_ptr_type_template<T>, std::ptrdiff_t>::value;
