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
#include <utility/math.h>
#include <utility/unit_math.h>

#ifndef _WIN32
#define TEMPLATE_TOKEN template
#else
#define TEMPLATE_TOKEN
#endif

enum class resource_t : int {
	array_type,
	uniform_type,
	swap_type,
	aggregate_uniform_type
};

enum struct memory_kind {
	particleData, cellData, customData, diffuseData, singleData
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
enum struct cell_structuring { hashed, MLM, complete };
enum struct neighbor_list { basic, constrained, cell_based, compactCell };

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

#define COMPACT_IDX 25
#define COMPACT_LEN (32 - COMPACT_IDX)
#define COMPACT_LEN_MAX ((1 << COMPACT_LEN)-1) 
#define COMPACT_IDX_MAX ((1 << COMPACT_IDX)-1) 

struct compactSpan {
	uint32_t idx : COMPACT_IDX;
	uint32_t len : COMPACT_LEN;
};


#define cudaAllocateMemory cudaMallocManaged