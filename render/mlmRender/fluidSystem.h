#pragma once
#include <utility/include_all.h>     
#include <render/mlmRender/voxelBVH.h>

struct FluidSystem {
	int32_t numptcls;
	float3 bounds[2];

	CFBVHNode* fluidBVH;
};

struct FluidMemory {
	int3 grid_size;
	float3 cell_size;
	float3 min_coord;

	int32_t hash_entries;
	int32_t mlm_schemes;

	int32_t num_ptcls;
	int32_t max_numptcls;
	float timestep;
	float radius;
	float rest_density;
	float auxScale;

	arrays::compactHashMap::type* compactHashMap;
	arrays::compactCellSpan::type* compactCellSpan;
	compactListEntry* cellSpan;
	compactListEntry* hashMap;
	int32_t* MLMResolution;
	float4* position;
	float* volume;
	float* auxDistance;
};
