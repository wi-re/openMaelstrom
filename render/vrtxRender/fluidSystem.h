#pragma once
#include <utility/include_all.h>     
#include <render/vrtxRender/voxelBVH.h>
namespace vrtx {
	enum Refl_t { DIFF, METAL, SPEC, REFR, COAT };
}

struct vrtxFluidArrays{
	float3 min_coord;
	float3 max_coord;
	float3 cell_size;
	int3 grid_size;
	int32_t hash_entries;
	int32_t mlm_schemes;

	int32_t num_ptcls;
	int32_t max_numptcls;
	float timestep;
	float renderRadius;
	float rest_density;

	float minMap;
	float maxMap;
	int32_t transferFn;
	int32_t mappingFn;

	arrays::compactHashMap::type* compactHashMap;
	arrays::compactCellSpan::type* compactCellSpan;

	int32_t* MLMResolution;
	float4* position;
	float* volume;
	float4* renderArray;
	float* auxIsoDensity;
	float* density;

	float4* centerPosition;
	float* anisotropicMatrices;
};

struct vrtxFluidMemory {
	int3 grid_size;
	float3 cell_size;
	float3 min_coord;
	float3 max_coord;
	float3 cell_size_actual;
	int3 grid_size_actual;

	int32_t maxZ_coordx;
	int32_t maxZ_coordy;
	int32_t maxZ_coordz;

	int32_t hash_entries;
	int32_t mlm_schemes;

	int32_t num_ptcls;
	int32_t max_numptcls;
	int32_t surfaceTechnique;
	float timestep;
	float renderRadius;
	float rest_density;
	float auxScale;
	
	float IOR;
	float vrtxR;
	float fluidBias;
	int32_t bounces;

	float3 vrtxFluidColor;
	float3 vrtxDebeer;
	float vrtxDebeerScale;

	int32_t vrtxDepth;
	float vrtxDepthScale;

	compactListEntry* cellSpan;
	compactListEntry* hashMap;

	int32_t colorMapFlipped;
	int32_t colorMapLength;
	float4* colorMap;

	float wmin;
	float wmax;

	vrtx::Refl_t bvhMaterial;
	float3 bvhColor;

	float3 vrtxDomainMin;
	float3 vrtxDomainMax;
	float vrtxDomainEpsilon;
};
