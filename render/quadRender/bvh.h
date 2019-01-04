#pragma once
#include <vector>
#include <utility/math.h>
#include <render/quadRender/geometry.h>

struct BVHNode {
  float3 bottom;
  float3 top;
  virtual bool leafNode() = 0;
};

struct BVHInner : BVHNode {
  BVHNode *left;
  BVHNode *right;
  virtual bool leafNode() { return false; }
};

struct BVHLeaf : BVHNode {
  std::vector<const Triangle *> triangles;
  virtual bool leafNode() { return true; }
};

struct CacheFriendlyBVHNode {
  float3 bottom;
  float3 top;
  union {
    struct {
      int32_t idxLeft;
	  int32_t idxRight;
    } inner;
    struct {
		int32_t count;
		int32_t startIdx;
    } leaf;
  } u;
};

struct BVHLimits {
	float3 bottom;
	float3 top;
};

struct BBoxTmp {
	float3 bottom{ FLT_MAX, FLT_MAX, FLT_MAX };
	float3 top{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float3 center;
	const Triangle *triangles = nullptr;  
	BBoxTmp(){}
};
typedef std::vector<BBoxTmp> BBoxEntries; 

struct TriangleIntersectionData {
	float4 center;
	float4 normal;
	float4 e1d1;
	float4 e2d2;
	float4 e3d3;
};

struct gpuBVH {
	bool active = true;
	Vertex* vertices;
	Triangle *pTriangles;
	uint4 *cudaBVHindexesOrTrilists;
	BVHLimits *cudaBVHlimits;
	TriangleIntersectionData *cudaTriangleIntersectionData;
	int *cudaTriIdxList;

	int32_t g_triIndexListNo = 0;
	int32_t g_pCFBVH_No = 0;
	int32_t g_verticesNo;
	int32_t g_trianglesNo;
};