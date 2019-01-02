#pragma once
#include <vector>
#include <utility/math.h>
#include <render/quadRender/geometry.h>

// The nice version of the BVH - a shallow hierarchy of inner and leaf nodes
struct BVHNode {
  float3 bottom;
  float3 top;
  virtual bool leafNode() = 0; // pure virtual
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
  // bounding box
  float3 bottom;
  float3 top;

  // parameters for leafnodes and innernodes occupy same space (union) to save memory
  // top bit discriminates between leafnode and innernode
  // no pointers, but indices (int): faster

  union {
    // inner node - stores indexes to array of CacheFriendlyBVHNode
    struct {
      int32_t idxLeft;
	  int32_t idxRight;
    } inner;
    // leaf node: stores triangle count and starting index in triangle list
    struct {
		int32_t count; // Top-most bit set, leafnode if set, innernode otherwise
		int32_t startIdx;
    } leaf;
  } u;
};

struct BVHLimits {
	// bounding box
	float3 bottom;
	float3 top;
};

// Work item for creation of BVH:
struct BBoxTmp {
	// Bottom point (ie minx,miny,minz)
	float3 bottom{ FLT_MAX, FLT_MAX, FLT_MAX };
	// Top point (ie maxx,maxy,maxz)
	float3 top{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	// Center point, ie 0.5*(top-bottom)
	float3 center; // = bbox centroid
	// Triangle
	const Triangle *triangles = nullptr;  // triangle list
	BBoxTmp()
	{}
};

// BVH CONSTRUCTION
// This builds the BVH, finding optimal split planes for each depth
// uses binning: divide the work bounding box into a number of equally sized "bins" along one of the axes
// choose axis and splitting plane resulting in least cost (determined by surface area heuristic or SAH)
// SAH (surface area heuristic): the larger the surface area of a bounding box, the costlier it is to raytrace
// find the bbox with the minimum surface area
//
// I strongly recommend reading Ingo Wald's 2007 paper "On fast SAH based BVH construction",  
// http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf, to understand the code below


typedef std::vector<BBoxTmp> BBoxEntries;  // vector of triangle bounding boxes needed during BVH construction

struct TriangleIntersectionData {
	float4 center;
	float4 normal;
	float4 e1d1;
	float4 e2d2;
	float4 e3d3;
};

struct gpuBVH {
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