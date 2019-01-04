#pragma once
#include <render/quadRender/bvh.h>
#include <render/quadRender/geometry.h>
#include <string>
#include <utility/math.h>
#include <vector>

struct mesh {
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
};

class objectLoader {
  std::vector<mesh> meshes;
  mesh merged;
  bool hashable = true;
  std::size_t stored_hash = 0x0;   
  bool dirty = true;
  std::vector<BVHNode *> BVHnodes;
  BVHNode *BVHroot = nullptr;

  BVHNode *buildBVH(BBoxEntries &work, int depth = 0);
  int32_t boxCount(BVHNode *root);
  uint32_t triangleCount(BVHNode *root);
  void depthCount(BVHNode *root, int32_t depth, int32_t &maxDepth);
  void PopulateCacheFriendlyBVH(const Triangle *pFirstTriangle, BVHNode *root, unsigned &idxBoxes,
                                unsigned &idxTriList);
  void CreateBVH();

  int32_t triIndicesLength = 0;
  std::unique_ptr<int32_t[]> triIndices = nullptr;
  int32_t CFBVHLength = 0;
  std::unique_ptr<CacheFriendlyBVHNode[]> CFBVH = nullptr;

  Vertex *cuVertices = nullptr;
  Triangle *cuTriangles = nullptr;
  TriangleIntersectionData* cuTriIntersectionData = nullptr;
  int *cuTriIndices = nullptr;
  BVHLimits *cuBVHLimits = nullptr;
  uint4 *cuBVHIndices = nullptr;


  void CreateCFBVH(); 
  void createGPUArrays();
public:
	bool active = true;
  ~objectLoader();
  objectLoader(bool a = true, bool h = true) : active(a), hashable(h) {}
  void appendObject(std::string fileName);
  void appendMesh(mesh);
  mesh mergeMeshes();
  std::size_t hash();

  void tearDownMeshes();

  void reset();

  void buildBVH();
  gpuBVH getGPUArrays();
};