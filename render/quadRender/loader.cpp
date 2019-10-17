#include <utility/generation.h>
#include <utility/helpers/pathfinder.h>
#include <utility/helpers/log.h>
#include <render/quadRender/quadRender.h>
#include <render/quadRender/loader.h>
#include <experimental/filesystem>
#include <boost/container_hash/hash.hpp>
#include <fstream>
#include <string>
#include <memory>
#include <utility>
void objectLoader::appendMesh(mesh toAdd) {
	meshes.push_back(toAdd);
	dirty = true;
}

void objectLoader::appendObject(std::string fileName) {
  LOG_INFO << "Started loading object " << fileName << std::endl;
  auto [objVertices, objTriangles, edges, minp, maxp] = generation::fileToObj(resolveFile(fileName, { get<parameters::config_folder>() }));
  auto numVtx = objVertices.size();
  auto numTri = objTriangles.size();
  LOG_INFO << "Vertices:  " << numVtx << std::endl;
  LOG_INFO << "Triangles: " << numTri << std::endl;

  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;

  for (int32_t i = 0; i < numVtx; ++i) {
    vertices.push_back(Vertex{objVertices[i], float3{0.f, 0.f, 0.f}});
  }
  for (int32_t i = 0; i < numTri; ++i) {
    auto [idx1, idx2, idx3] = objTriangles[i];
    triangles.push_back(Triangle(idx1, idx2, idx3, vertices));
  }
  for (auto &v : vertices) {
    v.normal = math::normalize(v.normal);
  }
  for (auto &t : triangles) {
	  t.recalculate(vertices);
  }
  LOG_INFO << "Finished loading object " << fileName << std::endl;
  meshes.push_back(mesh{vertices, triangles});
  dirty = true;
}
mesh objectLoader::mergeMeshes() {
  std::vector<Vertex> mergedVertices;
  std::vector<Triangle> mergedTriangles;
  int32_t offset = 0;
  for (int32_t i = 0; i < meshes.size(); ++i) {
    auto &vtx = meshes[i].vertices;
    auto &tri = meshes[i].triangles;
    for (auto &v : vtx)
      mergedVertices.push_back(v);
    for (auto t : tri) {
      t.i0 += offset;
      t.i1 += offset;
      t.i2 += offset;
      mergedTriangles.push_back(t);
    }
    offset = (int32_t) mergedVertices.size();
  }
  merged = mesh{mergedVertices, mergedTriangles};
  hash();
  dirty = false;
  return merged;
}
namespace boost {
inline auto hash_value(float3 v);
inline auto hash_value(float4 v);
//inline auto hash_value(Vertex v);
//inline auto hash_value(Triangle t);
} // namespace boost
inline auto combine(std::size_t &seed) {}
template <typename T, typename... Ts> auto combine(std::size_t &seed, T arg, Ts &&... args) {
  boost::hash_combine(seed, boost::hash_value(arg));
  combine(seed, std::forward<Ts>(args)...);
}

namespace boost {
inline auto hash_value(float3 v) {
  std::size_t seed = 0x129037;
  ::combine(seed, v.x, v.y, v.z);
  return seed;
}
inline auto hash_value(float4 v) {
	std::size_t seed = 0x129037;
	::combine(seed, v.x, v.y, v.z, v.w);
	return seed;
}


//inline auto hash_value(Vertex v) {
//  std::size_t seed = 0x0983274;
//  ::combine(seed, v.normal, v.position);
//  return seed;
//}
//inline auto hash_value(Triangle t) {
//  std::size_t seed = 0x01892364;
//  ::combine(seed, t.center, t.bottom, t.color, t.normal);
//  ::combine(seed, t.e1, t.e2, t.e3, t.d, t.d2, t.d3);
//  ::combine(seed, t.i0, t.i1, t.i2, t.twoSided);
//  return seed;
//}
} // namespace boost

  inline auto hash_value(Vertex v) {
    std::size_t seed = 0x0983274;
    ::combine(seed, v.normal, v.position);
    return seed;
  }
  inline auto hash_value(Triangle t) {
    std::size_t seed = 0x01892364;
    ::combine(seed, t.center, t.bottom, t.color, t.normal);
    ::combine(seed, t.e1, t.e2, t.e3, t.d, t.d2, t.d3);
    ::combine(seed, t.i0, t.i1, t.i2, t.twoSided);
    return seed;
  }

inline std::size_t objectLoader::hash() {
  if (!dirty)
    return stored_hash;
  if (!hashable)
    return SIZE_MAX;
  stored_hash = boost::hash_range(merged.triangles.begin(), merged.triangles.end());
  boost::hash_range(stored_hash, merged.vertices.begin(), merged.vertices.end());
  return stored_hash;
}
BVHNode *objectLoader::buildBVH(BBoxEntries &work, int depth) {
  if (work.size() < 4) {
    BVHLeaf *leaf = new BVHLeaf;
    BVHnodes.push_back(leaf);
    for (auto &t : work)
      leaf->triangles.push_back(t.triangles);
    return leaf;
  }
  float3 bottom{FLT_MAX, FLT_MAX, FLT_MAX};
  float3 top{-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for(auto& v : work){
    bottom = math::min(bottom, v.bottom);
    top = math::max(top, v.top);
  }
  float side1 = top.x - bottom.x;
  float side2 = top.y - bottom.y;
  float side3 = top.z - bottom.z;

  float minCost = work.size() * (side1 * side2 + side2 * side3 + side3 * side1);

  float bestSplit = FLT_MAX;
  int bestAxis = -1;

  for (int axis = 0; axis < 3; axis++) {
    auto start = math::weak_at(bottom, axis + 1);
    auto stop = math::weak_at(top, axis + 1);

    if (fabsf(stop - start) < 1e-4)
      continue;

    auto step = (stop - start) / (1024.f / (depth + 1.f));
    for (float testSplit = start + step; testSplit < stop - step; testSplit += step) {
      float3 lbottom{FLT_MAX, FLT_MAX, FLT_MAX};
      float3 ltop{-FLT_MAX, -FLT_MAX, -FLT_MAX};

      float3 rbottom{FLT_MAX, FLT_MAX, FLT_MAX};
      float3 rtop{-FLT_MAX, -FLT_MAX, -FLT_MAX};

      int countLeft = 0, countRight = 0;
      for (auto &v : work) {
        float value = math::weak_at(v.center, axis + 1);
        if (value < testSplit) {
          lbottom = math::min(lbottom, v.bottom);
          ltop = math::max(ltop, v.top);
          countLeft++;
        } else {
          rbottom = math::min(rbottom, v.bottom);
          rtop = math::max(rtop, v.top);
          countRight++;
        }
      }
      if (countLeft <= 1 || countRight <= 1)
        continue;

      float lside1 = ltop.x - lbottom.x;
      float lside2 = ltop.y - lbottom.y;
      float lside3 = ltop.z - lbottom.z;

      float rside1 = rtop.x - rbottom.x;
      float rside2 = rtop.y - rbottom.y;
      float rside3 = rtop.z - rbottom.z;

      float surfaceLeft = lside1 * lside2 + lside2 * lside3 + lside3 * lside1;
      float surfaceRight = rside1 * rside2 + rside2 * rside3 + rside3 * rside1;

      float totalCost = surfaceLeft * countLeft + surfaceRight * countRight;

      if (totalCost < minCost) {
        minCost = totalCost;
        bestSplit = testSplit;
        bestAxis = axis;
      }
    }
  }
  if (bestAxis == -1) {
    BVHLeaf *leaf = new BVHLeaf;
    BVHnodes.push_back(leaf);
    for (auto &t : work)
      leaf->triangles.push_back(t.triangles);
    return leaf;
  }

  BBoxEntries left;
  BBoxEntries right;
  float3 lbottom{FLT_MAX, FLT_MAX, FLT_MAX};
  float3 ltop{-FLT_MAX, -FLT_MAX, -FLT_MAX};
  float3 rbottom{FLT_MAX, FLT_MAX, FLT_MAX};
  float3 rtop{-FLT_MAX, -FLT_MAX, -FLT_MAX};

  for (auto &v : work) {
    float value = math::weak_at(v.center, bestAxis + 1);
    if (value < bestSplit) {
      left.push_back(v);
      lbottom = math::min(lbottom, v.bottom);
      ltop = math::max(ltop, v.top);
    } else {
      right.push_back(v);
      rbottom = math::min(rbottom, v.bottom);
      rtop = math::max(rtop, v.top);
    }
  }
  BVHInner *inner = new BVHInner;
  BVHnodes.push_back(inner);

  inner->left = buildBVH(left, depth + 1);
  inner->left->bottom = lbottom;
  inner->left->top = ltop;

  inner->right = buildBVH(right, depth + 1);
  inner->right->bottom = rbottom;
  inner->right->top = rtop;

  return inner;
}
void objectLoader::CreateBVH() {
  std::vector<BBoxTmp> work;
  float3 bottom{FLT_MAX, FLT_MAX, FLT_MAX};
  float3 top{-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for(const auto& triangle : merged.triangles) {
    BBoxTmp b;
    b.triangles = &triangle;
	auto eval = [&](auto idx) {
		b.bottom = math::min(b.bottom, math::castTo<float3>(merged.vertices[idx].position));
		b.top = math::max(b.top, math::castTo<float3>(merged.vertices[idx].position));	
	};
	eval(triangle.i0);
	eval(triangle.i1);
	eval(triangle.i2);

	b.center = (b.top + b.bottom) * 0.5f;

    bottom = math::min(bottom, b.bottom);
    top = math::max(top, b.top);
    work.push_back(b);
  }
  BVHNode *root = buildBVH(work);
  root->bottom = bottom;
  root->top = top;

  BVHroot = root;
}
int32_t objectLoader::boxCount(BVHNode *root) {
  if (!root->leafNode()) {
    BVHInner *p = dynamic_cast<BVHInner *>(root);
    return 1 + boxCount(p->left) + boxCount(p->right);
  } else
    return 1;
}
uint32_t objectLoader::triangleCount(BVHNode *root) {
  if (!root->leafNode()) {
    BVHInner *p = dynamic_cast<BVHInner *>(root);
    return triangleCount(p->left) + triangleCount(p->right);
  } else {
    BVHLeaf *p = dynamic_cast<BVHLeaf *>(root);
    return (unsigned)p->triangles.size();
  }
}
void objectLoader::depthCount(BVHNode *root, int32_t depth, int32_t &maxDepth) {
  if (maxDepth < depth)
    maxDepth = depth;
  if (!root->leafNode()) {
    BVHInner *p = dynamic_cast<BVHInner *>(root);
	depthCount(p->left, depth + 1, maxDepth);
	depthCount(p->right, depth + 1, maxDepth);
  }
}
void objectLoader::PopulateCacheFriendlyBVH(const Triangle *pFirstTriangle, BVHNode *root, uint32_t &idxBoxes,
                                            uint32_t &idxTriList) {
  uint32_t currIdxBoxes = idxBoxes;
  CFBVH[currIdxBoxes].bottom = root->bottom;
  CFBVH[currIdxBoxes].top = root->top;
  if (!root->leafNode()) { // inner node
    BVHInner *p = dynamic_cast<BVHInner *>(root);
    int32_t idxLeft = ++idxBoxes;
    PopulateCacheFriendlyBVH(pFirstTriangle, p->left, idxBoxes, idxTriList);
    int32_t idxRight = ++idxBoxes;
    PopulateCacheFriendlyBVH(pFirstTriangle, p->right, idxBoxes, idxTriList);
    CFBVH[currIdxBoxes].u.inner.idxLeft = idxLeft;
    CFBVH[currIdxBoxes].u.inner.idxRight = idxRight;
  }
  else { // leaf
    BVHLeaf *p = dynamic_cast<BVHLeaf *>(root);
    uint32_t count = (uint32_t)p->triangles.size();
    CFBVH[currIdxBoxes].u.leaf.count =
        0x80000000 | count; // highest bit set indicates a leaf node (inner node if highest bit is 0)
    CFBVH[currIdxBoxes].u.leaf.startIdx = idxTriList;

    for (auto it = p->triangles.begin(); it != p->triangles.end(); it++) {
      triIndices[idxTriList++] = *it - pFirstTriangle;
    }
  }
}
void objectLoader::CreateCFBVH() {
  if (!BVHroot) {
    puts("Internal bug in CreateCFBVH, please report it...");
    fflush(stdout);
    exit(1);
  }

  unsigned idxTriList = 0;
  unsigned idxBoxes = 0;

  triIndicesLength = triangleCount(BVHroot);
  triIndices = std::make_unique<int[]>(triIndicesLength);

  CFBVHLength = boxCount(BVHroot);
  CFBVH = std::make_unique<CacheFriendlyBVHNode[]>(CFBVHLength); // array

  PopulateCacheFriendlyBVH(&merged.triangles[0], BVHroot, idxBoxes, idxTriList);

  if ((idxBoxes != CFBVHLength - 1) || (idxTriList != triIndicesLength)) {
    puts("Internal bug in CreateCFBVH, please report it...");
    fflush(stdout);
    exit(1);
  }

  int maxDepth = 0;
  depthCount(BVHroot, 0, maxDepth);
  if (maxDepth >= BVH_STACK_SIZE) {
    printf("Max depth of BVH was %d\n", maxDepth);
    puts("Recompile with BVH_STACK_SIZE set to more than that...");
    fflush(stdout);
    exit(1);
  }
}
namespace fs = std::experimental::filesystem;
void objectLoader::buildBVH() {
  std::string hash_string = std::to_string(hash());
  LOG_INFO << "Creating BVH for mesh with hash " << hash_string << std::endl;
  fs::path p(hash_string);
  p.replace_extension(".bvh");
  if (!BVHroot) {
    if (/*!hashable || !fs::exists(p)*/ true) {
      // No cached BVH data - we need to calculate them
      LOG_INFO << "Creating BVH for mesh with hash " << hash_string << std::endl;
      CreateBVH();
      // Now that the BVH has been created, copy its data into a more cache-friendly format
      // (CacheFriendlyBVHNode occupies exactly 32 bytes, i.e. a cache-line)
      LOG_INFO << "Creating CFBVH for mesh with hash " << hash_string << std::endl;
      CreateCFBVH();
      if (!hashable) {
        LOG_INFO << "Writing CFBVH to file for hash " << hash_string << std::endl;
        std::ofstream ofs(p, std::ios::out | std::ios::binary);
        auto write = [&](auto &arg, auto size) { ofs.write(reinterpret_cast<char *>(&arg), size); };
        write(CFBVHLength, sizeof(int32_t));
        write(triIndicesLength, sizeof(int32_t));
        write(CFBVH, CFBVHLength * sizeof(CacheFriendlyBVHNode));
        write(triIndices, triIndicesLength * sizeof(int32_t));
        ofs.close();
      }
    }
  } else { // BVH has been built already and stored in a file, read the file
    LOG_INFO << "Reading BVH from file for mehs with hash " << hash_string << std::endl;
    std::ifstream ifs(p, std::ios::in | std::ios::binary);
    auto read = [&](auto &arg, auto size) { ifs.read(reinterpret_cast<char *>(&arg), size); };
    read(CFBVHLength, sizeof(int32_t));
    read(triIndicesLength, sizeof(int32_t));
    CFBVH = std::make_unique<CacheFriendlyBVHNode[]>(CFBVHLength);
	triIndices = std::make_unique<int32_t[]>(triIndicesLength);
    read(CFBVH[0], CFBVHLength * sizeof(CacheFriendlyBVHNode));
    read(triIndices[0], triIndicesLength * sizeof(int32_t));
    ifs.close();
  }
  createGPUArrays();
}
void objectLoader::createGPUArrays() {
	if (!active)
		return;
  auto allocAndCopy = [&](auto &ptr, auto &data) {
    using T = typename std::decay_t<decltype(data)>::value_type;
    cudaMalloc((void **)&ptr, data.size() * sizeof(T));
    cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
  };

  std::vector<TriangleIntersectionData> intersectData;
  std::vector<BVHLimits> limits;
  std::vector<uint4> indexlist;
  for (const auto &t : merged.triangles) {
    TriangleIntersectionData current;
    current.center = float4{t.center.x, t.center.y, t.center.z, t.twoSided ? 1.f : 0.f};
    current.normal = float4{t.normal.x, t.normal.y, t.normal.z, t.d};
    current.e1d1 = float4{t.e1.x, t.e1.y, t.e1.z, t.d1};
    current.e2d2 = float4{t.e2.x, t.e2.y, t.e2.z, t.d2};
    current.e3d3 = float4{t.e3.x, t.e3.y, t.e3.z, t.d3};
    intersectData.push_back(current);
  }
  for (int32_t i = 0; i < CFBVHLength; i++) {
	  limits.push_back(BVHLimits{ CFBVH[i].top, CFBVH[i].bottom });
	  auto u = CFBVH[i].u;
	  indexlist.push_back(uint4{ (uint32_t)u.leaf.count, (uint32_t)u.inner.idxRight, (uint32_t)u.inner.idxLeft,
								(uint32_t)u.leaf.startIdx });
  }

  cudaMalloc((void **)&cuTriIndices, triIndicesLength * sizeof(int));
  cudaMemcpy(cuTriIndices, triIndices.get(), triIndicesLength * sizeof(int), cudaMemcpyHostToDevice);

  allocAndCopy(cuTriIntersectionData, intersectData);
  allocAndCopy(cuVertices, merged.vertices);
  allocAndCopy(cuTriangles, merged.triangles);
  allocAndCopy(cuBVHLimits, limits);
  allocAndCopy(cuBVHIndices, indexlist);
}
objectLoader::~objectLoader() {
  auto freeCUDA = [&](auto ptr) {
    if (ptr != nullptr)
      cudaFree(ptr);
  };
  freeCUDA(cuVertices);
  freeCUDA(cuTriangles);
  freeCUDA(cuTriIntersectionData);
  freeCUDA(cuTriIndices);
  freeCUDA(cuBVHLimits);
  freeCUDA(cuBVHIndices);
  for (auto &alloc : BVHnodes)
    free(alloc);
}
void objectLoader::reset() {
	auto freeCUDA = [&](auto ptr) {
		if (ptr != nullptr)
			cudaFree(ptr);
	};
	freeCUDA(cuVertices);
	freeCUDA(cuTriangles);
	freeCUDA(cuTriIntersectionData);
	freeCUDA(cuTriIndices);
	freeCUDA(cuBVHLimits);
	freeCUDA(cuBVHIndices);
	for (auto &alloc : BVHnodes)
		free(alloc);
	meshes.clear();
	merged.vertices.clear();
	merged.triangles.clear();
	BVHnodes.clear();
	BVHroot = nullptr;
	triIndicesLength = 0;
	triIndices = nullptr;
	CFBVHLength = 0;
	CFBVH = nullptr;
	cuVertices = nullptr;
	cuTriangles = nullptr;
	cuTriIntersectionData = nullptr;
	cuTriIndices = nullptr;
	cuBVHLimits = nullptr;
	cuBVHIndices = nullptr;
	dirty = true;
	stored_hash = 0x0;
}

gpuBVH objectLoader::getGPUArrays() {
	return gpuBVH{ active, cuVertices, cuTriangles , cuBVHIndices , cuBVHLimits , cuTriIntersectionData,
	cuTriIndices, triIndicesLength , CFBVHLength , (int32_t)merged.vertices.size() , (int32_t)merged.triangles.size() };
}

void objectLoader::tearDownMeshes() {
	int32_t vtxCounter = (int32_t)merged.vertices.size();
	for (auto& t : merged.triangles) {
		auto vtx0 = merged.vertices[t.i0];
		auto vtx1 = merged.vertices[t.i1];
		auto vtx2 = merged.vertices[t.i2];

		auto n0 = math::normalize(vtx0.normal);
		auto n1 = math::normalize(vtx1.normal);
		auto n2 = math::normalize(vtx2.normal);
	
		auto threshold = 1.f / 12.f * CUDART_PI_F;
		//threshold = 0.f;
		 
		auto angle0 = acosf(math::dot3(n0, t.normal));
		auto angle1 = acosf(math::dot3(n1, t.normal));
		auto angle2 = acosf(math::dot3(n2, t.normal));
		if (angle0 > threshold) {
			t.i0 = vtxCounter++;
			merged.vertices.push_back(Vertex{ vtx0.position, t.normal });
		}
		if (angle1 > threshold) {
			t.i1 = vtxCounter++;
			merged.vertices.push_back(Vertex{ vtx1.position, t.normal });
		}
		if (angle2 > threshold) {
			t.i2 = vtxCounter++;
			merged.vertices.push_back(Vertex{ vtx2.position, t.normal });
		}
	}
}