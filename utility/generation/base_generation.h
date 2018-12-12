#pragma once
#include <texture_types.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility/helpers/pathfinder.h>
#include <utility/mathv2.h>

#ifdef WIN32
#pragma warning(push, 0)
#endif
#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/ValueAccessor.h>
#ifdef WIN32
#pragma warning(pop)
#endif

enum struct genTechnique { hex_grid, square_grid, shell };

namespace generation {
// Helper namespace alias to reduce noise from experimental namespaces
namespace fs = std::experimental::filesystem;
namespace vdb = openvdb;
namespace vdbt = openvdb::tools;
namespace vdbm = openvdb::math;
// Helper struct for handling triangles when dealing with meshes
struct Triangle {
  int32_t i0, i1, i2;
};
// Helper struct for handling edges when dealing with meshes, used for shell generation
struct Edge {
  int32_t start, end;
};
// Helper struct required for shell generation
struct sort_Edge { 
  float3 start;
  float3 end;
  sort_Edge(const float3 v1, const float3 v2);
  bool operator<(const sort_Edge &rhs) const;
};
// Helper alias defines for shorter return types
using grid_tuple = std::tuple<vdb::FloatGrid::Ptr, vdb::Vec3d, vdb::Vec3d>;
using obj_tuple = std::tuple<std::vector<float3>, std::vector<Triangle>, std::vector<Edge>, float3, float3>;
using render_tuple =
    std::tuple<std::vector<float3>, std::vector<float3>, std::vector<Triangle>, std::vector<Edge>, float3, float3>;

} // namespace generation