#define NEW_STYLE
#define BOOST_USE_WINDOWS_H
//#define NO_OPERATORS
#include <utility/include_all.h>
#include <SPH/generation/volumeInlets.cuh>
#ifdef WIN32
#pragma warning(push, 0)
#endif
#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/ValueAccessor.h>
#ifdef WIN32
#pragma warning(pop)
#endif
#include <fstream>
#include <utility/template/for_struct.h>

void SPH::streamInlet::init(Memory) {
  std::vector<float4_u<SI::m>> generated_particles;
  std::vector<float4_u<SI::velocity>> velocities;

  for (auto &fluidVolume : get<parameters::inlet_volumes>()) {
    std::string vdbFile = get<parameters::config_folder>() + fluidVolume.fileName.value;
    openvdb::initialize();
    openvdb::io::File file(vdbFile);
    file.open();
    auto grids = file.getGrids();
    openvdb::GridBase::Ptr baseGrid;
    for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
      if (nameIter.gridName() == "surface")
        baseGrid = file.readGrid(nameIter.gridName());
    file.close();
    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
	openvdb::Vec3d min_vdb = grid->indexToWorld(box.getStart());
	openvdb::Vec3d max_vdb = grid->indexToWorld(box.getEnd());

	float_u<SI::m> radius{ fluidVolume.inlet_radius.value };
    if (radius <= 0.0_m)
		radius = get_u<parameters::radius>();
    fluidVolume.inlet_radius.value = radius.val;

    auto particleVolume = PI4O3 * math::power<3>(radius);
	auto H = support_from_volume(particleVolume);

    auto gen_position_offset = [](float3_u<SI::m> offset, float_u<SI::m> r, int32_t i, int32_t j, int32_t k) {
      float3_u<> initial{(float)i, (float)j, (float)k};
      return offset + initial * r;
    };
	auto spacing = math::power<ratio<1, 3>>(particleVolume) * 1.05f;
    
    openvdb::CoordBBox indexBB(openvdb::Coord(-50, -50, -20), openvdb::Coord(50, 50, 80));
	float3_u<SI::m> minVDB{ (float)min_vdb.x(),(float)min_vdb.y(), (float)min_vdb.z() };
	float3_u<SI::m> maxVDB{ (float)max_vdb.x(),(float)max_vdb.y(), (float)max_vdb.z() };
    int32_t i, j, k;
    i = j = k = 0;
	auto firstPoint = minVDB;
	float3_u<SI::m> inserted{ firstPoint.val.x,firstPoint.val.y,firstPoint.val.z };
    std::vector<float3_u<SI::m>> out_points;

    int32_t max_i = 0, max_j = 0, max_k = 0;
    do {
      do {
        do {
          inserted = gen_position_offset(firstPoint, spacing, i++, j, k);
		  
          if (sampler.wsSample(openvdb::Vec3d(inserted.val.x, inserted.val.y, inserted.val.z)) < 0.f)
            out_points.push_back(inserted);
          max_i = std::max(max_i, i);
          max_j = std::max(max_j, j);
          max_k = std::max(max_k, k);

        } while (inserted.val.x < maxVDB.val.x);
        i = 0;
        ++j;
      } while (inserted.val.y < maxVDB.val.y);
      j = 0;
      k++;
    } while (inserted.val.z < maxVDB.val.z);

    for (auto p : out_points)
		generated_particles.push_back(float4_u<SI::m>{p.val.x, p.val.y, p.val.z, H.val});
    auto vel = fluidVolume.emitter_velocity.value;
    velocities.push_back(float4_u<SI::velocity>{vel.x, vel.y, vel.z, vel.w});
    fluidVolume.particles_emitted.value = (int32_t) out_points.size();
  }
  if (generated_particles.size() > 0) {
    info<arrays::inletPositions>().allocate(generated_particles.size() * sizeof(float4));
    cuda::memcpy(get<arrays::inletPositions>(), generated_particles.data(), info<arrays::inletPositions>().alloc_size,
               cudaMemcpyHostToDevice);
    info<arrays::inletCounter>().allocate(sizeof(int32_t));
  }
}
