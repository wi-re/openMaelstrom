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
#include <utility/generation.h>

void SPH::streamInlet::init(Memory) {
  std::vector<float4_u<SI::m>> generated_particles;
  std::vector<float4_u<SI::velocity>> velocities;

  for (auto &fluidVolume : get<parameters::inletVolumes>()) {
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

    auto particleVolume = PI4O3 * math::power<3>(radius)* 1.f / get<parameters::adaptive::resolution>();
	auto H = support_from_volume(particleVolume);

    auto gen_position_offset = [](float3_u<SI::m> offset, float_u<SI::m> r, int32_t i, int32_t j, int32_t k) {
      float3_u<> initial{(float)i, (float)j, (float)k};
      return offset + initial * r;
    };
	auto spacing = math::power<ratio<1, 3>>(particleVolume) * 1.0f;
    
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

   // for (auto p : out_points)
	//	generated_particles.push_back(float4_u<SI::m>{p.val.x, p.val.y, p.val.z, H.val});

	//auto r = get<parameters::radius>() * 1.f / powf(get<parameters::adaptive::resolution>() / 8.f,1.f/3.f);
	auto r = get<parameters::radius>();
	auto volume = PI4O3 * math::power<3>(r);
	auto gPtcls = generation::generateParticles(fluidVolume.fileName.value, r, genTechnique::square_grid);
	//{
	//	using namespace generation;

	//	vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> sampler(*grid);

	//	auto v = PI4O3 * math::power<3>(r);
	//	auto[spacing, h, H] = getPacking(r);
	//	spacing *= 1.1f;
	//	auto[mind, maxd] = getDomain();

	//	auto gen_position_offset = [=, spacing = spacing](auto offset, auto i, auto j, auto k) {
	//		generation::vdb::Vec4f initial{ 2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)), 2.f * sqrtf(6.f) / 3.f * k,
	//						   0.f };
	//		return offset + initial * spacing;
	//	};
	//	vdb::Vec4f firstPoint{ (float)min_vdb.x() + 2.f * spacing, (float)min_vdb.y() + 2.f * spacing,
	//						  (float)min_vdb.z() + 2.f * spacing, h };

	//	std::vector<vdb::Vec4f> out_points;

	//	int32_t i, j, k;
	//	i = j = k = 0;
	//	vdb::Vec4f inserted;
	//	do {
	//		do {
	//			do {
	//				inserted = gen_position_offset(firstPoint, i++, j, k);
	//				if (inserted.x() < mind.x || inserted.y() < mind.y || inserted.z() < mind.z || inserted.x() > maxd.x ||
	//					inserted.y() > maxd.y || inserted.z() > maxd.z)
	//					continue;

	//				if (sampler.wsSample(vdb::Vec3d(inserted.x(), inserted.y(), inserted.z())) < -spacing * 1.25f)
	//					out_points.push_back(inserted);
	//			} while (inserted.x() < max_vdb.x() - 2.f * spacing);
	//			i = 0;
	//			++j;
	//		} while (inserted.y() < max_vdb.y() - 2.f * spacing);
	//		j = 0;
	//		k++;
	//	} while (inserted.z() < max_vdb.z() - 2.f * spacing);
	//	//gPtcls = out_points;

	//}
	//{
	//	using namespace generation;
	//	vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> sampler(*grid);

	//	auto v = PI4O3 * math::power<3>(r);
	//	auto[spacing, h, H] = getPacking(r);
	//	spacing = math::power<ratio<1, 3>>(v) * 1.3334f;

	//	auto[mind, maxd] = getDomain();

	//	auto gen_position_offset = [=, spacing = spacing](auto offset, int32_t i, int32_t j, int32_t k) {
	//		generation::vdb::Vec4f initial{ (float)i, (float)j, (float)k, 0.f };
	//		return offset + initial * spacing;
	//	};

	//	vdb::Vec4f firstPoint{ (float)min_vdb.x() + 2.f * spacing, (float)min_vdb.y() + 2.f * spacing,
	//						  (float)min_vdb.z() + 2.f * spacing, h };

	//	std::vector<vdb::Vec4f> out_points;

	//	int32_t i, j, k;
	//	i = j = k = 0;
	//	vdb::Vec4f inserted;
	//	do {
	//		do {
	//			do {
	//				inserted = gen_position_offset(firstPoint, i++, j, k);
	//				if (inserted.x() < mind.x || inserted.y() < mind.y || inserted.z() < mind.z || inserted.x() > maxd.x ||
	//					inserted.y() > maxd.y || inserted.z() > maxd.z)
	//					continue;

	//				if (sampler.wsSample(vdb::Vec3d(inserted.x(), inserted.y(), inserted.z())) < -0.f)
	//					out_points.push_back(inserted);
	//			} while (inserted.x() < max_vdb.x() - 2.f * spacing);
	//			i = 0;
	//			++j;
	//		} while (inserted.y() < max_vdb.y() - 2.f * spacing);
	//		j = 0;
	//		k++;
	//	} while (inserted.z() < max_vdb.z() - 2.f * spacing);
	//	gPtcls = out_points;
	//}
	auto inserted_particles = (int32_t)gPtcls.size();
	for (auto p : gPtcls)
		generated_particles.push_back(float4_u<SI::m>{p.x(), p.y(), p.z(), H.val});


    auto vel = fluidVolume.emitter_velocity.value;
    velocities.push_back(float4_u<SI::velocity>{vel.x, vel.y, vel.z, vel.w});
    fluidVolume.particles_emitted.value = (int32_t)generated_particles.size();
  }
  if (generated_particles.size() > 0) {
    info<arrays::inletPositions>().allocate(generated_particles.size() * sizeof(float4));
    cuda::memcpy(get<arrays::inletPositions>(), generated_particles.data(), info<arrays::inletPositions>().alloc_size,
               cudaMemcpyHostToDevice);
    info<arrays::inletCounter>().allocate(sizeof(int32_t));
  }
}
