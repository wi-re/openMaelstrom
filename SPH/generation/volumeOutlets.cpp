// As this module uses openVDB which can conflict with the custom math operators used for cuda we
// disable them here.
#define NEW_STYLE
//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <SPH/generation/volumeOutlets.cuh>
#include <utility/include_all.h>
#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/ValueAccessor.h>
#ifdef _WIN32
#pragma warning(pop)
#endif
// At some point this will have to be replaced with <filesystem>
#include <fstream>
#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

// This function is used to load the vdb files from disk and transforms them into cuda 3d textures.
void SPH::Outlet::init(Memory mem) {
  if (!valid(mem)) {
    return;
  }

  auto vols = get<parameters::outlet_volumes>();

  std::vector<cudaTextureObject_t> textures;
  std::vector<float4> minAABB;
  std::vector<float4> maxAABB;
  std::vector<int4> dims;
  std::vector<float> rates;

  for (auto boundaryVolume : vols) {
    std::string vdbFile = get<parameters::config_folder>() + boundaryVolume.fileName.value;
    auto std_file = fs::path(vdbFile);
    auto tmp_file = std_file;
    tmp_file.replace_extension(".vdb_tmp");

    float4 *host_mem = nullptr;
    openvdb::Vec3d min, max;
    int4 dimension;
    if (!fs::exists(tmp_file) || fs::last_write_time(std_file) > fs::last_write_time(tmp_file)) {
      logger(log_level::info) << "Parsing VDB file: " << std_file;
      openvdb::initialize();
      openvdb::io::File file(vdbFile);

      openvdb::FloatGrid::Ptr levelSet = openvdb::FloatGrid::create();
      levelSet->setTransform(openvdb::math::Transform::createLinearTransform(0.5));
      levelSet->setGridClass(openvdb::GRID_LEVEL_SET);
      levelSet->setName("LevelSet");

      openvdb::Vec3DGrid::Ptr normalSet = openvdb::Vec3dGrid::create();
      normalSet->setTransform(openvdb::math::Transform::createLinearTransform(0.5));
      normalSet->setName("NormalsLS");

      file.open();
      auto grids = file.getGrids();
      openvdb::GridBase::Ptr baseGrid;
      for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
        if (nameIter.gridName() == "surface") {
          baseGrid = file.readGrid(nameIter.gridName());
          // std::cout << "Found Grid" << std::endl;
        }
      }
      file.close();

      levelSet = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
      openvdb::FloatGrid::ConstAccessor c_accessor = levelSet->getConstAccessor();
      openvdb::Vec3dGrid::Accessor norm_accessor = normalSet->getAccessor();
      openvdb::math::UniformScaleMap map(0.5);

      openvdb::FloatGrid::ConstAccessor naccessor = levelSet->getConstAccessor();
      openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler> normalSampler(naccessor,
                                                                                                               levelSet->transform());

      openvdb::math::ScaleMap scalemap(levelSet->voxelSize());
      auto transformer = [&](auto iter, auto accessor) {
        openvdb::FloatGrid::ConstAccessor c_accessor = levelSet->getConstAccessor();
        openvdb::math::Vec3d isGrad = openvdb::math::ISGradient<openvdb::math::FD_HJWENO5>::result(c_accessor, iter.getCoord());
        openvdb::math::Vec3d wsGrad = scalemap.applyIJT(isGrad);
        wsGrad = openvdb::math::Gradient<openvdb::math::UniformScaleMap, openvdb::math::CD_6TH>::result(map, c_accessor, iter.getCoord());

        openvdb::Coord c = iter.getCoord();
        auto sample = [&](int32_t x, int32_t y, int32_t z) {
          // c_accessor.getValue
          return c_accessor.getValue(openvdb::Coord(c.x() + x, c.y() + y, c.z() + z));
        };

        openvdb::Coord c2(0, 0, 0);
        float max = -FLT_MAX;
        for (int32_t x = -1; x <= 1; ++x)
          for (int32_t y = -1; y <= 1; ++y)
            for (int32_t z = -1; z <= 1; ++z) {
              if (abs(x) + abs(y) + abs(z) == 3)
                continue;
              float s = sample(x, y, z);
              if (s > max) {
                c2 = openvdb::Coord(x, y, z);
                max = s;
              }
            }
        // wsGrad = openvdb::math::Vec3d(c2.x(), c2.y(), c2.z());

        float central_x = sample(1, 0, 0) - 2 * sample(0, 0, 0) + sample(-1, 0, 0);
        float central_y = sample(0, 1, 0) - 2 * sample(0, 0, 0) + sample(0, -1, 0);
        float central_z = sample(0, 0, 1) - 2 * sample(0, 0, 0) + sample(0, 0, -1);
        central_x = abs(central_x) > abs(central_y) && abs(central_x) > abs(central_z) ? central_x : 0.f;
        central_y = abs(central_y) > abs(central_x) && abs(central_y) > abs(central_z) ? central_y : 0.f;
        central_z = abs(central_z) > abs(central_y) && abs(central_z) > abs(central_x) ? central_z : 0.f;

        wsGrad.normalize();
        if (iter.isVoxelValue()) { // set a single voxel
          accessor.setValue(iter.getCoord(), wsGrad);
        } else { // fill an entire tile
          openvdb::CoordBBox bbox;
          iter.getBoundingBox(bbox);
          accessor.getTree()->fill(bbox, wsGrad);
        }
      };
      openvdb::tools::transformValues(levelSet->cbeginValueOn(), *normalSet, transformer);
      dimension = int4{256, 256, 256, 0};
      openvdb::CoordBBox box = levelSet->evalActiveVoxelBoundingBox();
      min = levelSet->indexToWorld(box.getStart());
      max = levelSet->indexToWorld(box.getEnd());

      openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> d_sampler(*levelSet);
      openvdb::tools::GridSampler<openvdb::Vec3dGrid, openvdb::tools::BoxSampler> n_sampler(*normalSet);

      host_mem = (float4 *)malloc(dimension.x * dimension.y * dimension.z * sizeof(float4));

      auto getWs = [&](uint32_t x, uint32_t y, uint32_t z) {
        openvdb::Vec3d pos = min;
        openvdb::Vec3d diff = max - min;
        openvdb::Vec3d offs{((double)x) / ((double)dimension.x), ((double)y) / ((double)dimension.y), ((double)z) / ((double)dimension.z)};
        return pos + diff * offs;
      };
      auto getSample = [&](uint32_t x, uint32_t y, uint32_t z) {
        auto ws = getWs(x, y, z);
        auto d = d_sampler.wsSample(ws);
        auto n = n_sampler.wsSample(ws);
        return float4{(float)n.x(), (float)n.y(), (float)n.z(), d};
      };

      for (int32_t x = 0; x < dimension.x; ++x)
        for (int32_t y = 0; y < dimension.y; ++y)
          for (int32_t z = 0; z < dimension.z; ++z)
            host_mem[x + dimension.x * (y + dimension.y * z)] = getSample(x, y, z);

      std::ofstream processed_vdb(tmp_file.string(), std::ios::out | std::ios::binary);
      processed_vdb.write(reinterpret_cast<char *>(&dimension), sizeof(int4));
      processed_vdb.write(reinterpret_cast<char *>(&min), sizeof(openvdb::Vec3d));
      processed_vdb.write(reinterpret_cast<char *>(&max), sizeof(openvdb::Vec3d));
      processed_vdb.write(reinterpret_cast<char *>(host_mem), dimension.x * dimension.y * dimension.z * sizeof(float4));
      processed_vdb.close();
    } else {
      logger(log_level::info) << "Loading TEMP VDB file: " << tmp_file << std::endl;
      std::ifstream processed_vdb(tmp_file.string(), std::ios::in | std::ios::binary);
      processed_vdb.read(reinterpret_cast<char *>(&dimension), sizeof(int4));
      processed_vdb.read(reinterpret_cast<char *>(&min), sizeof(openvdb::Vec3d));
      processed_vdb.read(reinterpret_cast<char *>(&max), sizeof(openvdb::Vec3d));
      host_mem = (float4 *)malloc(dimension.x * dimension.y * dimension.z * sizeof(float4));
      processed_vdb.read(reinterpret_cast<char *>(host_mem), dimension.x * dimension.y * dimension.z * sizeof(float4));
      processed_vdb.close();
    }
    cudaTextureDesc textureDescription;
    auto channelDesc = cudaCreateChannelDesc<float4>();
    memset(&textureDescription, 0, sizeof(cudaTextureDesc));
    textureDescription.normalizedCoords = true;
    textureDescription.filterMode = cudaFilterModeLinear;
    textureDescription.addressMode[0] = cudaAddressModeClamp; // clamp
    textureDescription.addressMode[1] = cudaAddressModeClamp;
    textureDescription.addressMode[2] = cudaAddressModeClamp;
    textureDescription.readMode = cudaReadModeElementType;

    cudaArray *arr_ptr;
    cudaPitchedPtr pitched_ptr;
    cudaTextureObject_t texture;

    cudaMalloc3DArray(&arr_ptr, &channelDesc, make_cudaExtent(dimension.x, dimension.y, dimension.y));
    pitched_ptr = make_cudaPitchedPtr(host_mem, dimension.x * sizeof(float4), dimension.x, dimension.y);

    cudaMemcpy3DParms copyParams;
    memset(&copyParams,0x00,sizeof(cudaMemcpy3DParms));
    copyParams.srcPtr = pitched_ptr;
    copyParams.dstArray = arr_ptr;
    copyParams.extent = make_cudaExtent(dimension.x, dimension.y, math::weak_get<3>(dimension));
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr_ptr;

    cudaCreateTextureObject(&texture, &texRes, &textureDescription, NULL);

    textures.push_back(texture);
    minAABB.push_back(float4{static_cast<float>(min.x()), static_cast<float>(min.y()), static_cast<float>(min.z()), 0.f});
    maxAABB.push_back(float4{static_cast<float>(max.x()), static_cast<float>(max.y()), static_cast<float>(max.z()), 0.f});
    dims.push_back(dimension);
    rates.push_back(boundaryVolume.flowRate.value);
    free(host_mem);
  }
  if (textures.size() > 0) {
    using dimensions = decltype(info<arrays::volumeOutletDimensions>());
    using volumes = decltype(info<arrays::volumeOutletVolumes>());
    using boundaryMins = decltype(info<arrays::volumeOutletMin>());
    using boundaryMaxs = decltype(info<arrays::volumeOutletMax>());
    using rate = decltype(info<arrays::volumeOutletRate>());
    using rateAcc = decltype(info<arrays::volumeOutletRateAccumulator>());

    get<parameters::volumeOutletCounter>() = (int32_t) textures.size();

    dimensions::allocate(sizeof(int4) * textures.size());
    volumes::allocate(sizeof(cudaTextureObject_t) * textures.size());
    boundaryMins::allocate(sizeof(float4) * textures.size());
    boundaryMaxs::allocate(sizeof(float4) * textures.size());
    rate::allocate(sizeof(float) * textures.size());
    rateAcc::allocate(sizeof(float) * textures.size());

#ifdef UNIFIED_MEMORY
	for (uint32_t i = 0; i < textures.size(); ++i) {
		dimensions::ptr[i] = dims[i];
		volumes::ptr[i] = textures[i];
		boundaryMins::ptr[i] = minAABB[i];
		boundaryMaxs::ptr[i] = maxAABB[i];
        rate::ptr[i] = rates[i];
        rateAcc::ptr[i] = 0.f;
  }
#else
	cudaMemcpy(dimensions::ptr, dims.data(), sizeof(decltype(dims)::value_type) * dims.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(volumes::ptr, textures.data(), sizeof(decltype(textures)::value_type) * textures.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryMins::ptr, minAABB.data(), sizeof(decltype(minAABB)::value_type) * minAABB.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryMaxs::ptr, maxAABB.data(), sizeof(decltype(maxAABB)::value_type) * maxAABB.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(rate::ptr, rates.data(), sizeof(decltype(rates)::value_type) * rates.size(), cudaMemcpyHostToDevice);
	cudaMemset(rateAcc::ptr,0x00,sizeof(float) * textures.size());
#endif
  }
  else {
	  using dimensions = decltype(info<arrays::volumeOutletDimensions>());
	  using volumes = decltype(info<arrays::volumeOutletVolumes>());
	  using boundaryMins = decltype(info<arrays::volumeOutletMin>());
	  using boundaryMaxs = decltype(info<arrays::volumeOutletMax>());
      using rate = decltype(info<arrays::volumeOutletRate>());
      using rateAcc = decltype(info<arrays::volumeOutletRateAccumulator>());

	  get<parameters::volumeOutletCounter>() = 0;

	  dimensions::allocate(sizeof(int4) * 1);
	  volumes::allocate(sizeof(cudaTextureObject_t) * 1);
	  boundaryMins::allocate(sizeof(float4) * 1);
	  boundaryMaxs::allocate(sizeof(float4) * 1);
      rate::allocate(sizeof(float) * 1);
      rateAcc::allocate(sizeof(float) * 1);
  }
}