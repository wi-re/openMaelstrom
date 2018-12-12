#pragma once
#include <utility/include_all.h>
#include <utility/generation.h>

namespace generation {
bool startsWith(std::string mainStr, std::string toMatch) {
  if (mainStr.find(toMatch) == 0)
    return true;
  else
    return false;
}
std::pair<float3, float3> getDomain() {
  return std::make_pair(get<parameters::min_domain>(), get<parameters::max_domain>());
}
std::tuple<float, float, float> getPacking(float r) {
  auto volume = PI4O3 * math::power<3>(r);
  auto h = support_from_volume(volume);
  auto H = h * ::kernelSize();

  auto gen_position = [&](auto r, int32_t i, int32_t j, int32_t k) {
    float4 initial{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)), 2.f * sqrtf(6.f) / 3.f * k, h / r};
    return initial * r; 
  };

  auto spacing = math::brentsMethod(
      [=](auto r) {
        float density = -1.0f;
        int32_t requiredSlices_x = (int32_t)math::ceilf(h / r);
        int32_t requiredSlices_y = (int32_t)math::ceilf(h / (sqrtf(3.f) * r));
        int32_t requiredSlices_z = (int32_t)math::ceilf(h / r * 3.f / (sqrtf(6.f) * 2.f));
        density = -1.f;
        float4 center_position{0.f, 0.f, 0.f, h};
        for (int32_t x_it = -requiredSlices_x; x_it <= requiredSlices_x; x_it++)
          for (int32_t y_it = -requiredSlices_y; y_it <= requiredSlices_y; y_it++)
            for (int32_t z_it = -requiredSlices_z; z_it <= requiredSlices_z; z_it++)
              density += volume * spline4_kernel(center_position, gen_position(r, x_it, y_it, z_it));
        return density;
      },
      r * 0.1f, r * 4.f, 1e-6f, 10000);

  return std::make_tuple(spacing, h, H);
}

std::tuple<cudaTextureObject_t, float4, float4, int4> cudaVolume(std::string fileName) {
  auto path = resolveFile(fileName, {get<parameters::config_folder>()});
  auto std_file = fs::path(path);
  std_file.replace_extension(".vdb_tmp");
  float4 *host_mem = nullptr;
  vdb::Vec3d min, max;
  int4 dimension;

  if (!fs::exists(std_file) || fs::last_write_time(path) > fs::last_write_time(std_file)) {
    logger(log_level::info) << "Creating TEMP VDB file: " << std_file << std::endl;
    auto [levelSet, min_vdb, max_vdb] = fileToVDB(path);
    vdb::Vec3DGrid::Ptr normalSet = vdb::Vec3dGrid::create();
    normalSet->setTransform(vdbm::Transform::createLinearTransform(0.5));
    normalSet->setName("NormalsLS");

    vdb::FloatGrid::ConstAccessor c_accessor = levelSet->getConstAccessor();
    vdb::Vec3dGrid::Accessor norm_accessor = normalSet->getAccessor();
    vdbm::UniformScaleMap map(0.5);

    vdb::FloatGrid::ConstAccessor naccessor = levelSet->getConstAccessor();
    vdbt::GridSampler<vdb::FloatGrid::ConstAccessor, vdbt::BoxSampler> normalSampler(naccessor, levelSet->transform());

    vdbm::ScaleMap scalemap(levelSet->voxelSize());
    auto transformer = [&](auto iter, auto accessor) {
      vdb::FloatGrid::ConstAccessor c_accessor = levelSet->getConstAccessor();
      vdbm::Vec3d isGrad = vdbm::ISGradient<vdbm::FD_HJWENO5>::result(c_accessor, iter.getCoord());
      vdbm::Vec3d wsGrad = scalemap.applyIJT(isGrad);
      wsGrad = vdbm::Gradient<vdbm::UniformScaleMap, vdbm::CD_6TH>::result(map, c_accessor, iter.getCoord());

      vdb::Coord c = iter.getCoord();
      auto sample = [&](int32_t x, int32_t y, int32_t z) {
        // c_accessor.getValue
        return c_accessor.getValue(openvdb::Coord(c.x() + x, c.y() + y, c.z() + z));
      };

      vdb::Coord c2(0, 0, 0);
      float max = -FLT_MAX;
      for (int32_t x = -1; x <= 1; ++x)
        for (int32_t y = -1; y <= 1; ++y)
          for (int32_t z = -1; z <= 1; ++z) {
            if (abs(x) + abs(y) + abs(z) == 3)
              continue;
            float s = sample(x, y, z);
            if (s > max) {
              c2 = vdb::Coord(x, y, z);
              max = s;
            }
          }

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
        vdb::CoordBBox bbox;
        iter.getBoundingBox(bbox);
        accessor.getTree()->fill(bbox, wsGrad);
      }
    };
    vdbt::transformValues(levelSet->cbeginValueOn(), *normalSet, transformer);
    dimension = int4{256, 256, 256, 0};
    vdb::CoordBBox box = levelSet->evalActiveVoxelBoundingBox();
    min = levelSet->indexToWorld(box.getStart());
    max = levelSet->indexToWorld(box.getEnd());

    vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> d_sampler(*levelSet);
    vdbt::GridSampler<vdb::Vec3dGrid, vdbt::BoxSampler> n_sampler(*normalSet);

    host_mem = (float4 *)malloc(dimension.x * dimension.y * dimension.z * sizeof(float4));

    auto getWs = [&](uint32_t x, uint32_t y, uint32_t z) {
      vdb::Vec3d pos = min;
      vdb::Vec3d diff = max - min;
      vdb::Vec3d offs{((double)x) / ((double)dimension.x), ((double)y) / ((double)dimension.y),
                      ((double)z) / ((double)dimension.z)};
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

    std::ofstream processed_vdb(std_file.string(), std::ios::out | std::ios::binary);
    processed_vdb.write(reinterpret_cast<char *>(&dimension), sizeof(int4));
    processed_vdb.write(reinterpret_cast<char *>(&min), sizeof(vdb::Vec3d));
    processed_vdb.write(reinterpret_cast<char *>(&max), sizeof(vdb::Vec3d));
    processed_vdb.write(reinterpret_cast<char *>(host_mem), dimension.x * dimension.y * dimension.z * sizeof(float4));
    processed_vdb.close();
  } else {
    logger(log_level::info) << "Loading TEMP VDB file: " << std_file << std::endl;
    std::ifstream processed_vdb(std_file.string(), std::ios::in | std::ios::binary);
    processed_vdb.read(reinterpret_cast<char *>(&dimension), sizeof(int4));
    processed_vdb.read(reinterpret_cast<char *>(&min), sizeof(vdb::Vec3d));
    processed_vdb.read(reinterpret_cast<char *>(&max), sizeof(vdb::Vec3d));
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
  memset(&copyParams, 0x00, sizeof(cudaMemcpy3DParms));
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
  auto min_f = float4{static_cast<float>(min.x()), static_cast<float>(min.y()), static_cast<float>(min.z()), 0.f};
  auto max_f = float4{static_cast<float>(max.x()), static_cast<float>(max.y()), static_cast<float>(max.z()), 0.f};
  free(host_mem);
  return std::make_tuple(texture, min_f, max_f, dimension);
}
} // namespace generation