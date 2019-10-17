#include <utility/include_all.h>
#include <utility/generation.h>

#define FILE_FORMAT_VERSION 0x1000A

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
      r * 0.01f, r * 4.f, 1e-6f, 10000);

  return std::make_tuple(spacing, h, H);
}

std::tuple<cudaTextureObject_t, float4, float4, int4, float4, Matrix4x4> cudaVolume(std::string fileName) {
	auto path = resolveFile(fileName, { get<parameters::config_folder>() });
	auto std_file = fs::path(path);
	std_file.replace_extension(".vdb_tmp");

	static std::map<std::string, std::tuple<cudaTextureObject_t, float4, float4, int4, float4, Matrix4x4>> buffer;
	if (buffer.find(std_file.string()) != buffer.end()) {
		return buffer[std_file.string()];
	}
  float4 *host_mem = nullptr;
  vdb::Vec3d min, max;
  int4 dimension;
  float volume;
  float4 centerOfMass;
  Matrix4x4 inertiaMatrix;
  bool readFromTmp = false;
  bool tmpFileExists = fs::exists(std_file) && fs::last_write_time(path) < fs::last_write_time(std_file);
  if (tmpFileExists) {
	  readFromTmp = true;
	  LOG_INFO << "Loading TEMP VDB file: " << std_file << std::endl;
	  std::ifstream processed_vdb(std_file.string(), std::ios::in | std::ios::binary);
	  int32_t marker = 0;
	  processed_vdb.read(reinterpret_cast<char *>(&marker), sizeof(int));
	  if (marker != 0) readFromTmp = false;
	  else {
		  processed_vdb.read(reinterpret_cast<char *>(&marker), sizeof(int));
		  if (marker != FILE_FORMAT_VERSION) readFromTmp = false;
		  else {
			  processed_vdb.read(reinterpret_cast<char *>(&dimension), sizeof(int4));
			  processed_vdb.read(reinterpret_cast<char *>(&min), sizeof(vdb::Vec3d));
			  processed_vdb.read(reinterpret_cast<char *>(&max), sizeof(vdb::Vec3d));
			  host_mem = (float4 *)malloc(dimension.x * dimension.y * dimension.z * sizeof(float4));
			  processed_vdb.read(reinterpret_cast<char *>(host_mem), dimension.x * dimension.y * dimension.z * sizeof(float4));
			  processed_vdb.read(reinterpret_cast<char *>(&volume), sizeof(float));
			  processed_vdb.read(reinterpret_cast<char *>(&centerOfMass), sizeof(float4));
			  processed_vdb.read(reinterpret_cast<char *>(inertiaMatrix.data), sizeof(Matrix4x4));
			  processed_vdb.close();
		  }
	  }
  }
  if (!readFromTmp) {
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
    auto transformer = [&, levelSet = levelSet ](auto iter, auto accessor) {
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
      //central_x = abs(central_x) > abs(central_y) && abs(central_x) > abs(central_z) ? central_x : 0.f;
      //central_y = abs(central_y) > abs(central_x) && abs(central_y) > abs(central_z) ? central_y : 0.f;
      //central_z = abs(central_z) > abs(central_y) && abs(central_z) > abs(central_x) ? central_z : 0.f;

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
	  n.normalize();
      return float4{(float)n.x(), (float)n.y(), (float)n.z(), d};
    };
	vdb::Vec3d diff = max - min;
	float3 dX{ (float)diff.x() / (float)dimension.x,(float)diff.y() / (float)dimension.y ,(float)diff.z() / (float)dimension.z };
	float dV = dX.x * dX.y * dX.z;
	float sumVolume = 0.f;
	float4 sumPosition{ 0.f,0.f,0.f,0.f };
	float samples = 0.f;
    for (int32_t x = 0; x < dimension.x; ++x)
      for (int32_t y = 0; y < dimension.y; ++y)
		  for (int32_t z = 0; z < dimension.z; ++z) {
			  auto sample = getSample(x, y, z);
			  host_mem[x + dimension.x * (y + dimension.y * z)] = sample;
		  }
	int3 dimensionCenters{ 256,256,256 };
	inertiaMatrix = Matrix4x4::identity();
	inertiaMatrix(0, 0) = 0.f;
	inertiaMatrix(1, 1) = 0.f;
	inertiaMatrix(2, 2) = 0.f;
	for (int32_t x = 0; x < dimensionCenters.x; ++x)
		for (int32_t y = 0; y < dimensionCenters.y; ++y)
			for (int32_t z = 0; z < dimensionCenters.z; ++z) {
				auto sample = getSample(x, y, z);
				if (sample.w < 0.f) {
					samples += 1.f;
					sumVolume += dV;
					vdb::Vec3d pos = min;
					vdb::Vec3d diff = max - min;
					vdb::Vec3d offs{ ((double)x) / ((double)dimensionCenters.x), ((double)y) / ((double)dimensionCenters.y),
									((double)z) / ((double)dimensionCenters.z) };
					auto ws = pos + diff * offs;
					sumPosition += float4{ (float)ws.x(), (float)ws.y(), (float)ws.z(), 0.f };

					float x = (float)ws.x();
					float y = (float)ws.y();
					float z = (float)ws.z();

					inertiaMatrix(0, 0) += (y * y + z * z) * dV;
					inertiaMatrix(1, 1) += (x * x + z * z) * dV;
					inertiaMatrix(2, 2) += (x * x + y * y) * dV;

					inertiaMatrix(0, 1) += -(x * y) * dV;
					inertiaMatrix(1, 0) += -(x * y) * dV;

					inertiaMatrix(0, 2) += -(x * z) * dV;
					inertiaMatrix(2, 0) += -(y * z) * dV;

					inertiaMatrix(2, 1) += -(y * z) * dV;
					inertiaMatrix(1, 2) += -(y * z) * dV;
				}
			}
	std::cout << fileName << std::endl;
	std::cout << "Min: " << min << std::endl;
	std::cout << "Max: " << max << std::endl;
	std::cout << "Summed volume: " << sumVolume << std::endl;
	std::cout << "Summed position: " << sumPosition / samples << std::endl;
	sumPosition /= samples;
	//min += openvdb::Vec3d(sumPosition.x, sumPosition.y, sumPosition.z);
	//max += openvdb::Vec3d(sumPosition.x, sumPosition.y, sumPosition.z);
	//sumPosition = float4{ 0.f,0.f,0.f,0.f };
    std::ofstream processed_vdb(std_file.string(), std::ios::out | std::ios::binary);
	int32_t marker = 0;
	processed_vdb.write(reinterpret_cast<char *>(&marker), sizeof(int));
	marker = FILE_FORMAT_VERSION;
	processed_vdb.write(reinterpret_cast<char *>(&marker), sizeof(int));
    processed_vdb.write(reinterpret_cast<char *>(&dimension), sizeof(int4));
    processed_vdb.write(reinterpret_cast<char *>(&min), sizeof(vdb::Vec3d));
    processed_vdb.write(reinterpret_cast<char *>(&max), sizeof(vdb::Vec3d));
    processed_vdb.write(reinterpret_cast<char *>(host_mem), dimension.x * dimension.y * dimension.z * sizeof(float4));
	processed_vdb.write(reinterpret_cast<char *>(&sumVolume),sizeof(float));
	processed_vdb.write(reinterpret_cast<char *>(&sumPosition), sizeof(float4));
	processed_vdb.write(reinterpret_cast<char *>(inertiaMatrix.data), sizeof(Matrix4x4));
    processed_vdb.close();
	centerOfMass = sumPosition;
	volume = sumVolume;
  } 
  centerOfMass.w = volume;
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
  auto min_f = float4{static_cast<float>(min.x()), static_cast<float>(min.y()), static_cast<float>(min.z()), 0.f} - centerOfMass;
  auto max_f = float4{static_cast<float>(max.x()), static_cast<float>(max.y()), static_cast<float>(max.z()), 0.f} -centerOfMass;
  free(host_mem);
  //std::cout << fileName << ": " << std::endl;
  //std::cout << "Inertia Matrix" << std::endl;
  //inertiaMatrix.print();
  //std::cout << "Inertia Matrix Inverse" << std::endl;
  //inertiaMatrix.inverse().print();
  auto tup = std::make_tuple(texture, min_f, max_f, dimension, centerOfMass, inertiaMatrix);
  buffer[std_file.string()] = tup;
  cuda::sync("Created volume");
  return tup;
}
} // namespace generation