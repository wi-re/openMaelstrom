#define BOOST_USE_WINDOWS_H
#include <SPH/boundary/volumeBoundary.cuh>
#include <utility/generation.h>
#include <utility/include_all.h>
// At some point this will have to be replaced with <filesystem>
#include <fstream>
// This function is used to load the vdb files from disk and transforms them into cuda 3d textures.
void SPH::volume::init_volumes(Memory mem) {
  if (!valid(mem)) {
    return;
  }
  auto vols = get<parameters::boundary_volumes>();

  std::vector<cudaTextureObject_t> textures;
  std::vector<float4> minAABB;
  std::vector<float4> maxAABB;
  std::vector<int4> dims;

  for (auto boundaryVolume : vols) {
    auto [texture, min, max, dimension] = generation::cudaVolume(boundaryVolume.fileName.value);
    textures.push_back(texture);
    minAABB.push_back(min);
    maxAABB.push_back(max);
    dims.push_back(dimension);
  }
  if (textures.size() > 0) {
    using dimensions = decltype(info<arrays::volumeBoundaryDimensions>());
    using volumes = decltype(info<arrays::volumeBoundaryVolumes>());
    using boundaryMins = decltype(info<arrays::volumeBoundaryMin>());
    using boundaryMaxs = decltype(info<arrays::volumeBoundaryMax>());

    get<parameters::volumeBoundaryCounter>() = (int32_t)textures.size();

    dimensions::allocate(sizeof(int4) * textures.size());
    volumes::allocate(sizeof(cudaTextureObject_t) * textures.size());
    boundaryMins::allocate(sizeof(float4) * textures.size());
    boundaryMaxs::allocate(sizeof(float4) * textures.size());

#ifdef UNIFIED_MEMORY
    for (uint32_t i = 0; i < textures.size(); ++i) {
      dimensions::ptr[i] = dims[i];
      volumes::ptr[i] = textures[i];
      boundaryMins::ptr[i] = minAABB[i];
      boundaryMaxs::ptr[i] = maxAABB[i];
    }
#else
    cudaMemcpy(dimensions::ptr, dims.data(), sizeof(decltype(dims)::value_type) * dims.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(volumes::ptr, textures.data(), sizeof(decltype(textures)::value_type) * textures.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(boundaryMins::ptr, minAABB.data(), sizeof(decltype(minAABB)::value_type) * minAABB.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(boundaryMaxs::ptr, maxAABB.data(), sizeof(decltype(maxAABB)::value_type) * maxAABB.size(),
               cudaMemcpyHostToDevice);
#endif
  } else {
    using dimensions = decltype(info<arrays::volumeBoundaryDimensions>());
    using volumes = decltype(info<arrays::volumeBoundaryVolumes>());
    using boundaryMins = decltype(info<arrays::volumeBoundaryMin>());
    using boundaryMaxs = decltype(info<arrays::volumeBoundaryMax>());

    get<parameters::volumeBoundaryCounter>() = 0;

    dimensions::allocate(sizeof(int4) * 1);
    volumes::allocate(sizeof(cudaTextureObject_t) * 1);
    boundaryMins::allocate(sizeof(float4) * 1);
    boundaryMaxs::allocate(sizeof(float4) * 1);
  }
}