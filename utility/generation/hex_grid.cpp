#pragma once
#include <utility/include_all.h>
#include <utility/generation.h>

namespace generation {
std::vector<vdb::Vec4f> VolumeToHex(vdb::FloatGrid::Ptr grid, vdb::Vec3d min_vdb, vdb::Vec3d max_vdb, float r) {
  vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> sampler(*grid);

  auto v = PI4O3 * math::power<3>(r);
  auto [spacing, h, H] = getPacking(r);
  auto [mind, maxd] = getDomain();

  auto gen_position_offset = [=](auto offset, auto i, auto j, auto k) {
    vdb::Vec4f initial{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)), 2.f * sqrtf(6.f) / 3.f * k,
                       0.f};
    return offset + initial * spacing;
  };
  vdb::Vec4f firstPoint{(float)min_vdb.x() + 2.f * spacing, (float)min_vdb.y() + 2.f * spacing,
                        (float)min_vdb.z() + 2.f * spacing, h};

  std::vector<vdb::Vec4f> out_points;

  int32_t i, j, k;
  i = j = k = 0;
  vdb::Vec4f inserted;
  do {
    do { 
      do {
        inserted = gen_position_offset(firstPoint, i++, j, k);
        if (inserted.x() < mind.x || inserted.y() < mind.y || inserted.z() < mind.z || inserted.x() > maxd.x ||
            inserted.y() > maxd.y || inserted.z() > maxd.z)
          continue;

        if (sampler.wsSample(vdb::Vec3d(inserted.x(), inserted.y(), inserted.z())) < -spacing * 1.25f)
          out_points.push_back(inserted);
      } while (inserted.x() < max_vdb.x() - 2.f * spacing);
      i = 0;
      ++j;
    } while (inserted.y() < max_vdb.y() - 2.f * spacing);
    j = 0;
    k++;
  } while (inserted.z() < max_vdb.z() - 2.f * spacing);

  return out_points;
} 
std::vector<vdb::Vec4f> particlesFromVDBHexagonal(fs::path fileName, float r) {
  auto [grid, min_vdb, max_vdb] = loadVDBFile(fileName);
  return VolumeToHex(grid, min_vdb, max_vdb, r);
}
std::vector<vdb::Vec4f> particlesFromOBJHexagonal(fs::path fileName, float r) {
  auto [grid, min_vdb, max_vdb] = VDBfromOBJ(fileName);
  return VolumeToHex(grid, min_vdb, max_vdb, r);
}
} // namespace generation