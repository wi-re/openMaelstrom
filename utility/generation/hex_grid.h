#pragma once
#include <utility/generation.h>

namespace generation {
// Grid to particle methods
// Takes vdb grid information and returns a particle set using a dense hexagonal grid sampling
std::vector<vdb::Vec4f> VolumeToHex(vdb::FloatGrid::Ptr grid, vdb::Vec3d min_vdb, vdb::Vec3d max_vdb, float r);
// Wrapper function which takes a path to a *.vdb or *.obj  file
std::vector<vdb::Vec4f> particlesFromVDBHexagonal(fs::path fileName, float r);
std::vector<vdb::Vec4f> particlesFromOBJHexagonal(fs::path fileName, float r);
} // namespace generation 