#pragma once
#include <utility/generation.h>

namespace generation {
// Takes vdb grid information and returns a particle set using a dense cubic grid sampling
std::vector<vdb::Vec4f> VolumeToRegular(vdb::FloatGrid::Ptr grid, vdb::Vec3d min_vdb, vdb::Vec3d max_vdb, float r);
// Wrapper function which takes a path to a *.vdb or *.obj file
std::vector<vdb::Vec4f> particlesFromVDBRegular(fs::path fileName, float r);
std::vector<vdb::Vec4f> particlesFromOBJRegular(fs::path fileName, float r);
} // namespace generation 