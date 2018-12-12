#pragma once
#include <utility/generation.h>

namespace generation{    
// Takes a path to either an *.obj or *.vdb file and returns a set of particles along the mesh surface
std::vector<vdb::Vec4f> ObjToShell(fs::path path, float r, float threshold = 0.75f);
} 