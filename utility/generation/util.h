#pragma once
#include <utility/generation.h>

namespace generation {
// ----------------------------------------------------------------
// Actual functions begin here
// Utility functions
// Used to check if a string starts with a given prefix, useful for OBJ parsing
bool startsWith(std::string mainStr, std::string toMatch);
// Used to parse a string into a list of space seperated values, useful for OBJ parsing
template <typename T> auto readValues(std::string str, std::string prefix = "") {
  if (!startsWith(str, prefix))
    return std::vector<T>{};
  std::string in = str.substr(prefix.length() + 1);
  std::istringstream iss(in);
  std::vector<T> res;
  T temp;
  while (iss >> temp)
    res.push_back(temp); 
  return res;
}
// Returns the minimum and maximum domain parameters, not related to volume information
std::pair<float3, float3> getDomain();
// Calculates the ideal packing ratio for a radius r using Brent's method.
// Return values are: packing, support radius, support radius * k
std::tuple<float, float, float> getPacking(float r);

// Takes the path to an *.obj or *.vdb file and returns the equivalent SDF parsed into a 256^3 CUDA 3D texture
// The fileName argument is parsed through resolveFiles with the configuration directory added.
std::tuple<cudaTextureObject_t, float4, float4, int4> cudaVolume(std::string fileName);
} // namespace generation