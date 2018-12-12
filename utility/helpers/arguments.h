#pragma once
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <utility/math.h>
#include <utility/template/metafunctions.h>

namespace arguments {

HAS_STATIC_MEMBER(rear_ptr);

template <typename T> std::string type_name() {
  if constexpr (std::is_same<T, float4>::value)
    return "float4";
  if constexpr (std::is_same<T, float3>::value)
    return "float3";
  if constexpr (std::is_same<T, float2>::value)
    return "float2";
  if constexpr (std::is_same<T, uint4>::value)
    return "uint4";
  if constexpr (std::is_same<T, uint3>::value)
    return "uint3";
  if constexpr (std::is_same<T, uint2>::value)
    return "uint2";
  if constexpr (std::is_same<T, float>::value)
    return "float";
  if constexpr (std::is_same<T, uint32_t>::value)
    return "uint";
  if constexpr (std::is_same<T, int32_t>::value)
    return "int";
  return typeid(T).name();
}

void loadbar(unsigned int x, unsigned int n, uint32_t w = 50, std::ostream &io = std::cout);

struct cmd {
  std::vector<std::string> jsons;
  
  std::string snapFile;
  bool snapped = false;

  static cmd &instance();
  bool headless;

  std::chrono::high_resolution_clock::time_point start;
  boost::program_options::variables_map vm;

  bool init(bool console, int argc, char *argv[]);

  void finalize();

  void parameter_stats();

  boost::program_options::variables_map &getVM();

  bool end_simulation_frame = false;
  int32_t timesteps = 50;

  bool end_simulation_time = false;
  double time_limit = 50.0;

  bool pause_simulation_time = false;
  double time_pause = 50.0;

  bool timers = true;
};

} // namespace arguments
