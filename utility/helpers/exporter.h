#pragma once
#include <fstream>
#include <utility/include_all.h>
#include <IO/config/parser.h>


namespace IO {
template<typename P>
auto writeArray(std::ostream& oss){
    if(!P::valid() || P::ptr == nullptr)
      return;
  	if (
		!(
			P::kind == memory_kind::particleData	||
			P::kind == memory_kind::rigidData		||
			P::kind == memory_kind::spareData		||
			P::kind == memory_kind::individualData
		)
		) return;
    
    std::string str(P::qualifiedName);
    auto size = (int32_t) str.size();
    oss.write(reinterpret_cast<char*>(&size), sizeof(size));
    oss.write(str.c_str(), size);
    std::size_t elemSize = sizeof(typename P::type);
    std::size_t allocSize = P::alloc_size;
    oss.write(reinterpret_cast<char*>(&elemSize), sizeof(elemSize));
    oss.write(reinterpret_cast<char*>(&allocSize), sizeof(allocSize));

    using T = typename P::type;
    auto p = P::ptr;
    T* ptr = new T[allocSize / elemSize];
    cudaDeviceSynchronize();
    //std::cout << P::qualifiedName << "= " << ptr << " : " << p << " -> " << allocSize << std::endl;
    cudaMemcpy(ptr, p, allocSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    oss.write(reinterpret_cast<char*>(ptr), allocSize);
    std::string comp = P::variableName;
    if (comp.compare("rigidFiles") != 0) 
      delete[] ptr;
}
template<typename P>
auto writeParameter(std::ostream& oss){
    std::string str(P::jsonName);
    auto size = (int32_t) str.size();
    oss.write(reinterpret_cast<char*>(&size), sizeof(size));
    oss.write(str.c_str(), size);
    std::size_t elemSize = sizeof(typename P::type);
    oss.write(reinterpret_cast<char*>(&elemSize), sizeof(elemSize));
    oss.write(reinterpret_cast<char*>(&elemSize), sizeof(elemSize));
    oss.write(reinterpret_cast<char*>(P::ptr), sizeof(typename P::type));
    std::string val = IO::config::convertToString(*P::ptr);
    auto vsize = (int32_t) val.size();
    oss.write(reinterpret_cast<char*>(&vsize), sizeof(vsize));
    oss.write(val.c_str(), vsize);
}
template<typename P, typename M>
auto writeArray(M&& memory, std::ostream& oss){
    if(!P::valid() || P::ptr == nullptr)
      return;
    std::string str(P::variableName);
    auto size = (int32_t) str.size();
    oss.write(reinterpret_cast<char*>(&size), sizeof(size));
    oss.write(str.c_str(), size);
    std::size_t elemSize = sizeof(typename P::type);
    std::size_t allocSize = P::alloc_size;
    oss.write(reinterpret_cast<char*>(&elemSize), sizeof(elemSize));
    oss.write(reinterpret_cast<char*>(&allocSize), sizeof(allocSize));

    using T = typename P::type;
    auto p = P::get_member(memory);
    T* ptr = new T[allocSize / elemSize];
    cudaDeviceSynchronize();
    //std::cout << P::variableName << "= " << ptr << " : " << p << " -> " << allocSize << std::endl;
    cudaMemcpy(ptr, p, allocSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    oss.write(reinterpret_cast<char*>(ptr), allocSize);
    delete[] ptr;
}


template <typename P> auto dumpMemory(P &&memory, std::string filename) {
  using T = std::decay_t<P>;
  std::ofstream dump_file;
  dump_file.open(filename, std::ios::binary);
  for_each_r(typename T::parameters{}, [&](auto x) { writeParameter<decltype(x)>(dump_file); });
  for_each_r(typename T::basic_info_params{}, [&](auto x) { writeParameter<decltype(x)>(dump_file); });
  for_each_r(typename T::virtual_info_params{}, [&](auto x) { writeParameter<decltype(x)>(dump_file); });
  for_each_r(typename T::boundaryInfo_params{}, [&](auto x) { writeParameter<decltype(x)>(dump_file); });
  for_each_r(typename T::cell_info_params{}, [&](auto x) { writeParameter<decltype(x)>(dump_file); });

  for_each_r(typename T::virtual_info_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::boundaryInfo_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::cell_info_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::neighbor_info_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::input_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::output_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::swap_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });
  for_each_r(typename T::temporary_arrays{}, [&](auto x) { writeArray<decltype(x)>(memory, dump_file); });

  if (memory.resort == true) {
    //for_each_r(sorting_list, [&](auto x) { writeArray<decltype(x)>(dump_file); });
  }
}

void dumpAll(std::string filename);

} // namespace IO
