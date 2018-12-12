#include <utility/helpers/exporter.h>

void IO::dumpAll(std::string filename) {
  std::ofstream dump_file;
  dump_file.open(filename, std::ios::binary);
  for_each_r(uniforms_list, [&](auto x){
      writeParameter<decltype(x)>(dump_file);
      });
  for_each_r(arrays_list, [&](auto x){
    using P = decltype(x);
    memory_kind k = P::kind;
    if(k == memory_kind::particleData)
        writeArray<decltype(x)>(dump_file);
});
}