#include <IO/config/config.h>
#include <IO/config/initialize.h>
#include <IO/config/parser.h>
#include <IO/config/snapshot.h>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <utility/helpers/arguments.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>
#include <utility/template/tuple_for_each.h>

void IO::config::show_config() {
  for_each(uniforms_list, [](auto x) {
    if constexpr (!decltype(x)::visible)
      return;
    auto json = decltype(x)::jsonName;
    logger(log_level::info) << json << " = " << convertToString(*x.ptr) << std::endl;
  });
}

void IO::config::load_config(std::string filepath) {
  std::string folder = filepath.substr(0, filepath.find_last_of(R"(/\)") + 1);

  std::stringstream ss;
  std::ifstream file(filepath);
  ss << file.rdbuf();
  file.close();

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(ss, pt);

  for (auto j : arguments::cmd::instance().jsons) {
    std::vector<std::string> splitVec;
    boost::split(splitVec, j, boost::is_any_of("="), boost::token_compress_on);
    if (splitVec.size() == 2) {
      pt.put(splitVec[0], splitVec[1]);
    }
  }

  std::ifstream iss;
  std::vector<std::tuple<std::string, std::size_t, void*>> arrs;
  if(arguments::cmd::instance().snapped){
  iss.open(arguments::cmd::instance().snapFile, std::ios::binary);
  while (iss.good()) {
    int32_t size;
    std::string name;
    iss.read(reinterpret_cast<char *>(&size), sizeof(size));
    name.resize(size);
    iss.read(&name[0], size * sizeof(char));
    std::size_t elemSize, allocSize;
    iss.read(reinterpret_cast<char *>(&elemSize), sizeof(elemSize));
    iss.read(reinterpret_cast<char *>(&allocSize), sizeof(allocSize));
    auto buffer = std::make_unique<char[]>(allocSize);
    iss.read(buffer.get(), allocSize);
    if (allocSize != elemSize) { // read an array
  for_each_r(arrays_list, [&](auto x){
    using P = std::decay_t<decltype(x)>;
    if(P::variableName == name){
          void *ptr = malloc(allocSize);
          memcpy(ptr, buffer.get(), allocSize);
          arrs.push_back(std::make_tuple(P::variableName, allocSize, ptr));
          //std::cout << "array: " << name << " eS=" << elemSize << ", aS=" << allocSize
          //          << ", elements = " << allocSize / elemSize << std::endl;
    }
});
    } else {
  for_each_r(uniforms_list, [&](auto x){
    using P = std::decay_t<decltype(x)>;
    if(P::variableName == name){
      int32_t llen;
      iss.read(reinterpret_cast<char *>(&llen), sizeof(llen));
      std::string val;
      val.resize(llen);
      iss.read(&val[0], llen * sizeof(char));
          //std::cout << "param: " << name << " eS=" << elemSize << ", aS=" << allocSize << " : " << val << std::endl;
          pt.put(P::jsonName, val);
    }
      });
    }
  }
  }

  for_each(uniforms_list, [&pt](auto x) {
    if constexpr (decltype(x)::visible == true)
      parse<decltype(x)>(pt);
  });

  get<parameters::config_file>() = filepath;
  get<parameters::config_folder>() = folder;

  if (get<parameters::max_numptcls>() == 0)
    get<parameters::max_numptcls>() = get<parameters::num_ptcls>();

  initKernel();

  initDomain();

  initBoundary();

  initParameters();

  defaultAllocate();
  for(auto& alloc : arrs){
  for_each_r(arrays_list, [&](auto x){
    using P = std::decay_t<decltype(x)>;
    if(P::variableName == std::get<std::string>(alloc) && P::ptr != nullptr && P::valid())
        memcpy(P::ptr, std::get<void*>(alloc), std::get<std::size_t>(alloc));
    });
  }


  initSnapshot();
  for(auto& alloc : arrs){
    free(std::get<void*>(alloc));
  }
}

void IO::config::save_config(std::string filepath) {
  std::string folder = filepath.substr(0, filepath.find_last_of(R"(/\)") + 1);

  std::stringstream ss;
  std::ofstream file(filepath);

  boost::property_tree::ptree pt2;
  for_each(uniforms_list, [&pt2](auto x) {
    if constexpr (decltype(x)::visible == true)
      parseStore<decltype(x)>(pt2);
  });
  boost::property_tree::write_json(file, pt2);
  file.close();
}

std::vector<IO::config::SnapShot *> IO::config::snaps;
bool has_snap = false;

void IO::config::take_snapshot() {
  for (auto &ptr : snaps)
    ptr->save();
  has_snap = true;
}

void IO::config::load_snapshot() {
  if (has_snap)
    for (auto &ptr : snaps)
      ptr->load();
}

void IO::config::clear_snapshot() {
  for (auto &ptr : snaps)
    ptr->clear();
  has_snap = false;
}

bool IO::config::has_snapshot() { return has_snap; }

std::string IO::config::bytesToString(size_t bytes) {
  double b = static_cast<double>(bytes);
  std::stringstream sstream;
  if (b < 1024.0)
    sstream << b << "B";
  else {
    sstream << std::setprecision(2) << std::fixed;
    if (b < 1024.0 * 1024.0)
      sstream << b / 1024.0 << "KiB";
    else if (b < 1024.0 * 1024.0 * 1024.0)
      sstream << b / 1024.0 / 1024.0 << "MiB";
    else if (b < 1024.0 * 1024.0 * 1024.0 * 1024.0)
      sstream << b / 1024.0 / 1024.0 / 1024.0 << "GiB";
  }
  return sstream.str();
}

std::string IO::config::numberToString(double b) {
  std::stringstream sstream;
  if (abs(b) < 1000.0 && abs(b) > 1.000)
    sstream << b;
  else if (abs(b) > 1000.0) {
    sstream << std::setprecision(2) << std::fixed;
    if (abs(b) < 1000.0 * 1000.0)
      sstream << b / 1000.0 << "K";
    else if (abs(b) < 1000.0 * 1000.0 * 1000.0)
      sstream << b / 1000.0 / 1000.0 << "M";
    else if (abs(b) < 1000.0 * 1000.0 * 1000.0 * 1000.0)
      sstream << b / 1000.0 / 1000.0 / 1000.0 << "G";
  } else if (abs(b) < 1.000) {
    sstream << std::setprecision(2) << std::fixed;
    if (abs(b) < 1.0 / 1000.0 / 1000.0)
      sstream << b / 1000.0 << "m";
    else if (abs(b) < 1000.0 / 1000.0 / 1000.0)
      sstream << b / 1000.0 / 1000.0 << "u";
    else if (abs(b) < 1000.0 / 1000.0 / 1000.0 / 1000.0)
      sstream << b / 1000.0 / 1000.0 / 1000.0 << "n";
  }
  return sstream.str();
}
