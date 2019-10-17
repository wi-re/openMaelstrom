#include <IO/config/config.h>
#include <IO/config/initialize.h>
#include <IO/config/parser.h>
#include <IO/config/snapshot.h>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <utility/helpers/arguments.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>
#include <utility/template/tuple_for_each.h>
#include <utility/volumeBullet.h>

template<typename Ty>
struct logUniform{
	void operator()() {
		if constexpr (!Ty::visible)
			return;
		auto json = Ty::jsonName;
		logger(log_level::info) << json << " = " << IO::config::convertToString(*Ty::ptr) << std::endl;
	}
};
template<typename P>
struct readArray {
	void operator()(std::string name, std::shared_ptr<char> buffer, std::size_t allocSize, std::vector<std::tuple<std::string, std::size_t, void*>>& arrs) {
		if (P::qualifiedName == name) {
			void *ptr = malloc(allocSize);
			memcpy(ptr, buffer.get(), allocSize);
			arrs.push_back(std::make_tuple(P::qualifiedName, allocSize, ptr));
			//std::cout << "array: " << name << " eS=" << elemSize << ", aS=" << allocSize
			//          << ", elements = " << allocSize / elemSize << std::endl;
		}
	}
};
template<typename Ty>
struct parseUniform {
	void operator()(boost::property_tree::ptree& pt) {
		using namespace IO;
		using namespace IO::config;
		if constexpr (Ty::visible == true) {
			//std::cout << "Parsing " << Ty::jsonName << std::endl;
			parse<Ty>(pt);
		}
	}
};
template<typename Ty>
struct storeUniform {
	void operator()(boost::property_tree::ptree& pt) {
		using namespace IO;
		using namespace IO::config;
		if constexpr (Ty::visible == true)
			parseStore<Ty>(pt);
	}
};



void IO::config::show_config() {
	//callOnTypes<logUniform>(uniforms_list);
  iterateParameters([](auto x) {
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
  std::vector < std::pair < std::string, std::string>> cmdArguments;
  for (auto j : arguments::cmd::instance().jsons) {
    std::vector<std::string> splitVec;
    boost::split(splitVec, j, boost::is_any_of("="), boost::token_compress_on);
    if (splitVec.size() == 2) {
      pt.put(splitVec[0], splitVec[1]);
	  cmdArguments.push_back(std::make_pair(splitVec[0], splitVec[1]));
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

	std::shared_ptr<char> buffer(new char[allocSize], std::default_delete<char[]>());
    //auto buffer = std::make_shared<char>(allocSize);
    iss.read(buffer.get(), allocSize);
    if (allocSize != elemSize) { // read an array
		callOnTypes<readArray>(arrays_list, name, buffer, allocSize, arrs);

  iterateArraysList([&](auto x){
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
  iterateParameters([&](auto x){
    using P = std::decay_t<decltype(x)>;
    if(P::jsonName == name){
	//	std::cout << P::jsonName << " -> ";
      int32_t llen = 0;
      iss.read(reinterpret_cast<char *>(&llen), sizeof(llen));
	  //std::cout << ": " << llen << " -> ";
      std::string val = "";
      val.resize(llen);
      iss.read(val.data(), llen * sizeof(char));
	  //std::cout << val << std::endl;
	  bool found = false;
	  for (const auto&[key, val] : cmdArguments) {
		  if (key == P::jsonName)
			  found = true;
	  }
	  if (name.find("$") != std::string::npos) return;
	  if(!found)
          pt.put(P::jsonName, val);
    }
      });
    }
  }
  defaultRigidAllocate();
  }
 // std::ostringstream oss;
 // boost::property_tree::json_parser::write_json(oss, pt);

 // std::string inifile_text = oss.str();
 // std::cout << inifile_text << std::endl;

 // std::cout << "Done reading dump file" << std::endl;
  //try {
	 // callOnTypes<parseUniform>(uniforms_list, pt);
  //}
  //catch (std::exception e) {
	 // std::cerr << e.what() << std::endl;
	 // throw;
  //}
  iterateParameters([&pt](auto x) {
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

  initVolumeBoundary();
  
  iterateArraysList([&](auto x) {
	  using P = decltype(x);
	  auto it = find_if(begin(arrs), end(arrs), [](const auto& e) {
		  return get<std::string>(e) == P::qualifiedName;
	  });
	  if (it == end(arrs)) return;
	  auto& alloc = *it;
	  if (P::qualifiedName == std::get<std::string>(alloc) && P::ptr != nullptr && P::valid())
		  memcpy(P::ptr, std::get<void*>(alloc), std::get<std::size_t>(alloc));
  });
  bt::World::instance().resetWorld();

 // for(auto& alloc : arrs){
 // for_each_r(arrays_list, [&](auto x){
 //   using P = std::decay_t<decltype(x)>;
	////std::cout << std::get<std::string>(alloc) << std::endl;
 //   if(P::qualifiedName == std::get<std::string>(alloc) && P::ptr != nullptr && P::valid())
 //       memcpy(P::ptr, std::get<void*>(alloc), std::get<std::size_t>(alloc));
 //   });
 // }


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
//  callOnTypes<storeUniform>(uniforms_list, pt2);
  iterateParameters([&pt2](auto x) {
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

#include <utility/volumeBullet.h>
void IO::config::load_snapshot() {
	if (has_snap) {
		for (auto &ptr : snaps)
			ptr->load();
		if(get<parameters::volumeBoundaryCounter>() > 0)
		bt::World::instance().resetWorld();
	}
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
