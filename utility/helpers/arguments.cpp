#include <IO/config/config.h>
#include <IO/config/parser.h>
#include <boost/format.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <random>
#include <sstream>
#include <thread>
#include <utility/MemoryManager.h>
#include <utility/helpers/arguments.h>
#include <utility/helpers/log.h>
#include <utility/helpers/timer.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/template/tuple_for_each.h>
#include <utility/macro.h>
#include <utility/helpers/pathfinder.h>
#include <utility/MemoryManager.h>

thread_local int3 h_threadIdx;
thread_local int3 h_blockDim;
thread_local int3 h_blockIdx;

struct param_stats_base {
  virtual std::string to_string(size_t field_len = 5) = 0;
  virtual void sample() = 0;
  virtual std::string name() = 0;
};
std::string pad(std::string str, const size_t num = 9, const char paddingChar = ' ') {
  if (num > str.size())
    str.insert(0, num - str.size(), paddingChar);
  if (str.size() > num) {
    str = str.substr(0, num);
  }
  return str;
}

template <typename param> struct param_stats : public param_stats_base {
  using T = typename param::type;

  std::vector<T> samples;

  std::string to_string(size_t field_len = 5) {
    if (samples.size() == 0)
      return "";
    std::vector<T> sam = samples;
    std::nth_element(sam.begin(), sam.begin() + sam.size() / 2, sam.end(),
                     [](auto lhs, auto rhs) { return lhs < rhs; });
    auto median = sam[sam.size() / 2];

    T min = samples[0];
    T max = samples[0];
    T avg{};
    double ctr = static_cast<double>(samples.size());

    for (auto s : samples) {
      min = math::min(s, min);
      max = math::max(s, max);
      avg += s;
    }
    avg /= static_cast<T>(ctr);
    T stddev{};
    for (auto s : samples) {
      auto diff = (s - avg) * (s - avg);
      stddev += diff;
    }
    stddev /= static_cast<T>(ctr) - static_cast<T>(1.0);
    stddev = static_cast<T>(sqrt(stddev));

    std::stringstream sstream;
    sstream << std::setw(field_len) << param::variableName;
    sstream << " ";
    sstream << pad(IO::config::convertToString(avg)) << " ";
    sstream << pad(IO::config::convertToString(median)) << " ";
    sstream << pad(IO::config::convertToString(min)) << " ";
    sstream << pad(IO::config::convertToString(max)) << " ";
    sstream << pad(IO::config::convertToString(stddev)) << " ";

    // sstream << std::endl;
    return sstream.str();
  }

  void sample() { samples.push_back(get<param>()); }

  std::string name() { return param::variableName; }
};
std::vector<param_stats_base *> param_watchlist;

void arguments::loadbar(unsigned int x, unsigned int n, uint32_t w, std::ostream &io) {
  if ((x != n) && (x % (n / 100 + 1) != 0))
    return;

  float ratio = x / (float)n;
  uint32_t c = static_cast<uint32_t>(ratio * w);

  io << std::setw(3) << (uint32_t)(ratio * 100) << "% [";
  for (uint32_t x = 0; x < c; x++)
    io << "=";
  for (uint32_t x = c; x < w; x++)
    io << " ";
  io << "]";

  static auto start_time = std::chrono::high_resolution_clock::now();

  auto current_time = std::chrono::high_resolution_clock::now();

  io << "\r"
     << boost::format("%8.2f") %
            std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

  float tpp =
      std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() / ratio;

  io << "s of est. " << boost::format("%8.2f") % tpp << "s" << std::flush;
}

arguments::cmd &arguments::cmd::instance() {
  static cmd c;
  return c;
}
bool arguments::cmd::init(bool console, int argc, char *argv[]) {
  headless = console;
  start = std::chrono::high_resolution_clock::now();
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
	  ("snap,s", po::value<std::string>(), "produce help message")
	  ("help,h", "produce help message")
	  ("frames,f", po::value<int>(), "set frame limiter")
	  ("time,t", po::value<double>(),"set time limiter")
	  ("pause", po::value<double>(),"set time limiter")
	  ("verbose,v", "louden the logger on the console")
	  ("no-timer,nt","silence the timer output on the console")
	  ("memory,mem", "print memory consumption at end")
	  ("info,i", "print total runtime at end")
	  ("log", po::value<std::string>(), "log to file")
	  ("config", po::value<std::string>(),"specify config file to open")
	  ("config_id,c", po::value<int>(), "load known config with given id")
	  ("option,o", po::value<int>(), "load optional parameters from config")
	  ("list,l", "list all known ids")("params,p", "watch parameters and print them")
	  ("record,r", po::value<std::string>(), "log to file")
	  ("config_log", po::value<std::string>(), "save the config at the end to file")
	  ("neighbordump", po::value<std::string>(),"writes a copy of the neighborlist at the end to file")
	  ("json,j", po::value<std::vector<std::string>>(&jsons)->multitoken(),"writes a copy of the neighborlist at the end to file")
	  ("deviceRegex,G", po::value<std::string>(), "regex to force functions to be called on device")
	  ("hostRegex,H", po::value<std::string>(), "regex to force functions to be called on host")
	  ("debugRegex,D", po::value<std::string>(), "regex to force functions to be called in debug mode");

  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

 

  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::cout << std::endl;
    std::cout << "modules usable for simulation ( use -j modules.name=option to configure them)"
              << std::endl;
    std::cout << "modules.sorting = hashed_cell|linear_cell" << std::endl;
    std::cout << "modules.hash_width = 32bit|64bit" << std::endl;
    std::cout << "modules.neighborhood = constrained|cell_based" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.pressure = IISPH|IISPH17|DFSPH" << std::endl;
    std::cout << "modules.drag = Gissler17" << std::endl;
    std::cout << "modules.tension = Akinci" << std::endl;
    std::cout << "modules.vorticity = Bender17" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.error_checking = true|false" << std::endl;
    std::cout << "modules.gl_record = true|false" << std::endl;
    std::cout << "modules.alembic_export = true|false" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.volumeBoundary = true|false" << std::endl;
    std::cout << "modules.xsph = true|false" << std::endl;
    std::cout << "modules.surfaceDistance = true|false" << std::endl;
    std::cout << "modules.sprayAndFoam = true|false" << std::endl;
    std::cout << "modules.adaptive = true|false" << std::endl;
    std::cout << "modules.viscosity = true|false" << std::endl;

    return false;
  }

  logger::silent = true;

  if (vm.count("params")) {
    param_watchlist.push_back(new param_stats<parameters::num_ptcls>());
    param_watchlist.push_back(new param_stats<parameters::timestep>());
    param_watchlist.push_back(new param_stats<parameters::iterations>());
    param_watchlist.push_back(new param_stats<parameters::density_error>());
  }

  if (vm.count("verbose"))
    logger::silent = false;
  if (vm.count("record")) {
    jsons.push_back("modules.gl_record=true");
    std::stringstream sstream;
    sstream << "render_settings.gl_file=" << vm["record"].as<std::string>();
    jsons.push_back(sstream.str());
  }
  if (vm.count("no-timer"))
    timers = false;
  if (vm.count("frames")) {
    timesteps = vm["frames"].as<int>();
    end_simulation_frame = true;
  }
  if (vm.count("time")) {
    time_limit = vm["time"].as<double>();
    end_simulation_time = true;
  }
  if (vm.count("pause")) {
	  time_pause = vm["pause"].as<double>();
	  pause_simulation_time = true;
  }
  if (vm.count("deviceRegex")) {
	  std::stringstream sstream;
	  sstream << "simulation_settings.deviceRegex=" << vm["deviceRegex"].as<std::string>();
	  jsons.push_back(sstream.str());
  }
  if (vm.count("hostRegex")) {
	  std::stringstream sstream;
	  sstream << "simulation_settings.hostRegex=" << vm["hostRegex"].as<std::string>();
	  jsons.push_back(sstream.str());
  }
  if (vm.count("debugRegex")) {
	  std::stringstream sstream;
	  sstream << "simulation_settings.debugRegex=" << vm["debugRegex"].as<std::string>();
	  jsons.push_back(sstream.str());
  }


  auto config_paths = resolveFile("cfg/paths.cfg");
  std::ifstream ifs(config_paths.string());
  std::vector<std::string> config_folders;
  std::copy(std::istream_iterator<std::string>(ifs),
            std::istream_iterator<std::string>(),
       	    std::back_inserter(config_folders));
  ifs.close();
  std::experimental::filesystem::path default_config = "";
  try{
    default_config = resolveFile("Flow/config.json", config_folders);
    get<parameters::config_file>() = default_config.string();
  }catch(...){
    std::cout << "Could not find default configuration" << std::endl;
  }

  if (vm.count("snap")) {
    snapFile = resolveFile(vm["snap"].as<std::string>(), config_folders).string();
    snapped = true;
  }

  if (vm.count("config")) {
    get<parameters::config_file>() = resolveFile(vm["config"].as<std::string>(), config_folders).string();
    std::ofstream file;
	auto path = resolveFile("cfg/configs.sph");
    file.open(resolveFile("cfg/configs.sph").string(), std::ios_base::app);
    file << vm["config"].as<std::string>() << std::endl;
    file.close();
  }
  if (vm.count("list")) {
    std::ifstream file(resolveFile("cfg/configs.sph").string());
    std::vector<std::string> Configs;
    std::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(),
              std::back_inserter(Configs));
    int32_t i = 0;
    for (auto c : Configs) {
      std::cout << i++ << "\t" << c << std::endl;
    }
    return false;
  }
  if (vm.count("config_id")) {
    std::ifstream file(resolveFile("cfg/configs.sph").string());
    std::vector<std::string> Configs;
    std::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(),
              std::back_inserter(Configs));
    int32_t i = vm["config_id"].as<int>();
    if (i >= (int32_t) Configs.size() || i < 0) {
      std::cerr << "Not a valid config id" << std::endl;
      return false;
    }
    get<parameters::config_file>() = resolveFile(Configs[i], config_folders).string();
  }
  if (vm.count("option")) {
	  std::string config = get<parameters::config_file>();

	  std::stringstream ss;
	  std::ifstream file(config);
	  ss << file.rdbuf();
	  file.close();

	  boost::property_tree::ptree pt;
	  boost::property_tree::read_json(ss, pt);

	  auto options = pt.get_child_optional("options");
	  if (options) {
		  int32_t idx = 0;
		  for (auto& child : options.get()) {
			  if (idx == vm["option"].as<int>()) {
				  for (auto& params : child.second) {
					  for (auto& param : params.second) {
						  std::stringstream option;
						  option << params.first << "." << param.first << "=" << param.second.get_value<std::string>();
						  std::cout << option.str() << std::endl;
						  jsons.push_back(option.str());
					  }
				  }
			  }
			  idx++;
		  }
	  }
  }

  return true;
}

void arguments::cmd::finalize() {
  if (timers) {
    struct TimerData {
      std::string name;
      float average, median, min, max, dev;
      double total;
    };

    std::vector<TimerData> t_data;

    size_t max_len = 0;
    double frame_time = 0.0;

    for (auto t : TimerManager::getTimers()) {
      double total_time = 0.0;
      for (const auto &sample : t->getSamples())
        total_time += sample.second;

      TimerData current{t->getDecriptor(), t->getAverage(), t->getMedian(), t->getMin(),
                        t->getMax(),       t->getStddev(),  total_time};
      max_len = std::max(max_len, current.name.size());
      t_data.push_back(current);
      if (t->getDecriptor() == "Frame")
        frame_time = total_time;
    }

    int32_t number_field = 6;
    std::cout << std::endl;
    std::cout << std::setw(max_len + 3) << "Name"
              << " " << std::setw(number_field + 2) << "avg(ms)" << std::setw(number_field + 2)
              << "med(ms)" << std::setw(number_field + 2) << "min(ms)"
              << std::setw(number_field + 2) << "max(ms)" << std::setw(number_field + 2)
              << "dev(ms)" << std::setw(8) << "%"
              << "\t Total time" << std::endl;

    for (const auto &t : t_data) {
      std::cout << std::setw(max_len + 3) << t.name;
      std::cout << " ";
      std::cout << boost::format("%8.2f") % t.average;
      std::cout << boost::format("%8.2f") % t.median;
      std::cout << boost::format("%8.2f") % t.min;
      std::cout << boost::format("%8.2f") % t.max;
      std::cout << boost::format("%8.2f") % t.dev;
      std::cout << boost::format("%8.2f") % ((t.total / frame_time) * 100.0);
      std::chrono::duration<double, std::milli> dur(t.total);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(dur);
      auto remainder = t.total - seconds.count() * 1000.0;
      auto ms = static_cast<int32_t>(remainder);
      auto s = seconds.count() - minutes.count() * 60;
      std::cout << "\t";
      std::cout << boost::format("%i:%02i.%03i") % minutes.count() % s % ms;

      std::cout << std::endl;
    }
  }
  if (vm.count("config_log")) {
    IO::config::save_config(vm["config_log"].as<std::string>());
  }
  if (vm.count("log")) {
    logger::write_log(vm["log"].as<std::string>());
  }
  if (vm.count("memory") || vm.count("info")) {
    struct MemoryInfo {
      std::string name, type_name;
      size_t allocSize, elements;
      bool swap;
    };

    std::vector<MemoryInfo> mem_info;
    size_t longest_name = 0;
    size_t longes_type = 0;

    for_each(arrays_list, [&](auto x) {
      using T = decltype(x);
      using type = typename T::type;
      auto allocSize = T::alloc_size;
      auto elements = allocSize / sizeof(type);
      std::string name = T::variableName;
      std::string t_name = type_name<type>();
      bool swap = false;

      if constexpr (has_rear_ptr<T>)
        swap = true;
      MemoryInfo current{name, t_name, allocSize, elements, swap};
      if (allocSize != 0) {
        longest_name = std::max(longest_name, current.name.size());
        longes_type = std::max(longes_type, current.type_name.size());
        mem_info.push_back(current);
      }
    });
    if (vm.count("memory")) {
      std::cout << std::endl;
      std::cout << std::setw(longest_name + 3) << "Name"
                << " " << std::setw(longes_type + 3) << "Type"
                << " " << std::setw(8) << "size" << std::endl;
    }
    size_t total = 0;
	size_t total_fixed = 0;
	size_t total_dynamic = 0;
    for (const auto &t : mem_info) {
      if (vm.count("memory")) {
        std::cout << std::setw(longest_name + 3) << t.name;
        std::cout << " ";
        std::cout << std::setw(longes_type + 3) << t.type_name;
        std::cout << " ";
        if (t.swap)
          std::cout << std::setw(8) << IO::config::bytesToString(2 * t.allocSize);
        else
          std::cout << std::setw(8) << IO::config::bytesToString(t.allocSize);
        std::cout << std::endl;
      }
      if (t.swap)
        total += 2 * t.allocSize;
      else
        total += t.allocSize;
	  if (t.swap)
		total_fixed += t.allocSize;
    }
	for (auto& alloc : MemoryManager::instance().allocations)
		total_dynamic += alloc.allocation_size;
	
    std::cout << "Total memory consumption: " << IO::config::bytesToString(total) << std::endl;
	std::cout << "const memory consumption: " << IO::config::bytesToString(total_fixed) << std::endl;
	std::cout << "dyn   memory consumption: " << IO::config::bytesToString(total_dynamic) << std::endl;
  }
  if (vm.count("info")) {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - start;

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(dur);
    auto ms = milliseconds.count() - seconds.count() * 1000;
    auto s = seconds.count() - minutes.count() * 60;
    std::cout << "Total time:               ";
    std::cout << boost::format("%i:%02i.%03i") % minutes.count() % s % ms;
    std::cout << std::endl;
    std::cout << "Number of particles:      "
              << IO::config::numberToString(get<parameters::num_ptcls>()) << " of "
              << IO::config::numberToString(get<parameters::max_numptcls>()) << std::endl;
    std::cout << "Frames:                   " << get<parameters::frame>() << std::endl;
    std::cout << "Simulated time:           " << get<parameters::simulationTime>() << "s" << std::endl;
  }

  if (vm.count("neighbordump")) {
    int32_t *neighbor_list_shared = get<arrays::neighborList>();
    int32_t alloc_size = (int32_t) info<arrays::neighborList>().alloc_size;
    void *neighbor_list_local = malloc(alloc_size);

    int32_t *offsetList_shared = get<arrays::neighborListLength>();
    int32_t alloc_size_offsetList = (int32_t)info<arrays::neighborListLength>().alloc_size;
    void *offsetList_local = malloc(alloc_size_offsetList);

	cuda::memcpy(neighbor_list_local, neighbor_list_shared, alloc_size, cudaMemcpyDeviceToHost);
	cuda::memcpy(offsetList_local, offsetList_shared, alloc_size_offsetList, cudaMemcpyDeviceToHost);

    std::string fileName = vm["neighbordump"].as<std::string>();

    std::ofstream ofs(fileName, std::ios::binary);
    int32_t max_num = get<parameters::max_numptcls>();
    int32_t num = get<parameters::num_ptcls>();
    ofs.write(reinterpret_cast<char *>(&max_num), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&num), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&alloc_size), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&alloc_size_offsetList), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(neighbor_list_local), alloc_size);
    ofs.write(reinterpret_cast<char *>(offsetList_local), alloc_size_offsetList);
    ofs.close();
  }
  if (vm.count("params")) {
    size_t max_len = 0;
    for (auto p : param_watchlist) {
      max_len = std::max(max_len, p->name().size());
    }

    int32_t number_field = 8;
    std::cout << std::endl;
    std::cout << std::setw(max_len + 3) << "Name"
              << "    " << std::setw(number_field + 2) << "avg    " << std::setw(number_field + 2)
              << "med    " << std::setw(number_field + 2) << "min    "
              << std::setw(number_field + 2) << "max    " << std::setw(number_field + 2)
              << "dev    " << std::endl;

    for (const auto &p : param_watchlist) {
      std::cout << p->to_string(max_len + 3) << std::endl;
    }
  }

  auto& mem = MemoryManager::instance();
  for(auto& allocation : mem.allocations){
    if(allocation.ptr == nullptr) continue;

    for_each(arrays_list, [allocation](auto x) { 
      using T = decltype(x);
      if(T::ptr == allocation.ptr)
        T::ptr = nullptr;
    });
    cudaFree(allocation.ptr);
  }
  // Graceful shutdown disabled due to issues on current linux versions.
  for_each(arrays_list, [](auto x) { 
    using T = decltype(x);
    if(T::valid() && T::ptr != nullptr)
      T::free();
    });
}

void arguments::cmd::parameter_stats() {
  if (vm.count("params")) {
    for (auto &p : param_watchlist)
      p->sample();
  }
}

boost::program_options::variables_map &arguments::cmd::getVM() { return vm; }
