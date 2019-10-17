#include <iostream>
#include <utility/helpers/log.h>

std::vector<std::tuple<log_level, std::chrono::system_clock::time_point, std::string>> logger::logs;
bool logger::silent = false;

std::function<void(std::string, log_level)> logger::log_fn = [](std::string message,
                                                                log_level log) {
  std::stringstream os;
  if (!silent) {
    switch (log) {
    case log_level::warning:
      os << "WARNING: ";
      break;
    case log_level::info:
      os << "INFO: ";
      break;
    case log_level::debug:
      os << "DEBUG: ";
      break;
    case log_level::error:
      os << "ERROR: ";
      break;
    case log_level::verbose:
      os << "VERBOSE: ";
      break;
    }
    os << message;
    std::cout << os.str();
  }
  logs.push_back(std::make_tuple(log, std::chrono::system_clock::now(), message));
};
#include <chrono>
#include <ctime>
#include <fstream>

void logger::write_log(std::string fileName) {
  std::ofstream file;
  file.open(fileName);

  for (auto log : logger::logs) {
    auto [level, time, message] = log;

    auto tt = std::chrono::system_clock::to_time_t(time);
    // Convert std::time_t to std::tm (popular extension)
    auto tm = std::localtime(&tt);

    std::stringstream sstream;
    switch (level) {
    case log_level::info:
      sstream << R"(<font color="Black">INFO: )";
      break;
    case log_level::error:
      sstream << R"(<font color="Red">ERROR: )";
      break;
    case log_level::debug:
      sstream << R"(<font color="green">DEBUG: )";
      break;
    case log_level::warning:
      sstream << R"(<font color="aqua">WARNING: )";
      break;
	case log_level::verbose:
		sstream << R"(<font color="blue">VERBOSE: )";
		break;
    }

    sstream << "</font> " << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << " -> "
            << message << R"(<br>)";
    file << sstream.str() << std::endl;
  }
  file.close();
}