#pragma once
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <functional>

enum struct log_level { warning, info, debug, error, verbose };

class logger : public std::ostream, std::streambuf {
  log_level logging_level = log_level::info;
  std::stringstream sstream;

public:
  logger(log_level _level = log_level::info) : std::ostream(this), logging_level(_level) {}

  static std::function<void(std::string, log_level)> log_fn;
  static std::vector<std::tuple<log_level, std::chrono::system_clock::time_point, std::string>>
      logs;
  static bool silent;

  static void write_log(std::string filename);

  int overflow(int c) {
    sstream << static_cast<char>(c);
    return 0;
  }

  ~logger() { log_fn(sstream.str(), logging_level); }
};

#define LOG_DEBUG logger(log_level::debug) << __FILE__ << "@" << __LINE__ << ": "
#define LOG_ERROR logger(log_level::error) << __FILE__ << "@" << __LINE__ << ": "
#define LOG_INFO logger(log_level::info) << __FILE__ << "@" << __LINE__ << ": "
#define LOG_WARNING logger(log_level::warning) << __FILE__ << "@" << __LINE__ << ": "
#define LOG_VERBOSE logger(log_level::verbose) << __FILE__ << "@" << __LINE__ << ": "
