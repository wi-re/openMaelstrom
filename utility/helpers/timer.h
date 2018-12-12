#pragma once
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <utility/helpers/color.h>
#include <vector>

class Timer {
public:
  using sample = std::pair<int32_t, float>;

protected:
  std::vector<sample> samples;

  std::string descriptor;
  Color graph_color;

  float average, min, max, stddev, sum, median;
  int32_t last_stats;

  int32_t last_sample;

public:
  std::mutex vec_lock;
  bool graph;

  Timer(std::string descriptor = "unnamed", Color c = Color::black, bool b = true);
  virtual void start() = 0;
  virtual void stop() = 0;

  inline const std::vector<sample> &getSamples() const { return samples; }
  inline Color getColor() const { return graph_color; }
  inline std::string getDecriptor() const { return descriptor; }

  inline void setSamples(std::vector<sample> arg) { samples = arg; }

  void generateStats();
  inline std::tuple<float, float, float, float> getStats() {
    generateStats();
    return std::make_tuple(min, max, average, stddev);
  }
  inline float getMin() {
    generateStats();
    return min;
  }
  inline float getMax() {
    generateStats();
    return max;
  }
  inline float getAverage() {
    generateStats();
    return average;
  }
  inline float getStddev() {
    generateStats();
    return stddev;
  }
  inline float getSum() {
    generateStats();
    return sum;
  }
  inline float getMedian() {
    generateStats();
    return median;
  }
};

class hostTimer : public Timer {
  std::chrono::high_resolution_clock::time_point last_tp;

public:
  hostTimer(std::string descriptor = "unnamed", Color c = Color::black, bool b = true);
  virtual void start() override;
  virtual void stop() override;
};

class cudaTimer : public Timer {
  bool timer_stopped = false;
  cudaEvent_t start_event, stop_event;

public:
  cudaTimer(std::string descriptor = "unnamed", Color c = Color::black, bool b = true);
  virtual void start() override;
  virtual void stop() override;
};

class TimerManager {
  static std::vector<Timer *> timers;

public:
  TimerManager() = delete;
  static hostTimer *createTimer(std::string name = "UNNAMED", Color c = Color::rosemadder,
                                bool b = true);

  static Timer *createHybridTimer(std::string name = "UNNAMED", Color c = Color::rosemadder,
                                  bool b = true);

  static cudaTimer *createGPUTimer(std::string name = "UNNAMED", Color c = Color::rosemadder,
                                   bool b = true);
  static inline std::vector<Timer *>& getTimers() { return timers; }
};

extern std::map<std::string, cudaTimer *> timer_map;

#define TIME_CODE(name, col, x)                                                                    \
  static hostTimer &t_##name = *TimerManager::createTimer(#name, col, false);                      \
  t_##name.start();                                                                                \
  x;                                                                                               \
  t_##name.stop();
#define GRAPH_CODE(name, col, x)                                                                   \
  static hostTimer &t_##name = *TimerManager::createTimer(#name, col, true);                       \
  t_##name.start();                                                                                \
  x;                                                                                               \
  t_##name.stop();

#define TIME_CODE_GPU(name, col, x)                                                                \
  static cudaTimer &t_##name = *TimerManager::createGPUTimer(#name, col, false);                   \
  t_##name.start();                                                                                \
  x;                                                                                               \
  t_##name.stop();
#define GRAPH_CODE_GPU(name, col, x)                                                               \
  static cudaTimer &t_##name = *TimerManager::createGPUTimer(#name, col, true);                    \
  t_##name.start();                                                                                \
  x;                                                                                               \
  t_##name.stop();
