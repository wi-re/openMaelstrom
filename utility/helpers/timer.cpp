#include <algorithm>
#include <utility/helpers/timer.h>
#include <utility/identifier/uniform.h>

std::map<std::string, cudaTimer *> timer_map;

Timer::Timer(std::string descr, Color c, bool b)
    : descriptor(descr), graph_color(c), last_stats(-1), last_sample(0), graph(b) {}

void Timer::generateStats() {
  if (last_stats != last_sample) {
    if (samples.size() == 0)
      return;
    std::vector<std::pair<int32_t, float>> sam = samples;
    std::nth_element(sam.begin(), sam.begin() + sam.size() / 2, sam.end(),
                     [](auto lhs, auto rhs) { return lhs.second < rhs.second; });
    median = sam[sam.size() / 2].second;
    min = FLT_MAX;
    max = -FLT_MAX;
    average = 0.f;
    sum = 0.f;
    float ctr = 0.f;
    for (auto s : samples) {
      auto val = s.second;
      min = std::min(min, val);
      max = std::max(max, val);
      sum += val;
      ctr++;
    }
    average = sum / ctr;
    stddev = 0.f;
    for (auto s : samples) {
      auto val = s.second;
      auto diff = (val - average) * (val - average);
      stddev += diff;
    }
    stddev /= ctr - 1;
    stddev = sqrt(stddev);
    last_stats = last_sample;
  }
}

std::vector<Timer *> TimerManager::timers;

hostTimer::hostTimer(std::string descr, Color c, bool b) : Timer(descr, c, b) {
  last_tp = std::chrono::high_resolution_clock::now();
}

hostTimer *TimerManager::createTimer(std::string name, Color c, bool b) {
  if (c == Color::rosemadder)
    c = getRandomColor();
  hostTimer *new_timer = new hostTimer(name, c, b);
  timers.push_back(new_timer);
  return new_timer;
}
cudaTimer *TimerManager::createGPUTimer(std::string name, Color c, bool b) {
  if (c == Color::rosemadder)
    c = getRandomColor();
  cudaTimer *new_timer = new cudaTimer(name, c, b);
  timers.push_back(new_timer);
  return new_timer;
}

Timer *TimerManager::createHybridTimer(std::string name, Color c, bool b) {
  if (parameters::target{} == launch_config::device)
    return createGPUTimer(name, c, b);
  else
    return createTimer(name, c, b);
}

void hostTimer::start() { last_tp = std::chrono::high_resolution_clock::now(); }
void hostTimer::stop() {
  auto tp = std::chrono::high_resolution_clock::now();
  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(tp - last_tp).count();
  std::lock_guard<std::mutex> guard(vec_lock);
  samples.push_back(std::make_pair(++last_sample, (static_cast<float>(dur_us) / 1000.f)));
}

cudaTimer::cudaTimer(std::string descr, Color c, bool b) : Timer(descr, c, b) {
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
}

void cudaTimer::start() {
  if (timer_stopped) {
    cudaEventSynchronize(stop_event);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    std::lock_guard<std::mutex> guard(vec_lock);
    samples.push_back(std::make_pair(++last_sample, (milliseconds)));
    timer_stopped = false;
  }
  cudaEventRecord(start_event);
}
void cudaTimer::stop() {
  cudaEventRecord(stop_event);

  timer_stopped = true;
}
