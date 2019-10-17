#pragma once
#include <mutex>
#include <vector>
#include <functional>
#include <utility/helpers/arguments.h>
#include <utility/helpers/color.h>
#include <utility/helpers/timer.h>
#include <utility/identifier/arrays.h>

struct simulationStep {
	bool graph = true;
	Timer * timer = nullptr;
	bool has_timer = true;

	std::string description = "";
	Color color = Color::alizarin_crimson;
	simulationStep(std::string name = "", Color c = Color::flatpink) : description(name), color(c) {
		has_timer = description != "";
	};

	virtual void call() = 0;
	virtual bool isValid() = 0;
	virtual std::vector<array_enum> getSwapArrays() = 0;
	virtual std::vector<array_enum> getInputsArrays() = 0;
	virtual std::vector<array_enum> getOutputArrays() = 0;
	virtual std::vector<array_enum> getTemporaryArrays() = 0;
	virtual std::vector<array_enum> getCellArrays() = 0;
	virtual std::vector<array_enum> getVirtualArrays() = 0;
	virtual std::vector<array_enum> getBoundaryArrays() = 0;
	virtual std::vector<array_enum> getNeighborArrays() = 0;
	virtual std::vector<array_enum> getArrays() {
		std::vector<array_enum> allArrays;
		auto append = [&](auto v) {
			allArrays.insert(allArrays.end(), v.begin(), v.end());
		};
		append(getSwapArrays());
		append(getInputsArrays());
		append(getOutputArrays());
		append(getTemporaryArrays());
		append(getCellArrays());
		append(getVirtualArrays());
		append(getBoundaryArrays());
		append(getNeighborArrays());
		return allArrays;
	}
};

template<typename T>
struct debugArrayEntry {
	using value_type = T;
	typename T::type* ptr = nullptr;
	bool valid = false;
};

struct debugArrays {
	debugArrayEntry<arrays::splitIndicator> splitIndicator{ nullptr, false };
	debugArrayEntry<arrays::parentIndex> parentIndex{ nullptr, false };
	debugArrayEntry<arrays::parentVolume> parentVolume{ nullptr, false };
	debugArrayEntry<arrays::parentPosition> parentPosition{ nullptr, false };
	debugArrayEntry<arrays::angularVelocity> angularVelocity{ nullptr, false };
	debugArrayEntry<arrays::distanceBuffer> distanceBuffer{ nullptr, false };
	debugArrayEntry<arrays::position> position{ nullptr, false };
	debugArrayEntry<arrays::acceleration> acceleration{ nullptr, false };
	debugArrayEntry<arrays::velocity> velocity{ nullptr, false };
	debugArrayEntry<arrays::particle_type> particle_type{ nullptr, false };
	debugArrayEntry<arrays::renderArray> renderArray{ nullptr, false };
	debugArrayEntry<arrays::debugArray> debugArray{ nullptr, false };
	debugArrayEntry<arrays::volume> volume{ nullptr, false };
	debugArrayEntry<arrays::lifetime> lifetime{ nullptr, false };
	debugArrayEntry<arrays::pressure> pressure{ nullptr, false };
	debugArrayEntry<arrays::density> density{ nullptr, false };
	debugArrayEntry<arrays::neighborListLength> neighborListLength{ nullptr, false };
};


struct cuda_particleSystem {
private:
  std::vector<simulationStep*> functions;
  std::vector<std::function<void()>> callbacks;
  cuda_particleSystem() = default;
  void setup_simulation();
  debugArrays m_debugArrays;
public:
	bool renderFlag = false;
  bool init = false;
  volatile bool running = true;
  volatile bool single = false;

  std::mutex simulation_lock;

  static cuda_particleSystem &instance();

  void storeDebugArrays();
  debugArrays getDebugArrays();

  void step();
  void* retainArray(std::string arrayName);
  void addCallback(std::function<void()> cb);
  void init_simulation();
};
