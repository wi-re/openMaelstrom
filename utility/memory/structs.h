#pragma once
#include <utility/memory/manager.h>
#include <utility/memory/functions.h>
enum struct inst_t { array_clear, function_call };
template <typename arr> struct array_clear {
	using type = arr;
	constexpr static const inst_t kind = inst_t::array_clear;
	std::function<void()> callFn;
	std::function<bool()> validFn;
	array_clear() {
		validFn = []() {
			if (!arr::valid())
				return false;
			return true;
		};
		callFn = []() {
			MemoryManager::instance().allocate(arr{});

			cuda::Memset(arr::ptr, 0, arr::alloc_size);
		};
	}
	void operator()() {
		if (!validFn())
			return;
		callFn();
	}
};
template <typename T> struct function_call {
	using type = T;
	constexpr static const inst_t kind = inst_t::function_call;
	std::string functionName = "";
	Color c = Color::alizarin_crimson;
	bool graph = true;
	Timer *timer = nullptr;
	bool has_timer = true;
	std::function<void()> callFn;
	std::function<bool()> validFn;

	function_call() {}

	function_call(void(*func)(T), std::string _functionName, Color _c = Color::alizarin_crimson,
		bool _graph = true)
		: functionName(_functionName), c(_c), graph(_graph) {
		bool time_fn = functionName != "";
		has_timer = time_fn;

		validFn = []() {
			T valid_test_mem{};
			if (!valid(valid_test_mem))
				return false;
			return true;
		};

		callFn = [func]() {
			auto memory = prepareMemory2<T>();

			func(memory);

			clearMemory2(memory);
		};
	}

	void operator()() {
		if (!validFn())
			return;
		if (this->timer == nullptr && this->has_timer)
			this->timer = TimerManager::createGPUTimer(functionName, c, graph);

		if (this->timer)
			this->timer->start();

		callFn();

		if (this->timer)
			this->timer->stop();
	}
};
