#pragma once
#include <utility/launcher.h>
#include <utility/identifier.h>
template <typename C, typename... Ts> __global__ void launch_cuda_function(C fn, Ts... args) {
	fn(args...);
}

template <launch_config cfg, typename cache_t> struct configuration_helper;
template <typename cache_t> struct configuration_helper<launch_config::device, cache_t> {
	template <typename T, typename C, typename... Vs>
	static void fn(T &launcher, C, std::function<void(Vs...)>) {
		cache_t cache;
		int32_t minGridSize;
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &launcher.m_blockSize,
			&launch_cuda_function<C, Vs...>, cache, 0);
		if (launcher.m_blockSize == 0)
			launcher.m_blockSize = 512;
		launcher.m_sharedMemory = (int32_t) cache(launcher.m_blockSize);
		logger(log_level::info) << "Device launching " << launcher.m_name << " with blocksize "
			<< launcher.m_blockSize << " and " << launcher.m_sharedMemory
			<< "B shared memory." << std::endl;
	}
};
template <typename cache_t>
struct configuration_helper<launch_config::_used_for_template_specializations, cache_t> {
	template <typename T, typename C, typename... Vs>
	static void fn(T &launcher, C, std::function<void(Vs...)>) {
		cache_t cache;
		int32_t minGridSize;
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &launcher.m_blockSize,
			&launch_cuda_function<C, Vs...>, cache, 0);
		if (launcher.m_blockSize == 0)
			launcher.m_blockSize = 512;
		launcher.m_sharedMemory = (int32_t) cache(launcher.m_blockSize);
		logger(log_level::info) << "Hybrid launching " << launcher.m_name << " with blocksize "
			<< launcher.m_blockSize << " and " << launcher.m_sharedMemory
			<< "B shared memory." << std::endl;
	}
};

template <typename cache_t> struct configuration_helper<launch_config::host, cache_t> {
	template <typename T, typename C, typename... Vs>
	static void fn(T &launcher, C, std::function<void(Vs...)>) {
		logger(log_level::info) << "Host launching " << launcher.m_name << std::endl;
	}
};

template <launch_config config, typename C, typename cache_t, typename... Vs>
struct computeLauncher {
	const std::string m_name;
	const std::string m_file;
	const int32_t m_line;
	const Color m_col;
	hostTimer *m_host_timer = nullptr;
	cudaTimer *m_device_timer = nullptr;
	bool m_timeCode = false;
	C m_kernel;
	int32_t m_blockSize;
	int32_t m_sharedMemory;

	hostOnly computeLauncher(C fun, std::function<void(Vs...)> fn, int line, std::string file,
		std::string name = "", bool time_code = false, Color c = Color::black)
		: m_name(name), m_file(file), m_line(line), m_col(c), m_timeCode(time_code), m_kernel(fun) {
		configuration_helper<config, cache_t>::fn(*this, fun, fn);
	}

	template <typename T> hostOnly void host_launch(int32_t elements, Vs... args) {
		if (m_timeCode && m_host_timer == nullptr)
			m_host_timer = TimerManager::createTimer(m_name, m_col, false);
		cuda::sync();
		h_blockDim = int3{ elements, 1, 1 };
		h_blockIdx = int3{ 0, 0, 0 };
		h_threadIdx = int3{ 1, 0, 0 };
		if (m_host_timer != nullptr)
			m_host_timer->start();
		tbb::parallel_for(tbb::blocked_range<int32_t>(0, elements),
			[&](const tbb::blocked_range<int32_t> &range) {
			for (int32_t i = range.begin(); i < range.end(); ++i) {
				h_threadIdx.x = i;
				m_kernel(args...);
			}
		});
		if (m_host_timer != nullptr)
			m_host_timer->stop();
		cuda::sync();
	}
	template <typename T> hostOnly void device_launch(int32_t elements, Vs... args) {
		if (m_timeCode && m_device_timer == nullptr)
			m_device_timer = TimerManager::createGPUTimer(m_name, m_col, false);
		int gridSize = (elements + m_blockSize - 1) / m_blockSize;
		if (m_device_timer != nullptr)
			m_device_timer->start();
#ifdef __CUDACC__
		launch_cuda_function <<<gridSize, m_blockSize, m_sharedMemory >>>(m_kernel, args...);
#endif
		if (m_device_timer != nullptr)
			m_device_timer->stop();
		if (get<parameters::error_checking>())
			cuda::checkMessages(m_name, m_file, m_line, error_level::thread_synchronize);
	}
	template <typename T> hostOnly void debug_launch(int32_t elements, Vs... args) {
		std::cout << "Starting function: " << m_name << " ( " << m_file << " @ " << m_line << ")"
			<< std::endl;
		if (m_host_timer == nullptr)
			m_host_timer = TimerManager::createTimer(m_name, m_col, false);
		cuda::sync(m_name);
		h_blockDim = int3{ elements, 1, 1 };
		h_blockIdx = int3{ 0, 0, 0 };
		h_threadIdx = int3{ 1, 0, 0 };
		if (m_host_timer != nullptr)
			m_host_timer->start();
		for (int32_t i = 0; i < elements; ++i) {
			h_threadIdx.x = i;
			m_kernel(args...);
		}
		if (m_host_timer != nullptr)
			m_host_timer->stop();
		cuda::sync(m_name);
		if (m_host_timer != nullptr) {
			auto &samples = m_host_timer->getSamples();
			auto last = samples[samples.size() - 1];
			std::cout << "Function " << m_name << " finished after: " << last.second << "ms."
				<< std::endl;
		}
		else {
			std::cout << "Function " << m_name << " finished." << std::endl;
		}
	}

	template <launch_config cfg, typename> struct Delegator;
	template <typename T> struct Delegator<launch_config::device, T> {
		template <typename L, typename... Us>
		static void call(L *launcher, int32_t elements, Us &&... args) {
			launcher->TEMPLATE_TOKEN device_launch<void>(elements, std::forward<Us>(args)...);
		}
	};
	template <typename T> struct Delegator<launch_config::host, T> {
		template <typename L, typename... Us>
		static void call(L *launcher, int32_t elements, Us &&... args) {
			launcher->TEMPLATE_TOKEN host_launch<void>(elements, std::forward<Us>(args)...);
		}
	};
	template <typename T> struct Delegator<launch_config::debug, T> {
		template <typename L, typename... Us>
		static void call(L *launcher, int32_t elements, Us &&... args) {
			launcher->TEMPLATE_TOKEN debug_launch<void>(elements, std::forward<Us>(args)...);
		}
	};
	template <typename T> struct Delegator<launch_config::pure_host, T> {
		template <typename L, typename... Us>
		static void call(L *launcher, int32_t elements, Us &&... args) {
			launcher->TEMPLATE_TOKEN host_launch<void>(elements, std::forward<Us>(args)...);
		}
	};
	template <typename T> struct Delegator<launch_config::_used_for_template_specializations, T> {
		template <typename L, typename... Us>
		static void call(L *launcher, int32_t elements, Us &&... args) {
			launch_config target = parameters::target{};
			if (parameters::regex_cfg{}) {
				auto matcher = [&](std::string str, launch_config cfg) {
					if (str == "")
						return;
					std::regex reg(str, std::regex_constants::ECMAScript | std::regex_constants::icase);
					if (std::regex_search(launcher->m_name, reg))
						target = cfg;
				};
				matcher(parameters::deviceRegex{}, launch_config::device);
				matcher(parameters::hostRegex{}, launch_config::host);
				matcher(parameters::debugRegex{}, launch_config::debug);
			}
			cudaDeviceSynchronize();
			switch (target) {
			case launch_config::device:
				launcher->TEMPLATE_TOKEN device_launch<void>(elements, std::forward<Us>(args)...);
				break;
			case launch_config::host:
				launcher->TEMPLATE_TOKEN host_launch<void>(elements, std::forward<Us>(args)...);
				break;
			case launch_config::debug:
				launcher->TEMPLATE_TOKEN debug_launch<void>(elements, std::forward<Us>(args)...);
				break;
			case launch_config::pure_host:
				launcher->TEMPLATE_TOKEN host_launch<void>(elements, std::forward<Us>(args)...);
				break;
			default:
				break;
			}
			cudaDeviceSynchronize();
		}
	};

	template <launch_config cfg> hostOnly void operator()(int32_t elements, Vs &&... args) {
		static_assert(cfg == config || config == launch_config::_used_for_template_specializations ||
			(config == launch_config::host &&
			(cfg == launch_config::host || cfg == launch_config::debug ||
				cfg == launch_config::pure_host)),
			"Improperly configured call (requested launch configuration does not match "
			"compiled configurations)");
		if (elements == 0)
			return;
		Delegator<cfg, void>::call(this, elements, std::forward<Vs>(args)...);
	}
};

template <launch_config cfg, typename C, typename... Vs, typename T, typename... Ts>
auto get_computeLauncher(C fun, std::function<void(Vs...)> fn, T, Ts... args) {
	return computeLauncher<cfg, C, T, Vs...>(fun, fn, args...);
}

template <launch_config cfg, typename C, typename... Vs, typename T, typename... Ts>
auto get_FunctionLauncher_i(C fun, std::function<void(Vs...)> fn, T&&, Ts... args) {
	return computeLauncher<cfg, C, std::decay_t<T>, Vs...>(fun, fn, args...);
}

template <launch_config cfg, typename C, template <typename...> typename Cs, typename... Vs,
	typename... Ts, typename... Us>
	auto get_FunctionLauncher(C fun, std::function<void(Vs...)> fn, int line, std::string file,
		std::string name, Cs<Us...> cache, Ts... args) {
	return get_FunctionLauncher_i<cfg>(fun, fn, cache, line, file, name, args...);
}

template <launch_config cfg, typename C, typename... Vs, typename... Ts>
auto get_FunctionLauncher(C fun, std::function<void(Vs...)> fn, int line, std::string file,
	std::string name, Ts... args) {
	return get_FunctionLauncher_i<cfg>(fun, fn, caches<>{}, line, file, name, args...);
}
