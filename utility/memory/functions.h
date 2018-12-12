#pragma once
#include <utility/MemoryManager.h>
template <typename T> auto prepareMemory2() {
	T memory;
	for_each_r(typename T::parameters{}, [&memory](auto x) {
		using P = decltype(x);
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = mem_t{ *P::ptr };
	});
	for_each_r(typename T::basic_info_params{}, [&memory](auto x) {
		using P = decltype(x);
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = mem_t{ *P::ptr };
	});
	for_each_r(typename T::virtual_info_params{}, [&memory](auto x) {
		using P = decltype(x);
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = mem_t{ *P::ptr };
	});
	for_each_r(typename T::boundaryInfo_params{}, [&memory](auto x) {
		using P = decltype(x);
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = mem_t{ *P::ptr };
	});
	for_each_r(typename T::cell_info_params{}, [&memory](auto x) {
		using P = decltype(x);
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = mem_t{ *P::ptr };
	});

	for_each_r(typename T::virtual_info_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory, true); });
	for_each_r(typename T::boundaryInfo_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory, true); });
	for_each_r(typename T::cell_info_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory, true); });
	for_each_r(typename T::neighbor_info_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory, true); });
	for_each_r(typename T::input_arrays{}, [&memory](auto x) {
		using P = decltype(x);
		if (P::ptr == nullptr) {
			LOG_ERROR << "Requesting memory ( " << P::variableName
				<< " ) as input that has not yet been set when filling" << typeid(T).name()
				<< std::endl;
		}
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		P::get_member(memory) = (mem_t)P::ptr;
	});
	for_each_r(typename T::output_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory); });
	for_each_r(typename T::swap_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocateRear(x, memory); });
	for_each_r(typename T::temporary_arrays{},
		[&memory](auto x) { MemoryManager::instance().allocate(x, memory); });

	if (memory.resort == true) {
		for_each_r(sorting_list, [](auto x) { MemoryManager::instance().allocateRear(x); });
	}

	return memory;
}
template <typename T> void clearMemory2(T &memory) {
	for_each_r(typename T::parameters{}, [](auto) {});
	for_each_r(typename T::basic_info_params{}, [](auto) {});
	for_each_r(typename T::virtual_info_params{}, [](auto) {});
	for_each_r(typename T::cell_info_params{}, [](auto) {});

	for_each_r(typename T::virtual_info_arrays{}, [](auto) {});
	for_each_r(typename T::cell_info_arrays{}, [](auto) {});
	for_each_r(typename T::neighbor_info_arrays{}, [](auto) {});
	for_each_r(typename T::input_arrays{}, [](auto) {});
	for_each_r(typename T::output_arrays{}, [](auto) {});
	for_each_r(typename T::swap_arrays{},
		[&memory](auto x) { MemoryManager::instance().deallocateRear(x, memory); });
	for_each_r(typename T::temporary_arrays{},
		[&memory](auto x) { MemoryManager::instance().deallocate(x, memory); });
	if (memory.resort == true) {
		for_each_r(sorting_list, [](auto x) { MemoryManager::instance().freeRear(x); });
	}
}
