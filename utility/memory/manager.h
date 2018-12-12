#pragma once
#include <utility/MemoryManager.h>
#include <IO/config/config.h>
struct MemoryManager {
	struct Allocation {
		size_t allocation_size = 0;
		void *ptr = nullptr;
		bool inUse = false;
		std::string last_allocation = "";
	};

	std::vector<Allocation> allocations;
	std::vector<std::string> persistentArrays;

	static MemoryManager &instance();

	template <typename P, typename U> void bindInput(P, U &memory) {
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		mem_t ptr = (mem_t)P::ptr;
		if (ptr != nullptr) {
			P::get_member(memory) = (mem_t)P::ptr;
			return;
		}
		if (!P::valid()) {
			LOG_WARNING << "Requesting memory that is not valid for " << P::variableName
				<< " while filling " << typeid(U).name() << std::endl;
			P::get_member(memory) = nullptr;
			return;
		}
		if (P::valid()) {
			LOG_ERROR << "Requesting memory as input that is not set for " << P::variableName
				<< " while filling " << typeid(U).name() << std::endl;
			;
		}
	}

	template <typename P, typename U> void allocateRear(P, U &memory) {
		using mem_t = std::decay_t<decltype(P::get_member(memory).first)>;
		mem_t ptr = (mem_t)P::ptr;
		mem_t rear_ptr = (mem_t)P::rear_ptr;
		if (ptr != nullptr && rear_ptr != nullptr) {
			P::get_member(memory) = std::make_pair((mem_t)P::ptr, (mem_t)P::rear_ptr);
			return;
		}
		if (!P::valid()) {
			LOG_WARNING << "Requesting memory that is not valid for " << P::variableName
				<< " while filling " << typeid(U).name() << std::endl;
			P::get_member(memory) = std::make_pair(nullptr, nullptr);
			return;
		}
		if (ptr == nullptr) {
			LOG_ERROR << "Swap memory with invalid front ptr for " << P::variableName << " while filling "
				<< typeid(U).name() << std::endl;
			P::get_member(memory) = std::make_pair(nullptr, nullptr);
		}
		for (auto &allocation : allocations) {
			if (allocation.inUse)
				continue;
			if (P::alloc_size == allocation.allocation_size) {
				P::rear_ptr = (decltype(P::rear_ptr))allocation.ptr;
				P::get_member(memory) = std::make_pair((mem_t)P::ptr, (mem_t)P::rear_ptr);
				;
				allocation.last_allocation = typeid(U).name();
				allocation.inUse = true;
				return;
			}
		}
		LOG_INFO << "Allocating new memory block of size " << IO::config::bytesToString(P::alloc_size)
			<< " to fill request for " << P::variableName << " while filling " << typeid(U).name()
			<< std::endl;
		void *allocated_memory = nullptr;
		cudaAllocateMemory(&allocated_memory, P::alloc_size);
		Allocation allocated{ P::alloc_size, allocated_memory, true };
		allocations.push_back(allocated);
		P::rear_ptr = (decltype(P::rear_ptr))allocated.ptr;
		P::get_member(memory) = std::make_pair((mem_t)P::ptr, (mem_t)P::rear_ptr);
		;
	}

	template <typename P> void allocateRear(P) {
		auto ptr = P::ptr;
		auto rear_ptr = P::rear_ptr;
		if (ptr != nullptr && rear_ptr != nullptr) {
			return;
		}
		if (!P::valid()) {
			return;
		}
		if (ptr == nullptr) {
			LOG_ERROR << "Swap memory with invalid front ptr for " << P::variableName << std::endl;
		}
		for (auto &allocation : allocations) {
			if (allocation.inUse)
				continue;
			if (P::alloc_size == allocation.allocation_size) {
				P::rear_ptr = (decltype(P::rear_ptr))allocation.ptr;
				allocation.inUse = true;
				allocation.last_allocation = "sort";
				return;
			}
		}
		LOG_INFO << "Allocating new memory block of size " << IO::config::bytesToString(P::alloc_size)
			<< " to fill request for " << P::variableName << std::endl;
		void *allocated_memory = nullptr;
		cudaAllocateMemory(&allocated_memory, P::alloc_size);
		Allocation allocated{ P::alloc_size, allocated_memory, true };
		allocations.push_back(allocated);
		P::rear_ptr = (decltype(P::rear_ptr))allocated.ptr;
	}

	template <typename P, typename U> void allocate(P, U &memory, bool ignore_invalid = false) {
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		mem_t ptr = (mem_t)P::ptr;
		if (ptr != nullptr) {
			P::get_member(memory) = (mem_t)P::ptr;
			return;
		}
		if (!P::valid()) {
			if (ignore_invalid)
				return;
			LOG_WARNING << "Requesting memory that is not valid for " << P::variableName
				<< " while filling " << typeid(U).name() << std::endl;
			P::get_member(memory) = nullptr;
			return;
		}
		for (auto &allocation : allocations) {
			if (allocation.inUse)
				continue;
			if (P::alloc_size == allocation.allocation_size) {
				P::ptr = (decltype(P::ptr))allocation.ptr;
				P::get_member(memory) = (mem_t)allocation.ptr;
				allocation.inUse = true;
				allocation.last_allocation = typeid(U).name();
				return;
			}
		}
		LOG_INFO << "Allocating new memory block of size " << IO::config::bytesToString(P::alloc_size)
			<< " to fill request for " << P::variableName << " while filling " << typeid(U).name()
			<< std::endl;
		void *allocated_memory = nullptr;
		cudaAllocateMemory(&allocated_memory, P::alloc_size);
		Allocation allocated{ P::alloc_size, allocated_memory, true };
		allocations.push_back(allocated);
		P::get_member(memory) = (mem_t)allocated.ptr;
		P::ptr = (decltype(P::ptr))allocated.ptr;
	}

	template <typename P> void allocate(P, bool ignore_invalid = false) {
		using mem_t = typename P::type;
		mem_t *ptr = (mem_t *)P::ptr;
		if (ptr != nullptr) {
			return;
		}
		if (!P::valid()) {
			if (ignore_invalid)
				return;
			LOG_WARNING << "Requesting memory that is not valid for " << P::variableName
				<< " while filling "
				<< "allocaiton request" << std::endl;
			return;
		}
		for (auto &allocation : allocations) {
			if (allocation.inUse)
				continue;
			if (P::alloc_size == allocation.allocation_size) {
				P::ptr = (decltype(P::ptr))allocation.ptr;
				allocation.inUse = true;
				// LOG_INFO << "Binding to " << P::variableName << "while filling " << typeid(U).name() << "
				// previously used for: " << allocation.last_allocation << std::endl;
				allocation.last_allocation = "allocation";
				return;
			}
		}
		LOG_INFO << "Allocating new memory block of size " << IO::config::bytesToString(P::alloc_size)
			<< " to fill request for " << P::variableName << " while filling "
			<< "allocation request" << std::endl;
		void *allocated_memory = nullptr;
		cudaAllocateMemory(&allocated_memory, P::alloc_size);
		Allocation allocated{ P::alloc_size, allocated_memory, true };
		allocations.push_back(allocated);
		P::ptr = (decltype(P::ptr))allocated.ptr;
	}
	template <typename P, typename U> void deallocate([[maybe_unused]] P arr, U &memory) {
		using mem_t = std::decay_t<decltype(P::get_member(memory))>;
		mem_t ptr = (mem_t)P::ptr;
		if (P::variableName == get<parameters::render_buffer>())
			return;
		for(const auto& str : persistentArrays){
			if(P::variableName == str)
				return;
		}
		for (auto &allocation : allocations) {
			if (!allocation.inUse)
				continue;
			if (allocation.ptr == (void *)ptr) {
				// LOG_INFO << "Freeing memory block from " << P::variableName << " previously used for " <<
				// allocation.last_allocation << std::endl;
				P::ptr = nullptr;
				allocation.inUse = false;
				return;
			}
		}
	}
	template <typename P, typename U> void deallocateRear(P, U &memory) {
		using mem_t = std::decay_t<decltype(P::get_member(memory).first)>;
		mem_t ptr = (mem_t)P::rear_ptr;
		for (auto &allocation : allocations) {
			if (!allocation.inUse)
				continue;
			if (allocation.ptr == (void *)ptr) {
				P::rear_ptr = nullptr;
				// LOG_INFO << "Freeing memory block from " << P::variableName << " previously used for " <<
				// allocation.last_allocation << std::endl;
				allocation.inUse = false;
				return;
			}
		}
	}
	template <typename P> void freeRear(P) {
		auto ptr = P::rear_ptr;
		for (auto &allocation : allocations) {
			if (!allocation.inUse)
				continue;
			if (allocation.ptr == (void *)ptr) {
				P::rear_ptr = nullptr;
				allocation.inUse = false;
				return;
			}
		}
	}

	void reclaimMemory();
};
