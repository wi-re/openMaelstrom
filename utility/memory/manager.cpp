#include <utility/MemoryManager.h>

MemoryManager &MemoryManager::instance() {
  static MemoryManager inst;
  return inst;
}

void MemoryManager::reclaimMemory() {
  for_each(arrays_list, [this](auto x) {
    using P = std::decay_t<decltype(x)>;
    if (P::valid() == false)
      return;
    //std::cout << P::variableName << " [" << P::valid() << "] -> " << P::ptr << std::endl;
    if (P::kind != memory_kind::particleData)
      return;
    if constexpr (has_rear_ptr<P>) {
    //std::cout << P::variableName << "|[" << P::valid() << "] -> " << P::ptr << std::endl;
      auto ptr = P::rear_ptr;
      for (auto &allocation : allocations) {
        if (!allocation.inUse)
          continue;
        //std::cout << allocation.inUse << " x " << allocation.last_allocation << " -> " << ptr << " : " << allocation.ptr << std::endl;
        if (allocation.ptr == (void *)ptr) {
          P::rear_ptr = nullptr;
          allocation.inUse = false;
          //std::cout << "Resetting allocation for " << P::variableName << std::endl;
          return;
        }
      }
    } else {
    //std::cout << P::variableName << "x[" << P::valid() << "] -> " << P::ptr << std::endl;
      if (P::variableName == get<parameters::render_buffer>())
        return;
		for(const auto& str : persistentArrays){
			if(P::variableName == str)
				return;
		}
      if (P::variableName == arrays::density::variableName)
        return;
    //std::cout << P::ptr << "=[" << P::valid() << "] -> " << P::ptr << std::endl;
      auto ptr = P::ptr;
      for (auto &allocation : allocations) {
        if (!allocation.inUse)
          continue;
        //std::cout << allocation.inUse << " x " << allocation.last_allocation << " -> " << ptr << " : " << allocation.ptr << std::endl;
        if (allocation.ptr == (void *)ptr) {
          P::ptr = nullptr;
          allocation.inUse = false;
          //std::cout << "Resetting allocation for " << P::variableName << std::endl;
          return;
        }
      }
    }
  });
}
