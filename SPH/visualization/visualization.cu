#include <SPH/visualization/visualization.cuh>
#include <utility/include_all.h>
#ifdef _WIN32
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 0> * = 0,
          typename = void>
hostDeviceInline float length3(T arg) {
  return static_cast<float>(arg);
}
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 1> * = 0,
          typename = void, typename = void>
hostDeviceInline float length3(T arg) {
  return arg.x;
}
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 2> * = 0,
          typename = void, typename = void, typename = void>
hostDeviceInline float length3(T arg) {
  return sqrtf(arg.x * arg.x + arg.y * arg.y);
}
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 3> * = 0,
          typename = void, typename = void, typename = void, typename = void>
hostDeviceInline float length3(T arg) {
  return static_cast<float>(math::length3(arg));
}
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 4> * = 0,
          typename = void, typename = void, typename = void, typename = void, typename = void>
hostDeviceInline float length3(T arg) {
  return static_cast<float>(math::length3(math::castTo<typename vector_t<float,math::dimension<T>::value>::type>(arg)));
  //return static_cast<float>(math::length3(math::to<vector_t<float,math::dimension<T>::value>::base_type>(arg)));
}
template <typename T, typename std::enable_if_t<math::dimension<T>::value == 0xDEADBEEF> * = 0>
hostDeviceInline float length3(T) {
  return 1.f;
}
#else
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float>::type 
length3(T) { 
  return 1.f;
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0, float>::type 
length3(T arg) { 
  return arg;
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 1, float>::type 
length3(T arg) { 
  return (float) arg.x;
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 2, float>::type 
length3(T arg) { 
  return sqrtf((float)(arg.x * arg.x + arg.y * arg.y));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 3, float>::type 
length3(T arg) { 
  return sqrt((float) math::dot3(arg, arg));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 4, float>::type 
length3(T arg) { 
  return sqrt((float) math::dot3(arg, arg));
}
#endif
#define returnW(val) float4{static_cast<float>(math::weak_get<1>(arg)), static_cast<float>(math::weak_get<2>(arg)), static_cast<float>(math::weak_get<3>(arg)), static_cast<float>(val)}

template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float4>::type
getX(T arg) {
	return float4{ 0.f,0.f,0.f,1.f };
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value != 0xDEADBEEF, float4>::type
getX(T arg) {
	return returnW(math::weak_get<1>(arg));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float4>::type
getY(T) {
	return float4{ 0.f,0.f,0.f,1.f };
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value != 0xDEADBEEF, float4>::type
getY(T arg) {
	return returnW(math::weak_get<2>(arg));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float4>::type
getZ(T) {
	return float4{ 0.f,0.f,0.f,1.f };
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value != 0xDEADBEEF, float4>::type
getZ(T arg) {
	return returnW(math::weak_get<3>(arg));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float4>::type
getW(T) {
	return float4{ 0.f,0.f,0.f,1.f };
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value != 0xDEADBEEF, float4>::type
getW(T arg) {
	return returnW(math::weak_get<4>(arg));
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value == 0xDEADBEEF, float4>::type
getLength3(T) {
	return float4{ 0.f,0.f,0.f,1.f };
}
template<typename T>
hostDeviceInline typename std::enable_if<math::dimension<T>::value != 0xDEADBEEF, float4>::type
getLength3(T arg) {
	return returnW(length3(arg));
}

template <typename T>
hostDeviceInline void transform_length3(SPH::Visualizer::Memory arrays, T *ptr) {
  checkedParticleIdx(i);
  arrays.renderArray[i] = getLength3(ptr[i]);
}
template <typename T> hostDeviceInline void transform_x(SPH::Visualizer::Memory arrays, T *ptr) {
  checkedParticleIdx(i);
  arrays.renderArray[i] = getX(ptr[i]);
}
template <typename T> hostDeviceInline void transform_y(SPH::Visualizer::Memory arrays, T *ptr) {
  checkedParticleIdx(i);
  arrays.renderArray[i] = getY(ptr[i]);
}
template <typename T> hostDeviceInline void transform_z(SPH::Visualizer::Memory arrays, T *ptr) {
  checkedParticleIdx(i);
  arrays.renderArray[i] = getZ(ptr[i]);
}
template <typename T> hostDeviceInline void transform_w(SPH::Visualizer::Memory arrays, T *ptr) {
  checkedParticleIdx(i);
  arrays.renderArray[i] = getW(ptr[i]);
}

basicFunctionType cudaVisualizeParticles(SPH::Visualizer::Memory arrays, float min, float max) {
  checkedParticleIdx(i);
  auto val = arrays.renderArray[i].w;
  auto mapValue = [](auto val, auto min, auto max) { return (val - min) / (max - min); };
  if (min < max) {
    val = math::min(val, max);
    val = math::max(val, min);
    arrays.renderArray[i].w =
        math::min(math::max(mapValue(arrays.renderArray[i].w, min, max), 0.f), 1.f);
  } else {
    val = math::min(val, min);
    val = math::max(val, max);
    arrays.renderArray[i].w =
        math::min(math::max(1.f - mapValue(arrays.renderArray[i].w, max, min), 0.f), 1.f);
  }
}

basicFunction(visualize, cudaVisualizeParticles, "visualize Particles");
basicFunction(transformLength3, transform_length3, "visualize Particles ( transform_length3 )");
basicFunction(transformX, transform_x, "visualize Particles ( transform_x )");
basicFunction(transformY, transform_y, "visualize Particles ( transform_y )");
basicFunction(transformZ, transform_z, "visualize Particles ( transform_z )");
basicFunction(transformW, transform_w, "visualize Particles ( transform_w )");

void SPH::Visualizer::visualizeParticles(Memory mem) {
  if (mem.num_ptcls == 0)
    return;
  bool found = false;
  bool dotted = get<parameters::render_buffer>().find(".") != std::string::npos;
  //std::cout << dotted << std::endl;
  iterateArraysList([&](auto val) {
    using T = decltype(val);
    //auto variableName = T::variableName;
	//std::cout << T::variableName << " : " << T::qualifiedName << " <-> " << get<parameters::render_buffer>() << " -> " << ((!dotted && T::variableName == get<parameters::render_buffer>()) || (dotted && T::qualifiedName == get<parameters::render_buffer>()) )<< std::endl;
    if ((!dotted && T::variableName == get<parameters::render_buffer>()) || (dotted && T::qualifiedName == get<parameters::render_buffer>())) {
	//if (T::variableName == get<parameters::render_buffer>()) {
      using info = T;
      if (!info::valid() || info::alloc_size < sizeof(typename info::type) * mem.num_ptcls)
        return;
      found = true;
      auto ptr = get<T>();
      if (ptr == nullptr)
        return;
	  //if (sizeof(*ptr) > 4)
		 // get<parameters::visualizingVector>() = 1;
	  //else
		 // get<parameters::visualizingVector>() = 0;

	  if(get<parameters::vectorMode>() == "length")
		  launch<transformLength3>(mem.num_ptcls, mem, T::ptr);
	  if (get<parameters::vectorMode>() == "x")
		  launch<transformX>(mem.num_ptcls, mem, T::ptr);
	  if (get<parameters::vectorMode>() == "y")
		  launch<transformY>(mem.num_ptcls, mem, T::ptr);
	  if (get<parameters::vectorMode>() == "z")
		  launch<transformZ>(mem.num_ptcls, mem, T::ptr);
	  if (get<parameters::vectorMode>() == "w")
		  launch<transformW>(mem.num_ptcls, mem, T::ptr);
     // launch<transformLength3>(mem.num_ptcls, mem, T::ptr);
      if (get<parameters::render_auto>()) {
		  auto min = algorithm::reduce_min(mem.renderArray, mem.num_ptcls);
		  auto max = algorithm::reduce_max(mem.renderArray, mem.num_ptcls);
		  get<parameters::render_min>() = min.w;
		  get<parameters::render_max>() = max.w;
      }
      //launch<visualize>(mem.num_ptcls, mem, get<parameters::render_min>(),
       //                 get<parameters::render_max>());
    }
  });
  if (found == false) {
    using T = arrays::density;
    if (T::ptr == nullptr)
      return;
	//get<parameters::visualizingVector>() = 0;
    launch<transformLength3>(mem.num_ptcls, mem, T::ptr);
    if (get<parameters::render_auto>()) {
		auto min = algorithm::reduce_min(mem.renderArray, mem.num_ptcls);
		auto max = algorithm::reduce_max(mem.renderArray, mem.num_ptcls);
		get<parameters::render_min>() = min.w;
		get<parameters::render_max>() = max.w;
    }
    launch<visualize>(mem.num_ptcls, mem, get<parameters::render_min>(),
                      get<parameters::render_max>());
  }
}
