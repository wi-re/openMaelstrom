#pragma once
#include <utility/launcher.h>
#include <utility/macro.h>
#define iterateNeighbors(var) for (const auto& var : neighbor_iterate_wrapper(i, arrays, false, neigh_tag_ty<neighborhood>{}))
#define iterateCells(position,var) for (const auto& var : cell_iterate<hash_width, order, structure>( position, arrays))
#define iterateAllCells(position,var) for (const auto& var : cell_iterate<hash_width, order, structure>( position, arrays, true))
#define iterateBoundaryPlanes(var) for (const auto& var : boundaryPlanes(arrays))
#define iterateVirtualParticles(position,volume,var) for (const auto& var : virtual_span_u(position, volume, arrays))

#define support_h(x) float_u<SI::m>(x.val.w)
#define support_H(x) float_u<SI::m>(x.val.w * Kernel<kernel_kind::spline4>::kernel_size())

#define INTERNAL_NEIGH_FUNCTION_TYPE(x) template <neighbor_list neighborhood, typename... Ts> x void
#define INTERNAL_CELL_FUNCTION_TYPE(x) template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename... Ts> x void
#define INTERNAL_GLOBAL_FUNCTION_TYPE(x) template<typename... Ts> x void
#define INTERNAL_HYBRID_FUNCTION_TYPE(x) x void

#define neighFunctionType INTERNAL_NEIGH_FUNCTION_TYPE(hostDevice)
#define cellFunctionType INTERNAL_CELL_FUNCTION_TYPE(hostDevice)
#define templateFunctionType INTERNAL_GLOBAL_FUNCTION_TYPE(hostDevice)
#define basicFunctionType INTERNAL_HYBRID_FUNCTION_TYPE(hostDevice)

#define neighFunctionHostType INTERNAL_NEIGH_FUNCTION_TYPE(hostOnly)
#define cellFunctionHostType INTERNAL_CELL_FUNCTION_TYPE(hostOnly)
#define templateFunctionHostType INTERNAL_GLOBAL_FUNCTION_TYPE(hostOnly)
#define basicFunctionHostType INTERNAL_HYBRID_FUNCTION_TYPE(hostOnly)

#define neighFunctionDeviceType INTERNAL_NEIGH_FUNCTION_TYPE(deviceOnly)
#define cellFunctionDeviceType INTERNAL_CELL_FUNCTION_TYPE(deviceOnly)
#define templateFunctionDeviceType INTERNAL_GLOBAL_FUNCTION_TYPE(deviceOnly)
#define basicFunctionDeviceType INTERNAL_HYBRID_FUNCTION_TYPE(deviceOnly)

#define INTERNAL_MACRO_CELL_FUNCTION(config, kind, name, x, ...)                                   \
  template <launch_config cfg, hash_length hash_width, cell_ordering order,                        \
            cell_structuring structure, typename... Vs>                                            \
  struct name {                                                                                    \
    template <typename... Ts> static void launch(int32_t threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<hash_width, order, structure, Vs...>(args...); },                \
          std::function<void(Ts...)>{}, __LINE__, __FILE__, __VA_ARGS__);                          \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
    template <typename... Ts> static void launch(int2 threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<hash_width, order, structure, Vs...>(args...); },                \
          std::function<void(Ts...)>{}, __LINE__, __FILE__, __VA_ARGS__);                          \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
  };

#define INTERNAL_MACRO_NEIGH_FUNCTION(config, kind, name, x, ...)                                  \
  template <launch_config cfg, neighbor_list neighborhood, typename... Vs> struct name {           \
    template <typename... Ts> static void launch(int32_t threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<neighborhood, Vs...>(args...); }, std::function<void(Ts...)>{},  \
          __LINE__, __FILE__, __VA_ARGS__);                                                        \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
    template <typename... Ts> static void launch(int2 threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<neighborhood, Vs...>(args...); }, std::function<void(Ts...)>{},  \
          __LINE__, __FILE__, __VA_ARGS__);                                                        \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
  };

#define INTERNAL_MACRO_TEMPLATE_FUNCTION(config, kind, name, x, ...)                               \
  template <launch_config cfg, typename... Vs> struct name {                                       \
    template <typename... Ts>                                                                      \
    static void launch(int32_t threads, Ts &&... args) {                        \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<Vs...>(args...); }, std::function<void(Ts...)>{}, __LINE__,      \
          __FILE__, __VA_ARGS__);                                                                  \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
    template <typename... Ts>                                                                      \
    static void launch(int2 threads, Ts &&... args) {                        \
      static auto launcher = get_FunctionLauncher<config>(                                         \
          [] kind(Ts... args) { x<Vs...>(args...); }, std::function<void(Ts...)>{}, __LINE__,      \
          __FILE__, __VA_ARGS__);                                                                  \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
  };

#define INTERNAL_MACRO_GLOBAL_FUNCTION(config, kind, name, x, ...)                                 \
  template <launch_config cfg> struct name {                                                       \
    template <typename... Ts> static void launch(int32_t threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>([] kind(Ts... args) { x(args...); },     \
                                                          std::function<void(Ts...)>{}, __LINE__,  \
                                                          __FILE__, __VA_ARGS__);                  \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
    template <typename... Ts> static void launch(int2 threads, Ts &&... args) {                 \
      static auto launcher = get_FunctionLauncher<config>([] kind(Ts... args) { x(args...); },     \
                                                          std::function<void(Ts...)>{}, __LINE__,  \
                                                          __FILE__, __VA_ARGS__);                  \
      launcher.TEMPLATE_TOKEN operator()<cfg>(threads, std::forward<Ts>(args)...);                                \
    }                                                                                              \
  };

#define cellFunction(name, x, ...) INTERNAL_MACRO_CELL_FUNCTION(launch_config::_used_for_template_specializations, hostDevice, name, x, __VA_ARGS__)
#define neighFunction(name, x, ...) INTERNAL_MACRO_NEIGH_FUNCTION(launch_config::_used_for_template_specializations, hostDevice, name, x, __VA_ARGS__)
#define templateFunction(name, x, ...) INTERNAL_MACRO_TEMPLATE_FUNCTION(launch_config::_used_for_template_specializations, hostDevice, name, x, __VA_ARGS__)
#define basicFunction(name, x, ...) INTERNAL_MACRO_GLOBAL_FUNCTION(launch_config::_used_for_template_specializations, hostDevice, name, x, __VA_ARGS__)

#define cellFunctionHost(name, x, ...) INTERNAL_MACRO_CELL_FUNCTION(launch_config::host, hostOnly, name, x, __VA_ARGS__)
#define neighFunctionHost(name, x, ...) INTERNAL_MACRO_NEIGH_FUNCTION(launch_config::host, hostOnly, name, x, __VA_ARGS__)
#define templateFunctionHost(name, x, ...) INTERNAL_MACRO_TEMPLATE_FUNCTION(launch_config::host, hostOnly, name, x, __VA_ARGS__)
#define basicFunctionHost(name, x, ...) INTERNAL_MACRO_GLOBAL_FUNCTION(launch_config::host, hostOnly, name, x, __VA_ARGS__)

#define cellFunctionDevice(name, x, ...) INTERNAL_MACRO_CELL_FUNCTION(launch_config::device, deviceOnly, name, x, __VA_ARGS__)
#define neighFunctionDevice(name, x, ...) INTERNAL_MACRO_NEIGH_FUNCTION(launch_config::device,deviceOnly, name, x, __VA_ARGS__)
#define templateFunctionDevice(name, x, ...) INTERNAL_MACRO_TEMPLATE_FUNCTION(launch_config::device,deviceOnly, name, x, __VA_ARGS__)
#define basicFunctionDevice(name, x, ...) INTERNAL_MACRO_GLOBAL_FUNCTION(launch_config::device,deviceOnly, name, x, __VA_ARGS__)

#include <utility/SPH/indication.h>


template <typename Func, typename T, typename U>
__host__ __device__ void cell_linear_interpolate(Func &&fn, T &&mem, U position) {
	uint32_t num_cells = math::unit_get<1>(mem.grid_size) * math::unit_get<2>(mem.grid_size) *
		math::unit_get<3>(mem.grid_size);
	uint3 blockIdx = position_to_idx3D(position, mem.min_domain, math::unit_get<1>(mem.cell_size));
	{
		uint arrayIdx = 0;
		int3 cellIdx;

		//#pragma unroll 3
		for (int z = -1; z < 2; ++z) {
			cellIdx.z = blockIdx.z + z;
			//#pragma unroll 3
			for (int y = -1; y < 2; ++y) {
				cellIdx.y = blockIdx.y + y;
				//#pragma unroll 3
				for (int x = -1; x < 2; ++x) {
					cellIdx.x = blockIdx.x + x;
					if (cellIdx.x > -1 && cellIdx.x < (int)math::unit_get<1>(mem.grid_size) &&
						cellIdx.y > -1 && cellIdx.y < (int)math::unit_get<2>(mem.grid_size) &&
						cellIdx.z > -1 && cellIdx.z < (int)math::unit_get<3>(mem.grid_size)) {
						uint idx = idx3D_to_linear(
							uint3{ static_cast<uint32_t>(cellIdx.x), static_cast<uint32_t>(cellIdx.y),
							static_cast<uint32_t>(cellIdx.z) },
							uint3{ static_cast<uint32_t>(math::unit_get<1>(mem.grid_size)),
							static_cast<uint32_t>(math::unit_get<2>(mem.grid_size)),
							static_cast<uint32_t>(math::unit_get<3>(mem.grid_size)) });
						uint2 cellspan;
						if (idx < num_cells) {
							cellspan.x = mem.cellBegin[idx];
							cellspan.y = mem.cellEnd[idx];
							uint32_t neighbor_idx = 0;
							for (neighbor_idx = cellspan.x; neighbor_idx < cellspan.y; ++neighbor_idx) {
								fn(neighbor_idx);
							}
						}
					}
					arrayIdx++;
				}
			}
		}
	}
}

template <typename Func, typename T, typename U>
__host__ __device__ void hash_z_interpolate(Func &&fn, T &&mem, U position) {
	auto idx = position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size));
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint64_t morton = idx3D_to_morton(cell);

				hash_span s = mem.hashMap[idx3D_to_hash(cell, mem.hash_entries)];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					auto cs = mem.cellSpan[i];
					auto hash = position_to_morton(mem.position[cs.beginning], mem);
					if (hash == morton) {
						for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
							fn(j);
						}
						break;
					}
				}
			}
		}
	}
}

template <typename Func, typename T, typename U>
__host__ __device__ void mlm_interpolate(Func &&fn, T &&mem, U position) {
	auto support = math::unit_get<4>(position) *kernelSize();
	int32_t resolution = (float)math::clamp(
		math::floorf(math::abs(math::log2(support / math::unit_get<1>(mem.cell_size)))), 0,
		mem.mlm_schemes - 1);
	float factor = powf(0.5f, ((float)resolution));

	auto idx =
		position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size) * factor);
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint64_t morton = idx3D_to_morton(cell);

				hash_span s =
					mem.hashMap[idx3D_to_hash(cell, mem.hash_entries) + mem.hash_entries * resolution];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					auto cs = mem.cellSpan[i + mem.max_numptcls * resolution];
					auto hash = position_to_morton(mem.position[cs.beginning], mem, factor);
					if (hash == morton) {
						for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
							fn(j);
						}
						break;
					}
				}
			}
		}
	}
}

template <typename Func, typename T, typename U>
__host__ __device__ void hash_z_interpolate_32_all(Func &&fn, T &&mem, U position) {
	auto idx = position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size));
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint32_t morton = idx3D_to_morton_32(cell);

				hash_span s = mem.hashMap[idx3D_to_hash(cell, mem.hash_entries)];
				if (s.beginning == -1)
					continue;
				cell_span cs = mem.cellSpan[s.beginning];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					cs = mem.cellSpan[i];
					for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
						fn(j);
					}
				}
			}
		}
	}
}
template <typename Func, typename T, typename U>
__host__ __device__ void hash_z_interpolate_all(Func &&fn, T &&mem, U position) {
	auto idx = position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size));
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint64_t morton = idx3D_to_morton(cell);

				hash_span s = mem.hashMap[idx3D_to_hash(cell, mem.hash_entries)];
				if (s.beginning == -1)
					continue;
				cell_span cs = mem.cellSpan[s.beginning];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					cs = mem.cellSpan[i];
					for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
						fn(j);
					}
				}
			}
		}
	}
}

template <typename Func, typename T, typename U>
__host__ __device__ void mlm_interpolate_all(Func &&fn, T &&mem, U position) {
	auto support = math::unit_get<4>(position) * kernelSize();
	int32_t resolution = (float)math::clamp(
		math::floorf(math::abs(math::log2(support / math::unit_get<1>(mem.cell_size)))), 0,
		mem.mlm_schemes - 1);
	float factor = powf(0.5f, ((float)resolution));

	auto idx =
		position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size) * factor);
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint64_t morton = idx3D_to_morton(cell);

				hash_span s =
					mem.hashMap[idx3D_to_hash(cell, mem.hash_entries) + mem.hash_entries * resolution];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					auto cs = mem.cellSpan[i + mem.max_numptcls * resolution];
					auto hash = position_to_morton(mem.position[cs.beginning], mem);
					for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
						fn(j);
					}
				}
			}
		}
	}
}

template <typename Func, typename T, typename U>
__host__ __device__ void hash_z_interpolate_32(Func &&fn, T &&mem, U position) {
	auto idx = position_to_idx3D_i(position, mem.min_coord, math::unit_get<1>(mem.cell_size));
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
					static_cast<uint32_t>(idx.z + k) };
				if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
					continue;
				uint32_t morton = idx3D_to_morton_32(cell);

				hash_span s = mem.hashMap[idx3D_to_hash(cell, mem.hash_entries)];
				for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
					auto cs = mem.cellSpan[i];
					auto hash = position_to_morton_32(mem.position[cs.beginning], mem);
					if (hash == morton) {
						for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
							fn(j);
						}
						break;
					}
				}
			}
		}
	}
}

namespace detail {
	template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename Func,
		typename T, typename U>
		struct F;

	template <typename Func, typename T, typename U>
	struct F<hash_length::bit_64, cell_ordering::z_order, cell_structuring::hashed, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			hash_z_interpolate(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F<hash_length::bit_32, cell_ordering::z_order, cell_structuring::hashed, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			hash_z_interpolate_32(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F<hash_length::bit_32, cell_ordering::linear_order, cell_structuring::complete, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			cell_linear_interpolate(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F<hash_length::bit_64, cell_ordering::z_order, cell_structuring::MLM, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			mlm_interpolate(fn, mem, position);
		}
	};

	template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename Func,
		typename T, typename U>
		struct F_all;

	template <typename Func, typename T, typename U>
	struct F_all<hash_length::bit_64, cell_ordering::z_order, cell_structuring::hashed, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			hash_z_interpolate_all(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F_all<hash_length::bit_32, cell_ordering::z_order, cell_structuring::hashed, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			hash_z_interpolate_32_all(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F_all<hash_length::bit_32, cell_ordering::linear_order, cell_structuring::complete, Func, T,
		U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			cell_linear_interpolate(fn, mem, position);
		}
	};
	template <typename Func, typename T, typename U>
	struct F_all<hash_length::bit_64, cell_ordering::z_order, cell_structuring::MLM, Func, T, U> {
		__host__ __device__ static void interpolate(Func &fn, T &mem, U position) {
			mlm_interpolate_all(fn, mem, position);
		}
	};
} // namespace detail

template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename Func,
	typename T, typename U>
	__host__ __device__ void cell_interpolate(Func &fn, T &mem, U position) {
	detail::F<hash_width, order, structure, Func, T, U>::interpolate(fn, mem, position);
}

template <hash_length hash_width, cell_ordering order, cell_structuring structure, typename Func,
	typename T, typename U>
	__host__ __device__ void cell_interpolate_all(Func &fn, T &mem, U position) {
	detail::F_all<hash_width, order, structure, Func, T, U>::interpolate(fn, mem, position);
}
