// Our compiler who art in the computer casing, cooled down be your temps, thy numbers sensible, and your results publishable.
#define NO_QT
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
//#if defined(__clang__) && defined(__CUDA__)
//#endif
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <render/vrtxRender/vrtxRender.h>
#include <sstream>
#include <texture_types.h>
#include <utility/math.h>
#include <vector_functions.h>
#include <vector_types.h>

//#define SMALL_SCALE_DETAIL
//#define ISO_DENSITY
#define ANISOTROPIC
#pragma region MACROS
//#ifdef __INTELLISENSE__
//#define gridDim                                                                                                         \
//  (int3 { 32, 1, 1 })
//#define blockDim                                                                                                       \
//  (int3 { 32, 1, 1 })
//#define threadIdx                                                                                                      \
//  (int3 { 0, 0, 0 })
//#define blockIdx                                                                                                       \
//  (int3 { 0, 0, 0 })
//#endif
#ifdef __INTELLISENSE__
#define CPSYMBOL(symbol, var)
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel
#else
#define CPSYMBOL(symbol, var) cudaMemcpyToSymbol(symbol, &var, sizeof(symbol))
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel<<<blocks, tpb, sm, stream>>>
#endif
#pragma endregion

#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR 1e-3f // epsilon
#define samps 1            // samples
#define BVH_STACK_SIZE 32

namespace vrtx {

	__device__ bool RayIntersectsBox(const gpuBVH& bvh, const float3 &originInWorldSpace, const float3 &rayInWorldSpace, int boxIdx) {
		float Tnear, Tfar;
		Tnear = -FLT_MAX;
		Tfar = FLT_MAX;

		float2 limits;

#define CHECK_NEAR_AND_FAR_INTERSECTION(c)                                                                             \
  if (rayInWorldSpace.c == 0.f) {                                                                                    \
    if (originInWorldSpace.c < limits.x)                                                                             \
      return false;                                                                                                    \
    if (originInWorldSpace.c > limits.y)                                                                             \
      return false;                                                                                                    \
  } else {                                                                                                             \
    float T1 = (limits.x - originInWorldSpace.c) / rayInWorldSpace.c;                                              \
    float T2 = (limits.y - originInWorldSpace.c) / rayInWorldSpace.c;                                              \
    if (T1 > T2) {                                                                                                     \
      float tmp = T1;                                                                                                  \
      T1 = T2;                                                                                                         \
      T2 = tmp;                                                                                                        \
    }                                                                                                                  \
    if (T1 > Tnear)                                                                                                    \
      Tnear = T1;                                                                                                      \
    if (T2 < Tfar)                                                                                                     \
      Tfar = T2;                                                                                                       \
    if (Tnear > Tfar)                                                                                                  \
      return false;                                                                                                    \
    if (Tfar < 0.f)                                                                                                    \
      return false;                                                                                                    \
  }
		auto lim = bvh.cudaBVHlimits[boxIdx];
		limits = float2{ lim.bottom.x, lim.top.x };
		CHECK_NEAR_AND_FAR_INTERSECTION(x)
			limits = float2{ lim.bottom.y, lim.top.y };
		CHECK_NEAR_AND_FAR_INTERSECTION(y)
			limits = float2{ lim.bottom.z, lim.top.z };
		CHECK_NEAR_AND_FAR_INTERSECTION(z)
			return true;
	}

	__device__ bool BVH_IntersectTriangles(gpuBVH& bvh, const float3 &origin, const float3 &ray,
		unsigned avoidSelf, int &pBestTriIdx, float3 &pointHitInWorldSpace, float &kAB,
		float &kBC, float &kCA, float &hitdist, float3 &boxnormal) {
		pBestTriIdx = -1;
		float bestTriDist;
		bestTriDist = FLT_MAX;
		int32_t stack[BVH_STACK_SIZE];
		int32_t stackIdx = 0;
		stack[stackIdx++] = 0;
		while (stackIdx) {
			int32_t boxIdx = stack[stackIdx - 1];
			stackIdx--;
			uint4 data = bvh.cudaBVHindexesOrTrilists[boxIdx];
			if (!(data.x & 0x80000000)) { // INNER NODE
				if (RayIntersectsBox(bvh, origin, ray, boxIdx)) {
					stack[stackIdx++] = data.y;
					stack[stackIdx++] = data.z;
					if (stackIdx > BVH_STACK_SIZE) {
						return false;
					}
				}
			}
			else {
				for (uint32_t i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {
					int32_t idx = bvh.cudaTriIdxList[i];
					if (avoidSelf == idx)
						continue;
					float4 normal = bvh.cudaTriangleIntersectionData[idx].normal;
					float d = math::sqlength3(normal);
					float k = math::dot3(normal, ray);
					if (k == 0.0f)
						continue;
					float s = (normal.w - math::dot3(normal, origin)) / k;
					if (s <= 0.0f)
						continue;
					if (s <= NUDGE_FACTOR)
						continue;
					float3 hit = ray * s;
					hit += origin;

					float4 ee1 = bvh.cudaTriangleIntersectionData[idx].e1d1;
					float kt1 = math::dot3(ee1, hit) - ee1.w;
					if (kt1 < 0.0f)
						continue;
					float4 ee2 = bvh.cudaTriangleIntersectionData[idx].e2d2;
					float kt2 = math::dot3(ee2, hit) - ee2.w;
					if (kt2 < 0.0f)
						continue;
					float4 ee3 = bvh.cudaTriangleIntersectionData[idx].e3d3;
					float kt3 = math::dot3(ee3, hit) - ee3.w;
					if (kt3 < 0.0f)
						continue;
					{
						float hitZ = math::sqdistance(origin, hit);
						if (hitZ < bestTriDist) {
							bestTriDist = hitZ;
							hitdist = sqrtf(bestTriDist);
							pBestTriIdx = idx;
							pointHitInWorldSpace = hit;
							kAB = kt1;
							kBC = kt2;
							kCA = kt3;
						}
					}
				}
			}
		}

		return pBestTriIdx != -1;
	}
	__shared__ extern float sm_data[];
	struct rayState {
		uint32_t rayDone : 1;
		uint32_t threadDone : 1;
		uint32_t rayBounced : 1;
		uint32_t rayHitFluidAABB : 1;
		uint32_t rayHitFluidSurface : 1;
		int32_t index : 27;
	};
#pragma region types
#pragma endregion
#pragma region globals
	Pixel *cuImage;
	worldRay *cuCurrentRays;
	worldRay *cuCompactedRays;
	//int32_t *cuInternalFlag;
	int32_t *rayCounter;
	uint32_t *cRNGSeeds;
	int32_t* cuResortIndex;
	int32_t *cuResortKey;
	float *cufluidDepth;
	float4 *cufluidIntersection;
	float4* cuFluidColor;
	Box* cuBoxes = nullptr;
	Sphere* cuSpheres = nullptr;

	__device__ __constant__ SceneInformation cScene;
	__device__ __constant__ vrtxFluidMemory fluidMemory;
	//__device__ __constant__ Box boxes[] = {
	//	//{{-25.f, -25.f, 96.f},{25.f,25.f, 132.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, DIFF},
	//	{{190.f, -192.f, -192.f},{192.f,192.f, 192.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, DIFF}
	//	,{ {-521, -FLT_MAX, -FLT_MAX},{-51, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, DIFF}
	//	//,{ {-FLT_MAX, -25.f, -FLT_MAX},{32, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, DIFF}
	//};
	//__device__ __constant__ Sphere spheres[] = {
	//    //{16, {192.0f, 192, 192}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, DIFF},
	//	{32, {-96, 0, 16}, {0, 0, 0}, {1.f, 1.f, 1.f}, SPEC},
	//	{32, {-96, -64, 16}, {0, 0, 0}, {0.5f, 0.f, 0.f}, DIFF},
	//	{32, {-96, 64, 64}, {0, 0, 0}, {1.0f, 1.f, 1.f}, REFR},
	//	{10000, {50.0f, 40.8f, -1060}, {0.55f, 0.55f, 0.55f}, {0.075f, 0.075f, 0.075f}, DIFF},
	//    //{10000, {50.0f, 40.8f, -1060}, {0.55, 0.55, 0.55}, {0.175f, 0.175f, 0.175f}, DIFF},
	//	//{10000, {50.0f, 40.8f, -1060}, {0.f,0.f,0.f}, {0.f,0.f,0.f}, DIFF},
	//
	//    {100000, {0.0f, 0, -100000.}, {0, 0, 0}, {0.2f, 0.2f, 0.2f}, DIFF},
	//    {100000, {0.0f, 0, -100000.1}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, DIFF}};
	__device__ __constant__ Box* cBoxes;
	__device__ __constant__ Sphere* cSpheres;
	__device__ __constant__ int32_t cNumBoxes;
	__device__ __constant__ int32_t cNumSpheres;

	__device__ __constant__ int32_t cNumRays;
	__device__ __constant__ worldRay *cRaysDepth;
	__device__ __constant__ worldRay *cCompactRays;
	//__device__ __constant__ int32_t *cInternalFlag;
	__device__ __constant__ int32_t *cRayCounter;
	__device__ __constant__ Pixel *cImage;
	__device__ __constant__ uint32_t *cuSeeds;
	__device__ __constant__ float *cfluidDepth;
	__device__ __constant__ int32_t *cResortIndex;
	__device__ __constant__ int32_t *cResortKey;
	__device__ __constant__ float4 *cfluidIntersection;
	__device__ __constant__ float4* cFluidColor;
	__device__ __constant__ vrtxFluidArrays arrays;

	__device__ auto radiusFromVolume(float volume) {
		return powf(volume * PI4O3_1, 1.f / 3.f);
	}
	__device__ auto radiusFromVolume(int32_t i) {
		return powf(arrays.volume[i] * PI4O3_1, 1.f / 3.f);
	}
#define G(i, x, y) (arrays.anisotropicMatrices[i * 9 + (x * 3 + y)])

	surface<void, cudaSurfaceType2D> surfaceWriteOut;
#pragma endregion
#pragma region helper_functions
	__device__ __host__ __inline__ int8_t sgn(float x) { return x > 0.f ? 1 : (x < 0.f ? -1 : 0); }
	__device__ auto randf(int32_t index) {
		auto x = cuSeeds[index];
		x ^= x >> 13;
		x ^= x << 17;
		x ^= x >> 5;
		cuSeeds[index] = x;
		auto r = (x & 0x007FFFFF) | 0x3F800000;
		return *reinterpret_cast<float *>(&r) - 1.f;
	}
	template <typename T, typename U>
	hostDeviceInline auto position_to_morton_32_i_actual(T p, U &arrays, float factor = 1.f) {
		return idx3D_to_morton_32(
			position_to_idx3D_i(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size_actual) * factor));
	}

	namespace common {
		__device__ auto generateCameraRay(int32_t x, int32_t y, curandState &randState, int32_t i) {
			float3 rendercampos = float3{ cScene.m_camera.position.x, cScene.m_camera.position.y, cScene.m_camera.position.z };

			int32_t pixelx = x;
			int32_t pixely = cScene.height - y - 1;

			// float3 finalcol = float3{ 0.0f, 0.0f, 0.0f };
			float3 rendercamview =
				math::normalize(float3{ cScene.m_camera.view.x, cScene.m_camera.view.y, cScene.m_camera.view.z });
			float3 rendercamup = math::normalize(float3{ cScene.m_camera.up.x, cScene.m_camera.up.y, cScene.m_camera.up.z });
			float3 horizontalAxis = math::normalize(math::cross(rendercamview, rendercamup));
			float3 verticalAxis = math::normalize(math::cross(horizontalAxis, rendercamview));

			float3 middle = rendercampos + rendercamview;
			float3 horizontal = horizontalAxis * tanf(cScene.m_camera.fov.x * 0.5f * (CUDART_PI_F / 180));
			float3 vertical = -verticalAxis * tanf(-cScene.m_camera.fov.y * 0.5f * (CUDART_PI_F / 180));

			float jitterValueX = curand_uniform(&randState) - 0.5f;
			float jitterValueY = curand_uniform(&randState) - 0.5f;
			float sx = (jitterValueX + pixelx) / (cScene.width - 1);
			float sy = (jitterValueY + pixely) / (cScene.height - 1);

			// compute pixel on screen
			float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
			float3 pointOnImagePlane =
				rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cScene.m_camera.focalDistance);

			float3 aperturePoint;
			if (cScene.m_camera.apertureRadius > 0.00001f) {
				float random1 = curand_uniform(&randState);
				float random2 = curand_uniform(&randState);
				float angle = 2.f * CUDART_PI_F * random1;
				float distance = cScene.m_camera.apertureRadius * sqrtf(random2);
				float apertureX = cos(angle) * distance;
				float apertureY = sin(angle) * distance;

				aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
			}
			else {
				aperturePoint = rendercampos;
			}
			float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
			apertureToImagePlane = math::normalize(apertureToImagePlane);
			float3 rayInWorldSpace = math::normalize(apertureToImagePlane);
			float3 originInWorldSpace = aperturePoint;

			return worldRay{ originInWorldSpace, 1e21f, rayInWorldSpace, 0u, 0u, (uint32_t)i };
		}
		__global__ void generateBlockedRays(int32_t seed, Pixel *image, worldRay *rays, worldRay *oldRays, int32_t msaa_factor) {
			int32_t x = blockIdx.x * blockDim.y + threadIdx.y;
			int32_t y = blockIdx.y * blockDim.z + threadIdx.z;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			int32_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
			int32_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
				(threadIdx.y * blockDim.x) + threadIdx.x;

			curandState randState;
			curand_init(seed + threadId, 0, 0, &randState);

			image[i] = Pixel{ float3{0.f, 0.f, 0.f}, float3{1.f, 1.f, 1.f} };
			auto worldRay = generateCameraRay(x, y, randState, i);
			rays[i * msaa_factor + threadIdx.x] = worldRay;
			cfluidIntersection[i] = float4{ 0.f,0.f,0.f, FLT_MAX };
			//cInternalFlag[i] = 0;
		}
		//#define DEBUG_NORMALS

		__global__ void toneMap(int32_t frameNumber, float3 *accumBuffer, Pixel *image, float rate) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			accumBuffer[i] += image[i].color / rate;
			float3 tempcol = (accumBuffer[i] / frameNumber);
			float3 colour = float3{ math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
								   math::clamp(tempcol.z, 0.0f, 1.0f) };
#ifdef DEBUG_NORMALS
			float4 out{ colour.x, colour.y, colour.z, 1.f };
#else
			float4 out{ (powf(colour.x, 1 / 2.2f)), (powf(colour.y, 1 / 2.2f)), (powf(colour.z, 1 / 2.2f)), 1.f };
#endif
			// out = float4{ colour.x, colour.y, colour.z, 1.f };
			#if defined(__CUDA_ARCH__)
			surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
			#endif
		}
		__global__ void toneMapNormals(int32_t frameNumber, float3 *accumBuffer, Pixel *image, float rate) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			accumBuffer[i] += image[i].color / rate;
			float3 tempcol = (accumBuffer[i] / frameNumber);
			float3 colour = float3{ math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
								   math::clamp(tempcol.z, 0.0f, 1.0f) };
			float4 out{ colour.x, colour.y, colour.z, 1.f };
			#if defined(__CUDA_ARCH__)
			surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
			#endif
		}
	} // namespace common
	namespace aabb {
		__device__ auto rayIntersectAABB(Ray r, float3 b_min, float3 b_max) {
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
			float t1 = (b_min.x - r.orig.x) / r.dir.x;
			float t2 = (b_max.x - r.orig.x) / r.dir.x;

			float tmin = MIN(t1, t2);
			float tmax = MAX(t1, t2);

			t1 = (b_min.y - r.orig.y) / r.dir.y;
			t2 = (b_max.y - r.orig.y) / r.dir.y;

			tmin = MAX(tmin, MIN(t1, t2));
			tmax = MIN(tmax, MAX(t1, t2));

			t1 = (b_min.z - r.orig.z) / r.dir.z;
			t2 = (b_max.z - r.orig.z) / r.dir.z;

			tmin = MAX(tmin, MIN(t1, t2));
			tmax = MIN(tmax, MAX(t1, t2));

			return AABBHit{ tmax > MAX(tmin, 0.f), tmin, tmax };
		}
	} // namespace aabb
	namespace traversal {
		hostDeviceInline auto idx3D_to_hash_local(int3 idx, uint32_t hash_entries) {
			int32_t morton = idx3D_to_morton_32(idx);
			return morton % hash_entries;
		}
		__device__ auto lookup_cell(const int3 &idx) {
			if (idx.x >= fluidMemory.grid_size.x || idx.y >= fluidMemory.grid_size.y || idx.z >= fluidMemory.grid_size.z)
				return INT_MAX;
			if (idx.x < 0 || idx.y < 0 || idx.z < 0)
				return INT_MAX;
			auto morton = idx3D_to_hash(idx, fluidMemory.hash_entries);
			auto s = fluidMemory.hashMap[idx3D_to_morton(idx) % fluidMemory.hash_entries];
			if (s.hash.kind == LIST_ALWAYS_FALSE)
				return INT_MAX;
			if (s.hash.kind == LIST_ALWAYS_TRUE)
				return 1;
			if (s.hash.kind == LIST_COMPACT)
				if (morton == s.cell.hash)
					return 1;
			for (int32_t ii = s.hash.beginning; ii < s.hash.beginning + s.hash.length; ++ii)
				if (fluidMemory.cellSpan[ii].cell.hash == morton)
					return 1;
			return INT_MAX;
		}
		__device__ __host__ float mod(float a, float N) {
			return a - N * floorf(a / N);
		}
		__device__ __host__ float intBound2_s(float s, float ds) {
			if (s == floorf(s) && ds < 0.f)
				return 0.f;
			if (ds < 0.f)
				return intBound2_s(-s, -ds);
			float s2 = mod(s, 1.f);
			return (1.f - s2) / ds;
		}
		__device__ float3 intBound(const float3& s, const float3& ds) {
			//return (ds > 0 ? Math.ceil(s) - s : s - Math.floor(s)) / Math.abs(ds);
			//return float3{
			//	(ds.x > 0.f ? ceilf(s.x) - s.x : s.x - floorf(s.x)) / fabsf(ds.x),
			//	(ds.y > 0.f ? ceilf(s.y) - s.y : s.y - floorf(s.y)) / fabsf(ds.y),
			//	(ds.z > 0.f ? ceilf(s.z) - s.z : s.z - floorf(s.z)) / fabsf(ds.z)
			//};
			return float3{
				intBound2_s(s.x, ds.x),
				intBound2_s(s.y, ds.y),
				intBound2_s(s.z, ds.z)
			};
		}
		__device__ float3 intBoundRay(const Ray &r) {
			return intBound((r.orig - fluidMemory.min_coord) / fluidMemory.cell_size.x, r.dir);
		}
		__device__ float3 intBoundRay(const Ray &r, float t) {
			return intBound((r(t) - fluidMemory.min_coord) / fluidMemory.cell_size.x, r.dir);
		}
	} // namespace traversal
#pragma endregion
	namespace render {
		__device__ int lanemask_lt(int lane) {
			return (1 << (lane)) - 1;
		}
		// increment the value at ptr by 1 and return the old value
		__device__ int atomicAggInc(int *p) {
			unsigned int writemask = __activemask();
			unsigned int total = __popc(writemask);
			unsigned int prefix = __popc(writemask & lanemask_lt(threadIdx.x & (warpSize - 1)));
			// Find the lowest-numbered active lane
			int elected_lane = __ffs(writemask) - 1;
			int base_offset = 0;
			if (prefix == 0) {
				base_offset = atomicAdd(p, total);
			}
			base_offset = __shfl_sync(writemask, base_offset, elected_lane);
			int thread_offset = prefix + base_offset;
			return thread_offset;
		}

		struct compactIterator {
			int3 central_idx;
			bool full_loop;

			hostDevice compactIterator(int3 position) :central_idx(position) {
			}
			struct cell_iterator {
				int3 idx;
				int32_t i = -1, j = -1, k = -1;
				uint32_t ii = 0;
				int32_t jj = 0;

				int32_t neighbor_idx;

				compactHashSpan s{ 0, UINT31_MAX, -1 };
				compact_cellSpan cs{ -1, -1 };

				hostDevice int32_t cs_loop() {
					if (cs.beginning != -1 && jj < cs.beginning + cs.length) {
						neighbor_idx = jj;
						++jj;
						return neighbor_idx;
					}
					return -1;
				}

				hostDevice int32_t s_loop() {
					if (s.beginning != UINT31_MAX) {
						uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
							static_cast<uint32_t>(idx.z + k) };
						if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
							return -1;
						auto morton = idx3D_to_morton(cell);
						if (s.compacted && ii < s.beginning + s.length) {
							cs = cell_span{ (int32_t)s.beginning, s.length };
							jj = cs.beginning;
							ii = s.beginning + s.length;
							if (
								position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, 1.f) == morton) {
								if (cs_loop() != -1) {
									return neighbor_idx;
								}
							}
						}
						for (; ii < s.beginning + s.length;) {
							cs = arrays.compactCellSpan[ii];
							++ii;
							jj = cs.beginning;
							if (
								position_to_morton(pairHelper(arrays.position)[cs.beginning], arrays, 1.f) == morton) {
								if (cs_loop() != -1) {
									return neighbor_idx;
								}
							}
						}
						++k;
					}
					return -1;
				}

				hostDevice void increment() {
					if (cs_loop() != -1)
						return;
					if (s_loop() != -1)
						return;

					for (; i <= 1; ++i) {
						for (; j <= 1; ++j) {
							for (; k <= 1;) {
								uint3 cell = uint3{ static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
									static_cast<uint32_t>(idx.z + k) };
								if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
									++k;
									continue;
								}
								auto morton = idx3D_to_morton(cell);

								s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries)];
								ii = s.beginning;
								if (s.beginning == UINT31_MAX) {
									++k;
									continue;
								}
								if (s_loop() != -1)
									return;
							}
							k = -1;
						}
						j = -1;
					}
				}

				hostDevice cell_iterator(int3 c_idx, int32_t _i = -1,
					int32_t _j = -1, int32_t _k = -1)
					: idx(c_idx), i(_i), j(_j), k(_k) {
					int32_t thread_idx = getThreadIdx();
					increment();
				}

				hostDeviceInline int32_t operator*() { return neighbor_idx; };
				hostDeviceInline bool operator==(const cell_iterator &rawIterator) const {
					return (i == rawIterator.i);
				}
				hostDeviceInline bool operator!=(const cell_iterator &rawIterator) const {
					return (i != rawIterator.i);
				}

				hostDeviceInline cell_iterator &operator++() {
					increment();
					return (*this);
				}
				hostDeviceInline cell_iterator operator++(int) {
					auto temp(*this);
					increment();
					return temp;
				}
			};

			hostDeviceInline cell_iterator begin() const {
				return cell_iterator(central_idx);
			}
			hostDeviceInline cell_iterator end() const {
				return cell_iterator(central_idx, 2, 2, 2);
			}
			hostDeviceInline cell_iterator cbegin() const {
				return cell_iterator(central_idx);
			}
			hostDeviceInline cell_iterator cend() const {
				return cell_iterator(central_idx, 2, 2, 2);
			}
		};

		template<typename T>
		__device__ __host__ __inline__ auto square(T&& x) {
			return x * x;
		}
		template<typename T>
		__device__ __host__ __inline__ auto cube(T&& x) {
			return x * x * x;
		}
		__device__ __host__ __inline__ float k(float4 x_i, float4 x_j) {
			auto h = (x_i.w + x_j.w) * 0.5f * kernelSize();
			auto d = math::distance3(x_i, x_j);
			auto s = d / h;
			return math::max(0.f, cube(1.f - square(s)));
		}
		__device__ __inline__ float turkAnisotropic(float4 position, int32_t j) {
			float4 xBar_j = arrays.centerPosition[j];
			float det = xBar_j.w;
			float3 diff = math::castTo<float3>(position - xBar_j);
			float a1 = G(j, 0, 0) * diff.x + G(j, 0, 1) * diff.y + G(j, 0, 2) * diff.z;
			float a2 = G(j, 1, 0) * diff.x + G(j, 1, 1) * diff.y + G(j, 1, 2) * diff.z;
			float a3 = G(j, 2, 0) * diff.x + G(j, 2, 1) * diff.y + G(j, 2, 2) * diff.z;
			float q = sqrtf(a1 * a1 + a2 * a2 + a3 * a3);
			float sigma = 16.f / CUDART_PI_F;
			float W = 0.f;
			if (q <= 0.5f) {
				auto q1 = 1.f - q;
				auto q2 = 0.5f - q;
				W = (q1 * q1 * q1) - 4.f * (q2 * q2 * q2);
			}
			else if ((q <= 1.0f) && (q > 0.5f)) {
				auto q1 = 1.f - q;
				W = q1 * q1 * q1;
			}
			return det * W;
		}
		__device__ __inline__ float3 turkAnisotropicGradient(float4 position, int32_t j) {
			float4 xBar_j = arrays.centerPosition[j];
			float det = xBar_j.w;
			float3 diff = math::castTo<float3>(position - xBar_j);
			float a1 = G(j, 0, 0) * diff.x + G(j, 0, 1) * diff.y + G(j, 0, 2) * diff.z;
			float a2 = G(j, 1, 0) * diff.x + G(j, 1, 1) * diff.y + G(j, 1, 2) * diff.z;
			float a3 = G(j, 2, 0) * diff.x + G(j, 2, 1) * diff.y + G(j, 2, 2) * diff.z;
			float q = sqrtf(a1 * a1 + a2 * a2 + a3 * a3);
			float sigma = 1.f / CUDART_PI_F;
			float W = 0.f;
			if (q <= 0.5f) {
				auto q1 = 1.f - q;
				auto q2 = 0.5f - q;
				W = -3.f * q1 * q1 + 12.f * q2 * q2;
			}
			else if ((q <= 1.0f) && (q > 0.5f)) {
				auto q1 = 1.f - q;
				W = -3.f * (q1 * q1);
			}
			return q >= 1e-12f ? 16.f * math::normalize3(diff) * det * W : float3{ 0.f,0.f,0.f };
		}

#define SM_POS_S(i) float4{ssmPositionX[wOffset + i], ssmPositionY[wOffset + i], ssmPositionZ[wOffset + i], ssmPositionW[wOffset + i]}
		__global__ void raySchedulerCalculateNormalsStatic() {
			__shared__ float ssmDepthSM[64];
			__shared__ float ssmCacheX[64];
			__shared__ float ssmCacheY[64];
			__shared__ float ssmCacheZ[64];
			__shared__ float ssmPositionX[64];
			__shared__ float ssmPositionY[64];
			__shared__ float ssmPositionZ[64];
			__shared__ float ssmPositionW[64];
			__shared__ int32_t ssmBegin[64];
			__shared__ int32_t ssmLength[64];
			int32_t rIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (rIdx < cNumRays) {
				worldRay idRay = cRaysDepth[rIdx];
				ssmDepthSM[threadIdx.x] = cfluidDepth[rIdx];
				ssmPositionX[threadIdx.x] = idRay.orig.x + ssmDepthSM[threadIdx.x] * idRay.dir.x;
				ssmPositionY[threadIdx.x] = idRay.orig.y + ssmDepthSM[threadIdx.x] * idRay.dir.y;
				ssmPositionZ[threadIdx.x] = idRay.orig.z + ssmDepthSM[threadIdx.x] * idRay.dir.z;
				ssmPositionW[threadIdx.x] = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
			}
			else {
				ssmDepthSM[threadIdx.x] = FLT_MAX;
				ssmPositionX[threadIdx.x] = FLT_MAX;
				ssmPositionY[threadIdx.x] = FLT_MAX;
				ssmPositionZ[threadIdx.x] = FLT_MAX;
				ssmPositionW[threadIdx.x] = FLT_MAX;
				rIdx = -1;
			}
			ssmCacheX[threadIdx.x] = 0.f;
			ssmCacheY[threadIdx.x] = 0.f;
			ssmCacheZ[threadIdx.x] = 0.f;

			const int32_t wIdx = threadIdx.x % 32;
			const int32_t wOffset = (threadIdx.x / 32) * 32;
			while (__any_sync(__activemask(), rIdx != -1)) {
				int32_t vote = __ballot_sync(__activemask(), ssmDepthSM[threadIdx.x] < 1e19f);
				if (vote != 0) {
					const int32_t idx = 31 - __clz(vote);
					int32_t pred = 0;
					{
						const int3 voxelPosition = position_to_idx3D_i(SM_POS_S(idx), fluidMemory.min_coord, fluidMemory.cell_size_actual.x);
						const int3 vT = position_to_idx3D_i(SM_POS_S(wIdx), fluidMemory.min_coord, fluidMemory.cell_size_actual.x);
						pred = vT == voxelPosition;
						if (wIdx < 27) {
							int32_t i = wIdx;
							int32_t z = i % 3;
							i /= 3;
							int32_t y = i % 3;
							i /= 3;
							int32_t x = i;
							int3 offsetVoxel = voxelPosition + int3{ x - 1,y - 1,z - 1 };
							auto s = arrays.compactHashMap[idx3D_to_hash(voxelPosition + int3{ x - 1, y - 1, z - 1 }, arrays.hash_entries)];
							if (s.compacted == 0 && s.beginning != UINT31_MAX) {
								//ssmBegin[wOffset + wIdx] = 0;
								//ssmLength[wOffset + wIdx] = 0;
								int32_t morton = idx3D_to_morton_32(voxelPosition + int3{ x - 1, y - 1, z - 1 });
								ssmBegin[wOffset + wIdx] = 0;
								ssmLength[wOffset + wIdx] = 0;
								for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
									auto cs = arrays.compactCellSpan[i];
									if (position_to_morton_32(arrays.position[cs.beginning], arrays) == morton) {
										ssmBegin[wOffset + wIdx] = (int32_t)cs.beginning;
										ssmLength[wOffset + wIdx] = cs.length;
										break;
									}
								}

							}
							else {
								ssmBegin[wOffset + wIdx] = (int32_t)s.beginning;
								ssmLength[wOffset + wIdx] = s.length;
							}
						}
					}
					int32_t mask = __ballot_sync(__activemask(), pred);

					for (int32_t id = 0; id < 27; ++id) {
						while (ssmLength[wOffset + id] > 0) {
							uint32_t msk = __brev(mask);
							//int32_t pred = wIdx < ssmLength[wOffset + id];
							float4 p = wIdx < ssmLength[wOffset + id] ? arrays.position[ssmBegin[wOffset + id] + wIdx] : float4{ FLT_MAX, FLT_MAX, FLT_MAX, 0.1f };
							float v = wIdx < ssmLength[wOffset + id] ? arrays.volume[ssmBegin[wOffset + id] + wIdx] : 0.f;
							while (msk != 0) {
								const int32_t offset = __clz(msk);
								msk = msk ^ (1 << (31 - offset));
								float4 res = v * spline4_gradient(SM_POS_S(offset), p);
								for (int i = 16; i > 0; i /= 2) {
									res.x += __shfl_down_sync(0xffffffff, res.x, i);
									res.y += __shfl_down_sync(0xffffffff, res.y, i);
									res.z += __shfl_down_sync(0xffffffff, res.z, i);
								}
								if (wIdx == 0) {
									ssmCacheX[wOffset + offset] += res.x;
									ssmCacheY[wOffset + offset] += res.y;
									ssmCacheZ[wOffset + offset] += res.z;
								}
								__syncwarp();
							}
							if (wIdx == 0) {
								atomicAdd(&ssmBegin[wOffset + id], 32);
								atomicSub(&ssmLength[wOffset + id], 32);
							}
						}
					}
					if (pred == 1) {
						if (rIdx != -1) {
							//cfluidIntersection[rIdx] = float4{0.01f * ssmDepthSM[threadIdx.x], 0.01f * ssmDepthSM[threadIdx.x], 0.01f * ssmDepthSM[threadIdx.x],
							//ssmDepthSM[threadIdx.x] };
							cfluidIntersection[rIdx] = float4{
							ssmCacheX[threadIdx.x],
							ssmCacheY[threadIdx.x],
							ssmCacheZ[threadIdx.x],
							ssmDepthSM[threadIdx.x] };
						}
						//rIdx = -2;
						rIdx = atomicAggInc(cRayCounter);
						if (rIdx < cNumRays) {
							Ray idRay = cRaysDepth[rIdx];
							ssmDepthSM[threadIdx.x] = cfluidDepth[rIdx];
							ssmPositionX[threadIdx.x] = idRay.orig.x + ssmDepthSM[threadIdx.x] * idRay.dir.x;
							ssmPositionY[threadIdx.x] = idRay.orig.y + ssmDepthSM[threadIdx.x] * idRay.dir.y;
							ssmPositionZ[threadIdx.x] = idRay.orig.z + ssmDepthSM[threadIdx.x] * idRay.dir.z;
							ssmPositionW[threadIdx.x] = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
						}
						else {
							rIdx = -1;
							ssmDepthSM[threadIdx.x] = FLT_MAX;
						}
						ssmCacheX[threadIdx.x] = 0.f;
						ssmCacheY[threadIdx.x] = 0.f;
						ssmCacheZ[threadIdx.x] = 0.f;
					}
				}
				__syncwarp();
				// this breaks.
				if (rIdx != -1 && (rIdx == -2 || ssmDepthSM[threadIdx.x] >= 1e19f)) {
					if (rIdx != -2 && ssmDepthSM[threadIdx.x] >= 1e19f) {
						cfluidIntersection[rIdx] = float4{ 1.f, 1.f,1.f,FLT_MAX };
					}
					rIdx = atomicAggInc(cRayCounter);
					if (rIdx < cNumRays) {
						Ray idRay = cRaysDepth[rIdx];
						ssmDepthSM[threadIdx.x] = cfluidDepth[rIdx];
						ssmPositionX[threadIdx.x] = idRay.orig.x + ssmDepthSM[threadIdx.x] * idRay.dir.x;
						ssmPositionY[threadIdx.x] = idRay.orig.y + ssmDepthSM[threadIdx.x] * idRay.dir.y;
						ssmPositionZ[threadIdx.x] = idRay.orig.z + ssmDepthSM[threadIdx.x] * idRay.dir.z;
						ssmPositionW[threadIdx.x] = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
					}
					else {
						rIdx = -1;
						ssmDepthSM[threadIdx.x] = FLT_MAX;
					}
					ssmCacheX[threadIdx.x] = 0.f;
					ssmCacheY[threadIdx.x] = 0.f;
					ssmCacheZ[threadIdx.x] = 0.f;
				}
			}
		}
		__global__ void raySchedulerCalculateNormalsSolo() {
			int32_t rIdx = threadIdx.x + blockIdx.x * blockDim.x;
			float depth;
			float4 position;
			if (rIdx < cNumRays) {
				Ray idRay = cRaysDepth[rIdx];
				depth = cfluidDepth[rIdx];
				position = math::castTo<float4>(idRay(depth));
				position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
			}
			else {
				rIdx = -1;
			}
			while (rIdx != -1) {
				if (depth < 1e19f) {
					float4 normal{ 0.f,0.f,0.f,0.f };
#ifdef ISO_DENSITY
					for (const auto& var : compactIterator(position)) {
						normal += arrays.volume[var] / arrays.density[var] * gradient(position, arrays.position[var]);
					}
					normal = -math::normalize3(normal);
					cfluidIntersection[rIdx] = float4{ normal.x, normal.y, normal.z, depth };
#endif
#ifdef ANISOTROPIC
					float levelSetx = 0.f, levelSety = 0.f, levelSetz = 0.f, levelSet = 0.f;
					float dr = 0.001f;
					//for (const auto& j : compactIterator(position)) 
					//iterateCells()
					auto vIdx = position_to_idx3D_i(position, arrays.min_coord, arrays.cell_size.x);
					for (const auto& j : compactIterator(vIdx))
					{
						//				normal += math::castTo<float4>(turkAnisotropicGradient(position, j));
						levelSet += turkAnisotropic(position, j);
						levelSetx += turkAnisotropic(position + float4{ dr, 0.f, 0.f, 0.f }, j);
						levelSety += turkAnisotropic(position + float4{ 0.f,  dr, 0.f, 0.f }, j);
						levelSetz += turkAnisotropic(position + float4{ 0.f, 0.f,  dr, 0.f }, j);
					}
					normal = float4{ levelSetx, levelSety, levelSetz, 0.f };
					normal = (normal - levelSet) / dr;
					normal = -math::normalize3(normal);
					cfluidIntersection[rIdx] = float4{ normal.x, normal.y, normal.z, depth };
#endif
				}
				if (rIdx != -1) {
					rIdx = atomicAggInc(cRayCounter);
					if (rIdx < cNumRays) {
						Ray idRay = cRaysDepth[rIdx];
						depth = cfluidDepth[rIdx];
						position = math::castTo<float4>(idRay(depth));
						position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
					}
					else {
						rIdx = -1;
					}
				}
			}
		}
		__device__ __inline__ float fresnel(const float3 & I, const float3 & N, float ior) {
			float kr;
			float cosi = math::dot(I, N);
			float etai = 1.f, etat = ior;
			if (cosi > 0.f) { etat = 1.f; etai = ior; }
			// Compute sini using Snell's law
			float sint = etai / etat * sqrtf(math::max(0.f, 1.f - cosi * cosi));
			// Total internal reflection
			if (sint >= 1.f) {
				kr = 1;
			}
			else {
				float cost = sqrtf(math::max(0.f, 1.f - sint * sint));
				cosi = fabsf(cosi);
				float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
				float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
				kr = (Rs * Rs + Rp * Rp) / 2.f;
			}
			// As a consequence of the conservation of energy, transmittance is given by:
			// kt = 1 - kr;
			return kr;
		}
		__device__ float3 refract(const float3 &I, const float3 &N, const float &ior) {
			float cosi = math::dot(I, N);
			float etai = 1.f, etat = ior;
			float3 n = N;
			if (cosi < 0.f) { cosi = -cosi; }
			else { etai = ior; etat = 1.f; n = -N; }
			float eta = etai / etat;
			float k = 1.f - eta * eta * (1.f - cosi * cosi);
			return k < 0.f ? float3{ 0.f,0.f,0.f } : eta * I + (eta * cosi - sqrtf(k)) * n;
		}
		__device__ float3 reflect(const float3 &I, const float3 &N) {
			return I - 2 * math::dot(I, N) * N;
		}
		__global__ void intersectAABB() {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			auto idRay = cRaysDepth[i];
			auto aabb = aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord);
			if (aabb.hit && idRay.bounces < fluidMemory.bounces) {
				cResortKey[i] = idRay.index;
				cRaysDepth[i].depth = aabb.tmin;
			}
			else {
				cResortKey[i] = cNumRays + i;
				cRaysDepth[i].depth = FLT_MAX;
			}
			cResortIndex[i] = i;
			cfluidIntersection[i] = float4{ 1.f,0.,0.f,FLT_MAX };
			cfluidDepth[i] = FLT_MAX;
			//if (blockIdx.x == 0 && blockIdx.y == 0) {
			//	printf("[%d %d] -> %d : %d\n", x, y, cResortKey[i], cResortIndex[i]);
			//}
		}
		__global__ void sort() {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			if (i != cResortIndex[i])
				cCompactRays[i] = cRaysDepth[cResortIndex[i]];
			cfluidDepth[i] = FLT_MAX;
			//if (blockIdx.x == 0 && blockIdx.y == 0) {
			//	printf("[%d %d] -> %d -> %d\n",x ,y, i, cResortIndex[i]);
			//}
		}
		__global__ void shadeAndBounceRays(int32_t seed, int32_t numBVHs, vrtx::gpuBVH* sceneBVH, vrtx::Refl_t fluidMaterial, bool useFluidColorArray) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			//Refl_t refltype = DIFF;
			Refl_t refltype = fluidMaterial;
			curandState randState;
			curand_init(seed + i, 0, 0, &randState);
			auto randstate = &randState;

			auto fluidIntersection = cfluidIntersection[i];
			float depth = fluidIntersection.w;
			bool fluidHit = depth < 1e19f;
			auto idRay = cRaysDepth[i];
			if (idRay.bounces >= fluidMemory.bounces) return;
			auto predicate = false;
			int32_t ctr = 0;
			//while (idRay.bounces < fluidMemory.bounces && !predicate && ctr++ < 32) 
			{

				//depth = 1e21f;
				float3 normal = math::castTo<float3>(fluidIntersection);
				//normal = -normal;
				normal = math::normalize(normal);
				float3 n = normal;
				// normal = math::dot(normal, idRay.dir) < 0 ? normal : normal * -1;
				float3 color = fluidMemory.vrtxFluidColor;
				if (useFluidColorArray)
					color = math::castTo<float3>(cFluidColor[i]);

				//float3 color = float3{ 1.f,1.f, 1.f };
				float3 emission = float3{ 0.f, 0.f, 0.f };

				int32_t idx = -1;
				float kAB = 0.f, kBC = 0.f, kCA = 0.f;
				int32_t bvh_idx = -1;
				int32_t pBestTriIdx = -1;
				for (int32_t i = 0; i < numBVHs; ++i) {
					if (!sceneBVH[i].active)
						continue;
					float ktAB = 0.f, ktBC = 0.f, ktCA = 0.f;
					float hitdistance = FLT_MAX;
					float3 boxnormal = float3{ 0, 0, 0 };
					float3 point;
					BVH_IntersectTriangles(sceneBVH[i], idRay.orig, idRay.dir, UINT32_MAX, pBestTriIdx, point,
						ktAB, ktBC, ktCA, hitdistance, boxnormal);

					if (pBestTriIdx != -1 && (hitdistance < depth /*|| (fluidHit && hitdistance < depth + fluidMemory.fluidBias)*/) && hitdistance > 0.002f) // EPSILON
					{
						fluidHit = false;
						depth = hitdistance;
						bvh_idx = i;
						kAB = ktAB;
						kBC = ktBC;
						kCA = ktCA;
						auto pBestTri = &sceneBVH[bvh_idx].pTriangles[pBestTriIdx];

						n = math::normalize(math::castTo<float3>(pBestTri->normal));
						auto i0 = pBestTri->i0;
						auto i1 = pBestTri->i1;
						auto i2 = pBestTri->i2;
						//printf("%p [%d @ %d] -> %d %d %d => %.2f [%.2f %.2f %.2f]\n", pBestTri, pBestTriIdx, bvh_idx, i0, i1, i2, hitdistance, kAB, kBC, kCA);
						auto v0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].position);
						auto v1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].position);
						auto v2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].position);
						auto n0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].normal);
						auto n1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].normal);
						auto n2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].normal);

						auto ab = v1 - v0;
						auto bc = v2 - v1;
						auto cross_ab_bc = math::cross(ab, bc);
						auto area = math::length(cross_ab_bc);

						auto ABx = kAB * math::distance(v0, v1);
						auto BCx = kBC * math::distance(v1, v2);
						auto CAx = kCA * math::distance(v2, v0);

						n0 *= BCx / area;
						n1 *= CAx / area;
						n2 *= ABx / area;

						n = math::normalize(n0 + n1 + n2);
						//return n;

					   // n = math::normalize(math::castTo<float3>(kBC * n0 + kCA * n1 + kAB * n2));

						normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
						color = fluidMemory.bvhColor;
						//color = float3{ 0.9f, 0.9f, 0.9f };
						refltype = fluidMemory.bvhMaterial;
						emission = float3{ 0, 0, 0 };
					}
				}
				for (int32_t sphere_id = 0; sphere_id < cNumSpheres; sphere_id++) {
					Sphere &sphere = cSpheres[sphere_id];
					float d = cSpheres[sphere_id].intersect(idRay);
					if (d && (d < depth || (fluidHit && d < depth + fluidMemory.fluidBias))) {
						fluidHit = false;
						depth = d;
						auto x = idRay.orig + depth * idRay.dir;
						n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
						normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
						color = sphere.col;
						idx = sphere_id;
						emission = sphere.emi;
						refltype = sphere.refl;
					}
				}
				for (int32_t box_id = 0; box_id < cNumBoxes; box_id++) {
					Box &box = cBoxes[box_id];
					float d = box.intersect(idRay);
					if (d && (d < depth || (fluidHit && d < depth + fluidMemory.fluidBias))) {
						fluidHit = false;
						depth = d;
						auto x = idRay.orig + depth * idRay.dir;
						n = box.normal(x);
						normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
						color = box.col;
						idx = box_id;
						emission = box.emi;
						refltype = box.refl;
					}
				}
				if (depth > 1e19f) {
					cImage[idRay.index].mask = float3{ 0.f,0.f,0.f };
					idRay = worldRay{ idRay.orig, 1e21f, idRay.dir, idRay.internal, (uint32_t) idRay.bounces + 100u, (uint32_t) idRay.index };
					cfluidIntersection[i] = float4{ 0.f,0.f,0.f, FLT_MAX };
					cRaysDepth[i] = idRay;
					return;
				}
#ifdef DEBUG_NORMALS
				cImage[i].color = normal;
				//cImage[i].color = float3{ depth * 0.002f, depth * 0.002f, depth * 0.002f };
				cImage[i].mask = color;
#endif
				//#define SIMPLE_SHADING
#ifdef SIMPLE_SHADING
				cImage[idRay.index].color += cImage[idRay.index].mask * emission;
				cImage[idRay.index].mask *= color;
				float phi = 2 * CUDART_PI_F * randf(i);
				float r2 = randf(i);
				float r2s = sqrtf(r2);
				float3 w = math::normalize(normal);
				idRay.orig += idRay.dir * depth + w * 0.1f;
				float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
				float3 v = math::cross(w, u);
				idRay.dir = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
#else
				{
					cImage[idRay.index].color += cImage[idRay.index].mask * emission;
					if (math::length(emission) > 1e-2f) {
						if (idRay.bounces == 0)
							cImage[idRay.index].color = float3{ 1.f,1.f,1.f };
						idRay.bounces = fluidMemory.bounces+100;
						cfluidIntersection[i] = float4{ 0.f,0.f,0.f, FLT_MAX };
						cRaysDepth[i] = idRay;
						return;
					}
					auto mask = cImage[idRay.index].mask;
					auto x = idRay.orig + depth * idRay.dir;
					auto pointHitInWorldSpace = idRay.orig;
					float3 dw;
					float3 rayInWorldSpace = idRay.dir;
					float3 nl = normal;
					float3 f = color;
					float bias = fluidMemory.fluidBias;
					if (refltype == DIFF) {
						float phi = 2 * CUDART_PI_F * curand_uniform(randstate);
						float r2 = curand_uniform(randstate);
						float r2s = sqrtf(r2);
						float3 w = math::normalize(nl);
						float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
						float3 v = math::cross(w, u);

						dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
						pointHitInWorldSpace = x + w * bias;
						mask *= f;
					}
					if (refltype == METAL) {
						float phi = 2.f * CUDART_PI_F * curand_uniform(randstate);
						float r2 = curand_uniform(randstate);
						float phongexponent = 20;
						float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
						float sinTheta = sqrtf(1 - cosTheta * cosTheta);
						float3 w = math::normalize(rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace));
						float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
						float3 v = math::cross(w, u);

						dw = math::normalize(u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta);
						pointHitInWorldSpace = x + w * bias;
						mask *= f;
					}
					if (refltype == SPEC) {
						dw = rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace);
						pointHitInWorldSpace = x + nl * bias;
						mask *= f;
					}
					if (refltype == COAT) {
						float rouletteRandomFloat = curand_uniform(randstate);
						float threshold = 0.05f;
						float3 specularColor = float3{ 1, 1, 1 }; // hard-coded
						bool reflectFromSurface = (rouletteRandomFloat < threshold);

						if (reflectFromSurface) {
							mask *= specularColor;
							dw = rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace);
							pointHitInWorldSpace = x + nl * bias;
						}
						else {
							float r1 = 2.f * CUDART_PI_F * curand_uniform(randstate);
							float r2 = curand_uniform(randstate);
							float r2s = sqrtf(r2);
							float3 w = math::normalize(nl);
							float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
							float3 v = math::cross(w, u);

							dw = math::normalize(u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2));
							pointHitInWorldSpace = x + nl * bias;
							mask *= f;
						}
					}
					if (idRay.internal != 0 && idx == -1) {
						//float3 k{ 1.f - 0.05098f, 1.f - 0.23137f, 1.f - 0.494177f };
						float3 k = fluidMemory.vrtxDebeer;
						float scale = fluidMemory.vrtxDebeerScale;
						mask *= float3{ expf(-k.x * depth * scale), expf(-k.y * depth * scale),expf(-k.z * depth * scale) };
					}
					//cInternalFlag[i] = 0;
					if (refltype == REFR) {
						// compute fresnel
						rayInWorldSpace = math::normalize(rayInWorldSpace);
						n = math::normalize(n);
						//		  if (math::dot(rayInWorldSpace, n) < 0) n = -n;
								  //if (idRay.internal) n = -n;

						float Re = fresnel(rayInWorldSpace, n, fluidMemory.IOR);

						//mask = float3{ 0.f,0.f,0.f };
						//cImage[i].color = float3{ Re, Re, Re };


						bool outside = math::dot(rayInWorldSpace, n) < 0;
						float3 bias3 = 1e-2f * n;
						float Tr = 1.f - Re;
						//float P = .25f + .5f * Re;
						//float RP = Re / P;
						//float TP = Tr / (1.f - P);
						//if (kr < 1.f) {
						//mask *= 1.f - kr;
						if (curand_uniform(randstate) < Tr)
						{
							mask *= f;
							//mask /= Tr;
							dw = math::normalize(refract(rayInWorldSpace, n, fluidMemory.IOR));
							pointHitInWorldSpace = x + dw * 1e-1f;// -(outside ? bias3 : -bias3);
							idRay.internal = !idRay.internal;
						}
						else
						{
							//mask /= Re;
							dw = math::normalize(reflect(rayInWorldSpace, n));
							pointHitInWorldSpace = x + dw * 1e-1f;// +(outside ? bias3 : -bias3);
							idRay.internal = idRay.internal;
						}
						//}
						//else {
						   // dw = reflect(rayInWorldSpace, n);
						   // pointHitInWorldSpace = x + (outside ? bias : -bias);
						   // cInternalFlag[i] = outside;
						//}
					}
					else {
						idRay.internal = 0;
					}
					cImage[idRay.index].mask = mask;
					idRay = worldRay{ pointHitInWorldSpace, 1e21f, dw, idRay.internal, (uint32_t) idRay.bounces + 1u, (uint32_t) idRay.index };
				}
				auto aabb = aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord);
				depth = FLT_MAX;
				predicate = aabb.hit;
			}
#endif
			cfluidIntersection[i] = float4{ 0.f,0.f,0.f, FLT_MAX };
			cRaysDepth[i] = idRay;
		}
		__global__ void shadeAndBounceRaysNormals(int32_t seed, int32_t numBVHs, vrtx::gpuBVH* sceneBVH) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			//Refl_t refltype = DIFF;
			curandState randState;
			curand_init(seed + i, 0, 0, &randState);
			//auto randstate = &randState;

			auto fluidIntersection = cfluidIntersection[i];
			auto idRay = cRaysDepth[i];
			float depth = fluidIntersection.w;
			//depth = 1e21f;
			float3 normal = math::castTo<float3>(fluidIntersection);
			cImage[idRay.index].color = float3{ normal.x,normal.y, normal.z };
			//cImage[i].color = float3{ normal.x,normal.y, normal.z };
			//return;
			normal = math::normalize(normal);
			float3 n = normal;
			normal = math::dot(normal, idRay.dir) < 0 ? normal : normal * -1;
			float3 color = float3{ 0.05098f, 0.23137f, 0.494177f };
			//float3 color = float3{ 1.f,1.f, 1.f };
			//float3 emission = float3{ 0.f, 0.f, 0.f };

			//int32_t idx = -1;
			for (int32_t sphere_id = 0; sphere_id < cNumSpheres; sphere_id++) {
				Sphere &sphere = cSpheres[sphere_id];
				float d = cSpheres[sphere_id].intersect(idRay);
				if (d && d < depth) {
					depth = d;
					auto x = idRay.orig + depth * idRay.dir;
					n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
					normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
					color = sphere.col;
					//idx = sphere_id;
					//emission = sphere.emi;
					//refltype = sphere.refl;
				}
			}
			for (int32_t box_id = 0; box_id < cNumBoxes; box_id++) {
				Box &box = cBoxes[box_id];
				float d = box.intersect(idRay);
				if (d && d < depth) {
					depth = d;
					auto x = idRay.orig + depth * idRay.dir;
					n = box.normal(x);
					normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
					color = box.col;
					//idx = box_id;
					//emission = box.emi;
					//refltype = box.refl;
				}
			}
			float kAB = 0.f, kBC = 0.f, kCA = 0.f;
			int32_t bvh_idx = -1;
			int32_t pBestTriIdx = -1;
			for (int32_t i = 0; i < numBVHs; ++i) {
				if (!sceneBVH[i].active)
					continue;
				float ktAB = 0.f, ktBC = 0.f, ktCA = 0.f;
				float hitdistance = 1e20f;
				float3 boxnormal = float3{ 0, 0, 0 };
				float3 point;
				BVH_IntersectTriangles(sceneBVH[i], idRay.orig, idRay.dir, UINT32_MAX, pBestTriIdx, point,
					ktAB, ktBC, ktCA, hitdistance, boxnormal);

				if (hitdistance < depth && hitdistance > 0.002f) // EPSILON
				{
					depth = hitdistance;
					bvh_idx = i;
					kAB = ktAB;
					kBC = ktBC;
					kCA = ktCA;
					auto pBestTri = &sceneBVH[bvh_idx].pTriangles[pBestTriIdx];

					n = math::normalize(math::castTo<float3>(pBestTri->normal));
					auto i0 = pBestTri->i0;
					auto i1 = pBestTri->i1;
					auto i2 = pBestTri->i2;

					auto v0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].position);
					auto v1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].position);
					auto v2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].position);
					auto n0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].normal);
					auto n1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].normal);
					auto n2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].normal);

					auto ab = v1 - v0;
					auto bc = v2 - v1;
					auto cross_ab_bc = math::cross(ab, bc);
					auto area = math::length(cross_ab_bc);

					auto ABx = kAB * math::distance(v0, v1);
					auto BCx = kBC * math::distance(v1, v2);
					auto CAx = kCA * math::distance(v2, v0);

					n0 *= BCx / area;
					n1 *= CAx / area;
					n2 *= ABx / area;

					n = math::normalize(n0 + n1 + n2);
					//return n;

				   // n = math::normalize(math::castTo<float3>(kBC * n0 + kCA * n1 + kAB * n2));

					normal = math::dot(n, idRay.dir) < 0 ? n : n * -1;
					color = float3{ 0.9f, 0.9f, 0.9f };
					//refltype = DIFF;
					//emission = float3{ 0, 0, 0 };
				}
			}
			if (fluidMemory.vrtxDepth)
				cImage[idRay.index].color = float3{ depth * fluidMemory.vrtxDepthScale, depth * fluidMemory.vrtxDepthScale, depth * fluidMemory.vrtxDepthScale };
			if (depth > 1e19f) {
				return;
			}
			if (!fluidMemory.vrtxDepth)
				cImage[idRay.index].color = normal;
			cImage[idRay.index].mask = color;
		}
		struct warpState {
			int3 voxel;
			float4 tMax;
			Ray idRay;
			rayState rs;
		};
		__device__ __constant__ int3 voxelOffset[] = {
			int3{-1,-1,-1}, int3{-1,-1, 0}, int3{-1,-1, 1},
			int3{-1, 0,-1}, int3{-1, 0, 0}, int3{-1, 0, 1},
			int3{-1, 1,-1}, int3{-1, 1, 0}, int3{-1, 1, 1},
			int3{ 0,-1,-1}, int3{ 0,-1, 0}, int3{ 0,-1, 1},
			int3{ 0, 0,-1}, int3{ 0, 0, 0}, int3{ 0, 0, 1},
			int3{ 0, 1,-1}, int3{ 0, 1, 0}, int3{ 0, 1, 1},
			int3{ 1,-1,-1}, int3{ 1,-1, 0}, int3{ 1,-1, 1},
			int3{ 1, 0,-1}, int3{ 1, 0, 0}, int3{ 1, 0, 1},
			int3{ 1, 1,-1}, int3{ 1, 1, 0}, int3{ 1, 1, 1},
			int3{ 0, 0, 0}, int3{ 0, 0, 0}, int3{ 0, 0, 0},
			int3{ 0, 0, 0}, int3{ 0, 0, 0}
		};
		__global__ void raySchedulerIntersectFluidStaticWarp() {
			__shared__ int32_t smBegin[64];
			__shared__ int32_t smLength[64];
			__shared__ warpState smState[2];
			//auto& state = smState[threadIdx.x / 32];
#define state smState[threadIdx.x / 32]

			const int32_t wIdx = threadIdx.x % 32;
			const int32_t wOffset = (threadIdx.x / 32) * 32;
			if (wIdx == 0) {
				state.rs = rayState{ 0u, 0u, 1u, 0u, 0u, (int32_t)(threadIdx.x / 32u + blockIdx.x) };
			}
			__syncwarp();
			int3 localOffset;
			{
				int32_t i = wIdx;
				int32_t z = i % 3;
				i /= 3;
				int32_t y = i % 3;
				i /= 3;
				int32_t x = i;
				localOffset = int3{ x - 1, y - 1, z - 1 };
				if (wIdx >= 27)
					localOffset = int3{ 0,0,0 };
			}
			if (state.rs.index >= cNumRays)
				return;
			if (wIdx == 0) {
				auto idray = cRaysDepth[state.rs.index];
				state.idRay = idray;
				if (idray.depth == FLT_MAX)
					state.rs.index = -1;
			}
			__syncwarp();
			while (state.rs.index != -1) {
				if (!state.rs.rayHitFluidAABB) {
					auto aabb = aabb::rayIntersectAABB(state.idRay, fluidMemory.min_coord, fluidMemory.max_coord);
					if (!aabb.hit) {
						if (wIdx == 0) {
							cfluidDepth[state.rs.index] = 1e21f;
							state.rs = rayState{ 0, 0, 1, 0, 0, atomicAdd(cRayCounter,1) };
							if (state.rs.index < cNumRays)
								state.idRay = cRaysDepth[state.rs.index];
						}
						__syncwarp();
						continue;
					}
					aabb.tmin = math::max(0.f, aabb.tmin - 0.1f);
					int3 voxelPosition = position_to_idx3D_i(state.idRay(aabb.tmin), fluidMemory.min_coord, fluidMemory.cell_size.x);
					if (wIdx == 0) {
						state.voxel = voxelPosition;
						state.rs.rayHitFluidAABB = 1;
						state.tMax = math::castTo<float4>(traversal::intBoundRay(state.idRay, aabb.tmin));
						state.tMax.w = aabb.tmax;
					}
					//if (state.rs.index >= cNumRays)
					//	return;
				}
				__syncwarp();
				while (!state.rs.rayDone) {
					if (!state.rs.rayBounced) {
						auto cell_idx = traversal::lookup_cell(state.voxel /*+ localOffset*/);
						if (__any_sync(__activemask(), cell_idx != INT_MAX)) {
							if (wIdx == 0) {
								state.rs.rayDone = 1;
								state.rs.rayHitFluidSurface = 1;
							}
							break;
						}
					}
					if (wIdx == 0) {
						state.rs.rayBounced = 0;
						if (!state.rs.rayDone) {
							auto& tMax = state.tMax;
							auto& rs = state.rs;
							auto& voxel = state.voxel;
							auto& idRay = state.idRay;
							if (tMax.x < tMax.y) {
								if (tMax.x < tMax.z) {
									if (tMax.x > tMax.w) {
										rs.rayDone = 1;
									}
									voxel.x += sgn(idRay.dir.x);
									tMax.x += sgn(idRay.dir.x) / idRay.dir.x;
								}
								else {
									if (tMax.z > tMax.w) {
										rs.rayDone = 1;
									}
									voxel.z += sgn(idRay.dir.z);
									tMax.z += sgn(idRay.dir.z) / idRay.dir.z;
								}
							}
							else {
								if (tMax.y < tMax.z) {
									if (tMax.y > tMax.w) {
										rs.rayDone = 1;
									}
									voxel.y += sgn(idRay.dir.y);
									tMax.y += sgn(idRay.dir.y) / idRay.dir.y;
								}
								else {
									if (tMax.z > tMax.w) {
										rs.rayDone = 1;
									}
									voxel.z += sgn(idRay.dir.z);
									tMax.z += sgn(idRay.dir.z) / idRay.dir.z;
								}
							}
						}
					}
					__syncwarp();
				}
				__syncwarp();
				if (state.rs.rayHitFluidSurface) {
					const auto cs = fluidMemory.cell_size.x;
					float3 min = fluidMemory.min_coord + math::castTo<float3>(state.voxel) * cs;
					auto rH = aabb::rayIntersectAABB(state.idRay, min, min + cs);
					rH.tmin = math::max(0.f, rH.tmin);
					float t = rH.tmin + (rH.tmax - rH.tmin) * (((float)(wIdx)) / 30.f);
					float4 position = math::castTo<float4>(state.idRay(t));
					position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));

					if (wIdx < 27) {
						auto s = arrays.compactHashMap[idx3D_to_hash(state.voxel + localOffset, arrays.hash_entries)];
						if (s.compacted == 0 && s.beginning != UINT31_MAX) {
							//ssmBegin[wOffset + wIdx] = 0;
							//ssmLength[wOffset + wIdx] = 0;
							int32_t morton = idx3D_to_morton_32(state.voxel + localOffset);
							smBegin[wOffset + wIdx] = 0;
							smLength[wOffset + wIdx] = 0;
							for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
								auto cs = arrays.compactCellSpan[i];
								if (position_to_morton_32(arrays.position[cs.beginning], arrays) == morton) {
									smBegin[wOffset + wIdx] = (int32_t)cs.beginning;
									smLength[wOffset + wIdx] = cs.length;
									break;
								}
							}

						}
						else if (s.compacted == 1) {
							smBegin[wOffset + wIdx] = (int32_t)s.beginning;
							smLength[wOffset + wIdx] = s.length;
						}
						else {
							smBegin[wOffset + wIdx] = 0;
							smLength[wOffset + wIdx] = 0;
						}
					}
					__syncwarp();
#if defined(ISO_DENSITY)
					float levelSet = 0.f;
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
							float4 p = arrays.position[i];
							float v = arrays.volume[i];
							levelSet += v * spline4_kernel(p, position);
							//if (levelSet > 0.2f) break;
						}
						//if (levelSet > 0.2f) break;
					}
					uint32_t mask = __ballot_sync(__activemask(), cRaysDepth[state.rs.index].internal == 0 ? levelSet > fluidMemory.vrtxR : levelSet <= fluidMemory.vrtxR);
#elif defined(BRIDSON_05)
					float kernelSum = 0.f;
					float4 xBar{ 0.f,0.f,0.f,0.f };
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
							float4 p = arrays.position[i];
							float v = arrays.volume[i];
							kernelSum += spline4_kernel(p, position);
							xBar += p * spline4_kernel(p, position);
						}
					}
					float levelSet = math::distance3(position, xBar / kernelSum) - fluidMemory.radius;
					uint32_t mask = __ballot_sync(__activemask(), levelSet < 0.f);
#elif defined(SMALL_SCALE_DETAIL)
					float4 C1{ 0.f,0.f,0.f,0.f };
					float C2 = 0.f;
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
							float4 p = arrays.position[j];
							float w = arrays.auxIsoDensity[j];
							C1 += kernel(p, position) / w * p;
							C2 += kernel(p, position) / w;
						}
					}
					auto Cp = C1 / C2;
					Cp.w = position.w;
					float w_c = 0.f;
					float levelSet = math::distance3(position, Cp);
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
							float4 p = arrays.position[j];
							float w = arrays.auxIsoDensity[j];
							w_c += kernel(p, Cp) / w;
						}
					}
					float decay = square(1.f - square(w_c - fluidMemory.wmax) / square(fluidMemory.wmax - fluidMemory.wmin));
					levelSet -= fluidMemory.radius * decay;
					//float levelSet = math::distance3(position, xBar / kernelSum) - fluidMemory.radius;
					uint32_t mask = __ballot_sync(__activemask(), cInternalFlag[state.rs.index] != 0 ? levelSet > 0.8f * fluidMemory.radius : levelSet < 0.8f * fluidMemory.radius);
					//uint32_t mask = __ballot_sync(__activemask(), levelSet < 0.f);
#endif
#ifdef ANISOTROPIC
					float levelSet = 0.f;
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
							if (levelSet > fluidMemory.vrtxR + 0.1f) continue;
							levelSet += turkAnisotropic(position, j);
						}
					}
					//uint32_t mask = __ballot_sync(__activemask(), cRaysDepth[state.rs.index].internal == 0 ? levelSet > fluidMemory.vrtxR : levelSet <= fluidMemory.vrtxR);
					uint32_t maskL = __ballot_sync(__activemask(), levelSet < fluidMemory.vrtxR);
					uint32_t maskH = __ballot_sync(__activemask(), levelSet > fluidMemory.vrtxR);
					uint32_t maskF = (maskL << 1) & maskH;
					uint32_t maskR = (~maskL << 1) & ~maskH;
					uint32_t idxF = __ffs(maskF);
					uint32_t idxR = __ffs(maskR);
					uint32_t idx = idxF > idxR || idxR == 0 ? idxF - 1 : idxR - 1;
					if (maskF != 0 || maskR != 0) {
						float x0 = (float)(idx - 1);
						float y0 = __shfl_sync(__activemask(), levelSet, idx - 1);
						float x1 = (float)idx;
						float y1 = __shfl_sync(__activemask(), levelSet, idx);
						float dy = y1 - y0;
						float alpha = (fluidMemory.vrtxR - y0) / dy;
						float t0 = __shfl_sync(__activemask(), t, idx - 1);
						float t1 = __shfl_sync(__activemask(), t, idx);
						t = t0 * (1.f - alpha) + t1 * alpha;
					}
#endif
					//uint32_t lane = __ffs(mask) - 1;
					//t = __shfl_sync(__activemask(), t, lane);
					if (wIdx == 0) {
						if (maskF != 0 || maskR != 0) {
							cfluidDepth[state.rs.index] = t;
							//cfluidIntersection[state.rs.index];
							state.rs = rayState{ 0, 0, 1, 0, 0, atomicAdd(cRayCounter,1) };
							if (state.rs.index < cNumRays) {
								auto idray = cRaysDepth[state.rs.index];
								state.idRay = idray;
								if (idray.depth == FLT_MAX)
									state.rs.index = -1;
							}
							else
								state.rs.index = -1;
						}
						else {
							state.rs.rayDone = 0;
							state.rs.rayBounced = 1;
							state.rs.rayHitFluidSurface = 0;
						}
					}
					__syncwarp();
					continue;
				}
				if (wIdx == 0) {
					if (state.rs.rayDone) {
						cfluidDepth[state.rs.index] = 1e21f;
						state.rs = rayState{ 0, 0, 1, 0, 0, atomicAdd(cRayCounter,1) };
						if (state.rs.index < cNumRays) {
							auto idray = cRaysDepth[state.rs.index];
							state.idRay = idray;
							if (idray.depth == FLT_MAX)
								state.rs.index = -1;
						}
						else
							state.rs.index = -1;
					}
				}
				__syncwarp();
			}
		}
		//cfluidIntersection[rIdx] = float4{ 1.f, 1.f,1.f,FLT_MAX };
		__device__ __constant__ int3 directionLUT[6][10] = { // x+ x- y+ y- z+ z-
			{ int3{ 0, 0, 0}, int3{ 1, 0, 0}, int3{ 1, 0,-1}, int3{ 1, 0, 1}, int3{ 1,-1, 0}, int3{ 1,-1,-1}, int3{ 1,-1, 1}, int3{ 1, 1, 0}, int3{ 1, 1,-1}, int3{ 1, 1, 1} },
			{ int3{ 0, 0, 0}, int3{-1, 0, 0}, int3{-1, 0,-1}, int3{-1, 0, 1}, int3{-1,-1, 0}, int3{-1,-1,-1}, int3{-1,-1, 1}, int3{-1, 1, 0}, int3{-1, 1,-1}, int3{-1, 1, 1} },
			{ int3{ 0, 0, 0}, int3{ 0, 1, 0}, int3{ 0, 1,-1}, int3{ 0, 1, 1}, int3{-1, 1, 0}, int3{-1, 1,-1}, int3{-1, 1, 1}, int3{ 1, 1, 0}, int3{ 1, 1,-1}, int3{ 1, 1, 1} },
			{ int3{ 0, 0, 0}, int3{ 0,-1, 0}, int3{ 0,-1,-1}, int3{ 0,-1, 1}, int3{-1,-1, 0}, int3{-1,-1,-1}, int3{-1,-1, 1}, int3{ 1,-1, 0}, int3{ 1,-1,-1}, int3{ 1,-1, 1} },
			{ int3{ 0, 0, 0}, int3{ 0, 0, 1}, int3{ 0,-1, 1}, int3{ 0, 1, 1}, int3{-1, 0, 1}, int3{-1,-1, 1}, int3{-1, 1, 1}, int3{ 1, 0, 1}, int3{ 1,-1, 1}, int3{ 1, 1, 1} },
			{ int3{ 0, 0, 0}, int3{ 0, 0,-1}, int3{ 0,-1,-1}, int3{ 0, 1,-1}, int3{-1, 0,-1}, int3{-1,-1,-1}, int3{-1, 1,-1}, int3{ 1, 0,-1}, int3{ 1,-1,-1}, int3{ 1, 1,-1}} };
		__device__ __constant__ int3 fullLoopLUT[]{
			int3{ 0, 0, 0}, int3{ 0, 0,-1}, int3{ 0, 0, 1}, int3{ 0,-1, 0}, int3{ 0,-1,-1}, int3{ 0,-1, 1}, int3{ 0, 1, 0}, int3{ 0, 1,-1}, int3{ 0, 1, 1},
			int3{-1, 0, 0}, int3{-1, 0,-1}, int3{-1, 0, 1}, int3{-1,-1, 0}, int3{-1,-1,-1}, int3{-1,-1, 1}, int3{-1, 1, 0}, int3{-1, 1,-1}, int3{-1, 1, 1},
			int3{ 1, 0, 0}, int3{ 1, 0,-1}, int3{ 1, 0, 1}, int3{ 1,-1, 0}, int3{ 1,-1,-1}, int3{ 1,-1, 1}, int3{ 1, 1, 0}, int3{ 1, 1,-1}, int3{ 1, 1, 1}
		};
		struct compactRayState {
			uint32_t rayDone : 1;
			uint32_t rayBounced : 1;
			uint32_t rayHitFluidAABB : 1;
			uint32_t rayHitFluidSurface : 1;
			int32_t index : 23;
		};
#define IDRAY(i) Ray{{smRayOrig_x[i], smRayOrig_y[i], smRayOrig_z[i]},{smRayDir_x[i], smRayDir_y[i], smRayDir_z[i]}}
#define VOXEL(i) int3{smVoxel_x[i], smVoxel_y[i], smVoxel_z[i]}
#define VOXEL_SHUFFLE(i) int3{__shfl_sync(__activemask(), voxel.x, i),__shfl_sync(__activemask(), voxel.y, i), __shfl_sync(__activemask(), voxel.z, i)}
#define RAY_SHUFFLE(i) Ray{{__shfl_sync(__activemask(), idRay.orig.x, i),__shfl_sync(__activemask(), idRay.orig.y, i), __shfl_sync(__activemask(), idRay.orig.z, i)},{__shfl_sync(__activemask(), idRay.dir.x, i),__shfl_sync(__activemask(), idRay.dir.y, i), __shfl_sync(__activemask(), idRay.dir.z, i)}}
		__global__ void raySchedulerIntersectFluidStatic6() {
			__shared__ float smRayOrig_x[64];
			__shared__ float smRayOrig_y[64];
			__shared__ float smRayOrig_z[64];
			__shared__ float smRayDir_x[64];
			__shared__ float smRayDir_y[64];
			__shared__ float smRayDir_z[64];
			__shared__ int32_t smBegin[64];
			__shared__ int32_t smLength[64];
			__shared__ int32_t smVoxel_x[64];
			__shared__ int32_t smVoxel_y[64];
			__shared__ int32_t smVoxel_z[64];

			const int32_t wIdx = threadIdx.x % 32;
			const int32_t wOffset = (threadIdx.x / 32) * 32;
			bool internal = false;
			compactRayState rs = compactRayState{ 0, 0, 0, 0, static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x) };
			if (rs.index >= cNumRays) {
				rs.index = -1;
			}
			else {
				auto idRay = cRaysDepth[rs.index];
				internal = idRay.internal;
				smRayOrig_x[threadIdx.x] = idRay.orig.x;
				smRayOrig_y[threadIdx.x] = idRay.orig.y;
				smRayOrig_z[threadIdx.x] = idRay.orig.z;
				idRay.dir = math::normalize3(idRay.dir);
				smRayDir_x[threadIdx.x] = idRay.dir.x;
				smRayDir_y[threadIdx.x] = idRay.dir.y;
				smRayDir_z[threadIdx.x] = idRay.dir.z;
				if (idRay.bounces >= fluidMemory.bounces)
					rs.index = -1;
			}
			float4 tMax;
			while (__any_sync(__activemask(), rs.index != -1)) {
				if (rs.index != -1 && !rs.rayHitFluidAABB) {
					auto aabb = aabb::rayIntersectAABB(IDRAY(threadIdx.x), fluidMemory.min_coord, fluidMemory.max_coord);
					aabb.tmin = math::max(0.f, aabb.tmin);
					int3 voxelPosition = position_to_idx3D_i(IDRAY(threadIdx.x)(aabb.tmin), fluidMemory.min_coord, fluidMemory.cell_size.x);
					smVoxel_x[threadIdx.x] = voxelPosition.x;
					smVoxel_y[threadIdx.x] = voxelPosition.y;
					smVoxel_z[threadIdx.x] = voxelPosition.z;
					if (aabb.hit) {
						rs.rayHitFluidAABB = 1;
						//rs.rayHitFluidSurface = 0;
						//rs.rayBounced = 0;
						tMax = math::castTo<float4>(traversal::intBoundRay(IDRAY(threadIdx.x), aabb.tmin));
						tMax.w = ((aabb.tmax - aabb.tmin) / fluidMemory.cell_size.x + 1.f);
						//rs.rayBounced = 1;
					}
					else {
						rs.rayHitFluidAABB = 0;
						rs.rayDone = 1;
					}
				}
				if (rs.index != -1 && !rs.rayDone) {
					int32_t ctr = 0;
					do {
						if (!rs.rayHitFluidSurface && !rs.rayBounced) {
							auto cell_idx = traversal::lookup_cell(VOXEL(threadIdx.x));
							if (cell_idx != INT_MAX) {
								float3 min = fluidMemory.min_coord + math::castTo<float3>(VOXEL(threadIdx.x)) * fluidMemory.cell_size;
								float3 max = min + fluidMemory.cell_size;
								auto rH = aabb::rayIntersectAABB(IDRAY(threadIdx.x), min, max);
								auto hitPosition = IDRAY(threadIdx.x).orig + rH.tmin * IDRAY(threadIdx.x).dir;
								if (hitPosition.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
									hitPosition.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
									hitPosition.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
									hitPosition.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
									hitPosition.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
									hitPosition.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon)
									cell_idx = INT_MAX;
							}
							if (cell_idx != INT_MAX) {
								rs.rayDone = 1;
								rs.rayHitFluidSurface = 1;
							}
						}
						if (!rs.rayDone) {
							rs.rayBounced = 0;
							if (tMax.x < tMax.y) {
								if (tMax.x < tMax.z) {
									if (tMax.x > tMax.w) {
										rs.rayDone = 1;
										continue;
									}
									smVoxel_x[threadIdx.x] += sgn(smRayDir_x[threadIdx.x]);
									tMax.x += sgn(smRayDir_x[threadIdx.x]) / smRayDir_x[threadIdx.x];
								}
								else {
									if (tMax.z > tMax.w) {
										rs.rayDone = 1;
										continue;
									}
									smVoxel_z[threadIdx.x] += sgn(smRayDir_z[threadIdx.x]);
									tMax.z += sgn(smRayDir_z[threadIdx.x]) / smRayDir_z[threadIdx.x];
								}
							}
							else {
								if (tMax.y < tMax.z) {
									if (tMax.y > tMax.w) {
										rs.rayDone = 1;
										continue;
									}
									smVoxel_y[threadIdx.x] += sgn(smRayDir_y[threadIdx.x]);
									tMax.y += sgn(smRayDir_y[threadIdx.x]) / smRayDir_y[threadIdx.x];
								}
								else {
									if (tMax.z > tMax.w) {
										rs.rayDone = 1;
										continue;
									}
									smVoxel_z[threadIdx.x] += sgn(smRayDir_z[threadIdx.x]);
									tMax.z += sgn(smRayDir_z[threadIdx.x]) / smRayDir_z[threadIdx.x];
								}
							}
						}
						if (++ctr > 1024) {
							rs.rayHitFluidAABB = 0;
							rs.rayDone = true;
						}
					} while (__all_sync(__activemask(), !rs.rayDone));
				}
				__syncwarp();
				float depth = 1e21f;
				uint32_t mask = __brev(__ballot_sync(__activemask(), rs.rayHitFluidSurface));
				__syncwarp();
				while ((mask & __activemask()) != 0) {
					const int32_t offset = __clz(mask);
					mask = mask ^ (1 << (31 - offset));
					__syncwarp();
					float4 position;
					float t;
					{
						const auto cs = fluidMemory.cell_size.x;
						float3 min = fluidMemory.min_coord + math::castTo<float3>(VOXEL(wOffset + offset)) *  fluidMemory.cell_size;
						auto rH = aabb::rayIntersectAABB(IDRAY(wOffset + offset), min, min + fluidMemory.cell_size);
						if (rH.hit == false)continue;
						rH.tmin = math::max(0.f, rH.tmin);
						t = rH.tmin + (rH.tmax - rH.tmin) * (((float)(wIdx)) / 31.f);
						position = math::castTo<float4>(IDRAY(wOffset + offset)(t));
						position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
					}
					__syncwarp();
					if (wIdx < 27) {
						smBegin[wOffset + wIdx] = 0;
						smLength[wOffset + wIdx] = 0;
						auto s = arrays.compactHashMap[idx3D_to_hash(VOXEL(wOffset + offset) + fullLoopLUT[wIdx], arrays.hash_entries)];
						if (s.compacted == 0 && s.beginning != UINT31_MAX) {
							int32_t morton = idx3D_to_morton_32(VOXEL(wOffset + offset) + fullLoopLUT[wIdx]);
							for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
								auto cs = arrays.compactCellSpan[i];
								if (position_to_morton_32(arrays.position[cs.beginning], arrays) == morton) {
									smBegin[wOffset + wIdx] = (int32_t)cs.beginning;
									smLength[wOffset + wIdx] = cs.length;
									break;
								}
							}

						}
						else if (s.beginning != UINT31_MAX) {
							smBegin[wOffset + wIdx] = (int32_t)s.beginning;
							smLength[wOffset + wIdx] = s.length;
						}
					}
					__syncwarp();
#ifdef ANISOTROPIC
					float levelSet = 0.f;
					int32_t flag = __shfl_sync(__activemask(), rs.index, offset);
					flag = cRaysDepth[0].internal;
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
							if (levelSet > fluidMemory.vrtxR * 1.05f || j > smBegin[wOffset + id] + 1024) break;
							levelSet += turkAnisotropic(position, j);
						}
					}
					__syncwarp();
					int32_t internal_flag = 0;
					internal_flag = __shfl_sync(__activemask(), rs.index, offset);
					if (cRaysDepth[internal_flag].internal == 0 ? levelSet > fluidMemory.vrtxR : levelSet <= fluidMemory.vrtxR) {
						if (position.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
							position.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
							position.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
							position.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
							position.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
							position.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon) {
							levelSet = cRaysDepth[internal_flag].internal ? -10.f : 10.f;
						}
					}
					uint32_t mask = __ballot_sync(__activemask(), cRaysDepth[internal_flag].internal == 0 ? levelSet > fluidMemory.vrtxR : levelSet <= fluidMemory.vrtxR);
#endif
#ifdef ISO_DENSITY
					float levelSet = 0.f;
					for (int32_t id = 0; id < 27; ++id) {
						for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
							float4 p = arrays.position[i];
							float v = arrays.volume[i];
							levelSet += v * spline4_kernel(p, position);
							//if (levelSet > 0.2f) break;
						}
						//if (levelSet > 0.2f) break;
					}
					int32_t internal_flag = 0;
					internal_flag = __shfl_sync(__activemask(), internal, offset);
					uint32_t mask = __ballot_sync(__activemask(), internal_flag == 0 ? levelSet > fluidMemory.vrtxR : levelSet < fluidMemory.vrtxR);
#endif
					uint32_t maskL = __ballot_sync(__activemask(), levelSet >= fluidMemory.vrtxR);
					uint32_t maskH = __ballot_sync(__activemask(), levelSet < fluidMemory.vrtxR);
					uint32_t maskF = (maskL << 1) & maskH;
					uint32_t maskR = (~maskL << 1) & ~maskH;
					uint32_t idx = maskL & 0x1 ? __ffs(~maskL) - 1 : __ffs(maskL) - 1;
					//uint32_t idx = __ffs(maskL) - 1;
					//uint32_t idxF = __ffs(maskF);
					//uint32_t idxR = __ffs(maskR);
					//uint32_t idx = idxF > idxR || idxR == 0 ? idxF - 1 : idxR - 1;
					if (maskL != 0 && maskL != __activemask())
						//if ((maskF != 0 || maskR != 0) && idx < 31)
					{
						float x0 = (float)(idx - 1);
						float y0 = __shfl_sync(__activemask(), levelSet, idx - 1);
						float x1 = (float)idx;
						float y1 = __shfl_sync(__activemask(), levelSet, idx);
						float dy = y1 - y0;
						float alpha = (fluidMemory.vrtxR - y0) / dy;
						float t0 = __shfl_sync(__activemask(), t, idx - 1);
						float t1 = __shfl_sync(__activemask(), t, idx);
						float t = t0 * (1.f - alpha) + t1 * alpha;
						if (wIdx == offset) {
							depth = t;
							rs.rayHitFluidSurface = 0;
							rs.rayBounced = 1;
						}
					}
					if (maskL != 0 && maskL != __activemask()) {
						if (wIdx == offset) {
							auto position = IDRAY(wOffset + offset)(depth);
							if (position.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
								position.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
								position.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
								position.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
								position.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
								position.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon) {
								depth = 1e21f;
								rs.rayHitFluidSurface = 1;
							}
						}
					}
					__syncwarp();
				}
				if ((rs.index != -1)) {
					if (rs.rayHitFluidSurface == 1 && depth > 1e19f) {
						rs.rayDone = 0;
						rs.rayBounced = 1;
						rs.rayHitFluidSurface = 0;
					}
					if (rs.rayDone) {
						cfluidDepth[rs.index] = depth;
						rs = compactRayState{ 0, 0, 0, 0, atomicAggInc(cRayCounter) };
						if (rs.index >= cNumRays) {
							rs.index = -1;
						}
						else {
							auto idRay = cRaysDepth[rs.index];
							smRayOrig_x[threadIdx.x] = idRay.orig.x;
							smRayOrig_y[threadIdx.x] = idRay.orig.y;
							smRayOrig_z[threadIdx.x] = idRay.orig.z;
							idRay.dir = math::normalize3(idRay.dir);
							smRayDir_x[threadIdx.x] = idRay.dir.x;
							smRayDir_y[threadIdx.x] = idRay.dir.y;
							smRayDir_z[threadIdx.x] = idRay.dir.z;
							internal = idRay.internal;
							if (idRay.bounces >= fluidMemory.bounces)
								rs.index = -1;
						}
					}
				}
			}
		}
		__device__ float intersectParticle(const Ray &r, const float3& pos, float rad) {
			float3 op = pos - r.orig; //
			float t, epsilon = 0.01f;
			float b = math::dot(op, r.dir);
			float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
			if (disc < 0)
				return 0;
			else
				disc = sqrtf(disc);
			return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
		}
		__global__ void raySchedulerIntersectFluidStaticParticles(bool particles) {
			//const int32_t wIdx = threadIdx.x % 32;
			//const int32_t wOffset = (threadIdx.x / 32) * 32;
			compactRayState rs = compactRayState{ 0u, 1u, 0u, 0u, (int32_t)(threadIdx.x + blockIdx.x * blockDim.x) };
			Ray idRay;
			int3 voxelPosition;
			float4 tMax;
			if (rs.index >= cNumRays) {
				rs.index = -1;
			}
			else {
				idRay = cRaysDepth[rs.index];
			}
			int32_t counter = 0;
			while (rs.index != -1) {
				if (rs.index != -1 && !rs.rayHitFluidAABB) {
					auto aabb = aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord);
					aabb.tmin = math::max(0.f, aabb.tmin);
					//aabb.tmin = 0.f;
					voxelPosition = position_to_idx3D_i(idRay(aabb.tmin), fluidMemory.min_coord, fluidMemory.cell_size.x);
					if (aabb.hit) {
						rs.rayHitFluidAABB = 1;
						tMax = math::castTo<float4>(traversal::intBoundRay(idRay, aabb.tmin));
						tMax.w = (aabb.tmax - aabb.tmin) / fluidMemory.cell_size.x + 1.f;
						//rs.rayBounced = 1;
					}
					else {
						rs.rayHitFluidAABB = 0;
						rs.rayDone = 1;
					}
				}
				while (rs.index != -1 && !rs.rayDone) {
					if (counter++ > (1 << 16)) {
						rs.rayDone = 1;
						rs.rayHitFluidSurface = 0;
					}
					if (!rs.rayBounced) {
						auto cell_idx = traversal::lookup_cell(voxelPosition);
						if (cell_idx != INT_MAX) {
							float3 min = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) * fluidMemory.cell_size;
							float3 max = min + fluidMemory.cell_size;
							auto rH = aabb::rayIntersectAABB(idRay, min, max);
							auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
							if (hitPosition.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
								hitPosition.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
								hitPosition.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
								hitPosition.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
								hitPosition.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
								hitPosition.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon)
								cell_idx = INT_MAX;
						}
						if (cell_idx != INT_MAX) {
							rs.rayDone = 1;
							rs.rayHitFluidSurface = 1;
						}
					}
					if (!rs.rayDone) {
						rs.rayBounced = 0;
						if (tMax.x < tMax.y) {
							if (tMax.x < tMax.z) {
								if (tMax.x > tMax.w) {
									rs.rayDone = 1;
									rs.rayHitFluidSurface = 0;
								}
								voxelPosition.x += sgn(idRay.dir.x);
								tMax.x += sgn(idRay.dir.x) / idRay.dir.x;
							}
							else {
								if (tMax.z > tMax.w) {
									rs.rayDone = 1;
									rs.rayHitFluidSurface = 0;
								}
								voxelPosition.z += sgn(idRay.dir.z);
								tMax.z += sgn(idRay.dir.z) / idRay.dir.z;
							}
						}
						else {
							if (tMax.y < tMax.z) {
								if (tMax.y > tMax.w) {
									rs.rayDone = 1;
									rs.rayHitFluidSurface = 0;
								}
								voxelPosition.y += sgn(idRay.dir.y);
								tMax.y += sgn(idRay.dir.y) / idRay.dir.y;
							}
							else {
								if (tMax.z > tMax.w) {
									rs.rayDone = 1;
									rs.rayHitFluidSurface = 0;
								}
								voxelPosition.z += sgn(idRay.dir.z);
								tMax.z += sgn(idRay.dir.z) / idRay.dir.z;
							}
						}
					}
				}
				__syncwarp();
				float depth = 1e20f;
				//float3 n;
				if (rs.rayHitFluidSurface) {
					constexpr auto epsilon = 1e-1f;
					float3 min = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) * fluidMemory.cell_size;
					float3 max = min + fluidMemory.cell_size;
					auto rH = aabb::rayIntersectAABB(idRay, min, max);
					auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
					auto c = (min + max) * 0.5f;
					auto prel = hitPosition - c;
					auto d = math::abs((min - max) * 0.5f);
					auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
					char3 nc = char3{ static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z) };
					if (!particles) {
						int32_t morton = idx3D_to_morton_32(voxelPosition);
						auto beginning = 0;
						auto length = 0;
						auto s = arrays.compactHashMap[idx3D_to_hash(voxelPosition, arrays.hash_entries)];
						if (s.compacted == 0 && s.beginning != UINT31_MAX) {
							for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
								auto cs = arrays.compactCellSpan[i];
								if (position_to_morton_32(arrays.position[cs.beginning], arrays) == morton) {
									beginning = (int32_t)cs.beginning;
									length = cs.length;
									break;
								}
							}

						}
						else if (s.beginning != UINT31_MAX) {
							if (position_to_morton_32(arrays.position[(int32_t)s.beginning], arrays) == morton) {
								beginning = (int32_t)s.beginning;
								length = s.length;
							}
						}

						float intensity = length > 0 ? arrays.renderArray[beginning].w : 0.f;
						
						int32_t min_level = INT32_MAX;
						for (int32_t i = beginning; i < beginning + length; ++i) {
							min_level = math::min(arrays.MLMResolution[i], min_level);
							intensity = math::min(intensity, arrays.renderArray[i].w);
						}

						for (int32_t i = beginning; i < beginning + length; ++i) {
							float factor = powf(1.f / 2.f, (float)min_level);
							auto pos = arrays.position[i];
							int3 voxel = position_to_idx3D_i(pos, fluidMemory.min_coord, fluidMemory.cell_size.x * factor);
							float3 min = fluidMemory.min_coord + math::castTo<float3>(voxel) * fluidMemory.cell_size * factor;
							float3 max = min + fluidMemory.cell_size * factor;
							auto rH = aabb::rayIntersectAABB(idRay, min, max);
							if (rH.hit && rH.tmin > 0.f && rH.tmin < depth) {
								auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
								if (pos.x < fluidMemory.vrtxDomainMin.x ||
									pos.y < fluidMemory.vrtxDomainMin.y ||
									pos.z < fluidMemory.vrtxDomainMin.z ||
									pos.x > fluidMemory.vrtxDomainMax.x ||
									pos.y > fluidMemory.vrtxDomainMax.y ||
									pos.z > fluidMemory.vrtxDomainMax.z)
									continue;
								auto c = (min + max) * 0.5f;
								auto prel = hitPosition - c;
								auto d = math::abs((min - max) * 0.5f);
								auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
								char3 nc = char3{ static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z) };

								float3 nf = math::castTo<float3>(nc);
								cfluidIntersection[rs.index] = float4{ (float)n.x,  (float)n.y,  (float)n.z, rH.tmin };
								float4 col{ 0.8f, 0.8f, 0.8f, 1.f };
								depth = rH.tmin;
							}
						}
						if (depth > 1e19f) {
							rs.rayDone = 0;
							rs.rayBounced = 1;
							rs.rayHitFluidSurface = 0;
						}
						else {
							//auto intensity = (float)min_level;
							auto mapValue = [&](float value, float min, float max, int32_t mode) {
								//if (mode == 0)
								//	return (value - min) / (max - min);
								//if (mode == 1)
								//	return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
								//if (mode == 2)
								//	return (value * value - min * min) / (max * max - min * min);
								//if (mode == 3)
								//	return (pow(value, 1.f / 3.f) - pow(min, 1.f / 3.f)) / (pow(max, 1.f / 3.f) - pow(min, 1.f / 3.f));
								//if (mode == 4)
								//	return (value * value * value - min * min * min) / (max * max * max - min * min * min);
								//if (mode == 5)
								//	return (log(value) - log(min)) / (log(max) - log(min));
								return (value - min) / (max - min);
							};
							if (arrays.minMap < arrays.maxMap)
								intensity = mapValue(intensity, arrays.minMap, arrays.maxMap, arrays.transferFn);
							else
								intensity = mapValue(intensity, arrays.maxMap, arrays.minMap, arrays.transferFn);
							intensity = math::clamp(intensity, 0.f, 1.f);

							if (fluidMemory.colorMapFlipped)
								intensity = 1.f - intensity;
							intensity  *= ((float)fluidMemory.colorMapLength);
							int32_t lower = floorf(intensity);
							int32_t upper = ceilf(intensity);
							float mod = intensity - (float)lower;
							float4 col = fluidMemory.colorMap[lower] * mod + fluidMemory.colorMap[upper] * (1.f - mod);
							cFluidColor[rs.index] = float4{ col.x, col.y, col.z, 1.f };
						}
						//float3 nf = math::castTo<float3>(nc);
						//cfluidIntersection[rs.index] = float4{ (float)n.x,  (float)n.y,  (float)n.z, rH.tmin };
						//float4 col{ 0.8f, 0.8f, 0.8f, 1.f };
						//cFluidColor[rs.index] = float4{ col.x, col.y, col.z, 1.f };
					}
					else {
						//	
						int32_t idx = -1;
						for (auto ii : compactIterator(voxelPosition)) {
							float3 pos = math::castTo<float3>(arrays.position[ii]);
							auto t = intersectParticle(idRay, pos, radiusFromVolume(ii));
							if (t && t < depth) {
								auto hitPosition = idRay.orig + t * idRay.dir;
								if (pos.x < fluidMemory.vrtxDomainMin.x ||
									pos.y < fluidMemory.vrtxDomainMin.y ||
									pos.z < fluidMemory.vrtxDomainMin.z ||
									pos.x > fluidMemory.vrtxDomainMax.x ||
									pos.y > fluidMemory.vrtxDomainMax.y ||
									pos.z > fluidMemory.vrtxDomainMax.z)
									continue;
								idx = ii;
								depth = t;
							}
						}
						if (idx != -1) {
							float3 pos = math::castTo<float3>(arrays.position[idx]);
							auto x = idRay.orig + depth * idRay.dir;
							//n = math::normalize(float3{x.x - pos.x, x.y - pos.y, x.z - pos.z});
							cfluidIntersection[rs.index] = float4{ x.x - pos.x, x.y - pos.y, x.z - pos.z, depth };
							auto mapValue = [&](float value, float min, float max, int32_t mode) {
								if (arrays.transferFn == 0)
									return (value - min) / (max - min);
								if (arrays.transferFn == 1)
									return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
								if (arrays.transferFn == 2)
									return (value * value - min * min) / (max * max - min * min);
								if (arrays.transferFn == 3)
									return (pow(value, 1.f / 3.f) - pow(min, 1.f / 3.f)) / (pow(max, 1.f / 3.f) - pow(min, 1.f / 3.f));
								if (arrays.transferFn == 4)
									return (value * value * value - min * min * min) / (max * max * max - min * min * min);
								if (arrays.transferFn == 5)
									return (log(value) - log(min)) / (log(max) - log(min));
								return (value - min) / (max - min);
							};
							float intensity = arrays.renderArray[idx].w;
							if (arrays.minMap < arrays.maxMap)
								intensity = mapValue(intensity, arrays.minMap, arrays.maxMap, arrays.transferFn);
							else
								intensity = mapValue(intensity, arrays.maxMap, arrays.minMap, arrays.transferFn);
							intensity = math::clamp(intensity, 0.f, 1.f);

							if (fluidMemory.colorMapFlipped)
								intensity = 1.f - intensity;
							intensity = mapValue(intensity, 0.f, 1.f, arrays.mappingFn);
							float scaled = intensity * ((float)fluidMemory.colorMapLength);
							int32_t lower = floorf(scaled);
							int32_t upper = ceilf(scaled);
							float mod = scaled - (float)lower;
							float4 col = fluidMemory.colorMap[lower] * mod + fluidMemory.colorMap[upper] * (1.f - mod);
							cFluidColor[rs.index] = float4{ col.x, col.y, col.z, 1.f };
						}
						else {
							rs.rayDone = 0;
							rs.rayBounced = 1;
							rs.rayHitFluidSurface = 0;
						}
					}
				}
				if (rs.index != -1) {
					if (rs.rayDone) {
						//cfluidIntersection[rs.index] = float4{ tMax.x / tMax.w, tMax.y / tMax.w, tMax.z / tMax.w, FLT_MAX };
						rs = compactRayState{ 0, 1, 0, 0, atomicAggInc(cRayCounter) };
						counter = 0;
						if (rs.index >= cNumRays) {
							rs.index = -1;
						}
						else {
							idRay = cRaysDepth[rs.index];
						}
					}
				}
			}
		}
	} // namespace render
} // namespace vrtx
#define TOKENPASTE(x, y) x ## y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)

struct valid_fn {
	deviceInline bool operator()(const int32_t x) { return x < vrtx::cNumRays; }
};

std::map<std::string, std::vector<float>> cuVRTXRender(SceneInformation scene, cudaGraphicsResource_t resource, vrtx::objectLoader& sceneMeshes, vrtxFluidMemory fmem, vrtxFluidArrays farrays, float3 *acc,
	unsigned framenumber, unsigned hashedframes, int32_t renderMode, int32_t bounces, bool fluidRender, int32_t renderGrid, int32_t surfacingTechnique, vrtx::Refl_t fluidMaterial, bool dirty, std::vector<vrtx::Sphere> spheres, std::vector<vrtx::Box> boxes) {
	// using namespace vrtx;
	static std::random_device rd;
	static std::uniform_int_distribution<uint32_t> dist(0, UINT_MAX);
	static bool once = true;
	static cudaStream_t stream;
	constexpr auto msaa = 1;
	constexpr int32_t blocks_1080 = 16 * 68;
	constexpr int32_t blockSize_1080 = 64;
	static vrtx::gpuBVH* bvhs = nullptr;
	int32_t width = static_cast<int32_t>(scene.width);
	int32_t height = static_cast<int32_t>(scene.height);
	int32_t numRays = width * height * msaa;
	int32_t num_blocks = blocks_1080 * blockSize_1080;
	std::map<std::string, std::vector<float>> timers;
	//std::cout << "vRTX renderer built at " << __TIMESTAMP__ << std::endl;
	if (once) {
		std::cout << "vRTX renderer built at " << __TIMESTAMP__ << std::endl;
		cudaStreamCreate(&stream);
		cudaMalloc(&vrtx::cuImage, sizeof(vrtx::Pixel) * width * height);
		cudaMalloc(&vrtx::cuCurrentRays, sizeof(vrtx::worldRay) * width * height);
		cudaMalloc(&vrtx::cuCompactedRays, sizeof(vrtx::worldRay) * width * height);
		cudaMalloc(&vrtx::rayCounter, sizeof(int32_t));
		cudaMalloc(&vrtx::cRNGSeeds, sizeof(uint32_t) * numRays);
		cudaMalloc(&vrtx::cuResortIndex, sizeof(int32_t) * numRays);
		cudaMalloc(&vrtx::cuResortKey, sizeof(int32_t) * numRays);
		cudaMalloc(&vrtx::cufluidDepth, sizeof(float) * width * height);
		//cudaMalloc(&vrtx::cuInternalFlag, sizeof(int32_t) * width * height);
		cudaMalloc(&vrtx::cuFluidColor, sizeof(float4) * width * height);
		cudaMalloc(&vrtx::cufluidIntersection, sizeof(float4) * width * height);
		cudaMalloc(&bvhs, sizeof(vrtx::gpuBVH));
		std::vector<int32_t> seeds;
		for (int32_t i = 0; i < numRays; ++i)
			seeds.push_back(dist(rd));
		cudaMemcpy(vrtx::cRNGSeeds, seeds.data(), sizeof(int32_t) * numRays, cudaMemcpyHostToDevice);
		cudaArray_t color_arr;
		cudaGraphicsMapResources(1, &resource, 0);
		cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
		cudaBindSurfaceToArray(vrtx::surfaceWriteOut, color_arr);
		once = false;
		CPSYMBOL(vrtx::cNumRays, numRays);
		CPSYMBOL(vrtx::cResortIndex, vrtx::cuResortIndex);
		CPSYMBOL(vrtx::cResortKey, vrtx::cuResortKey);
		CPSYMBOL(vrtx::cRaysDepth, vrtx::cuCurrentRays);
		CPSYMBOL(vrtx::cCompactRays, vrtx::cuCompactedRays);
		CPSYMBOL(vrtx::cImage, vrtx::cuImage);
		CPSYMBOL(vrtx::cRayCounter, vrtx::rayCounter);
		//CPSYMBOL(vrtx::cInternalFlag, vrtx::cuInternalFlag);
		CPSYMBOL(vrtx::cFluidColor, vrtx::cuFluidColor);
		CPSYMBOL(vrtx::cuSeeds, vrtx::cRNGSeeds);
		CPSYMBOL(vrtx::cfluidDepth, vrtx::cufluidDepth);
		CPSYMBOL(vrtx::cfluidIntersection, vrtx::cufluidIntersection);
	}
	if (dirty) {
		if (vrtx::cuBoxes != nullptr) cudaFree(vrtx::cuBoxes);
		if (vrtx::cuSpheres != nullptr) cudaFree(vrtx::cuSpheres);
		int32_t numBoxes = (int32_t)boxes.size();
		int32_t numSpheres = (int32_t)spheres.size();
		cudaMalloc(&vrtx::cuBoxes, sizeof(vrtx::Box) * numBoxes);
		cudaMalloc(&vrtx::cuSpheres, sizeof(vrtx::Sphere) * numSpheres);
		cudaMemcpy(vrtx::cuBoxes, boxes.data(), sizeof(vrtx::Box) * numBoxes, cudaMemcpyHostToDevice);
		cudaMemcpy(vrtx::cuSpheres, spheres.data(), sizeof(vrtx::Sphere) * numSpheres, cudaMemcpyHostToDevice);
		CPSYMBOL(vrtx::cBoxes, vrtx::cuBoxes);
		CPSYMBOL(vrtx::cNumBoxes, numBoxes);
		CPSYMBOL(vrtx::cSpheres, vrtx::cuSpheres);
		CPSYMBOL(vrtx::cNumSpheres, numSpheres);
	}
	// cuda::sync(std::to_string(__LINE__));
	// scene.m_camera.apertureRadius = 0.f;
	CPSYMBOL(vrtx::cScene, scene);
	CPSYMBOL(vrtx::fluidMemory, fmem);
	CPSYMBOL(vrtx::arrays, farrays);
	if (get<parameters::boundaryVolumes>().size() > 0) {
		vrtx::gpuBVH bvhs_host[] = { sceneMeshes.getGPUArrays() };
		cudaDeviceSynchronize();
		cudaMemcpy(bvhs, bvhs_host, sizeof(vrtx::gpuBVH), cudaMemcpyHostToDevice);
	}
	dim3 texturedim((uint32_t)scene.width, (uint32_t)scene.height, 1);
	dim3 blockdim(8, 8, 1);
	dim3 griddim(texturedim.x / blockdim.x, texturedim.y / blockdim.y, 1);
	if (texturedim.x % blockdim.x != 0)
		griddim.x += 1;
	if (texturedim.y % blockdim.y != 0)
		griddim.y += 1;
	//cuda::sync(std::to_string(__LINE__));
#define MEASURE_CUDA(name, str, x)\
	static cudaEvent_t TOKENPASTE2(start_ev, __LINE__), TOKENPASTE2(end_ev, __LINE__);\
	static bool TOKENPASTE2(onces, __LINE__) = true;\
	if(TOKENPASTE2(onces, __LINE__)){\
		cudaEventCreate(&TOKENPASTE2(start_ev, __LINE__));\
		cudaEventCreate(&TOKENPASTE2(end_ev, __LINE__));\
		TOKENPASTE2(onces, __LINE__) = false;\
	}\
	cudaEventRecord(TOKENPASTE2(start_ev, __LINE__));\
	x;\
	cudaEventRecord(TOKENPASTE2(end_ev, __LINE__));\
	name = [&]() mutable{ \
		cudaEventSynchronize(TOKENPASTE2(end_ev, __LINE__));\
		float TOKENPASTE2(milliseconds, __LINE__) = 0; \
		cudaEventElapsedTime(&TOKENPASTE2(milliseconds, __LINE__), TOKENPASTE2(start_ev, __LINE__), TOKENPASTE2(end_ev, __LINE__)); \
		return TOKENPASTE2(milliseconds, __LINE__);\
	};

	std::function<float()> generateRays, intersectFluid, calculateNormals, bounceRays, toneMap, sortRays;
	MEASURE_CUDA(generateRays, stream, LAUNCH(vrtx::common::generateBlockedRays, griddim, dim3(msaa, blockdim.x, blockdim.y), 0, stream)(hashedframes, vrtx::cuImage, vrtx::cuCurrentRays, vrtx::cuCurrentRays, msaa););
	auto resortRays = [&]() mutable {
		//cuda::sync(std::to_string(__LINE__));
		thrust::cuda::par.on(stream);
		//cuda::sync(std::to_string(__LINE__));
		vrtx::render::intersectAABB <<< griddim, dim3(blockdim.x, blockdim.y, 1), 0, stream >>> ();
		//cuda::sync(std::to_string(__LINE__));
		cudaDeviceSynchronize();
		algorithm::sort_by_key(numRays, vrtx::cuResortKey, vrtx::cuResortIndex);
		cuda::sync(std::to_string(__LINE__));
		cudaDeviceSynchronize();
		cudaMemcpy(vrtx::cuCompactedRays, vrtx::cuCurrentRays, sizeof(vrtx::worldRay) * numRays, cudaMemcpyDeviceToDevice);
		vrtx::render::sort <<< griddim, dim3(blockdim.x, blockdim.y, 1), 0, stream >>> ();
		//cuda::sync(std::to_string(__LINE__));
		std::swap(vrtx::cuCurrentRays, vrtx::cuCompactedRays);
		//cuda::sync(std::to_string(__LINE__));
		cudaMemcpyToSymbolAsync(vrtx::cRaysDepth, &vrtx::cuCurrentRays, sizeof(vrtx::cRaysDepth), 0, cudaMemcpyHostToDevice, stream);
		//cuda::sync(std::to_string(__LINE__));
		cudaMemcpyToSymbolAsync(vrtx::cCompactRays, &vrtx::cuCompactedRays, sizeof(vrtx::cCompactRays), 0, cudaMemcpyHostToDevice, stream);
		//cuda::sync(std::to_string(__LINE__));
		thrust::cuda::par.on(0);
		// cuda::sync(std::to_string(__LINE__));
		return (float)algorithm::count_if(vrtx::cuResortKey, numRays, valid_fn{});
		//std::cout << __LINE__ << std::endl;
		//cudaDeviceSynchronize();
		//getchar();
	};


	//MEASURE_CUDA(sortRays, stream, auto ctr = resortRays());
	//launch<sort>(diff, mem, diff, mem.auxCellSpan, (compactListEntry*)mem.compactCellSpanSwap);

	//cuda::sync(std::to_string(__LINE__));
	timers[std::string("gen rays     ")].push_back(generateRays());
	//timers[std::string("sort rays    ")].push_back(sortRays());
	//timers[std::string("collision ctr")].push_back(-ctr);
	for (int32_t i = 0; i < (get<parameters::vrtxRenderNormals>() ? 1 : get<parameters::vrtxBounces>()); ++i) {
		MEASURE_CUDA(sortRays, stream, auto ctr = resortRays());
		cuda::sync(std::to_string(__LINE__));
		if (get<parameters::vrtxRenderFluid>()) {
			if (get<parameters::vrtxRenderSurface>()) {
				//MEASURE_CUDA(intersectFluid, stream,
				   // cudaMemcpyAsync(vrtx::rayCounter, &blocks_1080, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
				   // LAUNCH(vrtx::render::raySchedulerIntersectFluidStaticWarp, blocks_1080, blockSize_1080, 0, stream)();
				//);
				MEASURE_CUDA(intersectFluid, stream,
					cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
				cudaStreamSynchronize(stream);
				LAUNCH(vrtx::render::raySchedulerIntersectFluidStatic6, blocks_1080, blockSize_1080, 0, stream)();
				cudaStreamSynchronize(stream);
				);
				//cuda::sync(std::to_string(__LINE__));
				MEASURE_CUDA(calculateNormals, stream,
					cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
				LAUNCH(vrtx::render::raySchedulerCalculateNormalsSolo, blocks_1080, blockSize_1080, 0, stream)();
				);
				//cuda::sync(std::to_string(__LINE__));
			}
			else {
				MEASURE_CUDA(intersectFluid, stream, cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
				LAUNCH(vrtx::render::raySchedulerIntersectFluidStaticParticles, blocks_1080, blockSize_1080, 0, stream)(!get<parameters::vrtxRenderGrid>());
				);
			}
		}
		if (get<parameters::vrtxRenderNormals>()) {
			MEASURE_CUDA(bounceRays, stream,
				LAUNCH(vrtx::render::shadeAndBounceRaysNormals, griddim, blockdim, 0, stream)(dist(rd), get<parameters::vrtxRenderBVH>() &&
					get<parameters::boundaryVolumes>().size() > 0 ? 1 : 0, bvhs);
			);
		}
		else {
			MEASURE_CUDA(bounceRays, stream,
				cudaStreamSynchronize(stream);
				LAUNCH(vrtx::render::shadeAndBounceRays, griddim, blockdim, 0, stream)(dist(rd),
					get<parameters::vrtxRenderBVH>() && get<parameters::boundaryVolumes>().size() > 0 ? 1 : 0, bvhs, fluidMaterial,
					get<parameters::vrtxRenderFluid>() && !get<parameters::vrtxRenderSurface>());
			);
		}
		//cuda::sync(std::to_string(__LINE__));


		//std::cout << i << " ";
		if (get<parameters::vrtxRenderFluid>()) {
			timers[std::string("intersection ")].push_back((float)intersectFluid());
			if (get<parameters::vrtxRenderSurface>())
				timers[std::string("fluid normal ")].push_back((float)calculateNormals());
		}
		timers[std::string("ray bouncing ")].push_back((float)bounceRays());
		timers[std::string("ray sorting  ")].push_back((float)sortRays());
		timers[std::string("collision ctr")].push_back(-ctr);
	}

	//std::cout << std::endl;
	if (get<parameters::vrtxRenderNormals>()) {
		MEASURE_CUDA(toneMap, stream,
			LAUNCH(vrtx::common::toneMapNormals, griddim, blockdim, 0, stream)(framenumber, (float3 *)acc, vrtx::cuImage, 1.f);
		);
	}
	else {
		MEASURE_CUDA(toneMap, stream,
			LAUNCH(vrtx::common::toneMap, griddim, blockdim, 0, stream)(framenumber, (float3 *)acc, vrtx::cuImage, 1.f);
		);
	}
	timers[std::string("tone mapping")].push_back(toneMap());

	//cuda::sync(std::to_string(__LINE__));
	cudaStreamSynchronize(stream);
	cuda::sync("end of vrtx step");
	return timers;
}