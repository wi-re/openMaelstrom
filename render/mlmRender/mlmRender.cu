#define NO_QT
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <render/mlmRender/mlmRender.h>
#include <sstream>
#include <texture_types.h>
#include <utility/math.h>
#include <vector_functions.h>
#include <vector_types.h>
//#define DEBUG_AABB

#pragma region MACROS
#define SMRAY rayCache[threadIdx.x]
#define IDRAY rayid[threadIdx.x]
#define IDRAY2D rayid[threadIdx.x + blockDim.x * threadIdx.y]
#ifdef __INTELLISENSE__
#define gridDim int3{32,1,1}
#define blockDim int3{32,1,1}
#define threadIdx int3{0,0,0}
#define blockIdx int3{0,0,0}
#endif
#define SMRAY_DIR_X		SMRAY.direction.x
#define SMRAY_DIR_Y		SMRAY.direction.y
#define SMRAY_DIR_Z		SMRAY.direction.z
#define SMRAY_ORIG_X	SMRAY.origin.x
#define SMRAY_ORIG_Y	SMRAY.origin.y
#define SMRAY_ORIG_Z	SMRAY.origin.z
#define SMRAY_IDX		SMRAY.index
#define SMRAY_BOUNCES	SMRAY.bounces
#define SMRAY_DEPTH		SMRAY.depth
#define SMRAY_MASK_X	SMRAY.mask.x
#define SMRAY_MASK_Y	SMRAY.mask.y
#define SMRAY_MASK_Z	SMRAY.mask.z
#define SMRAY_DIR		SMRAY.direction
#define SMRAY_ORIG		SMRAY.origin
#define SMRAY_MASK		SMRAY.mask

#define IDRAY_DIR_X		IDRAY.direction.x
#define IDRAY_DIR_Y		IDRAY.direction.y
#define IDRAY_DIR_Z		IDRAY.direction.z
#define IDRAY_ORIG_X	IDRAY.origin.x
#define IDRAY_ORIG_Y	IDRAY.origin.y
#define IDRAY_ORIG_Z	IDRAY.origin.z
#define IDRAY_IDX		IDRAY.index
#define IDRAY_BOUNCES	IDRAY.bounces
#define IDRAY_DEPTH		IDRAY.depth
#define IDRAY_DIR		IDRAY.direction
#define IDRAY_ORIG		IDRAY.origin

#define GET_NEXT_RAY {if(!scheduler::grabRay()) return; else continue;}
#define GET_NEXT_IDRAY {if(!scheduler::grabIDRay()) return; else continue;}
#pragma endregion

namespace rtx {
	enum Refl_t { DIFF, METAL, SPEC, REFR, COAT };
	struct uint10_3 {
		uint32_t x : 10;
		uint32_t y : 10;
		uint32_t z : 10;
		uint32_t valid : 1;
	};
	struct Ray {
		float3 orig, dir;  
	};
	struct ScheduledRay {
		float3 origin;
		int32_t index;
		float3 direction;
		float depth;
		float3 mask;
		int32_t bounces;
	};
	struct RayWithIndex {
		float3 origin;
		float3 direction;
		int32_t index;
	};
	struct RayWithIndexAndDepth {
		float3 origin;
		float depth;
		float3 direction;
		uint32_t index : 24;
		uint32_t bounces : 4;
		uint32_t geomType : 4;
	};
	struct FluidIntersection {
		float3 normal;
		float depth = 1e21f;
	};
	struct RayIntersection {
		float depth;
		Refl_t materialType;
		float3 surfaceColor;
		float3 surfaceNormal;
		float3 emission;
	};
	struct rayHit {
		float3 position;
		float depth;
		float3 normal;
		bool status;
		int32_t voxelIdx;
	};
	struct rayHitLean {
		int3 voxel;
		float depth;
	};
	struct rayHitSuperLean {
		float depth;
	};
	struct AABBHit {
		bool hit;
		float tmin;
		float tmax;
	};
	struct Pixel {
		float3 color;
		float3 mask;
	};
	struct bounceRay {
		int32_t pixelIdx;
		float3 orig;
		float3 dir;
	};
	struct Sphere {
		float rad;
		float3 pos, emi, col;
		Refl_t refl;
		__device__ float intersect(const Ray &r) const {
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
		__device__ float intersectSM() const;
		__device__ float intersectID() const;
		__device__ float intersectID2D() const;
	};

	curandState* cuRandStates;
	Pixel* cuImage;
	Ray* cuCurrentRays;
	RayIntersection* cuRayIntersections;
	RayWithIndex* cuScheduledRays;
	FluidIntersection* cuFluidIntersections;
	RayWithIndexAndDepth* cuBlockedRays;
	int32_t* rayCounter;
	uint32_t* cRNGSeeds;
	
	__device__ __constant__ SceneInformation cScene;
	__device__ __constant__ FluidSystem fluidSystem;
	__device__ __constant__ FluidMemory fluidMemory;
	__device__ __constant__ Sphere spheres[] = {
		//{16, {128.0f, 128, 128}, {6, 4, 2}, {0.f, 0.f, 0.f}, DIFF},
		 
		{10000, {50.0f, 40.8f, -1060.f}, {0.55f, 0.55f, 0.55f}, {0.175f, 0.175f, 0.175f}, DIFF},

		//{100000, {0.0f, 0, -100000.}, {0, 0, 0}, {0.5f, 0.0f, 0.0f}, DIFF},
		//{100000, {0.0f, 0, -100000.1}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, DIFF}
	};
	__device__ __constant__ int32_t cNumRays;
	__device__ __constant__ RayWithIndex* cRays;
	__device__ __constant__ RayWithIndexAndDepth* cRaysDepth;
	__device__ __constant__ int32_t* cRayCounter;
	__device__ __constant__ Pixel* cImage;
	__device__ __constant__ curandState* cRandStates;
	__device__ __constant__ FluidIntersection* cFluidIntersections;
	__device__ __constant__ uint32_t* cuSeeds;
	__device__ __constant__ int32_t cMsaaRate;

#pragma region MORTON_LUT
	__device__ __constant__ int32_t morton256_x[256] = {
	0x00000000,
	0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049, 0x00000200,
	0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249, 0x00001000,
	0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049, 0x00001200,
	0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249, 0x00008000,
	0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049, 0x00008200,
	0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249, 0x00009000,
	0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049, 0x00009200,
	0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249, 0x00040000,
	0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049, 0x00040200,
	0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249, 0x00041000,
	0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049, 0x00041200,
	0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249, 0x00048000,
	0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049, 0x00048200,
	0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249, 0x00049000,
	0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049, 0x00049200,
	0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249, 0x00200000,
	0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049, 0x00200200,
	0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249, 0x00201000,
	0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049, 0x00201200,
	0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249, 0x00208000,
	0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049, 0x00208200,
	0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249, 0x00209000,
	0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049, 0x00209200,
	0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249, 0x00240000,
	0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049, 0x00240200,
	0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249, 0x00241000,
	0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049, 0x00241200,
	0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249, 0x00248000,
	0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049, 0x00248200,
	0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249, 0x00249000,
	0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049, 0x00249200,
	0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249
	};
	__device__ __constant__  int32_t morton256_y[256] = {
	0x00000000,
	0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092, 0x00000400,
	0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492, 0x00002000,
	0x00002002, 0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092, 0x00002400,
	0x00002402, 0x00002410, 0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492, 0x00010000,
	0x00010002, 0x00010010, 0x00010012, 0x00010080, 0x00010082, 0x00010090, 0x00010092, 0x00010400,
	0x00010402, 0x00010410, 0x00010412, 0x00010480, 0x00010482, 0x00010490, 0x00010492, 0x00012000,
	0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082, 0x00012090, 0x00012092, 0x00012400,
	0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490, 0x00012492, 0x00080000,
	0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092, 0x00080400,
	0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492, 0x00082000,
	0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092, 0x00082400,
	0x00082402, 0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492, 0x00090000,
	0x00090002, 0x00090010, 0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092, 0x00090400,
	0x00090402, 0x00090410, 0x00090412, 0x00090480, 0x00090482, 0x00090490, 0x00090492, 0x00092000,
	0x00092002, 0x00092010, 0x00092012, 0x00092080, 0x00092082, 0x00092090, 0x00092092, 0x00092400,
	0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482, 0x00092490, 0x00092492, 0x00400000,
	0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090, 0x00400092, 0x00400400,
	0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492, 0x00402000,
	0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092, 0x00402400,
	0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492, 0x00410000,
	0x00410002, 0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092, 0x00410400,
	0x00410402, 0x00410410, 0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492, 0x00412000,
	0x00412002, 0x00412010, 0x00412012, 0x00412080, 0x00412082, 0x00412090, 0x00412092, 0x00412400,
	0x00412402, 0x00412410, 0x00412412, 0x00412480, 0x00412482, 0x00412490, 0x00412492, 0x00480000,
	0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082, 0x00480090, 0x00480092, 0x00480400,
	0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490, 0x00480492, 0x00482000,
	0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092, 0x00482400,
	0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492, 0x00490000,
	0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092, 0x00490400,
	0x00490402, 0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492, 0x00492000,
	0x00492002, 0x00492010, 0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092, 0x00492400,
	0x00492402, 0x00492410, 0x00492412, 0x00492480, 0x00492482, 0x00492490, 0x00492492
	};
	__device__ __constant__  int32_t morton256_z[256] = {
	0x00000000,
	0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124, 0x00000800,
	0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924, 0x00004000,
	0x00004004, 0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124, 0x00004800,
	0x00004804, 0x00004820, 0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924, 0x00020000,
	0x00020004, 0x00020020, 0x00020024, 0x00020100, 0x00020104, 0x00020120, 0x00020124, 0x00020800,
	0x00020804, 0x00020820, 0x00020824, 0x00020900, 0x00020904, 0x00020920, 0x00020924, 0x00024000,
	0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104, 0x00024120, 0x00024124, 0x00024800,
	0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920, 0x00024924, 0x00100000,
	0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124, 0x00100800,
	0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924, 0x00104000,
	0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124, 0x00104800,
	0x00104804, 0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924, 0x00120000,
	0x00120004, 0x00120020, 0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124, 0x00120800,
	0x00120804, 0x00120820, 0x00120824, 0x00120900, 0x00120904, 0x00120920, 0x00120924, 0x00124000,
	0x00124004, 0x00124020, 0x00124024, 0x00124100, 0x00124104, 0x00124120, 0x00124124, 0x00124800,
	0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904, 0x00124920, 0x00124924, 0x00800000,
	0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120, 0x00800124, 0x00800800,
	0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924, 0x00804000,
	0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124, 0x00804800,
	0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924, 0x00820000,
	0x00820004, 0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124, 0x00820800,
	0x00820804, 0x00820820, 0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924, 0x00824000,
	0x00824004, 0x00824020, 0x00824024, 0x00824100, 0x00824104, 0x00824120, 0x00824124, 0x00824800,
	0x00824804, 0x00824820, 0x00824824, 0x00824900, 0x00824904, 0x00824920, 0x00824924, 0x00900000,
	0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104, 0x00900120, 0x00900124, 0x00900800,
	0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920, 0x00900924, 0x00904000,
	0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124, 0x00904800,
	0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924, 0x00920000,
	0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124, 0x00920800,
	0x00920804, 0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924, 0x00924000,
	0x00924004, 0x00924020, 0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124, 0x00924800,
	0x00924804, 0x00924820, 0x00924824, 0x00924900, 0x00924904, 0x00924920, 0x00924924
	};
#pragma endregion

	__shared__ extern RayWithIndexAndDepth rayid[];
	__shared__ extern RayIntersection rayi[];
	__shared__ extern Ray raySM[];
	__shared__ extern ScheduledRay rayCache[];
	__shared__ extern float SMCache[];

	surface<void, cudaSurfaceType2D> surfaceWriteOut;

	__device__ __host__ __inline__ float sgn(float x) {
		return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f);
	}
	__device__ auto idx3D_to_mortonLUT(uint10_3 idx) {
		return morton256_x[idx.x] | morton256_y[idx.y] | morton256_z[idx.z];
	}
	__device__ auto idx3D_to_mortonLUT(int3 idx) {
		return morton256_x[idx.x] | morton256_y[idx.y] | morton256_z[idx.z];
	}
	template<typename T, typename U>
	__device__ auto position_to_mortonLUT(T p, U& arrays, float factor = 1.f) {
		return idx3D_to_mortonLUT(position_to_idx3D_i(p, arrays.min_coord, math::unit_get<1>(arrays.cell_size) * factor));
	}
	__device__ auto idx3D_to_hash(uint10_3 idx, uint32_t hash_entries) {
		return (idx.x * 73856093 + idx.y * 19349663 + idx.z * 83492791) % hash_entries;
	}
	__device__ float Sphere::intersectSM() const {
		float3 op = pos - SMRAY_ORIG; //
		float t, epsilon = 0.01f;
		float b = math::dot(op, SMRAY_DIR);
		float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
		if (disc < 0)
			return 0;
		else
			disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
	__device__ float Sphere::intersectID() const {
		float3 op = pos - IDRAY_ORIG; //
		float t, epsilon = 0.01f;
		float b = math::dot(op, IDRAY_DIR);
		float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
		if (disc < 0)
			return 0;
		else
			disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
	__device__ float Sphere::intersectID2D() const {
		float3 op = pos - IDRAY2D.origin; //
		float t, epsilon = 0.01f;
		float b = math::dot(op, IDRAY2D.direction);
		float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
		if (disc < 0)
			return 0;
		else
			disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
	__device__ auto randf() {
		auto x = cuSeeds[threadIdx.x + blockIdx.x * blockDim.x];
		x ^= x >> 13;
		x ^= x << 17;
		x ^= x >> 5;
		cuSeeds[threadIdx.x + blockIdx.x * blockDim.x] = x;
		auto r = (x & 0x007FFFFF) | 0x3F800000;
		return *reinterpret_cast<float*>(&r) - 1.f;
	}

	namespace common {
		__device__ Ray generateCameraRay(int32_t x, int32_t y, curandState& randState) {
			float3 rendercampos = float3{ cScene.m_camera.position.x, cScene.m_camera.position.y, cScene.m_camera.position.z };

			int32_t pixelx = x;
			int32_t pixely = cScene.height - y - 1;

			//float3 finalcol = float3{ 0.0f, 0.0f, 0.0f };
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
			else
			{
				aperturePoint = rendercampos;
			}
			float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
			apertureToImagePlane = math::normalize(apertureToImagePlane);
			float3 rayInWorldSpace = math::normalize(apertureToImagePlane);
			float3 originInWorldSpace = aperturePoint;

			return Ray{ originInWorldSpace, rayInWorldSpace };
		}
		__global__ void initRNGSeeds(uint32_t* rngStates, int32_t seed) {
			int32_t gIdx = threadIdx.x + blockIdx.x * blockDim.x;
			rngStates[gIdx] = gIdx ^ seed;
		}
		__global__ void generatePrimaryRays(int32_t seed, Pixel* image, Ray* rays) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			int32_t threadId =
				(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

			curandState randState;
			curand_init(seed + threadId, 0, 0, &randState);

			auto worldRay = generateCameraRay(x, y, randState);
			image[i] = Pixel{ float3{0.f,0.f,0.f}, float3{1.f,1.f,1.f} };
			rays[i] = Ray{ worldRay.orig, worldRay.dir };
		}
		__global__ void generateScheduledRays(int32_t seed, Pixel* image, RayWithIndex* rays, Ray* oldRays, int32_t msaa_factor) {
			int32_t x = blockIdx.x * blockDim.y + threadIdx.y;
			int32_t y = blockIdx.y * blockDim.z + threadIdx.z;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			int32_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
			int32_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				+ (threadIdx.z * (blockDim.x * blockDim.y))
				+ (threadIdx.y * blockDim.x)
				+ threadIdx.x;

			curandState randState;
			curand_init(seed + threadId, 0, 0, &randState);

			image[i] = Pixel{ float3{0.f,0.f,0.f}, float3{1.f,1.f,1.f} };
			auto worldRay = generateCameraRay(x, y, randState);
			rays[i  * msaa_factor + threadIdx.x] = RayWithIndex{ worldRay.orig, worldRay.dir, i };
		}
		__global__ void generateBlockedRays(int32_t seed, Pixel* image, RayWithIndexAndDepth* rays, Ray* oldRays, int32_t msaa_factor) {
			int32_t x = blockIdx.x * blockDim.y + threadIdx.y;
			int32_t y = blockIdx.y * blockDim.z + threadIdx.z;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;

			int32_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
			int32_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
				+ (threadIdx.z * (blockDim.x * blockDim.y))
				+ (threadIdx.y * blockDim.x)
				+ threadIdx.x;

			curandState randState;
			curand_init(seed + threadId, 0, 0, &randState);

			image[i] = Pixel{ float3{0.f,0.f,0.f}, float3{1.f,1.f,1.f} };
			auto worldRay = generateCameraRay(x, y, randState);
			rays[i  * msaa_factor + threadIdx.x] = RayWithIndexAndDepth{ worldRay.orig, FLT_MAX, worldRay.dir, (uint32_t) i, 0u, 0u};
		}
		__global__ void toneMap(int32_t frameNumber, float3* accumBuffer, Pixel* image, float rate) {
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
			float4 out{ (powf(colour.x, 1 / 2.2f)), (powf(colour.y, 1 / 2.2f)), (powf(colour.z, 1 / 2.2f)), 1.f };
			//out = float4{ colour.x, colour.y, colour.z, 1.f };
			surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
		}
	}

	namespace aabb {
		__device__ __host__ auto rayIntersectAABB(Ray worldRay, float3 aabb_min, float3 aabb_max) {
			//float tmin, tmax, tymin, tymax, tzmin, tzmax;

			//float invdirx = 1.f / worldRay.dir.x;
			//float invdiry = 1.f / worldRay.dir.y;
			//float invdirz = 1.f / worldRay.dir.z;

			float tmin = ((worldRay.dir.x < 0.f ? aabb_max.x : aabb_min.x) - worldRay.orig.x) / worldRay.dir.x;
			float tmax = ((worldRay.dir.x < 0.f ? aabb_min.x : aabb_max.x) - worldRay.orig.x) / worldRay.dir.x;
			float tymin = ((worldRay.dir.y < 0.f ? aabb_max.y : aabb_min.y) - worldRay.orig.y) / worldRay.dir.y;
			float tymax = ((worldRay.dir.y < 0.f ? aabb_min.y : aabb_max.y) - worldRay.orig.y) / worldRay.dir.y;
#ifndef __CUDA_ARCH__
			std::cout << worldRay.orig << worldRay.dir << aabb_min << aabb_max << std::endl;
			std::cout << tmin << " " << tmax << " " << tymin << " " << tymax << std::endl;
#endif
			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((worldRay.dir.z < 0.f ? aabb_max.z : aabb_min.z) - worldRay.orig.z) / worldRay.dir.z;
			float tzmax = ((worldRay.dir.z < 0.f ? aabb_min.z : aabb_max.z) - worldRay.orig.z) / worldRay.dir.z;

#ifndef __CUDA_ARCH__
			std::cout << tzmin << " " << tzmax << std::endl;
#endif
			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ AABBHit rayIntersectFluidAABB(Ray worldRay) {
			float tmin, tmax, tymin, tymax, tzmin, tzmax;

			float invdirx = 1.f / worldRay.dir.x;
			float invdiry = 1.f / worldRay.dir.y;
			float invdirz = 1.f / worldRay.dir.z;

			tmin = (fluidSystem.bounds[invdirx < 0.f].x - worldRay.orig.x) * invdirx;
			tmax = (fluidSystem.bounds[1 - (invdirx < 0.f)].x - worldRay.orig.x) * invdirx;
			tymin = (fluidSystem.bounds[invdiry < 0.f].y - worldRay.orig.y) * invdiry;
			tymax = (fluidSystem.bounds[1 - (invdiry < 0.f)].y - worldRay.orig.y) * invdiry;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			tzmin = (fluidSystem.bounds[invdirz < 0.f].z - worldRay.orig.z) * invdirz;
			tzmax = (fluidSystem.bounds[1 - (invdirz < 0.f)].z - worldRay.orig.z) * invdirz;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ AABBHit rayIntersectFluidAABBSMRAY() {
			const auto aabb_min = fluidSystem.bounds[0];
			const auto aabb_max = fluidSystem.bounds[1];
			float tmin = ((SMRAY_DIR_X < 0.f ? aabb_max.x : aabb_min.x) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tmax = ((SMRAY_DIR_X < 0.f ? aabb_min.x : aabb_max.x) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tymin = ((SMRAY_DIR_Y < 0.f ? aabb_max.y : aabb_min.y) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;
			float tymax = ((SMRAY_DIR_Y < 0.f ? aabb_min.y : aabb_max.y) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((SMRAY_DIR_Z < 0.f ? aabb_max.z : aabb_min.z) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;
			float tzmax = ((SMRAY_DIR_Z < 0.f ? aabb_min.z : aabb_max.z) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ AABBHit rayIntersectFluidAABBIDRAY() {
			const auto aabb_min = fluidSystem.bounds[0];
			const auto aabb_max = fluidSystem.bounds[1];
			float tmin = ((IDRAY_DIR_X < 0.f ? aabb_max.x : aabb_min.x) - IDRAY_ORIG_X) / IDRAY_DIR_X;
			float tmax = ((IDRAY_DIR_X < 0.f ? aabb_min.x : aabb_max.x) - IDRAY_ORIG_X) / IDRAY_DIR_X;
			float tymin = ((IDRAY_DIR_Y < 0.f ? aabb_max.y : aabb_min.y) - IDRAY_ORIG_Y) / IDRAY_DIR_Y;
			float tymax = ((IDRAY_DIR_Y < 0.f ? aabb_min.y : aabb_max.y) - IDRAY_ORIG_Y) / IDRAY_DIR_Y;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((IDRAY_DIR_Z < 0.f ? aabb_max.z : aabb_min.z) - IDRAY_ORIG_Z) / IDRAY_DIR_Z;
			float tzmax = ((IDRAY_DIR_Z < 0.f ? aabb_min.z : aabb_max.z) - IDRAY_ORIG_Z) / IDRAY_DIR_Z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ auto rayIntersectAABBSM(float3 aabb_min, float3 aabb_max) {
			float tmin = ((SMRAY_DIR_X < 0.f ? aabb_max.x : aabb_min.x) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tmax = ((SMRAY_DIR_X < 0.f ? aabb_min.x : aabb_max.x) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tymin = ((SMRAY_DIR_Y < 0.f ? aabb_max.y : aabb_min.y) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;
			float tymax = ((SMRAY_DIR_Y < 0.f ? aabb_min.y : aabb_max.y) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((SMRAY_DIR_Z < 0.f ? aabb_max.z : aabb_min.z) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;
			float tzmax = ((SMRAY_DIR_Z < 0.f ? aabb_min.z : aabb_max.z) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ auto rayIntersectAABBID(float3 aabb_min, float3 aabb_max) {
			float tmin = ((IDRAY_DIR_X < 0.f ? aabb_max.x : aabb_min.x) - IDRAY_ORIG_X) / IDRAY_DIR_X;
			float tmax = ((IDRAY_DIR_X < 0.f ? aabb_min.x : aabb_max.x) - IDRAY_ORIG_X) / IDRAY_DIR_X;
			float tymin = ((IDRAY_DIR_Y < 0.f ? aabb_max.y : aabb_min.y) - IDRAY_ORIG_Y) / IDRAY_DIR_Y;
			float tymax = ((IDRAY_DIR_Y < 0.f ? aabb_min.y : aabb_max.y) - IDRAY_ORIG_Y) / IDRAY_DIR_Y;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((IDRAY_DIR_Z < 0.f ? aabb_max.z : aabb_min.z) - IDRAY_ORIG_Z) / IDRAY_DIR_Z;
			float tzmax = ((IDRAY_DIR_Z < 0.f ? aabb_min.z : aabb_max.z) - IDRAY_ORIG_Z) / IDRAY_DIR_Z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ auto rayIntersectAABBSMU(float3 aabb_min, float len) {
			float tmin = ((SMRAY_DIR_X < 0.f ? aabb_min.x + len : aabb_min.x) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tmax = ((SMRAY_DIR_X < 0.f ? aabb_min.x : aabb_min.x + len) - SMRAY_ORIG_X) / SMRAY_DIR_X;
			float tymin = ((SMRAY_DIR_Y < 0.f ? aabb_min.y + len : aabb_min.y) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;
			float tymax = ((SMRAY_DIR_Y < 0.f ? aabb_min.y : aabb_min.y + len) - SMRAY_ORIG_Y) / SMRAY_DIR_Y;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((SMRAY_DIR_Z < 0.f ? aabb_min.z + len : aabb_min.z) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;
			float tzmax = ((SMRAY_DIR_Z < 0.f ? aabb_min.z : aabb_min.z + len) - SMRAY_ORIG_Z) / SMRAY_DIR_Z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
	}

	namespace traversal {
		__device__ auto lookup_cell(const int3 &idx) {
			if (idx.x >= fluidMemory.grid_size.x || idx.y >= fluidMemory.grid_size.y || idx.z >= fluidMemory.grid_size.z)
				return -1;
			if (idx.x < 0 || idx.y < 0 || idx.z < 0)
				return -1;
			auto morton = idx3D_to_hash(idx, fluidMemory.hash_entries);
			auto s = fluidMemory.hashMap[idx3D_to_morton(idx) % fluidMemory.hash_entries];
			if (s.hash.kind == LIST_ALWAYS_FALSE)
				return INT_MAX;
			if (s.hash.kind == LIST_ALWAYS_TRUE)
				return 1;
			if (s.hash.kind == LIST_COMPACT)
				if (morton == s.cell.hash)
					return 1;
			if(s.hash.kind == LIST_ITERATE)
			for (int32_t ii = s.hash.beginning; ii < s.hash.beginning + s.hash.length; ++ii)
				if (fluidMemory.cellSpan[ii].cell.hash == morton)
					return 1;
			return INT_MAX;
		}
		//__device__ auto lookup_cell(const uint10_3& idx) {
		//	if (idx.x >= fluidMemory.grid_size.x || idx.y >= fluidMemory.grid_size.y || idx.z >= fluidMemory.grid_size.z)
		//		return INT_MAX;
		//	//auto morton = idx3D_to_mortonLUT(idx);
		//	auto s = fluidMemory.hashMap[idx3D_to_hash(idx, fluidMemory.hash_entries)];
		//	if (s.compacted == 1 && s.beginning != UINT31_MAX) {
		//		auto cs = cell_span{ (int32_t)s.beginning, s.length };
		//		auto jj = cs.beginning;
		//		if (position_to_mortonLUT(fluidMemory.position[jj], fluidMemory, 1.f) == idx3D_to_mortonLUT(idx)) {
		//			return cs.beginning;
		//		}
		//	}
		//	else {
		//		auto ii = (int32_t)s.beginning;
		//		if (s.beginning == UINT31_MAX)
		//			return INT_MAX;
		//		int32_t l = idx3D_to_mortonLUT(idx);
		//		for (; ii < s.beginning + s.length;) {
		//			auto cs = fluidMemory.cellSpan[ii];
		//			++ii;
		//			auto jj = cs.beginning;
		//			if (position_to_mortonLUT(fluidMemory.position[jj], fluidMemory, 1.f) == l)
		//				return ii;
		//		}
		//	}

		//	return INT_MAX;
		//}
		//__device__ auto lookupVoxelCenter(const float3& voxelCenter) {
		//	if (voxelCenter.x < fluidSystem.bounds[0].x || voxelCenter.y < fluidSystem.bounds[0].y || voxelCenter.z < fluidSystem.bounds[0].z)
		//		return INT_MAX;
		//	if (voxelCenter.x > fluidSystem.bounds[1].x || voxelCenter.y > fluidSystem.bounds[1].y || voxelCenter.z > fluidSystem.bounds[1].z)
		//		return INT_MAX;
		//	auto s = fluidMemory.hashMap[position_to_hash(voxelCenter, fluidMemory)];
		//	if (s.compacted == 1 && s.beginning != UINT31_MAX) {
		//		auto d = math::max_elem(math::abs((math::castTo<float3>(fluidMemory.position[s.beginning]) - voxelCenter) / fluidMemory.cell_size.x * 0.5f));
		//		if (d < 1.f)
		//			return (int32_t)s.beginning;
		//	}
		//	else {
		//		if (s.beginning == UINT31_MAX)
		//			return INT_MAX;
		//		for (int32_t ii = s.beginning; ii < s.beginning + s.length; ++ii) {
		//			auto jj = fluidMemory.cellSpan[ii].beginning;
		//			auto d = math::max_elem(math::abs((math::castTo<float3>(fluidMemory.position[jj]) - voxelCenter) / fluidMemory.cell_size.x * 0.5f));
		//			if (d < 1.f)
		//				return jj;
		//		}
		//	}

		//	return INT_MAX;
		//}
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
		__device__ __host__ float3 intBound(const float3& s, const float3& ds) {
			return float3{
				intBound2_s(s.x, ds.x),
				intBound2_s(s.y, ds.y),
				intBound2_s(s.z, ds.z) 
			};
		}
		template<typename Func> __device__ void iterateVoxels(Func&& fn, const float3& start, const float3& dir, float aabb_min, const float& aabb_max) {
			aabb_min = math::max( 0.f, aabb_min);
			auto position = start + dir * aabb_min;
			auto offset = (position - fluidMemory.min_coord) / fluidMemory.cell_size.x;
			int3 voxelPosition = math::castTo<int3>(math::floorf(offset));
			float3 tDelta = float3{ sgn(dir.x) / dir.x,sgn(dir.y) / dir.y,sgn(dir.z)/dir.z };
			float3 tMax = intBound(offset, dir);
			float limit = (aabb_max - aabb_min) / fluidMemory.cell_size.x + 1.f;
			while (true) {
				auto cell_idx = lookup_cell(voxelPosition);
				if (cell_idx == 1) {
					if (fn(voxelPosition))
						return;
				}
				if (tMax.x < tMax.y) {
					if (tMax.x < tMax.z) {
						if (tMax.x > limit) return;
						voxelPosition.x += sgn(dir.x);
						tMax.x += tDelta.x;
					}
					else {
						if (tMax.z > limit) return;
						voxelPosition.z += sgn(dir.z);
						tMax.z += tDelta.z;
					}
				}
				else {
					if (tMax.y < tMax.z) {
						if (tMax.y > limit) return;
						voxelPosition.y += sgn(dir.y);
						tMax.y += tDelta.y;
					}
					else {
						if (tMax.z > limit) return;
						voxelPosition.z += sgn(dir.z);
						tMax.z += tDelta.z;
					}
				}
			}
		}
		//template<typename Func> __device__ void iterateVoxelsCompact(Func&& fn, const float3& start, const float3& dir, const float& aabb_min, const float& aabb_max) {
		//	//auto intBound = [](auto s, auto ds) { 
		//	//	auto sIsInteger = (roundf(s) == s);
		//	//	if (ds < 0 && sIsInteger)
		//	//		return 0.f;
		//	//	return (ds > 0 ? math::ceilf(s) - s : s - math::floorf(s)) / math::abs(ds);
		//	//};
		//	auto vP = position_to_idx3D(start, fluidMemory.min_coord, math::unit_get<1>(fluidMemory.cell_size));
		//	uint10_3 voxelPosition{ vP.x, vP.y, vP.z, 1 };
		//	auto offset = (start - fluidMemory.min_coord) / fluidMemory.cell_size;
		//	//float3 tMax{ intBound(offset.x, dir.x), intBound(offset.y, dir.y), intBound(offset.z, dir.z) };
		//	float3 tMax = intBound(offset, dir);
		//	//float3 tDelta = math::castTo<float3>(uchar3{ sgn(dir.x), sgn(dir.y), sgn(dir.z) }) / dir;
		//	while (true) {
		//		auto cell_idx = lookup_cell(voxelPosition);
		//		if (cell_idx != UINT31_MAX) {
		//			if (fn(voxelPosition))
		//				return;
		//		}
		//		if (tMax.x < tMax.y) {
		//			if (tMax.x < tMax.z) {
		//				if (tMax.x > aabb_max - aabb_min) return;
		//				voxelPosition.x += sgn(dir.x);
		//				tMax.x += sgn(dir.x) / dir.x;
		//			}
		//			else {
		//				if (tMax.z > aabb_max - aabb_min) return;
		//				voxelPosition.z += sgn(dir.z);
		//				tMax.z += sgn(dir.z) / dir.z;
		//			}
		//		}
		//		else {
		//			if (tMax.y < tMax.z) {
		//				if (tMax.y > aabb_max - aabb_min) return;
		//				voxelPosition.y += sgn(dir.y);
		//				tMax.y += sgn(dir.y) / dir.y;
		//			}
		//			else {
		//				if (tMax.z > aabb_max - aabb_min) return;
		//				voxelPosition.z += sgn(dir.z);
		//				tMax.z += sgn(dir.z) / dir.z;
		//			}
		//		}
		//	}
		//}
		//template<typename Func> __device__ void iterateVoxelCenters(Func&& fn, const float3& start, const float3& dir, const float& aabb_min, const float& aabb_max) {
		//	//auto intBound = [](auto s, auto ds) { 
		//	//	auto sIsInteger = (roundf(s) == s);
		//	//	if (ds < 0 && sIsInteger)
		//	//		return 0.f;
		//	//	return (ds > 0 ? math::ceilf(s) - s : s - math::floorf(s)) / math::abs(ds);
		//	//};
		//	int3 voxelPosition = position_to_idx3D_i(start, fluidMemory.min_coord, fluidMemory.cell_size.x);
		//	float3 voxelPos = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) * fluidMemory.cell_size.x + 0.5f * fluidMemory.cell_size.x;
		//	auto offset = (start - fluidMemory.min_coord) / fluidMemory.cell_size;
		//	//float3 tMax{ intBound(offset.x, dir.x), intBound(offset.y, dir.y), intBound(offset.z, dir.z) };
		//	float3 tMax = intBound(offset, dir);
		//	//float3 tDelta = math::castTo<float3>(uchar3{ sgn(dir.x), sgn(dir.y), sgn(dir.z) }) / dir;
		//	while (true) {
		//		auto cell_idx = lookupVoxelCenter(voxelPos);
		//		if (cell_idx != UINT31_MAX) {
		//			if (fn(cell_idx))
		//				return;
		//		}
		//		if (tMax.x < tMax.y) {
		//			if (tMax.x < tMax.z) {
		//				if (tMax.x > aabb_max - aabb_min) return;
		//				voxelPos.x += sgn(dir.x) * fluidMemory.cell_size.x;
		//				tMax.x += sgn(dir.x) / dir.x;
		//			}
		//			else {
		//				if (tMax.z > aabb_max - aabb_min) return;
		//				voxelPos.z += sgn(dir.z) * fluidMemory.cell_size.x;
		//				tMax.z += sgn(dir.z) / dir.z;
		//			}
		//		}
		//		else {
		//			if (tMax.y < tMax.z) {
		//				if (tMax.y > aabb_max - aabb_min) return;
		//				voxelPos.y += sgn(dir.y) * fluidMemory.cell_size.x;
		//				tMax.y += sgn(dir.y) / dir.y;
		//			}
		//			else {
		//				if (tMax.z > aabb_max - aabb_min) return;
		//				voxelPos.z += sgn(dir.z) * fluidMemory.cell_size.x;
		//				tMax.z += sgn(dir.z) / dir.z;
		//			}
		//		}
		//	}
		//}
		template<typename Func> __device__ void iterateVoxelsSMRAY(Func&& fn, const float& aabb_min, const float& aabb_max) {
			auto intBound = [](auto s, auto ds) {
				auto sIsInteger = (roundf(s) == s);
				if (ds < 0 && sIsInteger)
					return 0.f;
				return (ds > 0 ? math::ceilf(s) - s : s - math::floorf(s)) / math::abs(ds);
			};
			int3 voxelPosition = position_to_idx3D_i(SMRAY.origin, fluidMemory.min_coord, math::unit_get<1>(fluidMemory.cell_size));
			char4 step{ static_cast<char>(sgn(SMRAY.direction.x)),  static_cast<char>(sgn(SMRAY.direction.y)),  static_cast<char>(sgn(SMRAY.direction.z)), 1 };
			auto offset = (SMRAY.origin - fluidMemory.min_coord) / fluidMemory.cell_size;
			float3 tMax{ intBound(offset.x, SMRAY.direction.x), intBound(offset.y, SMRAY.direction.y), intBound(offset.z, SMRAY.direction.z) };
			float3 tDelta = math::castTo<float3>(step) / SMRAY.direction;
			while (true) {
				auto cell_idx = lookup_cell(voxelPosition);
				if (cell_idx != UINT31_MAX) {
					if (fn(voxelPosition))
						return;
				}
				if (tMax.x < tMax.y) {
					if (tMax.x < tMax.z) {
						if (tMax.x > aabb_max - aabb_min) return;
						voxelPosition.x += step.x;
						tMax.x += tDelta.x;
					}
					else {
						if (tMax.z > aabb_max - aabb_min) return;
						voxelPosition.z += step.z;
						tMax.z += tDelta.z;
					}
				}
				else {
					if (tMax.y < tMax.z) {
						if (tMax.y > aabb_max - aabb_min) return;
						voxelPosition.y += step.y;
						tMax.y += tDelta.y;
						step.w = 2;
					}
					else {
						if (tMax.z > aabb_max - aabb_min) return;
						voxelPosition.z += step.z;
						tMax.z += tDelta.z;
						step.w = 3;
					}
				}
			}
		}
	}
	namespace bvh {
		__device__ auto rayIntersectBVHNode(const CFBVHNode& node, Ray worldRay) {
			float tmin, tmax, tymin, tymax, tzmin, tzmax;

			float invdirx = 1.f / worldRay.dir.x;
			float invdiry = 1.f / worldRay.dir.y;
			float invdirz = 1.f / worldRay.dir.z;

			float3 max = {
				invdirx < 0.f ? node.min.x : node.max.x,
				invdiry < 0.f ? node.min.y : node.max.y,
				invdirz < 0.f ? node.min.z : node.max.z
			};
			float3 min = {
				invdirx < 0.f ? node.max.x : node.min.x,
				invdiry < 0.f ? node.max.y : node.min.y,
				invdirz < 0.f ? node.max.z : node.min.z
			};

			tmin = (min.x - worldRay.orig.x) * invdirx;
			tmax = (max.x - worldRay.orig.x) * invdirx;
			tymin = (min.y - worldRay.orig.y) * invdiry;
			tymax = (max.y - worldRay.orig.y) * invdiry;

			if ((tmin > tymax) || (tymin > tmax))
				return AABBHit{ false };
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			tzmin = (min.z - worldRay.orig.z) * invdirz;
			tzmax = (max.z - worldRay.orig.z) * invdirz;

			if ((tmin > tzmax) || (tzmin > tmax))
				return AABBHit{ false };
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return AABBHit{ (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
		}
		__device__ rayHit rayIntersectFluidBVH(Ray worldRay, int32_t oldIdx = -1) {
			//auto aabb = rayIntersectFluidAABB(worldRay);
			//if (aabb.hit == false)
			//	return rayHit{ float3{FLT_MAX, FLT_MAX, FLT_MAX}, FLT_MAX, float3{1.f,0.f,0.f}, false, -1 };

			auto nodeNormal = [](auto node, auto point) {
				constexpr auto epsilon = 1e-4f;
				auto c = (node.min + node.max) * 0.5f;
				auto prel = point - c;
				auto d = math::abs((node.min - node.max) * 0.5f);
				auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
				return float3{ (float)n.x, (float)n.y, (float)n.z };
			};
			auto getHitInformation = [=](auto node, auto tmin, auto tmax) {
				float3 aabb_min = worldRay.orig + tmin * worldRay.dir;
				float3 aabb_max = worldRay.orig + tmax * worldRay.dir;
				// DEBUG render for AABB
				if (tmin >= 0.f)
					return rayHit{ aabb_min, tmin, math::abs(nodeNormal(node, aabb_min)), true };
				else
					return rayHit{ aabb_max, tmax, math::abs(nodeNormal(node, aabb_max)), true };
			};

			constexpr auto BVH_STACK_SIZE = 32;
			int32_t hitVoxelIdx = -1;
			float hitTmin = FLT_MAX;
			float hitTmax = -FLT_MAX;

			int32_t bvhStack[BVH_STACK_SIZE];
			int32_t bvhStackIdx = 0;
			bvhStack[bvhStackIdx++] = 0;

			while (bvhStackIdx) {
				int32_t boxIdx = bvhStack[bvhStackIdx - 1];
				bvhStackIdx--;
				auto node = fluidSystem.fluidBVH[boxIdx];
				if (!(node.u.leaf.count & 0x80000000)) { // INNER NODE
					auto intersection = rayIntersectBVHNode(node, worldRay);
					if (intersection.hit && intersection.tmin < hitTmin) {
						bvhStack[bvhStackIdx++] = node.u.inner.idxLeft;
						bvhStack[bvhStackIdx++] = node.u.inner.idxRight;
						if (bvhStackIdx > BVH_STACK_SIZE) {
							return rayHit{ float3{FLT_MAX, FLT_MAX, FLT_MAX}, FLT_MAX, float3{1.f,0.f,0.f}, false, -1 };
						}
						//return getHitInformation(node, intersection.tmin, intersection.tmax);
					}
				}
				else {
					auto intersection = rayIntersectBVHNode(node, worldRay);
					if (intersection.hit && intersection.tmin < hitTmin) {
						hitVoxelIdx = boxIdx;
						hitTmin = intersection.tmin;
						hitTmax = intersection.tmax;
					}
				}
			}
			if (hitVoxelIdx != -1) {
				auto hitNode = fluidSystem.fluidBVH[hitVoxelIdx];
				return getHitInformation(hitNode, hitTmin, hitTmax);
			}
			else
				return rayHit{ float3{FLT_MAX, FLT_MAX, FLT_MAX}, FLT_MAX, float3{1.f,0.f,0.f}, false, -1 };
		}
	}
	namespace fluid {
		__device__  rayHit rayIntersectFluid(Ray& worldRay, int32_t oldIdx = -1) {
			//#define DEBUG_AABB
			auto aabb_center = (fluidSystem.bounds[1] - fluidSystem.bounds[0]) / 2.f;
			auto aabb_normal = [](auto v) {
				constexpr auto epsilon = 1e-5f;
				auto c = (fluidSystem.bounds[0] + fluidSystem.bounds[1]) * 0.5f;
				auto prel = v - c;
				auto d = math::abs((fluidSystem.bounds[0] - fluidSystem.bounds[1]) * 0.5f);
				auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
				return float3{ (float)n.x, (float)n.y, (float)n.z };
			};
			worldRay.dir = math::normalize(worldRay.dir);
			auto aabb = aabb::rayIntersectFluidAABB(worldRay);
			if (aabb.hit == true) {
#ifdef DEBUG_AABB
				float3 aabb_min = worldRay.orig + aabb.tmin * worldRay.dir;
				float3 aabb_max = worldRay.orig + aabb.tmax * worldRay.dir;
				// DEBUG render for AABB
				if (aabb.tmin >= 0.f)
					return rayHit{ aabb_min, aabb.tmin, math::abs(aabb_normal(aabb_min)), true };
				else
					return rayHit{ aabb_max, aabb.tmax, math::abs(aabb_normal(aabb_max)), true };
#endif
				float t = FLT_MAX;
				char3 nc{ 1,0,0 };
				//float3 normal;
				
				traversal::iterateVoxels([&](int3 voxel) {
					constexpr auto epsilon = 1e-1f;
					float3 min = fluidMemory.min_coord + math::castTo<float3>(voxel) * fluidMemory.cell_size;
					float3 max = min + fluidMemory.cell_size;
					auto rH = aabb::rayIntersectAABB(worldRay, min, max);
					auto hitPosition = worldRay.orig + rH.tmin * worldRay.dir;
					auto c = (min + max) * 0.5f;
					auto prel = hitPosition - c;
					auto d = math::abs((min - max) * 0.5f);
					auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
					nc = char3{ static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z) };
					//normal = math::abs(prel / d);
					//normal = math::abs(float3{
					//	math::abs(prel.x) > math::abs(prel.y) && math::abs(prel.x) > math::abs(prel.z) ? prel.x : 0.f,
					//	math::abs(prel.y) > math::abs(prel.x) && math::abs(prel.y) > math::abs(prel.z) ? prel.y : 0.f,
					//	math::abs(prel.z) > math::abs(prel.y) && math::abs(prel.z) > math::abs(prel.x) ? prel.z : 0.f
					//});
	/*				normal = math::castTo<float3>(voxel);
					normal.x *= 0.001f;
					normal.y *= 0.02f;
					normal.z *= 0.02f;*/
					t = rH.tmin;
					return true;
				}, worldRay.orig, worldRay.dir, aabb.tmin, aabb.tmax);
				float3 hitPosition = worldRay.orig + t * worldRay.dir;
				float3 normal = float3{ (float)nc.x, (float)nc.y, (float)nc.z };
				if (nc.x == nc.y && nc.y == nc.z && nc.z == 0)
					normal = float3{ 1.f,0.f,0.f };
				if (t < 1e19f)
					return rayHit{ hitPosition, math::distance3(worldRay.orig, hitPosition), normal, true };
			}
			return rayHit{ float3{FLT_MAX, FLT_MAX, FLT_MAX}, FLT_MAX, float3{1.f,0.f,0.f}, false };
		}
		__device__  rayHitLean rayIntersectFluidLean(Ray& worldRay, int32_t oldIdx = -1) {
			auto aabb = aabb::rayIntersectFluidAABB(worldRay);
			if (aabb.hit == true) {
				float t = FLT_MAX;
				int3 hitVoxel{ INT_MAX, INT_MAX, INT_MAX };
				traversal::iterateVoxels([&](int3 voxel) {
					hitVoxel = voxel;
					t = 0.f;
					return true;
				}, worldRay.orig, worldRay.dir, math::max(aabb.tmin, 0.f), aabb.tmax);
				if (t < 1e19f)
					return rayHitLean{ hitVoxel, 0.f };
			}
			return rayHitLean{ {INT_MAX, INT_MAX, INT_MAX}, FLT_MAX };
		}
		__device__  rayHitLean rayIntersectFluidLeanSM(int32_t oldIdx = -1) {
			int32_t ii = threadIdx.x + blockDim.x * threadIdx.y;
			auto aabb = aabb::rayIntersectFluidAABB(raySM[ii]);
			if (aabb.hit == true) {
				float t = FLT_MAX;
				int3 hitVoxel{ INT_MAX, INT_MAX, INT_MAX };
				traversal::iterateVoxels([&](int3 voxel) {
					hitVoxel = voxel;
					t = 0.f;
					return true;
				}, raySM[ii].orig, raySM[ii].dir, math::max(aabb.tmin, 0.f), aabb.tmax);
				if (t < 1e19f)
					return rayHitLean{ hitVoxel, 0.f };
			}
			return rayHitLean{ {INT_MAX, INT_MAX, INT_MAX}, FLT_MAX };
		}
		__device__  int3 rayIntersectFluidLeanSMRAY(int32_t oldIdx = -1) {
			auto aabb = aabb::rayIntersectFluidAABBSMRAY();// (Ray{ SMRAY_ORIG, SMRAY_DIR });
			if (aabb.hit == true) {
				//float t = FLT_MAX;
				int3 idx{ INT_MAX, INT_MAX, INT_MAX };
				traversal::iterateVoxels([&](int3 voxel) {
					idx = voxel;
					//t = 0.f;
					return true;
				}, SMRAY_ORIG, SMRAY_DIR, math::max(aabb.tmin, 0.f), aabb.tmax);
				if (idx.x != INT_MAX)
					return idx;
			}
			return int3{ INT_MAX, INT_MAX, INT_MAX };
		}		
		__device__  int3 rayIntersectFluidLeanIDRAY(int32_t oldIdx = -1) {
			auto aabb = aabb::rayIntersectFluidAABBIDRAY();// (Ray{ SMRAY_ORIG, SMRAY_DIR });
			if (aabb.hit == true) {
				//float t = FLT_MAX;
				int3 idx{ INT_MAX, INT_MAX, INT_MAX };
				traversal::iterateVoxels([&](int3 voxel) {
					idx = voxel;
					//t = 0.f;
					return true;
				}, IDRAY_ORIG, IDRAY_DIR, math::max(aabb.tmin, 0.f), aabb.tmax);
				if (idx.x != INT_MAX)
					return idx;
			}
			return int3{ INT_MAX, INT_MAX, INT_MAX };
		}
		//__device__  uint10_3 rayIntersectFluidLeanSMRAYSuper(int32_t oldIdx = -1) {
		//	auto aabb = aabb::rayIntersectFluidAABBSMRAY();// (Ray{ SMRAY_ORIG, SMRAY_DIR });
		//	if (aabb.hit == true) {
		//		uint10_3 t{ 0, 0, 0, 0 };
		//		traversal::iterateVoxelsCompact([&](uint10_3 particleIdx) {
		//			t = particleIdx;
		//			return true;
		//		}, SMRAY_ORIG, SMRAY_DIR, 0.f, aabb.tmax);
		//		if (t.valid == true)
		//			return t;
		//	}
		//	return uint10_3{ 0, 0, 0, 0 };
		//}
	}
	namespace scheduler {
		__device__ void update_ray(int32_t counter) {
			SMRAY = ScheduledRay{ cRays[counter].origin, cRays[counter].index, cRays[counter].direction, 1e21f, float3{1.f,1.f,1.f}, 0 };
		}
		__device__ void updateIDRay(int32_t counter) {
			IDRAY = cRaysDepth[counter];// RayWithIndexAndDepth{ cRaysDepth[counter].origin, cRaysDepth[counter].depth, cRaysDepth[counter].direction, cRaysDepth[counter].index, cRaysDepth[counter].bounces, cRaysDepth[counter].geomType };
		}
		__device__ bool grabRay() {
			cuda_atomic<int32_t> atomicCounter(cRayCounter);
			int32_t counter = atomicCounter.add(1);
			if (counter >= cNumRays) return false;
			update_ray(counter);
			return true;
		}
		__device__ bool grabIDRay() {
			cuda_atomic<int32_t> atomicCounter(cRayCounter);
			int32_t counter = atomicCounter.add(1);
			if (counter >= cNumRays) return false;
			updateIDRay(counter);
			return true;
		}
	}
	namespace render{
		__global__ void  rayScheduler() {
			//const int32_t tIdx = threadIdx.x;
			//const int32_t gIdx = threadIdx.x + blockIdx.x * blockDim.x;
			__syncwarp();
			if (threadIdx.x + blockIdx.x * blockDim.x >= cNumRays) return;
			__syncwarp();
			scheduler::update_ray(threadIdx.x + blockIdx.x * blockDim.x);
			__syncwarp();
			while (1) {
				__syncwarp();
				auto fluidHit = fluid::rayIntersectFluidLeanSMRAY();
				__syncwarp();
				float3 normal, color;
				if (fluidHit.x != INT_MAX) {
					constexpr auto epsilon = 1e-1f;
					const auto cs = fluidMemory.cell_size.x;
					//auto hitPosition = SMRAY_ORIG + fluidHit * SMRAY_DIR;
					//int3 vIdx = position_to_idx3D_i(fluidMemory.position[fluidHit], fluidMemory.min_coord, cs);
					float3 min = fluidSystem.bounds[0] + math::castTo<float3>(fluidHit) * cs;
					auto rH = aabb::rayIntersectAABBSM(min, min + cs);
					auto t = rH.tmin;
					if (t < SMRAY_DEPTH) {
						SMRAY_DEPTH = t;
						auto hitPosition = SMRAY_ORIG + rH.tmin*1.01f * SMRAY_DIR;
						auto c = min + cs * 0.5f;
						auto prel = hitPosition - c;
						auto d = cs * 0.5f;
						auto ni = math::castTo<int3>(prel / d * (1.f + epsilon));
						auto nc = char3{ static_cast<char>(ni.x), static_cast<char>(ni.y), static_cast<char>(ni.z) };
						auto n = math::castTo<float3>(nc);
						//float3 n{ 1.f,0.f,0.f }; 
						auto nl = math::dot(n, SMRAY_DIR) < 0 ? n : n * -1;
						auto f = float3{ 0.05098f, 0.23137f, 0.494177f };

						normal = nl;
						color = f;
					} 
				}
				__syncwarp();
				int32_t idx = -1;
				for (int32_t sphere_id = 0; sphere_id < int32_t(sizeof(spheres) / sizeof(Sphere)); sphere_id++) {
					Sphere &sphere = spheres[sphere_id];
					float d = spheres[sphere_id].intersectSM();
					if (d && d < SMRAY_DEPTH) {
						SMRAY_DEPTH = d;
						auto x = SMRAY_ORIG + SMRAY_DIR * SMRAY_DEPTH;
						auto n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
						normal = math::dot(n, SMRAY_DIR) < 0 ? n : n * -1;
						color = sphere.col;
						idx = sphere_id;
					}
				}
				if (idx != -1 && math::length3(spheres[idx].emi) > 0.1f) {
					cuda_atomic<float3> atomicColor(&cImage[SMRAY_IDX].color);
					atomicColor.x += SMRAY_MASK_X * spheres[idx].emi.x;
					atomicColor.y += SMRAY_MASK_Y * spheres[idx].emi.y;
					atomicColor.z += SMRAY_MASK_Z * spheres[idx].emi.z;
					//GET_NEXT_RAY;
				}
				__syncwarp();
				if (SMRAY_DEPTH > 1e19f) GET_NEXT_RAY;
				//cuda_atomic<float3> atomicColor(&cImage[SMRAY_IDX].color);
				//atomicColor.x += randf();
				//atomicColor.y += randf();
				//atomicColor.z += randf();
				//GET_NEXT_RAY;
				SMRAY_MASK *= color;
				auto position = SMRAY_ORIG + SMRAY_DIR * SMRAY_DEPTH;

				//float phi = 2 * CUDART_PI_F * curand_uniform(cRandStates + threadIdx.x + blockIdx.x * blockDim.x);
				float phi = 2 * CUDART_PI_F * randf();
				//float r2 = curand_uniform(cRandStates + threadIdx.x + blockIdx.x * blockDim.x);
				float r2 = randf();
				float r2s = sqrtf(r2);
				float3 w = math::normalize(normal);
				float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
				float3 v = math::cross(w, u);

				auto dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));


				//SMRAY_MASK_X *= color.x;
				//SMRAY_MASK_Y *= color.y;
				//SMRAY_MASK_Z *= color.z;
				SMRAY_BOUNCES++;
				if (SMRAY_BOUNCES == 5)
					GET_NEXT_RAY;
				SMRAY_ORIG = position + w * 0.01f;
				//SMRAY_ORIG_X = position.x + w.x * 0.01f;
				//SMRAY_ORIG_Y = position.y + w.y * 0.01f;
				//SMRAY_ORIG_Z = position.z + w.z * 0.01f;
				SMRAY_DIR = dw;
				//SMRAY_DIR_X = dw.x;
				//SMRAY_DIR_Y = dw.y;
				//SMRAY_DIR_Z = dw.z;
				SMRAY_DEPTH = 1e21f;
			}
		}
		__global__ void  IDrayScheduler() {
			if (threadIdx.x + blockIdx.x * blockDim.x >= cNumRays) return;
			cuda_atomic<int32_t> atomicCounter(cRayCounter);
			auto counter = threadIdx.x + blockIdx.x * blockDim.x;
			IDRAY = cRaysDepth[counter];
			while (1) {
				auto fluidHit = fluid::rayIntersectFluidLeanIDRAY();
				if (fluidHit.x != INT_MAX) {
					constexpr auto epsilon = 1e-1f;
					const auto cs = fluidMemory.cell_size.x;
					float3 min = fluidSystem.bounds[0] + math::castTo<float3>(fluidHit) * cs;
					auto rH = aabb::rayIntersectAABBID(min, min + cs);
					auto t = rH.tmin;
					if (t < IDRAY_DEPTH) {
						IDRAY_DEPTH = t;
						auto hitPosition = IDRAY_ORIG + rH.tmin * IDRAY_DIR;
						auto c = min + cs * 0.5f;
						auto prel = hitPosition - c;
						auto d = cs * 0.5f;
						auto ni = math::castTo<int3>(prel / d * (1.f + epsilon));
						auto nc = char3{ static_cast<char>(ni.x), static_cast<char>(ni.y), static_cast<char>(ni.z) };
						auto n = math::castTo<float3>(nc);
						//float3 n{ 1.f,0.f,0.f }; 
						auto nl = math::dot(n, IDRAY_DIR) < 0 ? n : n * -1;
						//auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
						cFluidIntersections[counter].depth = rH.tmin;
						cFluidIntersections[counter].normal = n;
						//cRaysDepth[counter].depth = rH.tmin;
						//cRaysDepth[counter].geomType = MAX_VAL_04BIT;
					}
					else {
						cFluidIntersections[counter].depth = FLT_MAX;
					}
				}
				else {
					cFluidIntersections[counter].depth = FLT_MAX;
				}
				counter = atomicCounter.add(1);
				if (counter >= cNumRays) return;
				IDRAY = cRaysDepth[counter];
			}
		}
		__global__ void intersectAndShadeRaysSM(Ray *rays, Pixel *image, int32_t seed) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			int32_t ii = threadIdx.x + blockDim.x * threadIdx.y;
			float scene_t = 1e20f;
			raySM[ii] = rays[i];

			auto fluidHit = fluid::rayIntersectFluidLeanSM();
			float3 normal, color, emission;
			if (fluidHit.depth < FLT_MAX) {
				constexpr auto epsilon = 1e-1f;
				float3 min = fluidSystem.bounds[0] + math::castTo<float3>(fluidHit.voxel) * fluidMemory.cell_size;
				float3 max = fluidSystem.bounds[0] + (math::castTo<float3>(fluidHit.voxel) + 1.f) * fluidMemory.cell_size;
				auto rH = aabb::rayIntersectAABB(raySM[ii], min, max);
				auto hitPosition = raySM[ii].orig + rH.tmin * raySM[ii].dir;
				auto c = (min + max) * 0.5f;
				auto prel = hitPosition - c;
				auto d = math::abs((min - max) * 0.5f);
				auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
				auto nc = char3{ static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z) };
				auto t = rH.tmin;
				if (t < scene_t) {
					scene_t = t;
					auto n = math::castTo<float3>(nc);
					auto nl = math::dot(n, raySM[ii].dir) < 0 ? n : n * -1;
					auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
					auto emit = float3{ 0.f, 0.f, 0.f };
					// rayi[ii] = RayIntersection{ fluidHit.depth, DIFF, f, nl, emit };
					normal = nl;
					emission = emit;
					color = f;
				}
			}
			for (int32_t sphere_id = 0; sphere_id < int32_t(sizeof(spheres) / sizeof(Sphere)); sphere_id++) {
				Sphere &sphere = spheres[sphere_id];
				float d = spheres[sphere_id].intersect(raySM[ii]);
				if (d && d < scene_t) {
					scene_t = d;
					auto x = raySM[ii].orig + raySM[ii].dir * scene_t;
					auto n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
					normal = math::dot(n, raySM[ii].dir) < 0 ? n : n * -1;
					color = sphere.col;
					emission = sphere.emi;
					// rayi[ii] = RayIntersection{ d, sphere.refl, sphere.col, nl, sphere.emi };
				}
			}
			if (scene_t > 1e19f)
				return;
			// rayi[ii] = RayIntersection{ 1e20f, DIFF, float3{0.f,0.f,0.f}, float3{1.f,0.f,0.f},  float3{0.f,0.f,0.f} };

			curandState randState;
			curand_init(seed + i, 0, 0, &randState);

			auto pixel = image[i];

			// auto worldRay = rays[i];
			auto position = raySM[ii].orig + raySM[ii].dir * scene_t;
			pixel.color += (pixel.mask * emission);

			float phi = 2 * CUDART_PI_F * curand_uniform(&randState);
			float r2 = curand_uniform(&randState);
			float r2s = sqrtf(r2);
			float3 w = math::normalize(normal);
			float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
			float3 v = math::cross(w, u);

			auto dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));

			pixel.mask *= color;

			raySM[ii].orig = position + w * 0.01f;
			raySM[ii].dir = dw;
			rays[i] = raySM[ii];
			image[i] = pixel;
			//#undef i
			//#undef ii
		}
		__global__ void intersectRays(Ray* rays, RayIntersection* intersections) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			auto worldRay = rays[i];

			float scene_t = 1e20f;

			for (int32_t sphere_id = 0; sphere_id < int32_t(sizeof(spheres) / sizeof(Sphere)); sphere_id++) {
				Sphere &sphere = spheres[sphere_id];
				float d = spheres[sphere_id].intersect(worldRay);
				if (d && d < scene_t) {
					scene_t = d;
					auto x = worldRay.orig + worldRay.dir * scene_t;
					auto n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
					auto nl = math::dot(n, worldRay.dir) < 0 ? n : n * -1;
					intersections[i] = RayIntersection{ d, sphere.refl, sphere.col, n, sphere.emi };
				}
			}

			auto fluidHit = fluid::rayIntersectFluid(worldRay);
			if (fluidHit.status && fluidHit.depth < scene_t) {
				scene_t = fluidHit.depth;
				auto n = fluidHit.normal;
				//return n;
				//auto nl = math::dot(n, worldRay.dir) < 0 ? n : n * -1;
				auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
				auto emit = float3{ 0.f, 0.f, 0.f };
				intersections[i] = RayIntersection{ fluidHit.depth, DIFF, f, n, emit };
			}
			auto dw = worldRay.dir;
			if (dw.x != dw.x || dw.y != dw.y || dw.z != dw.z || (dw.x == 0.f && dw.y == 0.f && dw.z == 0.f)) {
				//printf("x[%f %f %f] : [%f %f %f]\n", worldRay.orig.x, worldRay.orig.y, worldRay.orig.z, worldRay.dir.x, worldRay.dir.y, worldRay.dir.z);
			}

			if (scene_t > 1e19f) {
				//printf("y[%f %f %f] : [%f %f %f]\n", worldRay.orig.x, worldRay.orig.y, worldRay.orig.z, worldRay.dir.x, worldRay.dir.y, worldRay.dir.z);
				intersections[i] = RayIntersection{ 1e20f, DIFF, float3{0.f,0.f,0.f}, float3{1.f,0.f,0.f},  float3{0.f,0.f,0.f} };
			}


		}
		__global__ void intersectRaysSpheresScene(int32_t seed) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			for (int32_t ii = 0; ii < cMsaaRate; ++ii) {
				int32_t idx = i * cMsaaRate + ii;
				IDRAY2D = cRaysDepth[idx];
				float3 normal, emission, color, x;
				if (cFluidIntersections[idx].depth < 1e19f) {
					normal = cFluidIntersections[idx].normal;
					IDRAY2D.depth = cFluidIntersections[idx].depth;
					x = IDRAY2D.origin + IDRAY2D.direction * IDRAY2D.depth;
					auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
					color = f;
					emission = float3{ 0.f,0.f,0.f };
				}

				for (int32_t sphere_id = 0; sphere_id < int32_t(sizeof(spheres) / sizeof(Sphere)); sphere_id++) {
					//Sphere &sphere = spheres[sphere_id];
					float d = spheres[sphere_id].intersectID2D();
					if (d && d < IDRAY2D.depth) {
						IDRAY2D.depth = d;
						IDRAY2D.geomType = sphere_id;
						x = IDRAY2D.origin + IDRAY2D.direction * IDRAY2D.depth;
						Sphere &sphere = spheres[IDRAY2D.geomType];
						normal = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
						color = sphere.col;
						emission = sphere.emi;
					}
				}
					normal = math::dot(normal, IDRAY2D.direction) < 0 ? normal : normal * -1;

					curandState randState;
					int32_t threadId =
						(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
					curand_init(seed + threadId, 0, 0, &randState);

					//float phi = 2 * CUDART_PI_F * randf();
					//float r2 = randf();
					float phi = 2 * CUDART_PI_F * curand_uniform(&randState);
					float r2 = curand_uniform(&randState);
					float r2s = sqrtf(r2);
					float3 w = math::normalize(normal);
					float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
					float3 v = math::cross(w, u);
					auto dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));

					cImage[IDRAY2D.index].color += cImage[IDRAY2D.index].mask * emission;
					cImage[IDRAY2D.index].mask *= color;
					IDRAY2D.origin = x + w * 0.01f;
					IDRAY2D.direction = dw;
				
				//if (IDRAY2D.bounces == 0) { cImage[IDRAY2D.index].color = float3{ d * 0.005f, d*0.005f, d * 0.005f }; }
				IDRAY2D.depth = FLT_MAX;
				IDRAY2D.geomType = 0;
				IDRAY2D.bounces++;
				cRaysDepth[i * cMsaaRate + ii] = IDRAY2D;
			}
		}
		__global__ void updateIDRAY(int32_t seed) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			for (int32_t ii = 0; ii < cMsaaRate; ++ii) {
				int32_t idx = i * cMsaaRate + ii;
				IDRAY2D = cRaysDepth[idx];
				float3 normal, emission, color;
				auto d = IDRAY2D.depth;
				auto x = IDRAY2D.origin + IDRAY2D.direction * IDRAY2D.depth;
				if (IDRAY2D.depth < 1e19f) {
					if (IDRAY2D.geomType == MAX_VAL_04BIT) {
						constexpr auto epsilon = 1e-1f;
						const auto cs = fluidMemory.cell_size.x;
						//auto hitPosition = SMRAY_ORIG + fluidHit * SMRAY_DIR;
						//int3 vIdx = position_to_idx3D_i(fluidMemory.position[fluidHit], fluidMemory.min_coord, cs);
						auto voxelIdx = position_to_idx3D_i(x, fluidMemory.min_coord, fluidMemory.cell_size.x);
						float3 min = fluidSystem.bounds[0] + math::castTo<float3>(voxelIdx) * cs;
						auto c = min + cs * 0.5f;
						auto prel = x - c;
						auto d = cs * 0.5f;
						auto ni = math::castTo<int3>(prel / d * (1.f + epsilon));
						auto nc = char3{ static_cast<char>(ni.x), static_cast<char>(ni.y), static_cast<char>(ni.z) };
						normal = math::castTo<float3>(nc);
						//auto nl = math::dot(n, SMRAY_DIR) < 0 ? n : n * -1;
						auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
						//normal = nl;
						color = f;
						emission = float3{ 0.f,0.f,0.f };
					}
					else 
					{
						Sphere &sphere = spheres[IDRAY2D.geomType];
							normal = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
							color = sphere.col;
							emission = sphere.emi;
					}
					normal = math::dot(normal, IDRAY2D.direction) < 0 ? normal : normal * -1;

					curandState randState;
					int32_t threadId =
						(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
					curand_init(seed + threadId, 0, 0, &randState);

					//float phi = 2 * CUDART_PI_F * randf();
					//float r2 = randf();
					float phi = 2 * CUDART_PI_F * curand_uniform(&randState);
					float r2 = curand_uniform(&randState);
					float r2s = sqrtf(r2);
					float3 w = math::normalize(normal);
					float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
					float3 v = math::cross(w, u);
					auto dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));

					cImage[IDRAY2D.index].color += cImage[IDRAY2D.index].mask * emission;
					cImage[IDRAY2D.index].mask *= color;
					IDRAY2D.origin = x + w * 0.01f;
					IDRAY2D.direction = dw;
				}
				if (IDRAY2D.bounces == 0) { cImage[IDRAY2D.index].color = float3{ d * 0.005f, d*0.005f, d * 0.005f }; }
				IDRAY2D.depth = FLT_MAX;
				IDRAY2D.geomType = 0;
				IDRAY2D.bounces++;
				cRaysDepth[i * cMsaaRate + ii] = IDRAY2D;
			}
		}
		__global__ void intersectRaysBVH(Ray* rays, RayIntersection* intersections) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			auto worldRay = rays[i];

			float scene_t = 1e20f;

			for (int32_t sphere_id = 0; sphere_id < int32_t(sizeof(spheres) / sizeof(Sphere)); sphere_id++) {
				Sphere &sphere = spheres[sphere_id];
				float d = spheres[sphere_id].intersect(worldRay);
				if (d && d < scene_t) {
					scene_t = d;
					auto x = worldRay.orig + worldRay.dir * scene_t;
					auto n = math::normalize(float3{ x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z });
					auto nl = math::dot(n, worldRay.dir) < 0 ? n : n * -1;
					intersections[i] = RayIntersection{ d, sphere.refl, sphere.col, nl, sphere.emi };
				}
			}

			auto fluidHit = bvh::rayIntersectFluidBVH(worldRay);
			if (fluidHit.status && fluidHit.depth < scene_t) {
				scene_t = fluidHit.depth;
				auto n = fluidHit.normal;
				//return n;
				auto nl = math::dot(n, worldRay.dir) < 0 ? n : n * -1;
				auto f = float3{ 0.05098f, 0.23137f, 0.494177f };
				auto emit = float3{ 0.f, 0.f, 0.f };
				intersections[i] = RayIntersection{ fluidHit.depth, DIFF, f, nl, emit };
			}

			if (scene_t > 1e19f)
				intersections[i] = RayIntersection{ 1e20f, DIFF, float3{0.f,0.f,0.f}, float3{1.f,0.f,0.f},  float3{0.f,0.f,0.f} };


		}
		__global__ void shadeRays(int32_t seed, Pixel* image, Ray* rays, RayIntersection* intersections) {
			int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= cScene.width)
				return;
			if (y >= cScene.height)
				return;
			int32_t i = (cScene.height - y - 1) * cScene.width + x;
			curandState randState;
			int32_t threadId =
				(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
			curand_init(seed + threadId, 0, 0, &randState);

			auto worldRay = rays[i];
			auto inter = intersections[i];
			auto pixel = image[i];
			if (inter.depth >= 1e18f)
				return;

			auto position = worldRay.orig + worldRay.dir * inter.depth;
			pixel.color += (pixel.mask * inter.emission);
			//pixel.color = inter.surfaceNormal;
			//pixel.color = worldRay.dir;

			float3 n = math::normalize(inter.surfaceNormal);
			float3 nl = math::dot(n, worldRay.dir) < 0 ? n : n * -1;
			
		  float phi = 2 * CUDART_PI_F * curand_uniform(&randState);
		  float r2 = curand_uniform(&randState);
		  float r2s = sqrtf(r2);
		  float3 w = math::normalize(nl);
		  float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{ 0, 1, 0 } : float3{ 1, 0, 0 }), w));
		  float3 v = math::cross(w, u);

		  float3 dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
		  worldRay.orig = position + w * 0.01f;

			pixel.mask *= inter.surfaceColor;

			//dw = rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace);
			//worldRay.orig = position + nl * bias;

		//	worldRay.orig = position + w * 1.5f;
			worldRay.dir = dw;
			if (dw.x != dw.x || dw.y != dw.y || dw.z != dw.z || (dw.x == 0.f && dw.y == 0.f && dw.z == 0.f))
				worldRay.dir = float3{ 1,0,0 };
			rays[i] = worldRay;
			image[i] = pixel;
		}
	}
}
#ifdef __INTELLISENSE__
#define CPSYMBOL(symbol, var)
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel
#else
#define CPSYMBOL(symbol, var) cudaMemcpyToSymbol(symbol, &var, sizeof(symbol))
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel<<<blocks,tpb,sm,stream>>>
#endif

void cudaMLMRender(SceneInformation scene, cudaGraphicsResource_t resource, FluidMemory fmem, FluidSystem fsys, float3 *acc, unsigned framenumber,
                   unsigned hashedframes, int32_t renderMode) {
	using namespace rtx;
	static std::random_device rd;
	static std::uniform_int_distribution<int32_t> dist(INT_MIN, INT_MAX);
  static bool once = true;
  static cudaStream_t stream;
  constexpr auto msaa = 1;
  constexpr int32_t blocks_1080 = 32 * 20;
  constexpr int32_t blockSize_1080 = 64;
  constexpr auto bounces = 5;
  int32_t width = static_cast<int32_t>(scene.width);
  int32_t height = static_cast<int32_t>(scene.height);
  int32_t numRays = width * height * msaa;
  //int32_t num_blocks = blocks_1080 * blockSize_1080;

  if (once) {
	  std::cout << "MLM renderer built at " << __TIMESTAMP__ << std::endl;
	  cudaStreamCreate(&stream);
	  cudaMalloc(&cuImage, sizeof(Pixel) * width * height);
	  cudaMalloc(&cuCurrentRays, sizeof(Ray) * width * height);
	  cudaMalloc(&cuRayIntersections, sizeof(RayIntersection) * width * height);
	  cudaMalloc(&cuScheduledRays, sizeof(RayWithIndex) * width * height * msaa);
	  cudaMalloc(&cuBlockedRays, sizeof(RayWithIndexAndDepth) * width * height * msaa);
	  cudaMalloc(&cuFluidIntersections, sizeof(FluidIntersection) * width * height * msaa);
	  cudaMalloc(&rayCounter, sizeof(int32_t));
	  cudaMalloc(&cRNGSeeds, sizeof(uint32_t) * blocks_1080 * blockSize_1080);
	  cudaMalloc(&cuRandStates, sizeof(curandState) * blocks_1080 * blockSize_1080);
	  //initRNG <<<blocks_1080, blockSize_1080>>> (cuRandStates, dist(rd));
	  LAUNCH(common::initRNGSeeds, blocks_1080, blockSize_1080, 0 ,0)(cRNGSeeds, dist(rd));
    cudaArray_t color_arr;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
    cudaBindSurfaceToArray(surfaceWriteOut, color_arr);
    once = false;
  }
  //scene.m_camera.apertureRadius = 0.f;

  CPSYMBOL(cScene, scene);
  CPSYMBOL(fluidSystem, fsys);
  CPSYMBOL(fluidMemory, fmem);
  CPSYMBOL(cNumRays, numRays);
  CPSYMBOL(cRays, cuScheduledRays);
  CPSYMBOL(cRaysDepth, cuBlockedRays);
  CPSYMBOL(cImage, cuImage);
  CPSYMBOL(cRandStates, cuRandStates);
  CPSYMBOL(cFluidIntersections, cuFluidIntersections);
  CPSYMBOL(cMsaaRate, msaa);
  CPSYMBOL(cRayCounter, rayCounter);
  CPSYMBOL(cuSeeds, cRNGSeeds);
  

  dim3 texturedim((uint32_t)scene.width, (uint32_t)scene.height, 1);
  dim3 blockdim(8, 8, 1);
  dim3 griddim(texturedim.x / blockdim.x, texturedim.y / blockdim.y, 1);
  if (texturedim.x % blockdim.x != 0)
    griddim.x += 1;
  if (texturedim.y % blockdim.y != 0)
    griddim.y += 1;
  //CoreLoopPathTracingKernel<<<griddim, blockdim>>>((float3 *)acc, framenumber, hashedframes);
   
  //if (renderMode == 0) {
	 // LAUNCH(common::generateScheduledRays, griddim, dim3(msaa, blockdim.x, blockdim.y), 0, stream)(hashedframes, cuImage, cuScheduledRays, cuCurrentRays, msaa);
	 // cudaMemcpy(rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice);
	 // render::rayScheduler << <blocks_1080, blockSize_1080, sizeof(ScheduledRay) * blockSize_1080, stream >> >();
	 // common::toneMap << <griddim, blockdim, 0, stream >> > (framenumber, (float3*)acc, cuImage, (float)msaa);
  //}
  //else if (renderMode == 4) {
	  //cuda::sync("Test 1");
	 // LAUNCH(common::generateBlockedRays, griddim, dim3(msaa, blockdim.x, blockdim.y), 0, stream)(hashedframes, cuImage, cuBlockedRays, cuCurrentRays, msaa);
	 //// cuda::sync("Test 2");
	 // for (int32_t i = 0; i < bounces; ++i) {
		//  //std::cout << "Bounce: " << i << std::endl;
		//  cudaMemcpyAsync(rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
		// // cuda::sync("Test 3");
		//  render::IDrayScheduler << <blocks_1080, blockSize_1080, sizeof(RayWithIndexAndDepth) * blockSize_1080, stream >> > ();
		//  //cuda::sync("Test 4");
		//  render::intersectRaysSpheresScene << <griddim, blockdim, blockdim.x * blockdim.y * sizeof(RayWithIndexAndDepth), stream >>> (dist(rd));
		//  //cuda::sync("Test 5");
		//  //render::updateIDRAY << <griddim, blockdim, blockdim.x * blockdim.y * sizeof(RayWithIndexAndDepth), stream >> > (dist(rd));
		// // cuda::sync("Test 6");
	 // }
	 // common::toneMap << <griddim, blockdim, 0, stream >> > (framenumber, (float3*)acc, cuImage, 1.f);
	 // //cuda::sync("Test 7");
 // }
  //else{
	  common::generatePrimaryRays << <griddim, blockdim, 0, stream >> > (hashedframes, cuImage, cuCurrentRays);
	  for (int32_t i = 0; i < bounces; ++i) {
		  std::cout << i;
		  std::cout.flush();
		  cuda::sync(std::to_string(__LINE__));
            //if (renderMode == 1) {
            //  render::intersectAndShadeRaysSM<<<griddim, blockdim, sizeof(RayIntersection) * blockdim.x * blockdim.y, stream>>>(
            //      cuCurrentRays, cuImage, hashedframes);
            //} else if (renderMode == 2) {
		  if (renderMode == 3) {
			  render::intersectRaysBVH<<<griddim, blockdim>>>(cuCurrentRays, cuRayIntersections);
              render::shadeRays<<<griddim, blockdim>>>(dist(rd), cuImage, cuCurrentRays, cuRayIntersections);
    }
  else {
		//	  std::cout << ".\n";
	  render::intersectRays << <griddim, blockdim >> > (cuCurrentRays, cuRayIntersections);
	  cuda::sync(std::to_string(__LINE__)); 
	  //std::cout << "-\n";
	  render::shadeRays << <griddim, blockdim >> > (dist(rd), cuImage, cuCurrentRays, cuRayIntersections);
	  cuda::sync(std::to_string(__LINE__));
	  //std::cout << ":\n";
  }
			  //break;
     //       } else 
		  
	  }
	  std::cout << std::endl;
	  common::toneMap << <griddim, blockdim, 0, stream >> > (framenumber, (float3*)acc, cuImage, 1.f);
	  cuda::sync(std::to_string(__LINE__));
  //}
  cudaStreamSynchronize(stream);
}