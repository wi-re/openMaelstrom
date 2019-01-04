#pragma once
#include <type_traits>
#include <utility>
#include <tuple>
#include <vector>
#include <array>
#include <cstring>
#include <string>
#include <array>
#include <utility/template/nonesuch.h>
#include <utility/unit_math.h>
#include <texture_types.h>


enum struct array_enum{adaptiveMergeable, adaptiveMergeCounter, adaptiveNumPtcls, adaptiveClassification, adaptiveSplitIndicator, adaptiveSplitIndicatorCompacted, adaptiveParentIndex, adaptiveParentVolume, adaptiveParentPosition, maxVelocity, cflValue, particleNormal, angularVelocity, boundaryLUT, boundaryPressureLUT, xbarLUT, ctrLUT, boundaryPlanes, volumeBoundaryVolumes, volumeBoundaryDimensions, volumeBoundaryMin, volumeBoundaryMax, decisionBuffer, surface_idxBuffer, markerBuffer, distanceBuffer, changeBuffer, position, acceleration, velocity, renderIntensity, volume, lifetime, pressure, density, particleIndex, particleIndexCompact, resortArray4, resortArray, dfsphDpDt, dfsphAlpha, dfsphRhoStar, dfsphKappa, dfsphKappaDivergence, iisphSum, iisphDii, iisphAiiOld, omega, alpha, gamma, sigma, iisphDensityAdvection, iisphDensityIteration, kernelBuffer, velocityAdvection, iisphSource, iisphOmega, iisphVolume, iisphAii, iisphAcceleration, iisphVolumeError, inletPositions, inletCounter, volumeOutletVolumes, volumeOutletDimensions, volumeOutletMin, volumeOutletMax, volumeOutletRate, volumeOutletRateAccumulator, compactCellList, compactCellScale, spanNeighborList, neighborList, neighborListLength, classification, resortIndex, ZOrder_64, ZOrder_32, cellSpanSwap, cellSpan, MLMResolution, cellparticleIndex, compactparticleIndex, hashMap, particleparticleIndex, cellBegin, cellEnd, support, supportEstimate, neighborCount, supportMarker, supportMarkerCompacted, closestNeighbor, closestNeighbor_f, neighborOverhead, neighborOverheadCount, neighborListSwap};

#include <utility/identifier/resource_helper.h>
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
constexpr T info() { return T(); }
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
typename T::type* get() { return T().ptr; }

namespace arrays{



struct adaptiveMergeable{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveMergeable;
	static constexpr const auto variableName = "adaptiveMergeable";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Indiciates the merge-ability or share-ability of a particle.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveMergeable;}
};

struct adaptiveMergeCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveMergeCounter;
	static constexpr const auto variableName = "adaptiveMergeCounter";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Indiciates the number of merging/sharing partners.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveMergeCounter;}
};

struct adaptiveNumPtcls{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveNumPtcls;
	static constexpr const auto variableName = "adaptiveNumPtcls";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used as an atomic counter to increment the number of particles.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveNumPtcls;}
};

struct adaptiveClassification{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::adaptiveClassification;
	static constexpr const auto variableName = "adaptiveClassification";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the classification of every particle according to a decision function.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveClassification;}
};

struct adaptiveSplitIndicator{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveSplitIndicator;
	static constexpr const auto variableName = "adaptiveSplitIndicator";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains an indicator that is incremented over time or reset depending on the classification of a particle as L over time.";

	static void swap();
	static int32_t* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveSplitIndicator;}
};

struct adaptiveSplitIndicatorCompacted{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveSplitIndicatorCompacted;
	static constexpr const auto variableName = "adaptiveSplitIndicatorCompacted";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains an indicator that is incremented over time or reset depending on the classification of a particle as L over time.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveSplitIndicatorCompacted;}
};

struct adaptiveParentIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptiveParentIndex;
	static constexpr const auto variableName = "adaptiveParentIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains a unique identifier of the parent particle of a particle that was split particle.";

	static void swap();
	static int32_t* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveParentIndex;}
};

struct adaptiveParentVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::adaptiveParentVolume;
	static constexpr const auto variableName = "adaptiveParentVolume";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the volume of the parent particle of a particle that was split.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveParentVolume;}
};

struct adaptiveParentPosition{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::adaptiveParentPosition;
	static constexpr const auto variableName = "adaptiveParentPosition";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains an estimated propagated position Contains the volume of the parent particle of a particle that was split.";

	static void swap();
	static float4* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptiveParentPosition;}
};

struct maxVelocity{
	using type = float;
	using unit_type = value_unit<float, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::maxVelocity;
	static constexpr const auto variableName = "maxVelocity";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the length of the velocity of each particle, used in a reduce operation to determine the ideal timestep.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.maxVelocity;}
};

struct cflValue{
	using type = float;
	using unit_type = value_unit<float, SI::s>;
	static constexpr const array_enum identifier = array_enum::cflValue;
	static constexpr const auto variableName = "cflValue";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the length of the velocity of each particle, used in a reduce operation to determine the ideal timestep.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cflValue;}
};

struct particleNormal{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::particleNormal;
	static constexpr const auto variableName = "particleNormal";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Normal of the particle calculated to determine the surface tension of a particle.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.particleNormal;}
};

struct angularVelocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::angularVelocity;
	static constexpr const auto variableName = "angularVelocity";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the angular velocity of a particle, used to improve the vorticity of the simulaiton.";

	static void swap();
	static float4* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.angularVelocity;}
};

struct boundaryLUT{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryLUT;
	static constexpr const auto variableName = "boundaryLUT";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.boundaryLUT;}
};

struct boundaryPressureLUT{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::m>>>;
	static constexpr const array_enum identifier = array_enum::boundaryPressureLUT;
	static constexpr const auto variableName = "boundaryPressureLUT";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.boundaryPressureLUT;}
};

struct xbarLUT{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip_2<SI::m>>>;
	static constexpr const array_enum identifier = array_enum::xbarLUT;
	static constexpr const auto variableName = "xbarLUT";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.xbarLUT;}
};

struct ctrLUT{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::ctrLUT;
	static constexpr const auto variableName = "ctrLUT";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.ctrLUT;}
};

struct boundaryPlanes{
	using type = float4;
	using unit_type = value_unit<float4, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryPlanes;
	static constexpr const auto variableName = "boundaryPlanes";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains boundary planes of the simulation as implicit plane equations entry = (n.x,n.y,n.z,d).";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.boundaryPlanes;}
};

struct volumeBoundaryVolumes{
	using type = cudaTextureObject_t;
	using unit_type = cudaTextureObject_t;
	static constexpr const array_enum identifier = array_enum::volumeBoundaryVolumes;
	static constexpr const auto variableName = "volumeBoundaryVolumes";
	static cudaTextureObject_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains references to boundary volumes created from vdb volumes for more complex boundary objects than planes.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryVolumes;}
};

struct volumeBoundaryDimensions{
	using type = int4;
	using unit_type = int4;
	static constexpr const array_enum identifier = array_enum::volumeBoundaryDimensions;
	static constexpr const auto variableName = "volumeBoundaryDimensions";
	static int4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the dimension in voxels for each boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryDimensions;}
};

struct volumeBoundaryMin{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeBoundaryMin;
	static constexpr const auto variableName = "volumeBoundaryMin";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Low part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryMin;}
};

struct volumeBoundaryMax{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeBoundaryMax;
	static constexpr const auto variableName = "volumeBoundaryMax";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "High part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryMax;}
};

struct decisionBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::decisionBuffer;
	static constexpr const auto variableName = "decisionBuffer";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to indicate if a particle is at the surface or not and if the state of the particle has changed in the last surface distance iteration.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.decisionBuffer;}
};

struct surface_idxBuffer{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::surface_idxBuffer;
	static constexpr const auto variableName = "surface_idxBuffer";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for compaction of indices.";

	static void swap();
	static int32_t* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.surface_idxBuffer;}
};

struct markerBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::markerBuffer;
	static constexpr const auto variableName = "markerBuffer";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Similar to decisionBuffer, these two values are used in an A->B, B->A fashion.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.markerBuffer;}
};

struct distanceBuffer{
	using type = float;
	using unit_type = value_unit<float, SI::m>;
	static constexpr const array_enum identifier = array_enum::distanceBuffer;
	static constexpr const auto variableName = "distanceBuffer";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the signed distance of a particle to the closest fluid surface (distances are negative as fluid particles are inside the fluid volume).";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.distanceBuffer;}
};

struct changeBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::changeBuffer;
	static constexpr const auto variableName = "changeBuffer";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to count particles that have changed in the last iteration.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.changeBuffer;}
};

struct position{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::position;
	static constexpr const auto variableName = "position";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the position of every particle in world space coordinates.";

	static void swap();
	static float4* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.position;}
};

struct acceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::acceleration>;
	static constexpr const array_enum identifier = array_enum::acceleration;
	static constexpr const auto variableName = "acceleration";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the acceleration of every particle. This value needs to be set to zero on every time step as the old value is resorted as the initial value for certain functions but this is not desirable in calculalting the new accelerations later in the timestep.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.acceleration;}
};

struct velocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::velocity;
	static constexpr const auto variableName = "velocity";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the velocity of every particle. The velocity of the previous timestep will be sorted as the initial value in the new timestep automatically.";

	static void swap();
	static float4* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.velocity;}
};

struct renderIntensity{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::renderIntensity;
	static constexpr const auto variableName = "renderIntensity";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the visualization information for every particle generated by the visualization module. This is a value ranging from 0 to 1 for easy visualization in OpenGL.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.renderIntensity;}
};

struct volume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::volume;
	static constexpr const auto variableName = "volume";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the volume a particle represents based on the particle radius not the particle support radius.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volume;}
};

struct lifetime{
	using type = float;
	using unit_type = value_unit<float, SI::s>;
	static constexpr const array_enum identifier = array_enum::lifetime;
	static constexpr const auto variableName = "lifetime";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the lifetime of a particle since it's creation, starts at 0 for emitted particles but particles created from adaptivity might start at values less than 0 for blending.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.lifetime;}
};

struct pressure{
	using type = float;
	using unit_type = value_unit<float, SI::Pa>;
	static constexpr const array_enum identifier = array_enum::pressure;
	static constexpr const auto variableName = "pressure";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the pressure of a particle calculated by IISPH, DFSPH does not calculate the pressure directly.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.pressure;}
};

struct density{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::density;
	static constexpr const auto variableName = "density";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the density of a particle, calculated in the density module. This value is not filtered by any shepard or related filters.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.density;}
};

struct particleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::particleIndex;
	static constexpr const auto variableName = "particleIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for selective functions";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.particleIndex;}
};

struct particleIndexCompact{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::particleIndexCompact;
	static constexpr const auto variableName = "particleIndexCompact";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for selective functions";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.particleIndexCompact;}
};

struct resortArray4{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::resortArray4;
	static constexpr const auto variableName = "resortArray4";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for compact resorting";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.resortArray4;}
};

struct resortArray{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::resortArray;
	static constexpr const auto variableName = "resortArray";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for compact resorting";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.resortArray;}
};

struct dfsphDpDt{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::density,SI::Hz>>;
	static constexpr const array_enum identifier = array_enum::dfsphDpDt;
	static constexpr const auto variableName = "dfsphDpDt";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Density change with respect to time estimated for DFSPH solver.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.dfsphDpDt;}
};

struct dfsphAlpha{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit < SI::area, SI::recip<SI::density>>>;
	static constexpr const array_enum identifier = array_enum::dfsphAlpha;
	static constexpr const auto variableName = "dfsphAlpha";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Common factor used in the solve step of both density and divergence of DFSPH.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.dfsphAlpha;}
};

struct dfsphRhoStar{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::dfsphRhoStar;
	static constexpr const auto variableName = "dfsphRhoStar";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Estimated density including advection forces at the new timestep.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.dfsphRhoStar;}
};

struct dfsphKappa{
	using type = float;
	using unit_type = value_unit<float,  SI::derived_unit < SI::area, SI::recip_2<SI::s>>>;
	static constexpr const array_enum identifier = array_enum::dfsphKappa;
	static constexpr const auto variableName = "dfsphKappa";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stiffness parameter of the incompressibility solver, contains the old value in the new timestep for a warm start.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.dfsphKappa;}
};

struct dfsphKappaDivergence{
	using type = float;
	using unit_type = value_unit<float,  SI::derived_unit < SI::area, SI::recip_2<SI::s>>>;
	static constexpr const array_enum identifier = array_enum::dfsphKappaDivergence;
	static constexpr const auto variableName = "dfsphKappaDivergence";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stiffness parameter of the divergence free solver, contains the old value in the new timestep for a warm start.";

	static void swap();
	static float* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.dfsphKappaDivergence;}
};

struct iisphSum{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::iisphSum;
	static constexpr const auto variableName = "iisphSum";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the sum element of the pressure forces (sum_j d_ij p_j).";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphSum;}
};

struct iisphDii{
	using type = float4;
	using unit_type = value_unit<float4, SI::derived_unit<SI::m, SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisphDii;
	static constexpr const auto variableName = "iisphDii";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the diagional elements of the pressure force d_ii";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphDii;}
};

struct iisphAiiOld{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit< SI::density, SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisphAiiOld;
	static constexpr const auto variableName = "iisphAiiOld";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the diagonal elements of the incompressibility matrix.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphAiiOld;}
};

struct omega{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::omega;
	static constexpr const auto variableName = "omega";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Corrective term used to adjust the solver for changing support radii.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.omega;}
};

struct alpha{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::alpha;
	static constexpr const auto variableName = "alpha";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Temporary value used to improve the performance.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.alpha;}
};

struct gamma{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::gamma;
	static constexpr const auto variableName = "gamma";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Temporary value used to improve the performance.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.gamma;}
};

struct sigma{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::sigma;
	static constexpr const auto variableName = "sigma";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Temporary value used to improve the performance.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.sigma;}
};

struct iisphDensityAdvection{
	using type = float;
	using unit_type = value_unit<float, SI::density>;
	static constexpr const array_enum identifier = array_enum::iisphDensityAdvection;
	static constexpr const auto variableName = "iisphDensityAdvection";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Estimated density of a particle after it has been influenced by advection forces.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphDensityAdvection;}
};

struct iisphDensityIteration{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisphDensityIteration;
	static constexpr const auto variableName = "iisphDensityIteration";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Density of a particle in the current iteration of the incompressibility solver.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphDensityIteration;}
};

struct kernelBuffer{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::kernelBuffer;
	static constexpr const auto variableName = "kernelBuffer";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Temporary value used to improve the performance.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.kernelBuffer;}
};

struct velocityAdvection{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::velocityAdvection;
	static constexpr const auto variableName = "velocityAdvection";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Velocity of a particle including influence of advection forces.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.velocityAdvection;}
};

struct iisphSource{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisphSource;
	static constexpr const auto variableName = "iisphSource";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Source term used for solving IISPH.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphSource;}
};

struct iisphOmega{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisphOmega;
	static constexpr const auto variableName = "iisphOmega";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Corrective term used to adjust the solver for changing support radii.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphOmega;}
};

struct iisphVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::iisphVolume;
	static constexpr const auto variableName = "iisphVolume";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the actual volume of a sample V_i = (V_0 / rho_i).";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphVolume;}
};

struct iisphAii{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisphAii;
	static constexpr const auto variableName = "iisphAii";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Diagonal elements of the matrix used to solve for incompressibility.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphAii;}
};

struct iisphAcceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::acceleration>;
	static constexpr const array_enum identifier = array_enum::iisphAcceleration;
	static constexpr const auto variableName = "iisphAcceleration";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Diagonal elements of the matrix used to solve for incompressibility.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphAcceleration;}
};

struct iisphVolumeError{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisphVolumeError;
	static constexpr const auto variableName = "iisphVolumeError";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Error of the incompressibility solver used as a threshold.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.iisphVolumeError;}
};

struct inletPositions{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::inletPositions;
	static constexpr const auto variableName = "inletPositions";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the values of all positions of all emitters that particles count be emitted from.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.inletPositions;}
};

struct inletCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::inletCounter;
	static constexpr const auto variableName = "inletCounter";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Atomic counter used to keep track of the particles that were created.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.inletCounter;}
};

struct volumeOutletVolumes{
	using type = cudaTextureObject_t;
	using unit_type = cudaTextureObject_t;
	static constexpr const array_enum identifier = array_enum::volumeOutletVolumes;
	static constexpr const auto variableName = "volumeOutletVolumes";
	static cudaTextureObject_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains references to boundary volumes created from vdb volumes for more complex boundary objects than planes.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletVolumes;}
};

struct volumeOutletDimensions{
	using type = int4;
	using unit_type = int4;
	static constexpr const array_enum identifier = array_enum::volumeOutletDimensions;
	static constexpr const auto variableName = "volumeOutletDimensions";
	static int4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the dimension in voxels for each boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletDimensions;}
};

struct volumeOutletMin{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeOutletMin;
	static constexpr const auto variableName = "volumeOutletMin";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Low part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletMin;}
};

struct volumeOutletMax{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeOutletMax;
	static constexpr const auto variableName = "volumeOutletMax";
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "High part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletMax;}
};

struct volumeOutletRate{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::volumeOutletRate;
	static constexpr const auto variableName = "volumeOutletRate";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Low part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletRate;}
};

struct volumeOutletRateAccumulator{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::volumeOutletRateAccumulator;
	static constexpr const auto variableName = "volumeOutletRateAccumulator";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Low part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeOutletRateAccumulator;}
};

struct compactCellList{
	using type = compactSpan;
	using unit_type = compactSpan;
	static constexpr const array_enum identifier = array_enum::compactCellList;
	static constexpr const auto variableName = "compactCellList";
	static compactSpan* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the spans of candidate neighbors of all particles. At most this could contain 27 entries and the end marker. The spans work similar to hash_span and cell_span.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.compactCellList;}
};

struct compactCellScale{
	using type = int8_t;
	using unit_type = int8_t;
	static constexpr const array_enum identifier = array_enum::compactCellScale;
	static constexpr const auto variableName = "compactCellScale";
	static int8_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the spans of candidate neighbors of all particles. At most this could contain 27 entries and the end marker. The spans work similar to hash_span and cell_span.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.compactCellScale;}
};

struct spanNeighborList{
	using type = neigh_span;
	using unit_type = neigh_span;
	static constexpr const array_enum identifier = array_enum::spanNeighborList;
	static constexpr const auto variableName = "spanNeighborList";
	static neigh_span* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the spans of candidate neighbors of all particles. At most this could contain 27 entries and the end marker. The spans work similar to hash_span and cell_span.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.spanNeighborList;}
};

struct neighborList{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborList;
	static constexpr const auto variableName = "neighborList";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the real neighbors of all particles as their indices plus an end marker entry.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborList;}
};

struct neighborListLength{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborListLength;
	static constexpr const auto variableName = "neighborListLength";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the number of neighbors of a particle, could be implicitly replaced by using the end marker but this option provides faster execution due to using for instead of while loops.";

	static void swap();
	static int32_t* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborListLength;}
};

struct classification{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::classification;
	static constexpr const auto variableName = "classification";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to store keys for resorting algorithms, e.g. a Z-Order Index in resort/resort. Required by all sorting methods in some way but used differently by them.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.classification;}
};

struct resortIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::resortIndex;
	static constexpr const auto variableName = "resortIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to store keys for resorting algorithms, e.g. a Z-Order Index in resort/resort. Required by all sorting methods in some way but used differently by them.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.resortIndex;}
};

struct ZOrder_64{
	using type = int64_t;
	using unit_type = int64_t;
	static constexpr const array_enum identifier = array_enum::ZOrder_64;
	static constexpr const auto variableName = "ZOrder_64";
	static int64_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the Morton Code of particles for hashed based restorting algorithms, this version allows for 64 bit Z-Ordering (21 bit per dimension resolution)";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.ZOrder_64;}
};

struct ZOrder_32{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::ZOrder_32;
	static constexpr const auto variableName = "ZOrder_32";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the Morton Code of particles for hashed based restorting algorithms, this version allows for 32 bit Z-Ordering (10 bit per dimension resolution)";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.ZOrder_32;}
};

struct cellSpanSwap{
	using type = cell_span;
	using unit_type = cell_span;
	static constexpr const array_enum identifier = array_enum::cellSpanSwap;
	static constexpr const auto variableName = "cellSpanSwap";
	static cell_span* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores information about used cells with one entry per particle. Cell Spans consist of a .beginning and .length entry that can be used to iterate over the particles within a cell. For MLM cellSpan is used as a temporary array only.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cellSpanSwap;}
};

struct cellSpan{
	using type = cell_span;
	using unit_type = cell_span;
	static constexpr const array_enum identifier = array_enum::cellSpan;
	static constexpr const auto variableName = "cellSpan";
	static cell_span* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Same functionality as cellSpan but is allocated to a multiple of the particle count based on the mlm scheme count.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cellSpan;}
};

struct MLMResolution{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::MLMResolution;
	static constexpr const auto variableName = "MLMResolution";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the resolution level of a particle when using MLM resorting.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.MLMResolution;}
};

struct cellparticleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::cellparticleIndex;
	static constexpr const auto variableName = "cellparticleIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the cell index of all particles, including duplicates. Size = number of particles + 1.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cellparticleIndex;}
};

struct compactparticleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::compactparticleIndex;
	static constexpr const auto variableName = "compactparticleIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Compacted version of cellparticleIndex only containing unique entries. Size = number of particles + 1.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.compactparticleIndex;}
};

struct hashMap{
	using type = hash_span;
	using unit_type = hash_span;
	static constexpr const array_enum identifier = array_enum::hashMap;
	static constexpr const auto variableName = "hashMap";
	static hash_span* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Similar to cellSpan, however: hashMap is not used by MLM sorting at all.";

	static constexpr const memory_kind kind = memory_kind::singleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.hashMap;}
};

struct particleparticleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::particleparticleIndex;
	static constexpr const auto variableName = "particleparticleIndex";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used for creating indices from 0:n for resorting with key-value sorting algorithms.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.particleparticleIndex;}
};

struct cellBegin{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::cellBegin;
	static constexpr const auto variableName = "cellBegin";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the first particle that is contained within a cell. This is as large as the domain as no compacting is done here. If the entry is -1 then there are no particles in this cell.";

	static constexpr const memory_kind kind = memory_kind::cellData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cellBegin;}
};

struct cellEnd{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::cellEnd;
	static constexpr const auto variableName = "cellEnd";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Similalr to cellBegin. Contains the last particle that is contained within a cell.";

	static constexpr const memory_kind kind = memory_kind::cellData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cellEnd;}
};

struct support{
	using type = float;
	using unit_type = value_unit<float, SI::m>;
	static constexpr const array_enum identifier = array_enum::support;
	static constexpr const auto variableName = "support";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the final support radius of each particle. Used as a temporary value.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.support;}
};

struct supportEstimate{
	using type = float;
	using unit_type = value_unit<float, SI::m>;
	static constexpr const array_enum identifier = array_enum::supportEstimate;
	static constexpr const auto variableName = "supportEstimate";
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the estimated support radius of each particle. Used as a temporary value.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.supportEstimate;}
};

struct neighborCount{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborCount;
	static constexpr const auto variableName = "neighborCount";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the number of neighbors a particle has even if those cannot be contained in the contrained neighborlist.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborCount;}
};

struct supportMarker{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::supportMarker;
	static constexpr const auto variableName = "supportMarker";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to mark particles as having too many neighbors to fit into the neighborlist.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.supportMarker;}
};

struct supportMarkerCompacted{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::supportMarkerCompacted;
	static constexpr const auto variableName = "supportMarkerCompacted";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Compacted version of the marker list for particles with too many neighbors.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.supportMarkerCompacted;}
};

struct closestNeighbor{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::closestNeighbor;
	static constexpr const auto variableName = "closestNeighbor";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Misused to increment counters for the overhead list.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.closestNeighbor;}
};

struct closestNeighbor_f{
	using type = int32_t;
	using unit_type = value_unit<int32_t, SI::m>;
	static constexpr const array_enum identifier = array_enum::closestNeighbor_f;
	static constexpr const auto variableName = "closestNeighbor_f";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the distance of a particle to the nearest neighbor.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.closestNeighbor_f;}
};

struct neighborOverhead{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborOverhead;
	static constexpr const auto variableName = "neighborOverhead";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Temporary buffer for neighbors that might be able to get merged into the normal neighbor list later.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborOverhead;}
};

struct neighborOverheadCount{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborOverheadCount;
	static constexpr const auto variableName = "neighborOverheadCount";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Number of temporary entries per particle";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborOverheadCount;}
};

struct neighborListSwap{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::neighborListSwap;
	static constexpr const auto variableName = "neighborListSwap";
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used as an internal switch in the constrained neighborlist method.";

	static constexpr const memory_kind kind = memory_kind::singleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.neighborListSwap;}
};




}
extern std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::compactCellList, arrays::compactCellScale, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::hashMap, arrays::particleparticleIndex, arrays::cellBegin, arrays::cellEnd, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount, arrays::neighborListSwap> allocations_list;
extern std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::boundaryLUT, arrays::boundaryPressureLUT, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryPlanes, arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::inletPositions, arrays::inletCounter, arrays::volumeOutletVolumes, arrays::volumeOutletDimensions, arrays::volumeOutletMin, arrays::volumeOutletMax, arrays::volumeOutletRate, arrays::volumeOutletRateAccumulator, arrays::compactCellList, arrays::compactCellScale, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::hashMap, arrays::particleparticleIndex, arrays::cellBegin, arrays::cellEnd, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount, arrays::neighborListSwap> arrays_list;
extern std::tuple<arrays::adaptiveSplitIndicator, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::angularVelocity, arrays::distanceBuffer, arrays::position, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::neighborListLength> sorting_list;
extern std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::compactCellList, arrays::compactCellScale, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::particleparticleIndex, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount> swapping_list;
template<typename T, bool b>
struct arrays_add_const;

template <typename T >
struct arrays_add_const<T, false> {
	using type = T;
};

template < typename T >
struct arrays_add_const<T, true> {
	using type = const T;
};

template< typename T>
using const_array = typename arrays_add_const<typename T::type, true>::type* __restrict;
template< typename T>
using write_array = typename arrays_add_const<typename T::type, false>::type* __restrict;
template< typename T, bool b = false>
using swap_array = std::pair<typename arrays_add_const<typename T::type, b>::type* __restrict__, typename arrays_add_const<typename T::type, b>::type* __restrict__>;

template< typename T>
using const_array_u = typename arrays_add_const<typename T::unit_type, true>::type* __restrict;
template< typename T>
using write_array_u = typename arrays_add_const<typename T::unit_type, false>::type* __restrict;
template< typename T, bool b = false>
using swap_array_u = std::pair<typename arrays_add_const<typename T::unit_type, b>::type* __restrict__, typename arrays_add_const<typename T::unit_type, b>::type* __restrict__>;

