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


enum struct array_enum{adaptive_mergeable, adaptive_mergeCounter, adaptive_ptclCounter, adaptive_classification, adaptive_splitIndicator, adaptive_adaptivityCounter, adaptive_splitIndicatorCompacted, adaptive_parentIndex, adaptive_parentVolume, adaptive_parentPosition, advectionArrays_maxVelocity, advectionArrays_cflValue, advectionArrays_particleNormal, advectionArrays_angularVelocity, rigidBodyArrays_rigidDensities, rigidBodyArrays_rigidVolumes, rigidBodyArrays_rigidLinearVelocities, rigidBodyArrays_rigidAVelocities, rigidBodyArrays_rigidOrigins, rigidBodyArrays_rigidQuaternions, boundaryArrays_splineLUT, boundaryArrays_offsetLUT, boundaryArrays_spikyLUT, boundaryArrays_adhesionLUT, boundaryArrays_volumeLUT, boundaryArrays_spline2LUT, boundaryArrays_splineGradientLUT, boundaryArrays_spikyGradientLUT, boundaryArrays_cohesionLUT, boundaryArrays_boundaryPlanes, boundaryArrays_boundaryPlaneVelocity, boundaryArrays_volumeBoundaryVolumes, boundaryArrays_volumeBoundaryDimensions, boundaryArrays_volumeBoundaryMin, boundaryArrays_volumeBoundaryMax, boundaryArrays_volumeBoundaryDensity, boundaryArrays_volumeBoundaryVolume, boundaryArrays_volumeBoundaryVelocity, boundaryArrays_volumeBoundaryAngularVelocity, boundaryArrays_volumeBoundaryKind, boundaryArrays_volumeBoundaryPosition, boundaryArrays_volumeBoundaryQuaternion, boundaryArrays_volumeBoundaryTransformMatrix, boundaryArrays_volumeBoundaryTransformMatrixInverse, boundaryArrays_volumeBoundaryInertiaMatrix, boundaryArrays_volumeBoundaryInertiaMatrixInverse, boundaryArrays_volumeBoundaryAcceleration, boundaryArrays_volumeBoundaryAngularAcceleration, surfaceArrays_decisionBuffer, surfaceArrays_surface_idxBuffer, surfaceArrays_markerBuffer, surfaceArrays_distanceBuffer, surfaceArrays_changeBuffer, basicArrays_matrixTest, basicArrays_position, basicArrays_acceleration, basicArrays_velocity, basicArrays_particle_type, basicArrays_renderArray, basicArrays_debugArray, basicArrays_volume, basicArrays_lifetime, basicArrays_pressure, basicArrays_density, basicArrays_fluidDensity, basicArrays_particleIndex, basicArrays_particleIndexCompact, basicArrays_resortArray4, basicArrays_resortArray, dfsphArrays_dfsphSource, dfsphArrays_dfsphAlpha, dfsphArrays_dfsphDpDt, dfsphArrays_dfsphRhoStar, iisphArrays_iisphSum, iisphArrays_iisphDii, iisphArrays_iisphAiiOld, iisphArrays_omega, iisphArrays_alpha, iisphArrays_gamma, iisphArrays_sigma, iisphArrays_iisphDensityAdvection, iisphArrays_iisphDensityIteration, iisphArrays_kernelBuffer, iisphArrays_velocityAdvection, iisph17_sourceTerm, iisph17_boundaryPressure, iisph17_omega, iisph17_apparentVolume, iisph17_Aii, iisph17_predictedAcceleration, iisph17_volumeError, iisph17_band_rb__sourceTerm, iisph17_band_rb__omega, iisph17_band_rb__apparentVolume, iisph17_band_rb__Aii, iisph17_band_rb__predictedAcceleration, iisph17_band_rb__volumeError, volumeInletArrays_inletPositions, volumeInletArrays_inletCounter, volumeInletArrays_volumeOutletVolumes, volumeInletArrays_volumeOutletDimensions, volumeInletArrays_volumeOutletMin, volumeInletArrays_volumeOutletMax, volumeInletArrays_volumeOutletRate, volumeInletArrays_volumeOutletRateAccumulator, structureArrays_compactCellList, structureArrays_neighborMask, structureArrays_compactCellScale, structureArrays_spanNeighborList, structureArrays_neighborList, structureArrays_neighborListLength, renderArrays_anisotropicMatrices, renderArrays_centerPosition, renderArrays_auxTest, renderArrays_auxIsoDensity, renderArrays_auxDistance, renderArrays_auxHashMap, renderArrays_auxCellSpan, renderArrays_auxCellInformation, renderArrays_auxCellSurface, structureArrays_classification, structureArrays_resortIndex, structureArrays_ZOrder_64, structureArrays_ZOrder_32, structureArrays_cellSpanSwap, structureArrays_cellSpan, structureArrays_compactCellSpanSwap, structureArrays_compactCellSpan, structureArrays_MLMResolution, structureArrays_cellparticleIndex, structureArrays_compactparticleIndex, structureArrays_hashMap, structureArrays_compactHashMap, structureArrays_mlmScaling, structureArrays_particleparticleIndex, structureArrays_cellBegin, structureArrays_cellEnd, supportArrays_support, supportArrays_supportEstimate, supportArrays_neighborCount, supportArrays_supportMarker, supportArrays_supportMarkerCompacted, supportArrays_closestNeighbor, supportArrays_closestNeighbor_f, supportArrays_neighborOverhead, supportArrays_neighborOverheadCount, supportArrays_neighborListSwap};

#include <utility/identifier/resource_helper.h>
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
constexpr T info() { return T(); }
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
typename T::type* get() { return T().ptr; }

namespace arrays{
namespace adaptive{
struct mergeable{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_mergeable;
	static constexpr const auto variableName = "mergeable";
	static constexpr const auto qualifiedName = "adaptive.mergeable";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.mergeable;}
};
struct mergeCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_mergeCounter;
	static constexpr const auto variableName = "mergeCounter";
	static constexpr const auto qualifiedName = "adaptive.mergeCounter";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.mergeCounter;}
};
struct ptclCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_ptclCounter;
	static constexpr const auto variableName = "ptclCounter";
	static constexpr const auto qualifiedName = "adaptive.ptclCounter";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.ptclCounter;}
};
struct classification{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::adaptive_classification;
	static constexpr const auto variableName = "classification";
	static constexpr const auto qualifiedName = "adaptive.classification";
	static constexpr const auto ambiguous = true;
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
struct splitIndicator{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_splitIndicator;
	static constexpr const auto variableName = "splitIndicator";
	static constexpr const auto qualifiedName = "adaptive.splitIndicator";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.splitIndicator;}
};
struct adaptivityCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_adaptivityCounter;
	static constexpr const auto variableName = "adaptivityCounter";
	static constexpr const auto qualifiedName = "adaptive.adaptivityCounter";
	static constexpr const auto ambiguous = false;
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains an indicator that is incremented over time or reset depending on the classification of a particle as L over time.";

	static constexpr const memory_kind kind = memory_kind::singleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adaptivityCounter;}
};
struct splitIndicatorCompacted{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_splitIndicatorCompacted;
	static constexpr const auto variableName = "splitIndicatorCompacted";
	static constexpr const auto qualifiedName = "adaptive.splitIndicatorCompacted";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.splitIndicatorCompacted;}
};
struct parentIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::adaptive_parentIndex;
	static constexpr const auto variableName = "parentIndex";
	static constexpr const auto qualifiedName = "adaptive.parentIndex";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.parentIndex;}
};
struct parentVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::adaptive_parentVolume;
	static constexpr const auto variableName = "parentVolume";
	static constexpr const auto qualifiedName = "adaptive.parentVolume";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.parentVolume;}
};
struct parentPosition{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::adaptive_parentPosition;
	static constexpr const auto variableName = "parentPosition";
	static constexpr const auto qualifiedName = "adaptive.parentPosition";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.parentPosition;}
};
}
namespace advectionArrays{
struct maxVelocity{
	using type = float;
	using unit_type = value_unit<float, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::advectionArrays_maxVelocity;
	static constexpr const auto variableName = "maxVelocity";
	static constexpr const auto qualifiedName = "advectionArrays.maxVelocity";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::advectionArrays_cflValue;
	static constexpr const auto variableName = "cflValue";
	static constexpr const auto qualifiedName = "advectionArrays.cflValue";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::advectionArrays_particleNormal;
	static constexpr const auto variableName = "particleNormal";
	static constexpr const auto qualifiedName = "advectionArrays.particleNormal";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::advectionArrays_angularVelocity;
	static constexpr const auto variableName = "angularVelocity";
	static constexpr const auto qualifiedName = "advectionArrays.angularVelocity";
	static constexpr const auto ambiguous = false;
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
}
namespace basicArrays{
struct fluidDensity{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::basicArrays_fluidDensity;
	static constexpr const auto variableName = "fluidDensity";
	static constexpr const auto qualifiedName = "basicArrays.fluidDensity";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the density of a particle, calculated in the density module. This value is not filtered by any shepard or related filters.";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.fluidDensity;}
};
struct particleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::basicArrays_particleIndex;
	static constexpr const auto variableName = "particleIndex";
	static constexpr const auto qualifiedName = "basicArrays.particleIndex";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_particleIndexCompact;
	static constexpr const auto variableName = "particleIndexCompact";
	static constexpr const auto qualifiedName = "basicArrays.particleIndexCompact";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_resortArray4;
	static constexpr const auto variableName = "resortArray4";
	static constexpr const auto qualifiedName = "basicArrays.resortArray4";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_resortArray;
	static constexpr const auto variableName = "resortArray";
	static constexpr const auto qualifiedName = "basicArrays.resortArray";
	static constexpr const auto ambiguous = false;
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
struct volume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::basicArrays_volume;
	static constexpr const auto variableName = "volume";
	static constexpr const auto qualifiedName = "basicArrays.volume";
	static constexpr const auto ambiguous = false;
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
struct matrixTest{
	using type = Matrix4x4;
	using unit_type = Matrix4x4;
	static constexpr const array_enum identifier = array_enum::basicArrays_matrixTest;
	static constexpr const auto variableName = "matrixTest";
	static constexpr const auto qualifiedName = "basicArrays.matrixTest";
	static constexpr const auto ambiguous = false;
	static Matrix4x4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Used to test arrays of matrices.";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.matrixTest;}
};
struct position{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::basicArrays_position;
	static constexpr const auto variableName = "position";
	static constexpr const auto qualifiedName = "basicArrays.position";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_acceleration;
	static constexpr const auto variableName = "acceleration";
	static constexpr const auto qualifiedName = "basicArrays.acceleration";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the acceleration of every particle. This value needs to be set to zero on every time step as the old value is resorted as the initial value for certain functions but this is not desirable in calculalting the new accelerations later in the timestep.";

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
	template<class T> static inline auto& get_member(T& var) { return var.acceleration;}
};
struct velocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::basicArrays_velocity;
	static constexpr const auto variableName = "velocity";
	static constexpr const auto qualifiedName = "basicArrays.velocity";
	static constexpr const auto ambiguous = false;
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
struct particle_type{
	using type = int;
	using unit_type = int;
	static constexpr const array_enum identifier = array_enum::basicArrays_particle_type;
	static constexpr const auto variableName = "particle_type";
	static constexpr const auto qualifiedName = "basicArrays.particle_type";
	static constexpr const auto ambiguous = false;
	static int* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the type of every particle. The type of the previous timestep will be sorted as the initial value in the new timestep automatically.";

	static void swap();
	static int* rear_ptr;
	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.particle_type;}
};
struct renderArray{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::basicArrays_renderArray;
	static constexpr const auto variableName = "renderArray";
	static constexpr const auto qualifiedName = "basicArrays.renderArray";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the visualization information for every particle generated by the visualization module. This is a value ranging from 0 to 1 for easy visualization in OpenGL.";

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
	template<class T> static inline auto& get_member(T& var) { return var.renderArray;}
};
struct debugArray{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::basicArrays_debugArray;
	static constexpr const auto variableName = "debugArray";
	static constexpr const auto qualifiedName = "basicArrays.debugArray";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Stores the visualization information for every particle generated by the visualization module. This is a value ranging from 0 to 1 for easy visualization in OpenGL.";

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
	template<class T> static inline auto& get_member(T& var) { return var.debugArray;}
};
struct lifetime{
	using type = float;
	using unit_type = value_unit<float, SI::s>;
	static constexpr const array_enum identifier = array_enum::basicArrays_lifetime;
	static constexpr const auto variableName = "lifetime";
	static constexpr const auto qualifiedName = "basicArrays.lifetime";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_pressure;
	static constexpr const auto variableName = "pressure";
	static constexpr const auto qualifiedName = "basicArrays.pressure";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::basicArrays_density;
	static constexpr const auto variableName = "density";
	static constexpr const auto qualifiedName = "basicArrays.density";
	static constexpr const auto ambiguous = false;
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
}
namespace boundaryArrays{
struct volumeBoundaryMin{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryMin;
	static constexpr const auto variableName = "volumeBoundaryMin";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryMin";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Low part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryMin;}
};
struct boundaryPlaneVelocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_boundaryPlaneVelocity;
	static constexpr const auto variableName = "boundaryPlaneVelocity";
	static constexpr const auto qualifiedName = "boundaryArrays.boundaryPlaneVelocity";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains boundary planes of the simulation as implicit plane equations entry = (n.x,n.y,n.z,d).";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.boundaryPlaneVelocity;}
};
struct volumeBoundaryVolumes{
	using type = cudaTextureObject_t;
	using unit_type = cudaTextureObject_t;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryVolumes;
	static constexpr const auto variableName = "volumeBoundaryVolumes";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryVolumes";
	static constexpr const auto ambiguous = false;
	static cudaTextureObject_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains references to boundary volumes created from vdb volumes for more complex boundary objects than planes.";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
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
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryDimensions;
	static constexpr const auto variableName = "volumeBoundaryDimensions";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryDimensions";
	static constexpr const auto ambiguous = false;
	static int4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Contains the dimension in voxels for each boundary volume.";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryDimensions;}
};
struct volumeBoundaryVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryVolume;
	static constexpr const auto variableName = "volumeBoundaryVolume";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryVolume";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryVolume;}
};
struct volumeBoundaryDensity{
	using type = float;
	using unit_type = value_unit<float, SI::density>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryDensity;
	static constexpr const auto variableName = "volumeBoundaryDensity";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryDensity";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryDensity;}
};
struct splineLUT{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_splineLUT;
	static constexpr const auto variableName = "splineLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.splineLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.splineLUT;}
};
struct offsetLUT{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_offsetLUT;
	static constexpr const auto variableName = "offsetLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.offsetLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.offsetLUT;}
};
struct spikyLUT{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_spikyLUT;
	static constexpr const auto variableName = "spikyLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.spikyLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.spikyLUT;}
};
struct adhesionLUT{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_adhesionLUT;
	static constexpr const auto variableName = "adhesionLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.adhesionLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.adhesionLUT;}
};
struct volumeLUT{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeLUT;
	static constexpr const auto variableName = "volumeLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeLUT;}
};
struct spline2LUT{
	using type = float;
	using unit_type = value_unit<float, SI::m_5>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_spline2LUT;
	static constexpr const auto variableName = "spline2LUT";
	static constexpr const auto qualifiedName = "boundaryArrays.spline2LUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.spline2LUT;}
};
struct splineGradientLUT{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::m>>>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_splineGradientLUT;
	static constexpr const auto variableName = "splineGradientLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.splineGradientLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.splineGradientLUT;}
};
struct spikyGradientLUT{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::m>>>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_spikyGradientLUT;
	static constexpr const auto variableName = "spikyGradientLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.spikyGradientLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.spikyGradientLUT;}
};
struct cohesionLUT{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_cohesionLUT;
	static constexpr const auto variableName = "cohesionLUT";
	static constexpr const auto qualifiedName = "boundaryArrays.cohesionLUT";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "Look-Up-Table to handle simulation boundaries more efficiently.";

	static constexpr const memory_kind kind = memory_kind::customData;

	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.cohesionLUT;}
};
struct boundaryPlanes{
	using type = float4;
	using unit_type = value_unit<float4, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_boundaryPlanes;
	static constexpr const auto variableName = "boundaryPlanes";
	static constexpr const auto qualifiedName = "boundaryArrays.boundaryPlanes";
	static constexpr const auto ambiguous = false;
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
struct volumeBoundaryVelocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryVelocity;
	static constexpr const auto variableName = "volumeBoundaryVelocity";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryVelocity";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryVelocity;}
};
struct volumeBoundaryAngularVelocity{
	using type = float4;
	using unit_type = value_unit<float4, SI::Hz>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryAngularVelocity;
	static constexpr const auto variableName = "volumeBoundaryAngularVelocity";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryAngularVelocity";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryAngularVelocity;}
};
struct volumeBoundaryKind{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryKind;
	static constexpr const auto variableName = "volumeBoundaryKind";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryKind";
	static constexpr const auto ambiguous = false;
	static int32_t* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryKind;}
};
struct volumeBoundaryPosition{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryPosition;
	static constexpr const auto variableName = "volumeBoundaryPosition";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryPosition";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryPosition;}
};
struct volumeBoundaryQuaternion{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryQuaternion;
	static constexpr const auto variableName = "volumeBoundaryQuaternion";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryQuaternion";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryQuaternion;}
};
struct volumeBoundaryTransformMatrix{
	using type = Matrix4x4;
	using unit_type = Matrix4x4;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryTransformMatrix;
	static constexpr const auto variableName = "volumeBoundaryTransformMatrix";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryTransformMatrix";
	static constexpr const auto ambiguous = false;
	static Matrix4x4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryTransformMatrix;}
};
struct volumeBoundaryMax{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryMax;
	static constexpr const auto variableName = "volumeBoundaryMax";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryMax";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "High part of the AABB of every boundary volume.";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryMax;}
};
struct volumeBoundaryTransformMatrixInverse{
	using type = Matrix4x4;
	using unit_type = Matrix4x4;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse;
	static constexpr const auto variableName = "volumeBoundaryTransformMatrixInverse";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryTransformMatrixInverse";
	static constexpr const auto ambiguous = false;
	static Matrix4x4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryTransformMatrixInverse;}
};
struct volumeBoundaryInertiaMatrix{
	using type = Matrix4x4;
	using unit_type = Matrix4x4;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryInertiaMatrix;
	static constexpr const auto variableName = "volumeBoundaryInertiaMatrix";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryInertiaMatrix";
	static constexpr const auto ambiguous = false;
	static Matrix4x4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryInertiaMatrix;}
};
struct volumeBoundaryInertiaMatrixInverse{
	using type = Matrix4x4;
	using unit_type = Matrix4x4;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse;
	static constexpr const auto variableName = "volumeBoundaryInertiaMatrixInverse";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryInertiaMatrixInverse";
	static constexpr const auto ambiguous = false;
	static Matrix4x4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryInertiaMatrixInverse;}
};
struct volumeBoundaryAcceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::acceleration>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryAcceleration;
	static constexpr const auto variableName = "volumeBoundaryAcceleration";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryAcceleration";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryAcceleration;}
};
struct volumeBoundaryAngularAcceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::s_2>;
	static constexpr const array_enum identifier = array_enum::boundaryArrays_volumeBoundaryAngularAcceleration;
	static constexpr const auto variableName = "volumeBoundaryAngularAcceleration";
	static constexpr const auto qualifiedName = "boundaryArrays.volumeBoundaryAngularAcceleration";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::individualData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryAngularAcceleration;}
};
}
namespace dfsphArrays{
struct dfsphAlpha{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::dfsphArrays_dfsphAlpha;
	static constexpr const auto variableName = "dfsphAlpha";
	static constexpr const auto qualifiedName = "dfsphArrays.dfsphAlpha";
	static constexpr const auto ambiguous = false;
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
struct dfsphDpDt{
	using type = float;
	using unit_type = value_unit<float, SI::Hz>;
	static constexpr const array_enum identifier = array_enum::dfsphArrays_dfsphDpDt;
	static constexpr const auto variableName = "dfsphDpDt";
	static constexpr const auto qualifiedName = "dfsphArrays.dfsphDpDt";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.dfsphDpDt;}
};
struct dfsphRhoStar{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::dfsphArrays_dfsphRhoStar;
	static constexpr const auto variableName = "dfsphRhoStar";
	static constexpr const auto qualifiedName = "dfsphArrays.dfsphRhoStar";
	static constexpr const auto ambiguous = false;
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
struct dfsphSource{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::dfsphArrays_dfsphSource;
	static constexpr const auto variableName = "dfsphSource";
	static constexpr const auto qualifiedName = "dfsphArrays.dfsphSource";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.dfsphSource;}
};
}
namespace iisph17{
struct omega{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_omega;
	static constexpr const auto variableName = "omega";
	static constexpr const auto qualifiedName = "iisph17.omega";
	static constexpr const auto ambiguous = true;
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
	template<class T> static inline auto& get_member(T& var) { return var.iisph17Omega;}
};
struct boundaryPressure{
	using type = float;
	using unit_type = value_unit<float, SI::Pa>;
	static constexpr const array_enum identifier = array_enum::iisph17_boundaryPressure;
	static constexpr const auto variableName = "boundaryPressure";
	static constexpr const auto qualifiedName = "iisph17.boundaryPressure";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.boundaryPressure;}
};
struct volumeError{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_volumeError;
	static constexpr const auto variableName = "volumeError";
	static constexpr const auto qualifiedName = "iisph17.volumeError";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.volumeError;}
};
struct predictedAcceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::acceleration>;
	static constexpr const array_enum identifier = array_enum::iisph17_predictedAcceleration;
	static constexpr const auto variableName = "predictedAcceleration";
	static constexpr const auto qualifiedName = "iisph17.predictedAcceleration";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.predictedAcceleration;}
};
struct Aii{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisph17_Aii;
	static constexpr const auto variableName = "Aii";
	static constexpr const auto qualifiedName = "iisph17.Aii";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.Aii;}
};
struct apparentVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::iisph17_apparentVolume;
	static constexpr const auto variableName = "apparentVolume";
	static constexpr const auto qualifiedName = "iisph17.apparentVolume";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.apparentVolume;}
};
struct sourceTerm{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_sourceTerm;
	static constexpr const auto variableName = "sourceTerm";
	static constexpr const auto qualifiedName = "iisph17.sourceTerm";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var.sourceTerm;}
};
}
namespace iisph17_band_rb{
struct _apparentVolume{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__apparentVolume;
	static constexpr const auto variableName = "_apparentVolume";
	static constexpr const auto qualifiedName = "iisph17_band_rb._apparentVolume";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._apparentVolume;}
};
struct _Aii{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit<SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__Aii;
	static constexpr const auto variableName = "_Aii";
	static constexpr const auto qualifiedName = "iisph17_band_rb._Aii";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._Aii;}
};
struct _predictedAcceleration{
	using type = float4;
	using unit_type = value_unit<float4, SI::acceleration>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__predictedAcceleration;
	static constexpr const auto variableName = "_predictedAcceleration";
	static constexpr const auto qualifiedName = "iisph17_band_rb._predictedAcceleration";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._predictedAcceleration;}
};
struct _volumeError{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__volumeError;
	static constexpr const auto variableName = "_volumeError";
	static constexpr const auto qualifiedName = "iisph17_band_rb._volumeError";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._volumeError;}
};
struct _sourceTerm{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__sourceTerm;
	static constexpr const auto variableName = "_sourceTerm";
	static constexpr const auto qualifiedName = "iisph17_band_rb._sourceTerm";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._sourceTerm;}
};
struct _omega{
	using type = float;
	using unit_type = value_unit<float, void_unit_ty>;
	static constexpr const array_enum identifier = array_enum::iisph17_band_rb__omega;
	static constexpr const auto variableName = "_omega";
	static constexpr const auto qualifiedName = "iisph17_band_rb._omega";
	static constexpr const auto ambiguous = false;
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
	template<class T> static inline auto& get_member(T& var) { return var._omega;}
};
}
namespace iisphArrays{
struct iisphAiiOld{
	using type = float;
	using unit_type = value_unit<float, SI::derived_unit< SI::density, SI::recip<SI::Pa>>>;
	static constexpr const array_enum identifier = array_enum::iisphArrays_iisphAiiOld;
	static constexpr const auto variableName = "iisphAiiOld";
	static constexpr const auto qualifiedName = "iisphArrays.iisphAiiOld";
	static constexpr const auto ambiguous = false;
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
struct sigma{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::iisphArrays_sigma;
	static constexpr const auto variableName = "sigma";
	static constexpr const auto qualifiedName = "iisphArrays.sigma";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_iisphDensityAdvection;
	static constexpr const auto variableName = "iisphDensityAdvection";
	static constexpr const auto qualifiedName = "iisphArrays.iisphDensityAdvection";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_iisphDensityIteration;
	static constexpr const auto variableName = "iisphDensityIteration";
	static constexpr const auto qualifiedName = "iisphArrays.iisphDensityIteration";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_kernelBuffer;
	static constexpr const auto variableName = "kernelBuffer";
	static constexpr const auto qualifiedName = "iisphArrays.kernelBuffer";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_velocityAdvection;
	static constexpr const auto variableName = "velocityAdvection";
	static constexpr const auto qualifiedName = "iisphArrays.velocityAdvection";
	static constexpr const auto ambiguous = false;
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
struct iisphSum{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::iisphArrays_iisphSum;
	static constexpr const auto variableName = "iisphSum";
	static constexpr const auto qualifiedName = "iisphArrays.iisphSum";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_iisphDii;
	static constexpr const auto variableName = "iisphDii";
	static constexpr const auto qualifiedName = "iisphArrays.iisphDii";
	static constexpr const auto ambiguous = false;
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
struct omega{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::iisphArrays_omega;
	static constexpr const auto variableName = "omega";
	static constexpr const auto qualifiedName = "iisphArrays.omega";
	static constexpr const auto ambiguous = true;
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
	template<class T> static inline auto& get_member(T& var) { return var.iisphArraysOmega;}
};
struct alpha{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::iisphArrays_alpha;
	static constexpr const auto variableName = "alpha";
	static constexpr const auto qualifiedName = "iisphArrays.alpha";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::iisphArrays_gamma;
	static constexpr const auto variableName = "gamma";
	static constexpr const auto qualifiedName = "iisphArrays.gamma";
	static constexpr const auto ambiguous = false;
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
}
namespace renderArrays{
struct auxIsoDensity{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxIsoDensity;
	static constexpr const auto variableName = "auxIsoDensity";
	static constexpr const auto qualifiedName = "renderArrays.auxIsoDensity";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxIsoDensity;}
};
struct anisotropicMatrices{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::renderArrays_anisotropicMatrices;
	static constexpr const auto variableName = "anisotropicMatrices";
	static constexpr const auto qualifiedName = "renderArrays.anisotropicMatrices";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.anisotropicMatrices;}
};
struct auxTest{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxTest;
	static constexpr const auto variableName = "auxTest";
	static constexpr const auto qualifiedName = "renderArrays.auxTest";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxTest;}
};
struct centerPosition{
	using type = float4;
	using unit_type = float4;
	static constexpr const array_enum identifier = array_enum::renderArrays_centerPosition;
	static constexpr const auto variableName = "centerPosition";
	static constexpr const auto qualifiedName = "renderArrays.centerPosition";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.centerPosition;}
};
struct auxCellSurface{
	using type = cellSurface;
	using unit_type = cellSurface;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxCellSurface;
	static constexpr const auto variableName = "auxCellSurface";
	static constexpr const auto qualifiedName = "renderArrays.auxCellSurface";
	static constexpr const auto ambiguous = false;
	static cellSurface* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxCellSurface;}
};
struct auxCellInformation{
	using type = cellInformation;
	using unit_type = cellInformation;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxCellInformation;
	static constexpr const auto variableName = "auxCellInformation";
	static constexpr const auto qualifiedName = "renderArrays.auxCellInformation";
	static constexpr const auto ambiguous = false;
	static cellInformation* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxCellInformation;}
};
struct auxCellSpan{
	using type = compactListEntry;
	using unit_type = compactListEntry;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxCellSpan;
	static constexpr const auto variableName = "auxCellSpan";
	static constexpr const auto qualifiedName = "renderArrays.auxCellSpan";
	static constexpr const auto ambiguous = false;
	static compactListEntry* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxCellSpan;}
};
struct auxHashMap{
	using type = compactListEntry;
	using unit_type = compactListEntry;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxHashMap;
	static constexpr const auto variableName = "auxHashMap";
	static constexpr const auto qualifiedName = "renderArrays.auxHashMap";
	static constexpr const auto ambiguous = false;
	static compactListEntry* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxHashMap;}
};
struct auxDistance{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::renderArrays_auxDistance;
	static constexpr const auto variableName = "auxDistance";
	static constexpr const auto qualifiedName = "renderArrays.auxDistance";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = ".";

	static constexpr const memory_kind kind = memory_kind::particleData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.auxDistance;}
};
}
namespace rigidBodyArrays{
struct rigidVolumes{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidVolumes;
	static constexpr const auto variableName = "rigidVolumes";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidVolumes";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidVolumes;}
};
struct rigidLinearVelocities{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidLinearVelocities;
	static constexpr const auto variableName = "rigidLinearVelocities";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidLinearVelocities";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidLinearVelocities;}
};
struct rigidAVelocities{
	using type = float3;
	using unit_type = value_unit<float3, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidAVelocities;
	static constexpr const auto variableName = "rigidAVelocities";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidAVelocities";
	static constexpr const auto ambiguous = false;
	static float3* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidAVelocities;}
};
struct rigidDensities{
	using type = float;
	using unit_type = value_unit<float, SI::density>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidDensities;
	static constexpr const auto variableName = "rigidDensities";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidDensities";
	static constexpr const auto ambiguous = false;
	static float* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidDensities;}
};
struct rigidOrigins{
	using type = float3;
	using unit_type = value_unit<float3, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidOrigins;
	static constexpr const auto variableName = "rigidOrigins";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidOrigins";
	static constexpr const auto ambiguous = false;
	static float3* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidOrigins;}
};
struct rigidQuaternions{
	using type = float4;
	using unit_type = value_unit<float4, SI::velocity>;
	static constexpr const array_enum identifier = array_enum::rigidBodyArrays_rigidQuaternions;
	static constexpr const auto variableName = "rigidQuaternions";
	static constexpr const auto qualifiedName = "rigidBodyArrays.rigidQuaternions";
	static constexpr const auto ambiguous = false;
	static float4* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "TODO";

	static constexpr const memory_kind kind = memory_kind::rigidData;

	static void defaultAllocate();
	static void leanAllocate();
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.rigidQuaternions;}
};
}
namespace structureArrays{
struct mlmScaling{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::structureArrays_mlmScaling;
	static constexpr const auto variableName = "mlmScaling";
	static constexpr const auto qualifiedName = "structureArrays.mlmScaling";
	static constexpr const auto ambiguous = false;
	static int32_t* ptr;
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
	template<class T> static inline auto& get_member(T& var) { return var.mlmScaling;}
};
struct particleparticleIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::structureArrays_particleparticleIndex;
	static constexpr const auto variableName = "particleparticleIndex";
	static constexpr const auto qualifiedName = "structureArrays.particleparticleIndex";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_cellBegin;
	static constexpr const auto variableName = "cellBegin";
	static constexpr const auto qualifiedName = "structureArrays.cellBegin";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_cellEnd;
	static constexpr const auto variableName = "cellEnd";
	static constexpr const auto qualifiedName = "structureArrays.cellEnd";
	static constexpr const auto ambiguous = false;
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
struct compactCellList{
	using type = compactSpan;
	using unit_type = compactSpan;
	static constexpr const array_enum identifier = array_enum::structureArrays_compactCellList;
	static constexpr const auto variableName = "compactCellList";
	static constexpr const auto qualifiedName = "structureArrays.compactCellList";
	static constexpr const auto ambiguous = false;
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
struct neighborMask{
	using type = compactCellNeighbors;
	using unit_type = compactCellNeighbors;
	static constexpr const array_enum identifier = array_enum::structureArrays_neighborMask;
	static constexpr const auto variableName = "neighborMask";
	static constexpr const auto qualifiedName = "structureArrays.neighborMask";
	static constexpr const auto ambiguous = false;
	static compactCellNeighbors* ptr;
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
	template<class T> static inline auto& get_member(T& var) { return var.neighborMask;}
};
struct compactCellScale{
	using type = int8_t;
	using unit_type = int8_t;
	static constexpr const array_enum identifier = array_enum::structureArrays_compactCellScale;
	static constexpr const auto variableName = "compactCellScale";
	static constexpr const auto qualifiedName = "structureArrays.compactCellScale";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_spanNeighborList;
	static constexpr const auto variableName = "spanNeighborList";
	static constexpr const auto qualifiedName = "structureArrays.spanNeighborList";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_neighborList;
	static constexpr const auto variableName = "neighborList";
	static constexpr const auto qualifiedName = "structureArrays.neighborList";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_neighborListLength;
	static constexpr const auto variableName = "neighborListLength";
	static constexpr const auto qualifiedName = "structureArrays.neighborListLength";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_classification;
	static constexpr const auto variableName = "classification";
	static constexpr const auto qualifiedName = "structureArrays.classification";
	static constexpr const auto ambiguous = true;
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
	template<class T> static inline auto& get_member(T& var) { return var.structureArraysClassification;}
};
struct resortIndex{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::structureArrays_resortIndex;
	static constexpr const auto variableName = "resortIndex";
	static constexpr const auto qualifiedName = "structureArrays.resortIndex";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_ZOrder_64;
	static constexpr const auto variableName = "ZOrder_64";
	static constexpr const auto qualifiedName = "structureArrays.ZOrder_64";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_ZOrder_32;
	static constexpr const auto variableName = "ZOrder_32";
	static constexpr const auto qualifiedName = "structureArrays.ZOrder_32";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_cellSpanSwap;
	static constexpr const auto variableName = "cellSpanSwap";
	static constexpr const auto qualifiedName = "structureArrays.cellSpanSwap";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_cellSpan;
	static constexpr const auto variableName = "cellSpan";
	static constexpr const auto qualifiedName = "structureArrays.cellSpan";
	static constexpr const auto ambiguous = false;
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
struct compactCellSpanSwap{
	using type = compact_cellSpan;
	using unit_type = compact_cellSpan;
	static constexpr const array_enum identifier = array_enum::structureArrays_compactCellSpanSwap;
	static constexpr const auto variableName = "compactCellSpanSwap";
	static constexpr const auto qualifiedName = "structureArrays.compactCellSpanSwap";
	static constexpr const auto ambiguous = false;
	static compact_cellSpan* ptr;
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
	template<class T> static inline auto& get_member(T& var) { return var.compactCellSpanSwap;}
};
struct compactCellSpan{
	using type = compact_cellSpan;
	using unit_type = compact_cellSpan;
	static constexpr const array_enum identifier = array_enum::structureArrays_compactCellSpan;
	static constexpr const auto variableName = "compactCellSpan";
	static constexpr const auto qualifiedName = "structureArrays.compactCellSpan";
	static constexpr const auto ambiguous = false;
	static compact_cellSpan* ptr;
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
	template<class T> static inline auto& get_member(T& var) { return var.compactCellSpan;}
};
struct MLMResolution{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::structureArrays_MLMResolution;
	static constexpr const auto variableName = "MLMResolution";
	static constexpr const auto qualifiedName = "structureArrays.MLMResolution";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_cellparticleIndex;
	static constexpr const auto variableName = "cellparticleIndex";
	static constexpr const auto qualifiedName = "structureArrays.cellparticleIndex";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_compactparticleIndex;
	static constexpr const auto variableName = "compactparticleIndex";
	static constexpr const auto qualifiedName = "structureArrays.compactparticleIndex";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::structureArrays_hashMap;
	static constexpr const auto variableName = "hashMap";
	static constexpr const auto qualifiedName = "structureArrays.hashMap";
	static constexpr const auto ambiguous = false;
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
struct compactHashMap{
	using type = compactHashSpan;
	using unit_type = compactHashSpan;
	static constexpr const array_enum identifier = array_enum::structureArrays_compactHashMap;
	static constexpr const auto variableName = "compactHashMap";
	static constexpr const auto qualifiedName = "structureArrays.compactHashMap";
	static constexpr const auto ambiguous = false;
	static compactHashSpan* ptr;
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
	template<class T> static inline auto& get_member(T& var) { return var.compactHashMap;}
};
}
namespace supportArrays{
struct support{
	using type = float;
	using unit_type = value_unit<float, SI::m>;
	static constexpr const array_enum identifier = array_enum::supportArrays_support;
	static constexpr const auto variableName = "support";
	static constexpr const auto qualifiedName = "supportArrays.support";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_supportEstimate;
	static constexpr const auto variableName = "supportEstimate";
	static constexpr const auto qualifiedName = "supportArrays.supportEstimate";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_neighborCount;
	static constexpr const auto variableName = "neighborCount";
	static constexpr const auto qualifiedName = "supportArrays.neighborCount";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_supportMarker;
	static constexpr const auto variableName = "supportMarker";
	static constexpr const auto qualifiedName = "supportArrays.supportMarker";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_supportMarkerCompacted;
	static constexpr const auto variableName = "supportMarkerCompacted";
	static constexpr const auto qualifiedName = "supportArrays.supportMarkerCompacted";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_closestNeighbor;
	static constexpr const auto variableName = "closestNeighbor";
	static constexpr const auto qualifiedName = "supportArrays.closestNeighbor";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_closestNeighbor_f;
	static constexpr const auto variableName = "closestNeighbor_f";
	static constexpr const auto qualifiedName = "supportArrays.closestNeighbor_f";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_neighborOverhead;
	static constexpr const auto variableName = "neighborOverhead";
	static constexpr const auto qualifiedName = "supportArrays.neighborOverhead";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_neighborOverheadCount;
	static constexpr const auto variableName = "neighborOverheadCount";
	static constexpr const auto qualifiedName = "supportArrays.neighborOverheadCount";
	static constexpr const auto ambiguous = false;
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
	static constexpr const array_enum identifier = array_enum::supportArrays_neighborListSwap;
	static constexpr const auto variableName = "neighborListSwap";
	static constexpr const auto qualifiedName = "supportArrays.neighborListSwap";
	static constexpr const auto ambiguous = false;
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
namespace surfaceArrays{
struct distanceBuffer{
	using type = float;
	using unit_type = value_unit<float, SI::m>;
	static constexpr const array_enum identifier = array_enum::surfaceArrays_distanceBuffer;
	static constexpr const auto variableName = "distanceBuffer";
	static constexpr const auto qualifiedName = "surfaceArrays.distanceBuffer";
	static constexpr const auto ambiguous = false;
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
struct markerBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::surfaceArrays_markerBuffer;
	static constexpr const auto variableName = "markerBuffer";
	static constexpr const auto qualifiedName = "surfaceArrays.markerBuffer";
	static constexpr const auto ambiguous = false;
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
struct surface_idxBuffer{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::surfaceArrays_surface_idxBuffer;
	static constexpr const auto variableName = "surface_idxBuffer";
	static constexpr const auto qualifiedName = "surfaceArrays.surface_idxBuffer";
	static constexpr const auto ambiguous = false;
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
struct decisionBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::surfaceArrays_decisionBuffer;
	static constexpr const auto variableName = "decisionBuffer";
	static constexpr const auto qualifiedName = "surfaceArrays.decisionBuffer";
	static constexpr const auto ambiguous = false;
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
struct changeBuffer{
	using type = float;
	using unit_type = float;
	static constexpr const array_enum identifier = array_enum::surfaceArrays_changeBuffer;
	static constexpr const auto variableName = "changeBuffer";
	static constexpr const auto qualifiedName = "surfaceArrays.changeBuffer";
	static constexpr const auto ambiguous = false;
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
}
namespace volumeInletArrays{
struct volumeOutletMax{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletMax;
	static constexpr const auto variableName = "volumeOutletMax";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletMax";
	static constexpr const auto ambiguous = false;
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
struct volumeOutletRateAccumulator{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletRateAccumulator;
	static constexpr const auto variableName = "volumeOutletRateAccumulator";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletRateAccumulator";
	static constexpr const auto ambiguous = false;
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
struct volumeOutletRate{
	using type = float;
	using unit_type = value_unit<float, SI::volume>;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletRate;
	static constexpr const auto variableName = "volumeOutletRate";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletRate";
	static constexpr const auto ambiguous = false;
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
struct volumeOutletMin{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletMin;
	static constexpr const auto variableName = "volumeOutletMin";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletMin";
	static constexpr const auto ambiguous = false;
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
struct volumeOutletDimensions{
	using type = int4;
	using unit_type = int4;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletDimensions;
	static constexpr const auto variableName = "volumeOutletDimensions";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletDimensions";
	static constexpr const auto ambiguous = false;
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
struct volumeOutletVolumes{
	using type = cudaTextureObject_t;
	using unit_type = cudaTextureObject_t;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_volumeOutletVolumes;
	static constexpr const auto variableName = "volumeOutletVolumes";
	static constexpr const auto qualifiedName = "volumeInletArrays.volumeOutletVolumes";
	static constexpr const auto ambiguous = false;
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
struct inletCounter{
	using type = int32_t;
	using unit_type = int32_t;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_inletCounter;
	static constexpr const auto variableName = "inletCounter";
	static constexpr const auto qualifiedName = "volumeInletArrays.inletCounter";
	static constexpr const auto ambiguous = false;
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
struct inletPositions{
	using type = float4;
	using unit_type = value_unit<float4, SI::m>;
	static constexpr const array_enum identifier = array_enum::volumeInletArrays_inletPositions;
	static constexpr const auto variableName = "inletPositions";
	static constexpr const auto qualifiedName = "volumeInletArrays.inletPositions";
	static constexpr const auto ambiguous = false;
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
}


}
namespace arrays{
	using namespace adaptive;
	using namespace advectionArrays;
	using namespace basicArrays;
	using namespace boundaryArrays;
	using namespace dfsphArrays;
	using namespace iisph17;
	using namespace iisph17_band_rb;
	using namespace iisphArrays;
	using namespace renderArrays;
	using namespace rigidBodyArrays;
	using namespace structureArrays;
	using namespace supportArrays;
	using namespace surfaceArrays;
	using namespace volumeInletArrays;
}

extern std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::adaptivityCounter, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::rigidBodyArrays::rigidDensities, arrays::rigidBodyArrays::rigidVolumes, arrays::rigidBodyArrays::rigidLinearVelocities, arrays::rigidBodyArrays::rigidAVelocities, arrays::rigidBodyArrays::rigidOrigins, arrays::rigidBodyArrays::rigidQuaternions, arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::matrixTest, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::hashMap, arrays::structureArrays::compactHashMap, arrays::structureArrays::mlmScaling, arrays::structureArrays::particleparticleIndex, arrays::structureArrays::cellBegin, arrays::structureArrays::cellEnd, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount, arrays::supportArrays::neighborListSwap> allocations_list;
extern std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::adaptivityCounter, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::rigidBodyArrays::rigidDensities, arrays::rigidBodyArrays::rigidVolumes, arrays::rigidBodyArrays::rigidLinearVelocities, arrays::rigidBodyArrays::rigidAVelocities, arrays::rigidBodyArrays::rigidOrigins, arrays::rigidBodyArrays::rigidQuaternions, arrays::boundaryArrays::splineLUT, arrays::boundaryArrays::offsetLUT, arrays::boundaryArrays::spikyLUT, arrays::boundaryArrays::adhesionLUT, arrays::boundaryArrays::volumeLUT, arrays::boundaryArrays::spline2LUT, arrays::boundaryArrays::splineGradientLUT, arrays::boundaryArrays::spikyGradientLUT, arrays::boundaryArrays::cohesionLUT, arrays::boundaryArrays::boundaryPlanes, arrays::boundaryArrays::boundaryPlaneVelocity, arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::matrixTest, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::volumeInletArrays::inletPositions, arrays::volumeInletArrays::inletCounter, arrays::volumeInletArrays::volumeOutletVolumes, arrays::volumeInletArrays::volumeOutletDimensions, arrays::volumeInletArrays::volumeOutletMin, arrays::volumeInletArrays::volumeOutletMax, arrays::volumeInletArrays::volumeOutletRate, arrays::volumeInletArrays::volumeOutletRateAccumulator, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::hashMap, arrays::structureArrays::compactHashMap, arrays::structureArrays::mlmScaling, arrays::structureArrays::particleparticleIndex, arrays::structureArrays::cellBegin, arrays::structureArrays::cellEnd, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount, arrays::supportArrays::neighborListSwap> arrays_list;
extern std::tuple<arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::basicArrays::matrixTest> individual_list;
extern std::tuple<arrays::adaptive::splitIndicator, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::angularVelocity, arrays::surfaceArrays::distanceBuffer, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::structureArrays::neighborListLength> sorting_list;
extern std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::particleparticleIndex, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount> swapping_list;
template<typename C, typename... Ts>
auto iterateAllocationsList(C&& fn, Ts&&... args){
	fn(arrays::adaptive::mergeable(), args...);
	fn(arrays::adaptive::mergeCounter(), args...);
	fn(arrays::adaptive::ptclCounter(), args...);
	fn(arrays::adaptive::classification(), args...);
	fn(arrays::adaptive::splitIndicator(), args...);
	fn(arrays::adaptive::adaptivityCounter(), args...);
	fn(arrays::adaptive::splitIndicatorCompacted(), args...);
	fn(arrays::adaptive::parentIndex(), args...);
	fn(arrays::adaptive::parentVolume(), args...);
	fn(arrays::adaptive::parentPosition(), args...);
	fn(arrays::advectionArrays::maxVelocity(), args...);
	fn(arrays::advectionArrays::cflValue(), args...);
	fn(arrays::advectionArrays::particleNormal(), args...);
	fn(arrays::advectionArrays::angularVelocity(), args...);
	fn(arrays::rigidBodyArrays::rigidDensities(), args...);
	fn(arrays::rigidBodyArrays::rigidVolumes(), args...);
	fn(arrays::rigidBodyArrays::rigidLinearVelocities(), args...);
	fn(arrays::rigidBodyArrays::rigidAVelocities(), args...);
	fn(arrays::rigidBodyArrays::rigidOrigins(), args...);
	fn(arrays::rigidBodyArrays::rigidQuaternions(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVolumes(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDimensions(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMin(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMax(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDensity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVolume(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryKind(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryPosition(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryQuaternion(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAcceleration(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularAcceleration(), args...);
	fn(arrays::surfaceArrays::decisionBuffer(), args...);
	fn(arrays::surfaceArrays::surface_idxBuffer(), args...);
	fn(arrays::surfaceArrays::markerBuffer(), args...);
	fn(arrays::surfaceArrays::distanceBuffer(), args...);
	fn(arrays::surfaceArrays::changeBuffer(), args...);
	fn(arrays::basicArrays::matrixTest(), args...);
	fn(arrays::basicArrays::position(), args...);
	fn(arrays::basicArrays::acceleration(), args...);
	fn(arrays::basicArrays::velocity(), args...);
	fn(arrays::basicArrays::particle_type(), args...);
	fn(arrays::basicArrays::renderArray(), args...);
	fn(arrays::basicArrays::debugArray(), args...);
	fn(arrays::basicArrays::volume(), args...);
	fn(arrays::basicArrays::lifetime(), args...);
	fn(arrays::basicArrays::pressure(), args...);
	fn(arrays::basicArrays::density(), args...);
	fn(arrays::basicArrays::fluidDensity(), args...);
	fn(arrays::basicArrays::particleIndex(), args...);
	fn(arrays::basicArrays::particleIndexCompact(), args...);
	fn(arrays::basicArrays::resortArray4(), args...);
	fn(arrays::basicArrays::resortArray(), args...);
	fn(arrays::dfsphArrays::dfsphSource(), args...);
	fn(arrays::dfsphArrays::dfsphAlpha(), args...);
	fn(arrays::dfsphArrays::dfsphDpDt(), args...);
	fn(arrays::dfsphArrays::dfsphRhoStar(), args...);
	fn(arrays::iisphArrays::iisphSum(), args...);
	fn(arrays::iisphArrays::iisphDii(), args...);
	fn(arrays::iisphArrays::iisphAiiOld(), args...);
	fn(arrays::iisphArrays::omega(), args...);
	fn(arrays::iisphArrays::alpha(), args...);
	fn(arrays::iisphArrays::gamma(), args...);
	fn(arrays::iisphArrays::sigma(), args...);
	fn(arrays::iisphArrays::iisphDensityAdvection(), args...);
	fn(arrays::iisphArrays::iisphDensityIteration(), args...);
	fn(arrays::iisphArrays::kernelBuffer(), args...);
	fn(arrays::iisphArrays::velocityAdvection(), args...);
	fn(arrays::iisph17::sourceTerm(), args...);
	fn(arrays::iisph17::boundaryPressure(), args...);
	fn(arrays::iisph17::omega(), args...);
	fn(arrays::iisph17::apparentVolume(), args...);
	fn(arrays::iisph17::Aii(), args...);
	fn(arrays::iisph17::predictedAcceleration(), args...);
	fn(arrays::iisph17::volumeError(), args...);
	fn(arrays::iisph17_band_rb::_sourceTerm(), args...);
	fn(arrays::iisph17_band_rb::_omega(), args...);
	fn(arrays::iisph17_band_rb::_apparentVolume(), args...);
	fn(arrays::iisph17_band_rb::_Aii(), args...);
	fn(arrays::iisph17_band_rb::_predictedAcceleration(), args...);
	fn(arrays::iisph17_band_rb::_volumeError(), args...);
	fn(arrays::structureArrays::compactCellList(), args...);
	fn(arrays::structureArrays::neighborMask(), args...);
	fn(arrays::structureArrays::compactCellScale(), args...);
	fn(arrays::structureArrays::spanNeighborList(), args...);
	fn(arrays::structureArrays::neighborList(), args...);
	fn(arrays::structureArrays::neighborListLength(), args...);
	fn(arrays::renderArrays::anisotropicMatrices(), args...);
	fn(arrays::renderArrays::centerPosition(), args...);
	fn(arrays::renderArrays::auxTest(), args...);
	fn(arrays::renderArrays::auxIsoDensity(), args...);
	fn(arrays::renderArrays::auxDistance(), args...);
	fn(arrays::renderArrays::auxHashMap(), args...);
	fn(arrays::renderArrays::auxCellSpan(), args...);
	fn(arrays::renderArrays::auxCellInformation(), args...);
	fn(arrays::renderArrays::auxCellSurface(), args...);
	fn(arrays::structureArrays::classification(), args...);
	fn(arrays::structureArrays::resortIndex(), args...);
	fn(arrays::structureArrays::ZOrder_64(), args...);
	fn(arrays::structureArrays::ZOrder_32(), args...);
	fn(arrays::structureArrays::cellSpanSwap(), args...);
	fn(arrays::structureArrays::cellSpan(), args...);
	fn(arrays::structureArrays::compactCellSpanSwap(), args...);
	fn(arrays::structureArrays::compactCellSpan(), args...);
	fn(arrays::structureArrays::MLMResolution(), args...);
	fn(arrays::structureArrays::cellparticleIndex(), args...);
	fn(arrays::structureArrays::compactparticleIndex(), args...);
	fn(arrays::structureArrays::hashMap(), args...);
	fn(arrays::structureArrays::compactHashMap(), args...);
	fn(arrays::structureArrays::mlmScaling(), args...);
	fn(arrays::structureArrays::particleparticleIndex(), args...);
	fn(arrays::structureArrays::cellBegin(), args...);
	fn(arrays::structureArrays::cellEnd(), args...);
	fn(arrays::supportArrays::support(), args...);
	fn(arrays::supportArrays::supportEstimate(), args...);
	fn(arrays::supportArrays::neighborCount(), args...);
	fn(arrays::supportArrays::supportMarker(), args...);
	fn(arrays::supportArrays::supportMarkerCompacted(), args...);
	fn(arrays::supportArrays::closestNeighbor(), args...);
	fn(arrays::supportArrays::closestNeighbor_f(), args...);
	fn(arrays::supportArrays::neighborOverhead(), args...);
	fn(arrays::supportArrays::neighborOverheadCount(), args...);
	fn(arrays::supportArrays::neighborListSwap(), args...);
}

template<typename C, typename... Ts>
auto iterateArraysList(C&& fn, Ts&&... args){
	fn(arrays::adaptive::mergeable(), args...);
	fn(arrays::adaptive::mergeCounter(), args...);
	fn(arrays::adaptive::ptclCounter(), args...);
	fn(arrays::adaptive::classification(), args...);
	fn(arrays::adaptive::splitIndicator(), args...);
	fn(arrays::adaptive::adaptivityCounter(), args...);
	fn(arrays::adaptive::splitIndicatorCompacted(), args...);
	fn(arrays::adaptive::parentIndex(), args...);
	fn(arrays::adaptive::parentVolume(), args...);
	fn(arrays::adaptive::parentPosition(), args...);
	fn(arrays::advectionArrays::maxVelocity(), args...);
	fn(arrays::advectionArrays::cflValue(), args...);
	fn(arrays::advectionArrays::particleNormal(), args...);
	fn(arrays::advectionArrays::angularVelocity(), args...);
	fn(arrays::rigidBodyArrays::rigidDensities(), args...);
	fn(arrays::rigidBodyArrays::rigidVolumes(), args...);
	fn(arrays::rigidBodyArrays::rigidLinearVelocities(), args...);
	fn(arrays::rigidBodyArrays::rigidAVelocities(), args...);
	fn(arrays::rigidBodyArrays::rigidOrigins(), args...);
	fn(arrays::rigidBodyArrays::rigidQuaternions(), args...);
	fn(arrays::boundaryArrays::splineLUT(), args...);
	fn(arrays::boundaryArrays::offsetLUT(), args...);
	fn(arrays::boundaryArrays::spikyLUT(), args...);
	fn(arrays::boundaryArrays::adhesionLUT(), args...);
	fn(arrays::boundaryArrays::volumeLUT(), args...);
	fn(arrays::boundaryArrays::spline2LUT(), args...);
	fn(arrays::boundaryArrays::splineGradientLUT(), args...);
	fn(arrays::boundaryArrays::spikyGradientLUT(), args...);
	fn(arrays::boundaryArrays::cohesionLUT(), args...);
	fn(arrays::boundaryArrays::boundaryPlanes(), args...);
	fn(arrays::boundaryArrays::boundaryPlaneVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVolumes(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDimensions(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMin(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMax(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDensity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVolume(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryKind(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryPosition(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryQuaternion(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAcceleration(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularAcceleration(), args...);
	fn(arrays::surfaceArrays::decisionBuffer(), args...);
	fn(arrays::surfaceArrays::surface_idxBuffer(), args...);
	fn(arrays::surfaceArrays::markerBuffer(), args...);
	fn(arrays::surfaceArrays::distanceBuffer(), args...);
	fn(arrays::surfaceArrays::changeBuffer(), args...);
	fn(arrays::basicArrays::matrixTest(), args...);
	fn(arrays::basicArrays::position(), args...);
	fn(arrays::basicArrays::acceleration(), args...);
	fn(arrays::basicArrays::velocity(), args...);
	fn(arrays::basicArrays::particle_type(), args...);
	fn(arrays::basicArrays::renderArray(), args...);
	fn(arrays::basicArrays::debugArray(), args...);
	fn(arrays::basicArrays::volume(), args...);
	fn(arrays::basicArrays::lifetime(), args...);
	fn(arrays::basicArrays::pressure(), args...);
	fn(arrays::basicArrays::density(), args...);
	fn(arrays::basicArrays::fluidDensity(), args...);
	fn(arrays::basicArrays::particleIndex(), args...);
	fn(arrays::basicArrays::particleIndexCompact(), args...);
	fn(arrays::basicArrays::resortArray4(), args...);
	fn(arrays::basicArrays::resortArray(), args...);
	fn(arrays::dfsphArrays::dfsphSource(), args...);
	fn(arrays::dfsphArrays::dfsphAlpha(), args...);
	fn(arrays::dfsphArrays::dfsphDpDt(), args...);
	fn(arrays::dfsphArrays::dfsphRhoStar(), args...);
	fn(arrays::iisphArrays::iisphSum(), args...);
	fn(arrays::iisphArrays::iisphDii(), args...);
	fn(arrays::iisphArrays::iisphAiiOld(), args...);
	fn(arrays::iisphArrays::omega(), args...);
	fn(arrays::iisphArrays::alpha(), args...);
	fn(arrays::iisphArrays::gamma(), args...);
	fn(arrays::iisphArrays::sigma(), args...);
	fn(arrays::iisphArrays::iisphDensityAdvection(), args...);
	fn(arrays::iisphArrays::iisphDensityIteration(), args...);
	fn(arrays::iisphArrays::kernelBuffer(), args...);
	fn(arrays::iisphArrays::velocityAdvection(), args...);
	fn(arrays::iisph17::sourceTerm(), args...);
	fn(arrays::iisph17::boundaryPressure(), args...);
	fn(arrays::iisph17::omega(), args...);
	fn(arrays::iisph17::apparentVolume(), args...);
	fn(arrays::iisph17::Aii(), args...);
	fn(arrays::iisph17::predictedAcceleration(), args...);
	fn(arrays::iisph17::volumeError(), args...);
	fn(arrays::iisph17_band_rb::_sourceTerm(), args...);
	fn(arrays::iisph17_band_rb::_omega(), args...);
	fn(arrays::iisph17_band_rb::_apparentVolume(), args...);
	fn(arrays::iisph17_band_rb::_Aii(), args...);
	fn(arrays::iisph17_band_rb::_predictedAcceleration(), args...);
	fn(arrays::iisph17_band_rb::_volumeError(), args...);
	fn(arrays::volumeInletArrays::inletPositions(), args...);
	fn(arrays::volumeInletArrays::inletCounter(), args...);
	fn(arrays::volumeInletArrays::volumeOutletVolumes(), args...);
	fn(arrays::volumeInletArrays::volumeOutletDimensions(), args...);
	fn(arrays::volumeInletArrays::volumeOutletMin(), args...);
	fn(arrays::volumeInletArrays::volumeOutletMax(), args...);
	fn(arrays::volumeInletArrays::volumeOutletRate(), args...);
	fn(arrays::volumeInletArrays::volumeOutletRateAccumulator(), args...);
	fn(arrays::structureArrays::compactCellList(), args...);
	fn(arrays::structureArrays::neighborMask(), args...);
	fn(arrays::structureArrays::compactCellScale(), args...);
	fn(arrays::structureArrays::spanNeighborList(), args...);
	fn(arrays::structureArrays::neighborList(), args...);
	fn(arrays::structureArrays::neighborListLength(), args...);
	fn(arrays::renderArrays::anisotropicMatrices(), args...);
	fn(arrays::renderArrays::centerPosition(), args...);
	fn(arrays::renderArrays::auxTest(), args...);
	fn(arrays::renderArrays::auxIsoDensity(), args...);
	fn(arrays::renderArrays::auxDistance(), args...);
	fn(arrays::renderArrays::auxHashMap(), args...);
	fn(arrays::renderArrays::auxCellSpan(), args...);
	fn(arrays::renderArrays::auxCellInformation(), args...);
	fn(arrays::renderArrays::auxCellSurface(), args...);
	fn(arrays::structureArrays::classification(), args...);
	fn(arrays::structureArrays::resortIndex(), args...);
	fn(arrays::structureArrays::ZOrder_64(), args...);
	fn(arrays::structureArrays::ZOrder_32(), args...);
	fn(arrays::structureArrays::cellSpanSwap(), args...);
	fn(arrays::structureArrays::cellSpan(), args...);
	fn(arrays::structureArrays::compactCellSpanSwap(), args...);
	fn(arrays::structureArrays::compactCellSpan(), args...);
	fn(arrays::structureArrays::MLMResolution(), args...);
	fn(arrays::structureArrays::cellparticleIndex(), args...);
	fn(arrays::structureArrays::compactparticleIndex(), args...);
	fn(arrays::structureArrays::hashMap(), args...);
	fn(arrays::structureArrays::compactHashMap(), args...);
	fn(arrays::structureArrays::mlmScaling(), args...);
	fn(arrays::structureArrays::particleparticleIndex(), args...);
	fn(arrays::structureArrays::cellBegin(), args...);
	fn(arrays::structureArrays::cellEnd(), args...);
	fn(arrays::supportArrays::support(), args...);
	fn(arrays::supportArrays::supportEstimate(), args...);
	fn(arrays::supportArrays::neighborCount(), args...);
	fn(arrays::supportArrays::supportMarker(), args...);
	fn(arrays::supportArrays::supportMarkerCompacted(), args...);
	fn(arrays::supportArrays::closestNeighbor(), args...);
	fn(arrays::supportArrays::closestNeighbor_f(), args...);
	fn(arrays::supportArrays::neighborOverhead(), args...);
	fn(arrays::supportArrays::neighborOverheadCount(), args...);
	fn(arrays::supportArrays::neighborListSwap(), args...);
}

template<typename C, typename... Ts>
auto iterateIndividualList(C&& fn, Ts&&... args){
	fn(arrays::boundaryArrays::volumeBoundaryVolumes(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDimensions(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMin(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryMax(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryDensity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVolume(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularVelocity(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryKind(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryPosition(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryQuaternion(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrix(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAcceleration(), args...);
	fn(arrays::boundaryArrays::volumeBoundaryAngularAcceleration(), args...);
	fn(arrays::basicArrays::matrixTest(), args...);
}

template<typename C, typename... Ts>
auto iterateSortingList(C&& fn, Ts&&... args){
	fn(arrays::adaptive::splitIndicator(), args...);
	fn(arrays::adaptive::parentIndex(), args...);
	fn(arrays::adaptive::parentVolume(), args...);
	fn(arrays::adaptive::parentPosition(), args...);
	fn(arrays::advectionArrays::angularVelocity(), args...);
	fn(arrays::surfaceArrays::distanceBuffer(), args...);
	fn(arrays::basicArrays::position(), args...);
	fn(arrays::basicArrays::acceleration(), args...);
	fn(arrays::basicArrays::velocity(), args...);
	fn(arrays::basicArrays::particle_type(), args...);
	fn(arrays::basicArrays::renderArray(), args...);
	fn(arrays::basicArrays::debugArray(), args...);
	fn(arrays::basicArrays::volume(), args...);
	fn(arrays::basicArrays::lifetime(), args...);
	fn(arrays::basicArrays::pressure(), args...);
	fn(arrays::basicArrays::density(), args...);
	fn(arrays::structureArrays::neighborListLength(), args...);
}

template<typename C, typename... Ts>
auto iterateSwappingList(C&& fn, Ts&&... args){
	fn(arrays::adaptive::mergeable(), args...);
	fn(arrays::adaptive::mergeCounter(), args...);
	fn(arrays::adaptive::ptclCounter(), args...);
	fn(arrays::adaptive::classification(), args...);
	fn(arrays::adaptive::splitIndicator(), args...);
	fn(arrays::adaptive::splitIndicatorCompacted(), args...);
	fn(arrays::adaptive::parentIndex(), args...);
	fn(arrays::adaptive::parentVolume(), args...);
	fn(arrays::adaptive::parentPosition(), args...);
	fn(arrays::advectionArrays::maxVelocity(), args...);
	fn(arrays::advectionArrays::cflValue(), args...);
	fn(arrays::advectionArrays::particleNormal(), args...);
	fn(arrays::advectionArrays::angularVelocity(), args...);
	fn(arrays::surfaceArrays::decisionBuffer(), args...);
	fn(arrays::surfaceArrays::surface_idxBuffer(), args...);
	fn(arrays::surfaceArrays::markerBuffer(), args...);
	fn(arrays::surfaceArrays::distanceBuffer(), args...);
	fn(arrays::surfaceArrays::changeBuffer(), args...);
	fn(arrays::basicArrays::position(), args...);
	fn(arrays::basicArrays::acceleration(), args...);
	fn(arrays::basicArrays::velocity(), args...);
	fn(arrays::basicArrays::particle_type(), args...);
	fn(arrays::basicArrays::renderArray(), args...);
	fn(arrays::basicArrays::debugArray(), args...);
	fn(arrays::basicArrays::volume(), args...);
	fn(arrays::basicArrays::lifetime(), args...);
	fn(arrays::basicArrays::pressure(), args...);
	fn(arrays::basicArrays::density(), args...);
	fn(arrays::basicArrays::fluidDensity(), args...);
	fn(arrays::basicArrays::particleIndex(), args...);
	fn(arrays::basicArrays::particleIndexCompact(), args...);
	fn(arrays::basicArrays::resortArray4(), args...);
	fn(arrays::basicArrays::resortArray(), args...);
	fn(arrays::dfsphArrays::dfsphSource(), args...);
	fn(arrays::dfsphArrays::dfsphAlpha(), args...);
	fn(arrays::dfsphArrays::dfsphDpDt(), args...);
	fn(arrays::dfsphArrays::dfsphRhoStar(), args...);
	fn(arrays::iisphArrays::iisphSum(), args...);
	fn(arrays::iisphArrays::iisphDii(), args...);
	fn(arrays::iisphArrays::iisphAiiOld(), args...);
	fn(arrays::iisphArrays::omega(), args...);
	fn(arrays::iisphArrays::alpha(), args...);
	fn(arrays::iisphArrays::gamma(), args...);
	fn(arrays::iisphArrays::sigma(), args...);
	fn(arrays::iisphArrays::iisphDensityAdvection(), args...);
	fn(arrays::iisphArrays::iisphDensityIteration(), args...);
	fn(arrays::iisphArrays::kernelBuffer(), args...);
	fn(arrays::iisphArrays::velocityAdvection(), args...);
	fn(arrays::iisph17::sourceTerm(), args...);
	fn(arrays::iisph17::boundaryPressure(), args...);
	fn(arrays::iisph17::omega(), args...);
	fn(arrays::iisph17::apparentVolume(), args...);
	fn(arrays::iisph17::Aii(), args...);
	fn(arrays::iisph17::predictedAcceleration(), args...);
	fn(arrays::iisph17::volumeError(), args...);
	fn(arrays::iisph17_band_rb::_sourceTerm(), args...);
	fn(arrays::iisph17_band_rb::_omega(), args...);
	fn(arrays::iisph17_band_rb::_apparentVolume(), args...);
	fn(arrays::iisph17_band_rb::_Aii(), args...);
	fn(arrays::iisph17_band_rb::_predictedAcceleration(), args...);
	fn(arrays::iisph17_band_rb::_volumeError(), args...);
	fn(arrays::structureArrays::compactCellList(), args...);
	fn(arrays::structureArrays::neighborMask(), args...);
	fn(arrays::structureArrays::compactCellScale(), args...);
	fn(arrays::structureArrays::spanNeighborList(), args...);
	fn(arrays::structureArrays::neighborList(), args...);
	fn(arrays::structureArrays::neighborListLength(), args...);
	fn(arrays::renderArrays::anisotropicMatrices(), args...);
	fn(arrays::renderArrays::centerPosition(), args...);
	fn(arrays::renderArrays::auxTest(), args...);
	fn(arrays::renderArrays::auxIsoDensity(), args...);
	fn(arrays::renderArrays::auxDistance(), args...);
	fn(arrays::renderArrays::auxHashMap(), args...);
	fn(arrays::renderArrays::auxCellSpan(), args...);
	fn(arrays::renderArrays::auxCellInformation(), args...);
	fn(arrays::renderArrays::auxCellSurface(), args...);
	fn(arrays::structureArrays::classification(), args...);
	fn(arrays::structureArrays::resortIndex(), args...);
	fn(arrays::structureArrays::ZOrder_64(), args...);
	fn(arrays::structureArrays::ZOrder_32(), args...);
	fn(arrays::structureArrays::cellSpanSwap(), args...);
	fn(arrays::structureArrays::cellSpan(), args...);
	fn(arrays::structureArrays::compactCellSpanSwap(), args...);
	fn(arrays::structureArrays::compactCellSpan(), args...);
	fn(arrays::structureArrays::MLMResolution(), args...);
	fn(arrays::structureArrays::cellparticleIndex(), args...);
	fn(arrays::structureArrays::compactparticleIndex(), args...);
	fn(arrays::structureArrays::particleparticleIndex(), args...);
	fn(arrays::supportArrays::support(), args...);
	fn(arrays::supportArrays::supportEstimate(), args...);
	fn(arrays::supportArrays::neighborCount(), args...);
	fn(arrays::supportArrays::supportMarker(), args...);
	fn(arrays::supportArrays::supportMarkerCompacted(), args...);
	fn(arrays::supportArrays::closestNeighbor(), args...);
	fn(arrays::supportArrays::closestNeighbor_f(), args...);
	fn(arrays::supportArrays::neighborOverhead(), args...);
	fn(arrays::supportArrays::neighborOverheadCount(), args...);
}

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
struct sortingArray{
	std::pair<arrays::adaptive::splitIndicator::type*, arrays::adaptive::splitIndicator::type*> AdaptiveSplitIndicator;
	std::pair<arrays::adaptive::parentIndex::type*, arrays::adaptive::parentIndex::type*> AdaptiveParentIndex;
	std::pair<arrays::adaptive::parentVolume::type*, arrays::adaptive::parentVolume::type*> AdaptiveParentVolume;
	std::pair<arrays::adaptive::parentPosition::type*, arrays::adaptive::parentPosition::type*> AdaptiveParentPosition;
	std::pair<arrays::advectionArrays::angularVelocity::type*, arrays::advectionArrays::angularVelocity::type*> AdvectionArraysAngularVelocity;
	std::pair<arrays::surfaceArrays::distanceBuffer::type*, arrays::surfaceArrays::distanceBuffer::type*> SurfaceArraysDistanceBuffer;
	std::pair<arrays::basicArrays::position::type*, arrays::basicArrays::position::type*> BasicArraysPosition;
	std::pair<arrays::basicArrays::acceleration::type*, arrays::basicArrays::acceleration::type*> BasicArraysAcceleration;
	std::pair<arrays::basicArrays::velocity::type*, arrays::basicArrays::velocity::type*> BasicArraysVelocity;
	std::pair<arrays::basicArrays::particle_type::type*, arrays::basicArrays::particle_type::type*> BasicArraysParticleType;
	std::pair<arrays::basicArrays::renderArray::type*, arrays::basicArrays::renderArray::type*> BasicArraysRenderArray;
	std::pair<arrays::basicArrays::debugArray::type*, arrays::basicArrays::debugArray::type*> BasicArraysDebugArray;
	std::pair<arrays::basicArrays::volume::type*, arrays::basicArrays::volume::type*> BasicArraysVolume;
	std::pair<arrays::basicArrays::lifetime::type*, arrays::basicArrays::lifetime::type*> BasicArraysLifetime;
	std::pair<arrays::basicArrays::pressure::type*, arrays::basicArrays::pressure::type*> BasicArraysPressure;
	std::pair<arrays::basicArrays::density::type*, arrays::basicArrays::density::type*> BasicArraysDensity;
	std::pair<arrays::structureArrays::neighborListLength::type*, arrays::structureArrays::neighborListLength::type*> StructureArraysNeighborListLength;

	hostInline void fillArray(){
		AdaptiveSplitIndicator = std::make_pair( arrays::adaptive::splitIndicator::ptr, arrays::adaptive::splitIndicator::rear_ptr);
		AdaptiveParentIndex = std::make_pair( arrays::adaptive::parentIndex::ptr, arrays::adaptive::parentIndex::rear_ptr);
		AdaptiveParentVolume = std::make_pair( arrays::adaptive::parentVolume::ptr, arrays::adaptive::parentVolume::rear_ptr);
		AdaptiveParentPosition = std::make_pair( arrays::adaptive::parentPosition::ptr, arrays::adaptive::parentPosition::rear_ptr);
		AdvectionArraysAngularVelocity = std::make_pair( arrays::advectionArrays::angularVelocity::ptr, arrays::advectionArrays::angularVelocity::rear_ptr);
		SurfaceArraysDistanceBuffer = std::make_pair( arrays::surfaceArrays::distanceBuffer::ptr, arrays::surfaceArrays::distanceBuffer::rear_ptr);
		BasicArraysPosition = std::make_pair( arrays::basicArrays::position::ptr, arrays::basicArrays::position::rear_ptr);
		BasicArraysAcceleration = std::make_pair( arrays::basicArrays::acceleration::ptr, arrays::basicArrays::acceleration::rear_ptr);
		BasicArraysVelocity = std::make_pair( arrays::basicArrays::velocity::ptr, arrays::basicArrays::velocity::rear_ptr);
		BasicArraysParticleType = std::make_pair( arrays::basicArrays::particle_type::ptr, arrays::basicArrays::particle_type::rear_ptr);
		BasicArraysRenderArray = std::make_pair( arrays::basicArrays::renderArray::ptr, arrays::basicArrays::renderArray::rear_ptr);
		BasicArraysDebugArray = std::make_pair( arrays::basicArrays::debugArray::ptr, arrays::basicArrays::debugArray::rear_ptr);
		BasicArraysVolume = std::make_pair( arrays::basicArrays::volume::ptr, arrays::basicArrays::volume::rear_ptr);
		BasicArraysLifetime = std::make_pair( arrays::basicArrays::lifetime::ptr, arrays::basicArrays::lifetime::rear_ptr);
		BasicArraysPressure = std::make_pair( arrays::basicArrays::pressure::ptr, arrays::basicArrays::pressure::rear_ptr);
		BasicArraysDensity = std::make_pair( arrays::basicArrays::density::ptr, arrays::basicArrays::density::rear_ptr);
		StructureArraysNeighborListLength = std::make_pair( arrays::structureArrays::neighborListLength::ptr, arrays::structureArrays::neighborListLength::rear_ptr);
	}

	template<typename C> hostDeviceInline void callOnArray(C func){
		func(AdaptiveSplitIndicator.first, AdaptiveSplitIndicator.second);
		func(AdaptiveParentIndex.first, AdaptiveParentIndex.second);
		func(AdaptiveParentVolume.first, AdaptiveParentVolume.second);
		func(AdaptiveParentPosition.first, AdaptiveParentPosition.second);
		func(AdvectionArraysAngularVelocity.first, AdvectionArraysAngularVelocity.second);
		func(SurfaceArraysDistanceBuffer.first, SurfaceArraysDistanceBuffer.second);
		func(BasicArraysPosition.first, BasicArraysPosition.second);
		func(BasicArraysAcceleration.first, BasicArraysAcceleration.second);
		func(BasicArraysVelocity.first, BasicArraysVelocity.second);
		func(BasicArraysParticleType.first, BasicArraysParticleType.second);
		func(BasicArraysRenderArray.first, BasicArraysRenderArray.second);
		func(BasicArraysDebugArray.first, BasicArraysDebugArray.second);
		func(BasicArraysVolume.first, BasicArraysVolume.second);
		func(BasicArraysLifetime.first, BasicArraysLifetime.second);
		func(BasicArraysPressure.first, BasicArraysPressure.second);
		func(BasicArraysDensity.first, BasicArraysDensity.second);
		func(StructureArraysNeighborListLength.first, StructureArraysNeighborListLength.second);
	}
};
auto inline getArrayQualifiedName(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return arrays::adaptive::mergeable::qualifiedName;
		case array_enum::adaptive_mergeCounter: return arrays::adaptive::mergeCounter::qualifiedName;
		case array_enum::adaptive_ptclCounter: return arrays::adaptive::ptclCounter::qualifiedName;
		case array_enum::adaptive_classification: return arrays::adaptive::classification::qualifiedName;
		case array_enum::adaptive_splitIndicator: return arrays::adaptive::splitIndicator::qualifiedName;
		case array_enum::adaptive_adaptivityCounter: return arrays::adaptive::adaptivityCounter::qualifiedName;
		case array_enum::adaptive_splitIndicatorCompacted: return arrays::adaptive::splitIndicatorCompacted::qualifiedName;
		case array_enum::adaptive_parentIndex: return arrays::adaptive::parentIndex::qualifiedName;
		case array_enum::adaptive_parentVolume: return arrays::adaptive::parentVolume::qualifiedName;
		case array_enum::adaptive_parentPosition: return arrays::adaptive::parentPosition::qualifiedName;
		case array_enum::advectionArrays_maxVelocity: return arrays::advectionArrays::maxVelocity::qualifiedName;
		case array_enum::advectionArrays_cflValue: return arrays::advectionArrays::cflValue::qualifiedName;
		case array_enum::advectionArrays_particleNormal: return arrays::advectionArrays::particleNormal::qualifiedName;
		case array_enum::advectionArrays_angularVelocity: return arrays::advectionArrays::angularVelocity::qualifiedName;
		case array_enum::rigidBodyArrays_rigidDensities: return arrays::rigidBodyArrays::rigidDensities::qualifiedName;
		case array_enum::rigidBodyArrays_rigidVolumes: return arrays::rigidBodyArrays::rigidVolumes::qualifiedName;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return arrays::rigidBodyArrays::rigidLinearVelocities::qualifiedName;
		case array_enum::rigidBodyArrays_rigidAVelocities: return arrays::rigidBodyArrays::rigidAVelocities::qualifiedName;
		case array_enum::rigidBodyArrays_rigidOrigins: return arrays::rigidBodyArrays::rigidOrigins::qualifiedName;
		case array_enum::rigidBodyArrays_rigidQuaternions: return arrays::rigidBodyArrays::rigidQuaternions::qualifiedName;
		case array_enum::boundaryArrays_splineLUT: return arrays::boundaryArrays::splineLUT::qualifiedName;
		case array_enum::boundaryArrays_offsetLUT: return arrays::boundaryArrays::offsetLUT::qualifiedName;
		case array_enum::boundaryArrays_spikyLUT: return arrays::boundaryArrays::spikyLUT::qualifiedName;
		case array_enum::boundaryArrays_adhesionLUT: return arrays::boundaryArrays::adhesionLUT::qualifiedName;
		case array_enum::boundaryArrays_volumeLUT: return arrays::boundaryArrays::volumeLUT::qualifiedName;
		case array_enum::boundaryArrays_spline2LUT: return arrays::boundaryArrays::spline2LUT::qualifiedName;
		case array_enum::boundaryArrays_splineGradientLUT: return arrays::boundaryArrays::splineGradientLUT::qualifiedName;
		case array_enum::boundaryArrays_spikyGradientLUT: return arrays::boundaryArrays::spikyGradientLUT::qualifiedName;
		case array_enum::boundaryArrays_cohesionLUT: return arrays::boundaryArrays::cohesionLUT::qualifiedName;
		case array_enum::boundaryArrays_boundaryPlanes: return arrays::boundaryArrays::boundaryPlanes::qualifiedName;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return arrays::boundaryArrays::boundaryPlaneVelocity::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return arrays::boundaryArrays::volumeBoundaryVolumes::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return arrays::boundaryArrays::volumeBoundaryDimensions::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryMin: return arrays::boundaryArrays::volumeBoundaryMin::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryMax: return arrays::boundaryArrays::volumeBoundaryMax::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return arrays::boundaryArrays::volumeBoundaryDensity::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return arrays::boundaryArrays::volumeBoundaryVolume::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return arrays::boundaryArrays::volumeBoundaryVelocity::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return arrays::boundaryArrays::volumeBoundaryAngularVelocity::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryKind: return arrays::boundaryArrays::volumeBoundaryKind::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return arrays::boundaryArrays::volumeBoundaryPosition::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return arrays::boundaryArrays::volumeBoundaryQuaternion::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return arrays::boundaryArrays::volumeBoundaryTransformMatrix::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return arrays::boundaryArrays::volumeBoundaryInertiaMatrix::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return arrays::boundaryArrays::volumeBoundaryAcceleration::qualifiedName;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return arrays::boundaryArrays::volumeBoundaryAngularAcceleration::qualifiedName;
		case array_enum::surfaceArrays_decisionBuffer: return arrays::surfaceArrays::decisionBuffer::qualifiedName;
		case array_enum::surfaceArrays_surface_idxBuffer: return arrays::surfaceArrays::surface_idxBuffer::qualifiedName;
		case array_enum::surfaceArrays_markerBuffer: return arrays::surfaceArrays::markerBuffer::qualifiedName;
		case array_enum::surfaceArrays_distanceBuffer: return arrays::surfaceArrays::distanceBuffer::qualifiedName;
		case array_enum::surfaceArrays_changeBuffer: return arrays::surfaceArrays::changeBuffer::qualifiedName;
		case array_enum::basicArrays_matrixTest: return arrays::basicArrays::matrixTest::qualifiedName;
		case array_enum::basicArrays_position: return arrays::basicArrays::position::qualifiedName;
		case array_enum::basicArrays_acceleration: return arrays::basicArrays::acceleration::qualifiedName;
		case array_enum::basicArrays_velocity: return arrays::basicArrays::velocity::qualifiedName;
		case array_enum::basicArrays_particle_type: return arrays::basicArrays::particle_type::qualifiedName;
		case array_enum::basicArrays_renderArray: return arrays::basicArrays::renderArray::qualifiedName;
		case array_enum::basicArrays_debugArray: return arrays::basicArrays::debugArray::qualifiedName;
		case array_enum::basicArrays_volume: return arrays::basicArrays::volume::qualifiedName;
		case array_enum::basicArrays_lifetime: return arrays::basicArrays::lifetime::qualifiedName;
		case array_enum::basicArrays_pressure: return arrays::basicArrays::pressure::qualifiedName;
		case array_enum::basicArrays_density: return arrays::basicArrays::density::qualifiedName;
		case array_enum::basicArrays_fluidDensity: return arrays::basicArrays::fluidDensity::qualifiedName;
		case array_enum::basicArrays_particleIndex: return arrays::basicArrays::particleIndex::qualifiedName;
		case array_enum::basicArrays_particleIndexCompact: return arrays::basicArrays::particleIndexCompact::qualifiedName;
		case array_enum::basicArrays_resortArray4: return arrays::basicArrays::resortArray4::qualifiedName;
		case array_enum::basicArrays_resortArray: return arrays::basicArrays::resortArray::qualifiedName;
		case array_enum::dfsphArrays_dfsphSource: return arrays::dfsphArrays::dfsphSource::qualifiedName;
		case array_enum::dfsphArrays_dfsphAlpha: return arrays::dfsphArrays::dfsphAlpha::qualifiedName;
		case array_enum::dfsphArrays_dfsphDpDt: return arrays::dfsphArrays::dfsphDpDt::qualifiedName;
		case array_enum::dfsphArrays_dfsphRhoStar: return arrays::dfsphArrays::dfsphRhoStar::qualifiedName;
		case array_enum::iisphArrays_iisphSum: return arrays::iisphArrays::iisphSum::qualifiedName;
		case array_enum::iisphArrays_iisphDii: return arrays::iisphArrays::iisphDii::qualifiedName;
		case array_enum::iisphArrays_iisphAiiOld: return arrays::iisphArrays::iisphAiiOld::qualifiedName;
		case array_enum::iisphArrays_omega: return arrays::iisphArrays::omega::qualifiedName;
		case array_enum::iisphArrays_alpha: return arrays::iisphArrays::alpha::qualifiedName;
		case array_enum::iisphArrays_gamma: return arrays::iisphArrays::gamma::qualifiedName;
		case array_enum::iisphArrays_sigma: return arrays::iisphArrays::sigma::qualifiedName;
		case array_enum::iisphArrays_iisphDensityAdvection: return arrays::iisphArrays::iisphDensityAdvection::qualifiedName;
		case array_enum::iisphArrays_iisphDensityIteration: return arrays::iisphArrays::iisphDensityIteration::qualifiedName;
		case array_enum::iisphArrays_kernelBuffer: return arrays::iisphArrays::kernelBuffer::qualifiedName;
		case array_enum::iisphArrays_velocityAdvection: return arrays::iisphArrays::velocityAdvection::qualifiedName;
		case array_enum::iisph17_sourceTerm: return arrays::iisph17::sourceTerm::qualifiedName;
		case array_enum::iisph17_boundaryPressure: return arrays::iisph17::boundaryPressure::qualifiedName;
		case array_enum::iisph17_omega: return arrays::iisph17::omega::qualifiedName;
		case array_enum::iisph17_apparentVolume: return arrays::iisph17::apparentVolume::qualifiedName;
		case array_enum::iisph17_Aii: return arrays::iisph17::Aii::qualifiedName;
		case array_enum::iisph17_predictedAcceleration: return arrays::iisph17::predictedAcceleration::qualifiedName;
		case array_enum::iisph17_volumeError: return arrays::iisph17::volumeError::qualifiedName;
		case array_enum::iisph17_band_rb__sourceTerm: return arrays::iisph17_band_rb::_sourceTerm::qualifiedName;
		case array_enum::iisph17_band_rb__omega: return arrays::iisph17_band_rb::_omega::qualifiedName;
		case array_enum::iisph17_band_rb__apparentVolume: return arrays::iisph17_band_rb::_apparentVolume::qualifiedName;
		case array_enum::iisph17_band_rb__Aii: return arrays::iisph17_band_rb::_Aii::qualifiedName;
		case array_enum::iisph17_band_rb__predictedAcceleration: return arrays::iisph17_band_rb::_predictedAcceleration::qualifiedName;
		case array_enum::iisph17_band_rb__volumeError: return arrays::iisph17_band_rb::_volumeError::qualifiedName;
		case array_enum::volumeInletArrays_inletPositions: return arrays::volumeInletArrays::inletPositions::qualifiedName;
		case array_enum::volumeInletArrays_inletCounter: return arrays::volumeInletArrays::inletCounter::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return arrays::volumeInletArrays::volumeOutletVolumes::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return arrays::volumeInletArrays::volumeOutletDimensions::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletMin: return arrays::volumeInletArrays::volumeOutletMin::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletMax: return arrays::volumeInletArrays::volumeOutletMax::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletRate: return arrays::volumeInletArrays::volumeOutletRate::qualifiedName;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return arrays::volumeInletArrays::volumeOutletRateAccumulator::qualifiedName;
		case array_enum::structureArrays_compactCellList: return arrays::structureArrays::compactCellList::qualifiedName;
		case array_enum::structureArrays_neighborMask: return arrays::structureArrays::neighborMask::qualifiedName;
		case array_enum::structureArrays_compactCellScale: return arrays::structureArrays::compactCellScale::qualifiedName;
		case array_enum::structureArrays_spanNeighborList: return arrays::structureArrays::spanNeighborList::qualifiedName;
		case array_enum::structureArrays_neighborList: return arrays::structureArrays::neighborList::qualifiedName;
		case array_enum::structureArrays_neighborListLength: return arrays::structureArrays::neighborListLength::qualifiedName;
		case array_enum::renderArrays_anisotropicMatrices: return arrays::renderArrays::anisotropicMatrices::qualifiedName;
		case array_enum::renderArrays_centerPosition: return arrays::renderArrays::centerPosition::qualifiedName;
		case array_enum::renderArrays_auxTest: return arrays::renderArrays::auxTest::qualifiedName;
		case array_enum::renderArrays_auxIsoDensity: return arrays::renderArrays::auxIsoDensity::qualifiedName;
		case array_enum::renderArrays_auxDistance: return arrays::renderArrays::auxDistance::qualifiedName;
		case array_enum::renderArrays_auxHashMap: return arrays::renderArrays::auxHashMap::qualifiedName;
		case array_enum::renderArrays_auxCellSpan: return arrays::renderArrays::auxCellSpan::qualifiedName;
		case array_enum::renderArrays_auxCellInformation: return arrays::renderArrays::auxCellInformation::qualifiedName;
		case array_enum::renderArrays_auxCellSurface: return arrays::renderArrays::auxCellSurface::qualifiedName;
		case array_enum::structureArrays_classification: return arrays::structureArrays::classification::qualifiedName;
		case array_enum::structureArrays_resortIndex: return arrays::structureArrays::resortIndex::qualifiedName;
		case array_enum::structureArrays_ZOrder_64: return arrays::structureArrays::ZOrder_64::qualifiedName;
		case array_enum::structureArrays_ZOrder_32: return arrays::structureArrays::ZOrder_32::qualifiedName;
		case array_enum::structureArrays_cellSpanSwap: return arrays::structureArrays::cellSpanSwap::qualifiedName;
		case array_enum::structureArrays_cellSpan: return arrays::structureArrays::cellSpan::qualifiedName;
		case array_enum::structureArrays_compactCellSpanSwap: return arrays::structureArrays::compactCellSpanSwap::qualifiedName;
		case array_enum::structureArrays_compactCellSpan: return arrays::structureArrays::compactCellSpan::qualifiedName;
		case array_enum::structureArrays_MLMResolution: return arrays::structureArrays::MLMResolution::qualifiedName;
		case array_enum::structureArrays_cellparticleIndex: return arrays::structureArrays::cellparticleIndex::qualifiedName;
		case array_enum::structureArrays_compactparticleIndex: return arrays::structureArrays::compactparticleIndex::qualifiedName;
		case array_enum::structureArrays_hashMap: return arrays::structureArrays::hashMap::qualifiedName;
		case array_enum::structureArrays_compactHashMap: return arrays::structureArrays::compactHashMap::qualifiedName;
		case array_enum::structureArrays_mlmScaling: return arrays::structureArrays::mlmScaling::qualifiedName;
		case array_enum::structureArrays_particleparticleIndex: return arrays::structureArrays::particleparticleIndex::qualifiedName;
		case array_enum::structureArrays_cellBegin: return arrays::structureArrays::cellBegin::qualifiedName;
		case array_enum::structureArrays_cellEnd: return arrays::structureArrays::cellEnd::qualifiedName;
		case array_enum::supportArrays_support: return arrays::supportArrays::support::qualifiedName;
		case array_enum::supportArrays_supportEstimate: return arrays::supportArrays::supportEstimate::qualifiedName;
		case array_enum::supportArrays_neighborCount: return arrays::supportArrays::neighborCount::qualifiedName;
		case array_enum::supportArrays_supportMarker: return arrays::supportArrays::supportMarker::qualifiedName;
		case array_enum::supportArrays_supportMarkerCompacted: return arrays::supportArrays::supportMarkerCompacted::qualifiedName;
		case array_enum::supportArrays_closestNeighbor: return arrays::supportArrays::closestNeighbor::qualifiedName;
		case array_enum::supportArrays_closestNeighbor_f: return arrays::supportArrays::closestNeighbor_f::qualifiedName;
		case array_enum::supportArrays_neighborOverhead: return arrays::supportArrays::neighborOverhead::qualifiedName;
		case array_enum::supportArrays_neighborOverheadCount: return arrays::supportArrays::neighborOverheadCount::qualifiedName;
		case array_enum::supportArrays_neighborListSwap: return arrays::supportArrays::neighborListSwap::qualifiedName;
		default: return "invalidEnum";
	}
}
auto inline getArrayVariableName(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return arrays::adaptive::mergeable::variableName;
		case array_enum::adaptive_mergeCounter: return arrays::adaptive::mergeCounter::variableName;
		case array_enum::adaptive_ptclCounter: return arrays::adaptive::ptclCounter::variableName;
		case array_enum::adaptive_classification: return arrays::adaptive::classification::variableName;
		case array_enum::adaptive_splitIndicator: return arrays::adaptive::splitIndicator::variableName;
		case array_enum::adaptive_adaptivityCounter: return arrays::adaptive::adaptivityCounter::variableName;
		case array_enum::adaptive_splitIndicatorCompacted: return arrays::adaptive::splitIndicatorCompacted::variableName;
		case array_enum::adaptive_parentIndex: return arrays::adaptive::parentIndex::variableName;
		case array_enum::adaptive_parentVolume: return arrays::adaptive::parentVolume::variableName;
		case array_enum::adaptive_parentPosition: return arrays::adaptive::parentPosition::variableName;
		case array_enum::advectionArrays_maxVelocity: return arrays::advectionArrays::maxVelocity::variableName;
		case array_enum::advectionArrays_cflValue: return arrays::advectionArrays::cflValue::variableName;
		case array_enum::advectionArrays_particleNormal: return arrays::advectionArrays::particleNormal::variableName;
		case array_enum::advectionArrays_angularVelocity: return arrays::advectionArrays::angularVelocity::variableName;
		case array_enum::rigidBodyArrays_rigidDensities: return arrays::rigidBodyArrays::rigidDensities::variableName;
		case array_enum::rigidBodyArrays_rigidVolumes: return arrays::rigidBodyArrays::rigidVolumes::variableName;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return arrays::rigidBodyArrays::rigidLinearVelocities::variableName;
		case array_enum::rigidBodyArrays_rigidAVelocities: return arrays::rigidBodyArrays::rigidAVelocities::variableName;
		case array_enum::rigidBodyArrays_rigidOrigins: return arrays::rigidBodyArrays::rigidOrigins::variableName;
		case array_enum::rigidBodyArrays_rigidQuaternions: return arrays::rigidBodyArrays::rigidQuaternions::variableName;
		case array_enum::boundaryArrays_splineLUT: return arrays::boundaryArrays::splineLUT::variableName;
		case array_enum::boundaryArrays_offsetLUT: return arrays::boundaryArrays::offsetLUT::variableName;
		case array_enum::boundaryArrays_spikyLUT: return arrays::boundaryArrays::spikyLUT::variableName;
		case array_enum::boundaryArrays_adhesionLUT: return arrays::boundaryArrays::adhesionLUT::variableName;
		case array_enum::boundaryArrays_volumeLUT: return arrays::boundaryArrays::volumeLUT::variableName;
		case array_enum::boundaryArrays_spline2LUT: return arrays::boundaryArrays::spline2LUT::variableName;
		case array_enum::boundaryArrays_splineGradientLUT: return arrays::boundaryArrays::splineGradientLUT::variableName;
		case array_enum::boundaryArrays_spikyGradientLUT: return arrays::boundaryArrays::spikyGradientLUT::variableName;
		case array_enum::boundaryArrays_cohesionLUT: return arrays::boundaryArrays::cohesionLUT::variableName;
		case array_enum::boundaryArrays_boundaryPlanes: return arrays::boundaryArrays::boundaryPlanes::variableName;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return arrays::boundaryArrays::boundaryPlaneVelocity::variableName;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return arrays::boundaryArrays::volumeBoundaryVolumes::variableName;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return arrays::boundaryArrays::volumeBoundaryDimensions::variableName;
		case array_enum::boundaryArrays_volumeBoundaryMin: return arrays::boundaryArrays::volumeBoundaryMin::variableName;
		case array_enum::boundaryArrays_volumeBoundaryMax: return arrays::boundaryArrays::volumeBoundaryMax::variableName;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return arrays::boundaryArrays::volumeBoundaryDensity::variableName;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return arrays::boundaryArrays::volumeBoundaryVolume::variableName;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return arrays::boundaryArrays::volumeBoundaryVelocity::variableName;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return arrays::boundaryArrays::volumeBoundaryAngularVelocity::variableName;
		case array_enum::boundaryArrays_volumeBoundaryKind: return arrays::boundaryArrays::volumeBoundaryKind::variableName;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return arrays::boundaryArrays::volumeBoundaryPosition::variableName;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return arrays::boundaryArrays::volumeBoundaryQuaternion::variableName;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return arrays::boundaryArrays::volumeBoundaryTransformMatrix::variableName;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::variableName;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return arrays::boundaryArrays::volumeBoundaryInertiaMatrix::variableName;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::variableName;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return arrays::boundaryArrays::volumeBoundaryAcceleration::variableName;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return arrays::boundaryArrays::volumeBoundaryAngularAcceleration::variableName;
		case array_enum::surfaceArrays_decisionBuffer: return arrays::surfaceArrays::decisionBuffer::variableName;
		case array_enum::surfaceArrays_surface_idxBuffer: return arrays::surfaceArrays::surface_idxBuffer::variableName;
		case array_enum::surfaceArrays_markerBuffer: return arrays::surfaceArrays::markerBuffer::variableName;
		case array_enum::surfaceArrays_distanceBuffer: return arrays::surfaceArrays::distanceBuffer::variableName;
		case array_enum::surfaceArrays_changeBuffer: return arrays::surfaceArrays::changeBuffer::variableName;
		case array_enum::basicArrays_matrixTest: return arrays::basicArrays::matrixTest::variableName;
		case array_enum::basicArrays_position: return arrays::basicArrays::position::variableName;
		case array_enum::basicArrays_acceleration: return arrays::basicArrays::acceleration::variableName;
		case array_enum::basicArrays_velocity: return arrays::basicArrays::velocity::variableName;
		case array_enum::basicArrays_particle_type: return arrays::basicArrays::particle_type::variableName;
		case array_enum::basicArrays_renderArray: return arrays::basicArrays::renderArray::variableName;
		case array_enum::basicArrays_debugArray: return arrays::basicArrays::debugArray::variableName;
		case array_enum::basicArrays_volume: return arrays::basicArrays::volume::variableName;
		case array_enum::basicArrays_lifetime: return arrays::basicArrays::lifetime::variableName;
		case array_enum::basicArrays_pressure: return arrays::basicArrays::pressure::variableName;
		case array_enum::basicArrays_density: return arrays::basicArrays::density::variableName;
		case array_enum::basicArrays_fluidDensity: return arrays::basicArrays::fluidDensity::variableName;
		case array_enum::basicArrays_particleIndex: return arrays::basicArrays::particleIndex::variableName;
		case array_enum::basicArrays_particleIndexCompact: return arrays::basicArrays::particleIndexCompact::variableName;
		case array_enum::basicArrays_resortArray4: return arrays::basicArrays::resortArray4::variableName;
		case array_enum::basicArrays_resortArray: return arrays::basicArrays::resortArray::variableName;
		case array_enum::dfsphArrays_dfsphSource: return arrays::dfsphArrays::dfsphSource::variableName;
		case array_enum::dfsphArrays_dfsphAlpha: return arrays::dfsphArrays::dfsphAlpha::variableName;
		case array_enum::dfsphArrays_dfsphDpDt: return arrays::dfsphArrays::dfsphDpDt::variableName;
		case array_enum::dfsphArrays_dfsphRhoStar: return arrays::dfsphArrays::dfsphRhoStar::variableName;
		case array_enum::iisphArrays_iisphSum: return arrays::iisphArrays::iisphSum::variableName;
		case array_enum::iisphArrays_iisphDii: return arrays::iisphArrays::iisphDii::variableName;
		case array_enum::iisphArrays_iisphAiiOld: return arrays::iisphArrays::iisphAiiOld::variableName;
		case array_enum::iisphArrays_omega: return arrays::iisphArrays::omega::variableName;
		case array_enum::iisphArrays_alpha: return arrays::iisphArrays::alpha::variableName;
		case array_enum::iisphArrays_gamma: return arrays::iisphArrays::gamma::variableName;
		case array_enum::iisphArrays_sigma: return arrays::iisphArrays::sigma::variableName;
		case array_enum::iisphArrays_iisphDensityAdvection: return arrays::iisphArrays::iisphDensityAdvection::variableName;
		case array_enum::iisphArrays_iisphDensityIteration: return arrays::iisphArrays::iisphDensityIteration::variableName;
		case array_enum::iisphArrays_kernelBuffer: return arrays::iisphArrays::kernelBuffer::variableName;
		case array_enum::iisphArrays_velocityAdvection: return arrays::iisphArrays::velocityAdvection::variableName;
		case array_enum::iisph17_sourceTerm: return arrays::iisph17::sourceTerm::variableName;
		case array_enum::iisph17_boundaryPressure: return arrays::iisph17::boundaryPressure::variableName;
		case array_enum::iisph17_omega: return arrays::iisph17::omega::variableName;
		case array_enum::iisph17_apparentVolume: return arrays::iisph17::apparentVolume::variableName;
		case array_enum::iisph17_Aii: return arrays::iisph17::Aii::variableName;
		case array_enum::iisph17_predictedAcceleration: return arrays::iisph17::predictedAcceleration::variableName;
		case array_enum::iisph17_volumeError: return arrays::iisph17::volumeError::variableName;
		case array_enum::iisph17_band_rb__sourceTerm: return arrays::iisph17_band_rb::_sourceTerm::variableName;
		case array_enum::iisph17_band_rb__omega: return arrays::iisph17_band_rb::_omega::variableName;
		case array_enum::iisph17_band_rb__apparentVolume: return arrays::iisph17_band_rb::_apparentVolume::variableName;
		case array_enum::iisph17_band_rb__Aii: return arrays::iisph17_band_rb::_Aii::variableName;
		case array_enum::iisph17_band_rb__predictedAcceleration: return arrays::iisph17_band_rb::_predictedAcceleration::variableName;
		case array_enum::iisph17_band_rb__volumeError: return arrays::iisph17_band_rb::_volumeError::variableName;
		case array_enum::volumeInletArrays_inletPositions: return arrays::volumeInletArrays::inletPositions::variableName;
		case array_enum::volumeInletArrays_inletCounter: return arrays::volumeInletArrays::inletCounter::variableName;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return arrays::volumeInletArrays::volumeOutletVolumes::variableName;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return arrays::volumeInletArrays::volumeOutletDimensions::variableName;
		case array_enum::volumeInletArrays_volumeOutletMin: return arrays::volumeInletArrays::volumeOutletMin::variableName;
		case array_enum::volumeInletArrays_volumeOutletMax: return arrays::volumeInletArrays::volumeOutletMax::variableName;
		case array_enum::volumeInletArrays_volumeOutletRate: return arrays::volumeInletArrays::volumeOutletRate::variableName;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return arrays::volumeInletArrays::volumeOutletRateAccumulator::variableName;
		case array_enum::structureArrays_compactCellList: return arrays::structureArrays::compactCellList::variableName;
		case array_enum::structureArrays_neighborMask: return arrays::structureArrays::neighborMask::variableName;
		case array_enum::structureArrays_compactCellScale: return arrays::structureArrays::compactCellScale::variableName;
		case array_enum::structureArrays_spanNeighborList: return arrays::structureArrays::spanNeighborList::variableName;
		case array_enum::structureArrays_neighborList: return arrays::structureArrays::neighborList::variableName;
		case array_enum::structureArrays_neighborListLength: return arrays::structureArrays::neighborListLength::variableName;
		case array_enum::renderArrays_anisotropicMatrices: return arrays::renderArrays::anisotropicMatrices::variableName;
		case array_enum::renderArrays_centerPosition: return arrays::renderArrays::centerPosition::variableName;
		case array_enum::renderArrays_auxTest: return arrays::renderArrays::auxTest::variableName;
		case array_enum::renderArrays_auxIsoDensity: return arrays::renderArrays::auxIsoDensity::variableName;
		case array_enum::renderArrays_auxDistance: return arrays::renderArrays::auxDistance::variableName;
		case array_enum::renderArrays_auxHashMap: return arrays::renderArrays::auxHashMap::variableName;
		case array_enum::renderArrays_auxCellSpan: return arrays::renderArrays::auxCellSpan::variableName;
		case array_enum::renderArrays_auxCellInformation: return arrays::renderArrays::auxCellInformation::variableName;
		case array_enum::renderArrays_auxCellSurface: return arrays::renderArrays::auxCellSurface::variableName;
		case array_enum::structureArrays_classification: return arrays::structureArrays::classification::variableName;
		case array_enum::structureArrays_resortIndex: return arrays::structureArrays::resortIndex::variableName;
		case array_enum::structureArrays_ZOrder_64: return arrays::structureArrays::ZOrder_64::variableName;
		case array_enum::structureArrays_ZOrder_32: return arrays::structureArrays::ZOrder_32::variableName;
		case array_enum::structureArrays_cellSpanSwap: return arrays::structureArrays::cellSpanSwap::variableName;
		case array_enum::structureArrays_cellSpan: return arrays::structureArrays::cellSpan::variableName;
		case array_enum::structureArrays_compactCellSpanSwap: return arrays::structureArrays::compactCellSpanSwap::variableName;
		case array_enum::structureArrays_compactCellSpan: return arrays::structureArrays::compactCellSpan::variableName;
		case array_enum::structureArrays_MLMResolution: return arrays::structureArrays::MLMResolution::variableName;
		case array_enum::structureArrays_cellparticleIndex: return arrays::structureArrays::cellparticleIndex::variableName;
		case array_enum::structureArrays_compactparticleIndex: return arrays::structureArrays::compactparticleIndex::variableName;
		case array_enum::structureArrays_hashMap: return arrays::structureArrays::hashMap::variableName;
		case array_enum::structureArrays_compactHashMap: return arrays::structureArrays::compactHashMap::variableName;
		case array_enum::structureArrays_mlmScaling: return arrays::structureArrays::mlmScaling::variableName;
		case array_enum::structureArrays_particleparticleIndex: return arrays::structureArrays::particleparticleIndex::variableName;
		case array_enum::structureArrays_cellBegin: return arrays::structureArrays::cellBegin::variableName;
		case array_enum::structureArrays_cellEnd: return arrays::structureArrays::cellEnd::variableName;
		case array_enum::supportArrays_support: return arrays::supportArrays::support::variableName;
		case array_enum::supportArrays_supportEstimate: return arrays::supportArrays::supportEstimate::variableName;
		case array_enum::supportArrays_neighborCount: return arrays::supportArrays::neighborCount::variableName;
		case array_enum::supportArrays_supportMarker: return arrays::supportArrays::supportMarker::variableName;
		case array_enum::supportArrays_supportMarkerCompacted: return arrays::supportArrays::supportMarkerCompacted::variableName;
		case array_enum::supportArrays_closestNeighbor: return arrays::supportArrays::closestNeighbor::variableName;
		case array_enum::supportArrays_closestNeighbor_f: return arrays::supportArrays::closestNeighbor_f::variableName;
		case array_enum::supportArrays_neighborOverhead: return arrays::supportArrays::neighborOverhead::variableName;
		case array_enum::supportArrays_neighborOverheadCount: return arrays::supportArrays::neighborOverheadCount::variableName;
		case array_enum::supportArrays_neighborListSwap: return arrays::supportArrays::neighborListSwap::variableName;
		default: return "invalidEnum";
	}
}
auto inline getArrayPtr(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return (void*) arrays::adaptive::mergeable::ptr;
		case array_enum::adaptive_mergeCounter: return (void*) arrays::adaptive::mergeCounter::ptr;
		case array_enum::adaptive_ptclCounter: return (void*) arrays::adaptive::ptclCounter::ptr;
		case array_enum::adaptive_classification: return (void*) arrays::adaptive::classification::ptr;
		case array_enum::adaptive_splitIndicator: return (void*) arrays::adaptive::splitIndicator::ptr;
		case array_enum::adaptive_adaptivityCounter: return (void*) arrays::adaptive::adaptivityCounter::ptr;
		case array_enum::adaptive_splitIndicatorCompacted: return (void*) arrays::adaptive::splitIndicatorCompacted::ptr;
		case array_enum::adaptive_parentIndex: return (void*) arrays::adaptive::parentIndex::ptr;
		case array_enum::adaptive_parentVolume: return (void*) arrays::adaptive::parentVolume::ptr;
		case array_enum::adaptive_parentPosition: return (void*) arrays::adaptive::parentPosition::ptr;
		case array_enum::advectionArrays_maxVelocity: return (void*) arrays::advectionArrays::maxVelocity::ptr;
		case array_enum::advectionArrays_cflValue: return (void*) arrays::advectionArrays::cflValue::ptr;
		case array_enum::advectionArrays_particleNormal: return (void*) arrays::advectionArrays::particleNormal::ptr;
		case array_enum::advectionArrays_angularVelocity: return (void*) arrays::advectionArrays::angularVelocity::ptr;
		case array_enum::rigidBodyArrays_rigidDensities: return (void*) arrays::rigidBodyArrays::rigidDensities::ptr;
		case array_enum::rigidBodyArrays_rigidVolumes: return (void*) arrays::rigidBodyArrays::rigidVolumes::ptr;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return (void*) arrays::rigidBodyArrays::rigidLinearVelocities::ptr;
		case array_enum::rigidBodyArrays_rigidAVelocities: return (void*) arrays::rigidBodyArrays::rigidAVelocities::ptr;
		case array_enum::rigidBodyArrays_rigidOrigins: return (void*) arrays::rigidBodyArrays::rigidOrigins::ptr;
		case array_enum::rigidBodyArrays_rigidQuaternions: return (void*) arrays::rigidBodyArrays::rigidQuaternions::ptr;
		case array_enum::boundaryArrays_splineLUT: return (void*) arrays::boundaryArrays::splineLUT::ptr;
		case array_enum::boundaryArrays_offsetLUT: return (void*) arrays::boundaryArrays::offsetLUT::ptr;
		case array_enum::boundaryArrays_spikyLUT: return (void*) arrays::boundaryArrays::spikyLUT::ptr;
		case array_enum::boundaryArrays_adhesionLUT: return (void*) arrays::boundaryArrays::adhesionLUT::ptr;
		case array_enum::boundaryArrays_volumeLUT: return (void*) arrays::boundaryArrays::volumeLUT::ptr;
		case array_enum::boundaryArrays_spline2LUT: return (void*) arrays::boundaryArrays::spline2LUT::ptr;
		case array_enum::boundaryArrays_splineGradientLUT: return (void*) arrays::boundaryArrays::splineGradientLUT::ptr;
		case array_enum::boundaryArrays_spikyGradientLUT: return (void*) arrays::boundaryArrays::spikyGradientLUT::ptr;
		case array_enum::boundaryArrays_cohesionLUT: return (void*) arrays::boundaryArrays::cohesionLUT::ptr;
		case array_enum::boundaryArrays_boundaryPlanes: return (void*) arrays::boundaryArrays::boundaryPlanes::ptr;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return (void*) arrays::boundaryArrays::boundaryPlaneVelocity::ptr;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return (void*) arrays::boundaryArrays::volumeBoundaryVolumes::ptr;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return (void*) arrays::boundaryArrays::volumeBoundaryDimensions::ptr;
		case array_enum::boundaryArrays_volumeBoundaryMin: return (void*) arrays::boundaryArrays::volumeBoundaryMin::ptr;
		case array_enum::boundaryArrays_volumeBoundaryMax: return (void*) arrays::boundaryArrays::volumeBoundaryMax::ptr;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return (void*) arrays::boundaryArrays::volumeBoundaryDensity::ptr;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return (void*) arrays::boundaryArrays::volumeBoundaryVolume::ptr;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return (void*) arrays::boundaryArrays::volumeBoundaryVelocity::ptr;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return (void*) arrays::boundaryArrays::volumeBoundaryAngularVelocity::ptr;
		case array_enum::boundaryArrays_volumeBoundaryKind: return (void*) arrays::boundaryArrays::volumeBoundaryKind::ptr;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return (void*) arrays::boundaryArrays::volumeBoundaryPosition::ptr;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return (void*) arrays::boundaryArrays::volumeBoundaryQuaternion::ptr;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return (void*) arrays::boundaryArrays::volumeBoundaryTransformMatrix::ptr;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return (void*) arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::ptr;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return (void*) arrays::boundaryArrays::volumeBoundaryInertiaMatrix::ptr;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return (void*) arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::ptr;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return (void*) arrays::boundaryArrays::volumeBoundaryAcceleration::ptr;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return (void*) arrays::boundaryArrays::volumeBoundaryAngularAcceleration::ptr;
		case array_enum::surfaceArrays_decisionBuffer: return (void*) arrays::surfaceArrays::decisionBuffer::ptr;
		case array_enum::surfaceArrays_surface_idxBuffer: return (void*) arrays::surfaceArrays::surface_idxBuffer::ptr;
		case array_enum::surfaceArrays_markerBuffer: return (void*) arrays::surfaceArrays::markerBuffer::ptr;
		case array_enum::surfaceArrays_distanceBuffer: return (void*) arrays::surfaceArrays::distanceBuffer::ptr;
		case array_enum::surfaceArrays_changeBuffer: return (void*) arrays::surfaceArrays::changeBuffer::ptr;
		case array_enum::basicArrays_matrixTest: return (void*) arrays::basicArrays::matrixTest::ptr;
		case array_enum::basicArrays_position: return (void*) arrays::basicArrays::position::ptr;
		case array_enum::basicArrays_acceleration: return (void*) arrays::basicArrays::acceleration::ptr;
		case array_enum::basicArrays_velocity: return (void*) arrays::basicArrays::velocity::ptr;
		case array_enum::basicArrays_particle_type: return (void*) arrays::basicArrays::particle_type::ptr;
		case array_enum::basicArrays_renderArray: return (void*) arrays::basicArrays::renderArray::ptr;
		case array_enum::basicArrays_debugArray: return (void*) arrays::basicArrays::debugArray::ptr;
		case array_enum::basicArrays_volume: return (void*) arrays::basicArrays::volume::ptr;
		case array_enum::basicArrays_lifetime: return (void*) arrays::basicArrays::lifetime::ptr;
		case array_enum::basicArrays_pressure: return (void*) arrays::basicArrays::pressure::ptr;
		case array_enum::basicArrays_density: return (void*) arrays::basicArrays::density::ptr;
		case array_enum::basicArrays_fluidDensity: return (void*) arrays::basicArrays::fluidDensity::ptr;
		case array_enum::basicArrays_particleIndex: return (void*) arrays::basicArrays::particleIndex::ptr;
		case array_enum::basicArrays_particleIndexCompact: return (void*) arrays::basicArrays::particleIndexCompact::ptr;
		case array_enum::basicArrays_resortArray4: return (void*) arrays::basicArrays::resortArray4::ptr;
		case array_enum::basicArrays_resortArray: return (void*) arrays::basicArrays::resortArray::ptr;
		case array_enum::dfsphArrays_dfsphSource: return (void*) arrays::dfsphArrays::dfsphSource::ptr;
		case array_enum::dfsphArrays_dfsphAlpha: return (void*) arrays::dfsphArrays::dfsphAlpha::ptr;
		case array_enum::dfsphArrays_dfsphDpDt: return (void*) arrays::dfsphArrays::dfsphDpDt::ptr;
		case array_enum::dfsphArrays_dfsphRhoStar: return (void*) arrays::dfsphArrays::dfsphRhoStar::ptr;
		case array_enum::iisphArrays_iisphSum: return (void*) arrays::iisphArrays::iisphSum::ptr;
		case array_enum::iisphArrays_iisphDii: return (void*) arrays::iisphArrays::iisphDii::ptr;
		case array_enum::iisphArrays_iisphAiiOld: return (void*) arrays::iisphArrays::iisphAiiOld::ptr;
		case array_enum::iisphArrays_omega: return (void*) arrays::iisphArrays::omega::ptr;
		case array_enum::iisphArrays_alpha: return (void*) arrays::iisphArrays::alpha::ptr;
		case array_enum::iisphArrays_gamma: return (void*) arrays::iisphArrays::gamma::ptr;
		case array_enum::iisphArrays_sigma: return (void*) arrays::iisphArrays::sigma::ptr;
		case array_enum::iisphArrays_iisphDensityAdvection: return (void*) arrays::iisphArrays::iisphDensityAdvection::ptr;
		case array_enum::iisphArrays_iisphDensityIteration: return (void*) arrays::iisphArrays::iisphDensityIteration::ptr;
		case array_enum::iisphArrays_kernelBuffer: return (void*) arrays::iisphArrays::kernelBuffer::ptr;
		case array_enum::iisphArrays_velocityAdvection: return (void*) arrays::iisphArrays::velocityAdvection::ptr;
		case array_enum::iisph17_sourceTerm: return (void*) arrays::iisph17::sourceTerm::ptr;
		case array_enum::iisph17_boundaryPressure: return (void*) arrays::iisph17::boundaryPressure::ptr;
		case array_enum::iisph17_omega: return (void*) arrays::iisph17::omega::ptr;
		case array_enum::iisph17_apparentVolume: return (void*) arrays::iisph17::apparentVolume::ptr;
		case array_enum::iisph17_Aii: return (void*) arrays::iisph17::Aii::ptr;
		case array_enum::iisph17_predictedAcceleration: return (void*) arrays::iisph17::predictedAcceleration::ptr;
		case array_enum::iisph17_volumeError: return (void*) arrays::iisph17::volumeError::ptr;
		case array_enum::iisph17_band_rb__sourceTerm: return (void*) arrays::iisph17_band_rb::_sourceTerm::ptr;
		case array_enum::iisph17_band_rb__omega: return (void*) arrays::iisph17_band_rb::_omega::ptr;
		case array_enum::iisph17_band_rb__apparentVolume: return (void*) arrays::iisph17_band_rb::_apparentVolume::ptr;
		case array_enum::iisph17_band_rb__Aii: return (void*) arrays::iisph17_band_rb::_Aii::ptr;
		case array_enum::iisph17_band_rb__predictedAcceleration: return (void*) arrays::iisph17_band_rb::_predictedAcceleration::ptr;
		case array_enum::iisph17_band_rb__volumeError: return (void*) arrays::iisph17_band_rb::_volumeError::ptr;
		case array_enum::volumeInletArrays_inletPositions: return (void*) arrays::volumeInletArrays::inletPositions::ptr;
		case array_enum::volumeInletArrays_inletCounter: return (void*) arrays::volumeInletArrays::inletCounter::ptr;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return (void*) arrays::volumeInletArrays::volumeOutletVolumes::ptr;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return (void*) arrays::volumeInletArrays::volumeOutletDimensions::ptr;
		case array_enum::volumeInletArrays_volumeOutletMin: return (void*) arrays::volumeInletArrays::volumeOutletMin::ptr;
		case array_enum::volumeInletArrays_volumeOutletMax: return (void*) arrays::volumeInletArrays::volumeOutletMax::ptr;
		case array_enum::volumeInletArrays_volumeOutletRate: return (void*) arrays::volumeInletArrays::volumeOutletRate::ptr;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return (void*) arrays::volumeInletArrays::volumeOutletRateAccumulator::ptr;
		case array_enum::structureArrays_compactCellList: return (void*) arrays::structureArrays::compactCellList::ptr;
		case array_enum::structureArrays_neighborMask: return (void*) arrays::structureArrays::neighborMask::ptr;
		case array_enum::structureArrays_compactCellScale: return (void*) arrays::structureArrays::compactCellScale::ptr;
		case array_enum::structureArrays_spanNeighborList: return (void*) arrays::structureArrays::spanNeighborList::ptr;
		case array_enum::structureArrays_neighborList: return (void*) arrays::structureArrays::neighborList::ptr;
		case array_enum::structureArrays_neighborListLength: return (void*) arrays::structureArrays::neighborListLength::ptr;
		case array_enum::renderArrays_anisotropicMatrices: return (void*) arrays::renderArrays::anisotropicMatrices::ptr;
		case array_enum::renderArrays_centerPosition: return (void*) arrays::renderArrays::centerPosition::ptr;
		case array_enum::renderArrays_auxTest: return (void*) arrays::renderArrays::auxTest::ptr;
		case array_enum::renderArrays_auxIsoDensity: return (void*) arrays::renderArrays::auxIsoDensity::ptr;
		case array_enum::renderArrays_auxDistance: return (void*) arrays::renderArrays::auxDistance::ptr;
		case array_enum::renderArrays_auxHashMap: return (void*) arrays::renderArrays::auxHashMap::ptr;
		case array_enum::renderArrays_auxCellSpan: return (void*) arrays::renderArrays::auxCellSpan::ptr;
		case array_enum::renderArrays_auxCellInformation: return (void*) arrays::renderArrays::auxCellInformation::ptr;
		case array_enum::renderArrays_auxCellSurface: return (void*) arrays::renderArrays::auxCellSurface::ptr;
		case array_enum::structureArrays_classification: return (void*) arrays::structureArrays::classification::ptr;
		case array_enum::structureArrays_resortIndex: return (void*) arrays::structureArrays::resortIndex::ptr;
		case array_enum::structureArrays_ZOrder_64: return (void*) arrays::structureArrays::ZOrder_64::ptr;
		case array_enum::structureArrays_ZOrder_32: return (void*) arrays::structureArrays::ZOrder_32::ptr;
		case array_enum::structureArrays_cellSpanSwap: return (void*) arrays::structureArrays::cellSpanSwap::ptr;
		case array_enum::structureArrays_cellSpan: return (void*) arrays::structureArrays::cellSpan::ptr;
		case array_enum::structureArrays_compactCellSpanSwap: return (void*) arrays::structureArrays::compactCellSpanSwap::ptr;
		case array_enum::structureArrays_compactCellSpan: return (void*) arrays::structureArrays::compactCellSpan::ptr;
		case array_enum::structureArrays_MLMResolution: return (void*) arrays::structureArrays::MLMResolution::ptr;
		case array_enum::structureArrays_cellparticleIndex: return (void*) arrays::structureArrays::cellparticleIndex::ptr;
		case array_enum::structureArrays_compactparticleIndex: return (void*) arrays::structureArrays::compactparticleIndex::ptr;
		case array_enum::structureArrays_hashMap: return (void*) arrays::structureArrays::hashMap::ptr;
		case array_enum::structureArrays_compactHashMap: return (void*) arrays::structureArrays::compactHashMap::ptr;
		case array_enum::structureArrays_mlmScaling: return (void*) arrays::structureArrays::mlmScaling::ptr;
		case array_enum::structureArrays_particleparticleIndex: return (void*) arrays::structureArrays::particleparticleIndex::ptr;
		case array_enum::structureArrays_cellBegin: return (void*) arrays::structureArrays::cellBegin::ptr;
		case array_enum::structureArrays_cellEnd: return (void*) arrays::structureArrays::cellEnd::ptr;
		case array_enum::supportArrays_support: return (void*) arrays::supportArrays::support::ptr;
		case array_enum::supportArrays_supportEstimate: return (void*) arrays::supportArrays::supportEstimate::ptr;
		case array_enum::supportArrays_neighborCount: return (void*) arrays::supportArrays::neighborCount::ptr;
		case array_enum::supportArrays_supportMarker: return (void*) arrays::supportArrays::supportMarker::ptr;
		case array_enum::supportArrays_supportMarkerCompacted: return (void*) arrays::supportArrays::supportMarkerCompacted::ptr;
		case array_enum::supportArrays_closestNeighbor: return (void*) arrays::supportArrays::closestNeighbor::ptr;
		case array_enum::supportArrays_closestNeighbor_f: return (void*) arrays::supportArrays::closestNeighbor_f::ptr;
		case array_enum::supportArrays_neighborOverhead: return (void*) arrays::supportArrays::neighborOverhead::ptr;
		case array_enum::supportArrays_neighborOverheadCount: return (void*) arrays::supportArrays::neighborOverheadCount::ptr;
		case array_enum::supportArrays_neighborListSwap: return (void*) arrays::supportArrays::neighborListSwap::ptr;
		default: return (void*) nullptr;
	}
}
auto inline getArrayAllocationSize(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return arrays::adaptive::mergeable::alloc_size;
		case array_enum::adaptive_mergeCounter: return arrays::adaptive::mergeCounter::alloc_size;
		case array_enum::adaptive_ptclCounter: return arrays::adaptive::ptclCounter::alloc_size;
		case array_enum::adaptive_classification: return arrays::adaptive::classification::alloc_size;
		case array_enum::adaptive_splitIndicator: return arrays::adaptive::splitIndicator::alloc_size;
		case array_enum::adaptive_adaptivityCounter: return arrays::adaptive::adaptivityCounter::alloc_size;
		case array_enum::adaptive_splitIndicatorCompacted: return arrays::adaptive::splitIndicatorCompacted::alloc_size;
		case array_enum::adaptive_parentIndex: return arrays::adaptive::parentIndex::alloc_size;
		case array_enum::adaptive_parentVolume: return arrays::adaptive::parentVolume::alloc_size;
		case array_enum::adaptive_parentPosition: return arrays::adaptive::parentPosition::alloc_size;
		case array_enum::advectionArrays_maxVelocity: return arrays::advectionArrays::maxVelocity::alloc_size;
		case array_enum::advectionArrays_cflValue: return arrays::advectionArrays::cflValue::alloc_size;
		case array_enum::advectionArrays_particleNormal: return arrays::advectionArrays::particleNormal::alloc_size;
		case array_enum::advectionArrays_angularVelocity: return arrays::advectionArrays::angularVelocity::alloc_size;
		case array_enum::rigidBodyArrays_rigidDensities: return arrays::rigidBodyArrays::rigidDensities::alloc_size;
		case array_enum::rigidBodyArrays_rigidVolumes: return arrays::rigidBodyArrays::rigidVolumes::alloc_size;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return arrays::rigidBodyArrays::rigidLinearVelocities::alloc_size;
		case array_enum::rigidBodyArrays_rigidAVelocities: return arrays::rigidBodyArrays::rigidAVelocities::alloc_size;
		case array_enum::rigidBodyArrays_rigidOrigins: return arrays::rigidBodyArrays::rigidOrigins::alloc_size;
		case array_enum::rigidBodyArrays_rigidQuaternions: return arrays::rigidBodyArrays::rigidQuaternions::alloc_size;
		case array_enum::boundaryArrays_splineLUT: return arrays::boundaryArrays::splineLUT::alloc_size;
		case array_enum::boundaryArrays_offsetLUT: return arrays::boundaryArrays::offsetLUT::alloc_size;
		case array_enum::boundaryArrays_spikyLUT: return arrays::boundaryArrays::spikyLUT::alloc_size;
		case array_enum::boundaryArrays_adhesionLUT: return arrays::boundaryArrays::adhesionLUT::alloc_size;
		case array_enum::boundaryArrays_volumeLUT: return arrays::boundaryArrays::volumeLUT::alloc_size;
		case array_enum::boundaryArrays_spline2LUT: return arrays::boundaryArrays::spline2LUT::alloc_size;
		case array_enum::boundaryArrays_splineGradientLUT: return arrays::boundaryArrays::splineGradientLUT::alloc_size;
		case array_enum::boundaryArrays_spikyGradientLUT: return arrays::boundaryArrays::spikyGradientLUT::alloc_size;
		case array_enum::boundaryArrays_cohesionLUT: return arrays::boundaryArrays::cohesionLUT::alloc_size;
		case array_enum::boundaryArrays_boundaryPlanes: return arrays::boundaryArrays::boundaryPlanes::alloc_size;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return arrays::boundaryArrays::boundaryPlaneVelocity::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return arrays::boundaryArrays::volumeBoundaryVolumes::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return arrays::boundaryArrays::volumeBoundaryDimensions::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryMin: return arrays::boundaryArrays::volumeBoundaryMin::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryMax: return arrays::boundaryArrays::volumeBoundaryMax::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return arrays::boundaryArrays::volumeBoundaryDensity::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return arrays::boundaryArrays::volumeBoundaryVolume::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return arrays::boundaryArrays::volumeBoundaryVelocity::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return arrays::boundaryArrays::volumeBoundaryAngularVelocity::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryKind: return arrays::boundaryArrays::volumeBoundaryKind::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return arrays::boundaryArrays::volumeBoundaryPosition::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return arrays::boundaryArrays::volumeBoundaryQuaternion::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return arrays::boundaryArrays::volumeBoundaryTransformMatrix::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return arrays::boundaryArrays::volumeBoundaryInertiaMatrix::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return arrays::boundaryArrays::volumeBoundaryAcceleration::alloc_size;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return arrays::boundaryArrays::volumeBoundaryAngularAcceleration::alloc_size;
		case array_enum::surfaceArrays_decisionBuffer: return arrays::surfaceArrays::decisionBuffer::alloc_size;
		case array_enum::surfaceArrays_surface_idxBuffer: return arrays::surfaceArrays::surface_idxBuffer::alloc_size;
		case array_enum::surfaceArrays_markerBuffer: return arrays::surfaceArrays::markerBuffer::alloc_size;
		case array_enum::surfaceArrays_distanceBuffer: return arrays::surfaceArrays::distanceBuffer::alloc_size;
		case array_enum::surfaceArrays_changeBuffer: return arrays::surfaceArrays::changeBuffer::alloc_size;
		case array_enum::basicArrays_matrixTest: return arrays::basicArrays::matrixTest::alloc_size;
		case array_enum::basicArrays_position: return arrays::basicArrays::position::alloc_size;
		case array_enum::basicArrays_acceleration: return arrays::basicArrays::acceleration::alloc_size;
		case array_enum::basicArrays_velocity: return arrays::basicArrays::velocity::alloc_size;
		case array_enum::basicArrays_particle_type: return arrays::basicArrays::particle_type::alloc_size;
		case array_enum::basicArrays_renderArray: return arrays::basicArrays::renderArray::alloc_size;
		case array_enum::basicArrays_debugArray: return arrays::basicArrays::debugArray::alloc_size;
		case array_enum::basicArrays_volume: return arrays::basicArrays::volume::alloc_size;
		case array_enum::basicArrays_lifetime: return arrays::basicArrays::lifetime::alloc_size;
		case array_enum::basicArrays_pressure: return arrays::basicArrays::pressure::alloc_size;
		case array_enum::basicArrays_density: return arrays::basicArrays::density::alloc_size;
		case array_enum::basicArrays_fluidDensity: return arrays::basicArrays::fluidDensity::alloc_size;
		case array_enum::basicArrays_particleIndex: return arrays::basicArrays::particleIndex::alloc_size;
		case array_enum::basicArrays_particleIndexCompact: return arrays::basicArrays::particleIndexCompact::alloc_size;
		case array_enum::basicArrays_resortArray4: return arrays::basicArrays::resortArray4::alloc_size;
		case array_enum::basicArrays_resortArray: return arrays::basicArrays::resortArray::alloc_size;
		case array_enum::dfsphArrays_dfsphSource: return arrays::dfsphArrays::dfsphSource::alloc_size;
		case array_enum::dfsphArrays_dfsphAlpha: return arrays::dfsphArrays::dfsphAlpha::alloc_size;
		case array_enum::dfsphArrays_dfsphDpDt: return arrays::dfsphArrays::dfsphDpDt::alloc_size;
		case array_enum::dfsphArrays_dfsphRhoStar: return arrays::dfsphArrays::dfsphRhoStar::alloc_size;
		case array_enum::iisphArrays_iisphSum: return arrays::iisphArrays::iisphSum::alloc_size;
		case array_enum::iisphArrays_iisphDii: return arrays::iisphArrays::iisphDii::alloc_size;
		case array_enum::iisphArrays_iisphAiiOld: return arrays::iisphArrays::iisphAiiOld::alloc_size;
		case array_enum::iisphArrays_omega: return arrays::iisphArrays::omega::alloc_size;
		case array_enum::iisphArrays_alpha: return arrays::iisphArrays::alpha::alloc_size;
		case array_enum::iisphArrays_gamma: return arrays::iisphArrays::gamma::alloc_size;
		case array_enum::iisphArrays_sigma: return arrays::iisphArrays::sigma::alloc_size;
		case array_enum::iisphArrays_iisphDensityAdvection: return arrays::iisphArrays::iisphDensityAdvection::alloc_size;
		case array_enum::iisphArrays_iisphDensityIteration: return arrays::iisphArrays::iisphDensityIteration::alloc_size;
		case array_enum::iisphArrays_kernelBuffer: return arrays::iisphArrays::kernelBuffer::alloc_size;
		case array_enum::iisphArrays_velocityAdvection: return arrays::iisphArrays::velocityAdvection::alloc_size;
		case array_enum::iisph17_sourceTerm: return arrays::iisph17::sourceTerm::alloc_size;
		case array_enum::iisph17_boundaryPressure: return arrays::iisph17::boundaryPressure::alloc_size;
		case array_enum::iisph17_omega: return arrays::iisph17::omega::alloc_size;
		case array_enum::iisph17_apparentVolume: return arrays::iisph17::apparentVolume::alloc_size;
		case array_enum::iisph17_Aii: return arrays::iisph17::Aii::alloc_size;
		case array_enum::iisph17_predictedAcceleration: return arrays::iisph17::predictedAcceleration::alloc_size;
		case array_enum::iisph17_volumeError: return arrays::iisph17::volumeError::alloc_size;
		case array_enum::iisph17_band_rb__sourceTerm: return arrays::iisph17_band_rb::_sourceTerm::alloc_size;
		case array_enum::iisph17_band_rb__omega: return arrays::iisph17_band_rb::_omega::alloc_size;
		case array_enum::iisph17_band_rb__apparentVolume: return arrays::iisph17_band_rb::_apparentVolume::alloc_size;
		case array_enum::iisph17_band_rb__Aii: return arrays::iisph17_band_rb::_Aii::alloc_size;
		case array_enum::iisph17_band_rb__predictedAcceleration: return arrays::iisph17_band_rb::_predictedAcceleration::alloc_size;
		case array_enum::iisph17_band_rb__volumeError: return arrays::iisph17_band_rb::_volumeError::alloc_size;
		case array_enum::volumeInletArrays_inletPositions: return arrays::volumeInletArrays::inletPositions::alloc_size;
		case array_enum::volumeInletArrays_inletCounter: return arrays::volumeInletArrays::inletCounter::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return arrays::volumeInletArrays::volumeOutletVolumes::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return arrays::volumeInletArrays::volumeOutletDimensions::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletMin: return arrays::volumeInletArrays::volumeOutletMin::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletMax: return arrays::volumeInletArrays::volumeOutletMax::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletRate: return arrays::volumeInletArrays::volumeOutletRate::alloc_size;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return arrays::volumeInletArrays::volumeOutletRateAccumulator::alloc_size;
		case array_enum::structureArrays_compactCellList: return arrays::structureArrays::compactCellList::alloc_size;
		case array_enum::structureArrays_neighborMask: return arrays::structureArrays::neighborMask::alloc_size;
		case array_enum::structureArrays_compactCellScale: return arrays::structureArrays::compactCellScale::alloc_size;
		case array_enum::structureArrays_spanNeighborList: return arrays::structureArrays::spanNeighborList::alloc_size;
		case array_enum::structureArrays_neighborList: return arrays::structureArrays::neighborList::alloc_size;
		case array_enum::structureArrays_neighborListLength: return arrays::structureArrays::neighborListLength::alloc_size;
		case array_enum::renderArrays_anisotropicMatrices: return arrays::renderArrays::anisotropicMatrices::alloc_size;
		case array_enum::renderArrays_centerPosition: return arrays::renderArrays::centerPosition::alloc_size;
		case array_enum::renderArrays_auxTest: return arrays::renderArrays::auxTest::alloc_size;
		case array_enum::renderArrays_auxIsoDensity: return arrays::renderArrays::auxIsoDensity::alloc_size;
		case array_enum::renderArrays_auxDistance: return arrays::renderArrays::auxDistance::alloc_size;
		case array_enum::renderArrays_auxHashMap: return arrays::renderArrays::auxHashMap::alloc_size;
		case array_enum::renderArrays_auxCellSpan: return arrays::renderArrays::auxCellSpan::alloc_size;
		case array_enum::renderArrays_auxCellInformation: return arrays::renderArrays::auxCellInformation::alloc_size;
		case array_enum::renderArrays_auxCellSurface: return arrays::renderArrays::auxCellSurface::alloc_size;
		case array_enum::structureArrays_classification: return arrays::structureArrays::classification::alloc_size;
		case array_enum::structureArrays_resortIndex: return arrays::structureArrays::resortIndex::alloc_size;
		case array_enum::structureArrays_ZOrder_64: return arrays::structureArrays::ZOrder_64::alloc_size;
		case array_enum::structureArrays_ZOrder_32: return arrays::structureArrays::ZOrder_32::alloc_size;
		case array_enum::structureArrays_cellSpanSwap: return arrays::structureArrays::cellSpanSwap::alloc_size;
		case array_enum::structureArrays_cellSpan: return arrays::structureArrays::cellSpan::alloc_size;
		case array_enum::structureArrays_compactCellSpanSwap: return arrays::structureArrays::compactCellSpanSwap::alloc_size;
		case array_enum::structureArrays_compactCellSpan: return arrays::structureArrays::compactCellSpan::alloc_size;
		case array_enum::structureArrays_MLMResolution: return arrays::structureArrays::MLMResolution::alloc_size;
		case array_enum::structureArrays_cellparticleIndex: return arrays::structureArrays::cellparticleIndex::alloc_size;
		case array_enum::structureArrays_compactparticleIndex: return arrays::structureArrays::compactparticleIndex::alloc_size;
		case array_enum::structureArrays_hashMap: return arrays::structureArrays::hashMap::alloc_size;
		case array_enum::structureArrays_compactHashMap: return arrays::structureArrays::compactHashMap::alloc_size;
		case array_enum::structureArrays_mlmScaling: return arrays::structureArrays::mlmScaling::alloc_size;
		case array_enum::structureArrays_particleparticleIndex: return arrays::structureArrays::particleparticleIndex::alloc_size;
		case array_enum::structureArrays_cellBegin: return arrays::structureArrays::cellBegin::alloc_size;
		case array_enum::structureArrays_cellEnd: return arrays::structureArrays::cellEnd::alloc_size;
		case array_enum::supportArrays_support: return arrays::supportArrays::support::alloc_size;
		case array_enum::supportArrays_supportEstimate: return arrays::supportArrays::supportEstimate::alloc_size;
		case array_enum::supportArrays_neighborCount: return arrays::supportArrays::neighborCount::alloc_size;
		case array_enum::supportArrays_supportMarker: return arrays::supportArrays::supportMarker::alloc_size;
		case array_enum::supportArrays_supportMarkerCompacted: return arrays::supportArrays::supportMarkerCompacted::alloc_size;
		case array_enum::supportArrays_closestNeighbor: return arrays::supportArrays::closestNeighbor::alloc_size;
		case array_enum::supportArrays_closestNeighbor_f: return arrays::supportArrays::closestNeighbor_f::alloc_size;
		case array_enum::supportArrays_neighborOverhead: return arrays::supportArrays::neighborOverhead::alloc_size;
		case array_enum::supportArrays_neighborOverheadCount: return arrays::supportArrays::neighborOverheadCount::alloc_size;
		case array_enum::supportArrays_neighborListSwap: return arrays::supportArrays::neighborListSwap::alloc_size;
		default: return (size_t)0u;
	}
}
auto inline getArrayKind(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return arrays::adaptive::mergeable::kind;
		case array_enum::adaptive_mergeCounter: return arrays::adaptive::mergeCounter::kind;
		case array_enum::adaptive_ptclCounter: return arrays::adaptive::ptclCounter::kind;
		case array_enum::adaptive_classification: return arrays::adaptive::classification::kind;
		case array_enum::adaptive_splitIndicator: return arrays::adaptive::splitIndicator::kind;
		case array_enum::adaptive_adaptivityCounter: return arrays::adaptive::adaptivityCounter::kind;
		case array_enum::adaptive_splitIndicatorCompacted: return arrays::adaptive::splitIndicatorCompacted::kind;
		case array_enum::adaptive_parentIndex: return arrays::adaptive::parentIndex::kind;
		case array_enum::adaptive_parentVolume: return arrays::adaptive::parentVolume::kind;
		case array_enum::adaptive_parentPosition: return arrays::adaptive::parentPosition::kind;
		case array_enum::advectionArrays_maxVelocity: return arrays::advectionArrays::maxVelocity::kind;
		case array_enum::advectionArrays_cflValue: return arrays::advectionArrays::cflValue::kind;
		case array_enum::advectionArrays_particleNormal: return arrays::advectionArrays::particleNormal::kind;
		case array_enum::advectionArrays_angularVelocity: return arrays::advectionArrays::angularVelocity::kind;
		case array_enum::rigidBodyArrays_rigidDensities: return arrays::rigidBodyArrays::rigidDensities::kind;
		case array_enum::rigidBodyArrays_rigidVolumes: return arrays::rigidBodyArrays::rigidVolumes::kind;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return arrays::rigidBodyArrays::rigidLinearVelocities::kind;
		case array_enum::rigidBodyArrays_rigidAVelocities: return arrays::rigidBodyArrays::rigidAVelocities::kind;
		case array_enum::rigidBodyArrays_rigidOrigins: return arrays::rigidBodyArrays::rigidOrigins::kind;
		case array_enum::rigidBodyArrays_rigidQuaternions: return arrays::rigidBodyArrays::rigidQuaternions::kind;
		case array_enum::boundaryArrays_splineLUT: return arrays::boundaryArrays::splineLUT::kind;
		case array_enum::boundaryArrays_offsetLUT: return arrays::boundaryArrays::offsetLUT::kind;
		case array_enum::boundaryArrays_spikyLUT: return arrays::boundaryArrays::spikyLUT::kind;
		case array_enum::boundaryArrays_adhesionLUT: return arrays::boundaryArrays::adhesionLUT::kind;
		case array_enum::boundaryArrays_volumeLUT: return arrays::boundaryArrays::volumeLUT::kind;
		case array_enum::boundaryArrays_spline2LUT: return arrays::boundaryArrays::spline2LUT::kind;
		case array_enum::boundaryArrays_splineGradientLUT: return arrays::boundaryArrays::splineGradientLUT::kind;
		case array_enum::boundaryArrays_spikyGradientLUT: return arrays::boundaryArrays::spikyGradientLUT::kind;
		case array_enum::boundaryArrays_cohesionLUT: return arrays::boundaryArrays::cohesionLUT::kind;
		case array_enum::boundaryArrays_boundaryPlanes: return arrays::boundaryArrays::boundaryPlanes::kind;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return arrays::boundaryArrays::boundaryPlaneVelocity::kind;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return arrays::boundaryArrays::volumeBoundaryVolumes::kind;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return arrays::boundaryArrays::volumeBoundaryDimensions::kind;
		case array_enum::boundaryArrays_volumeBoundaryMin: return arrays::boundaryArrays::volumeBoundaryMin::kind;
		case array_enum::boundaryArrays_volumeBoundaryMax: return arrays::boundaryArrays::volumeBoundaryMax::kind;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return arrays::boundaryArrays::volumeBoundaryDensity::kind;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return arrays::boundaryArrays::volumeBoundaryVolume::kind;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return arrays::boundaryArrays::volumeBoundaryVelocity::kind;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return arrays::boundaryArrays::volumeBoundaryAngularVelocity::kind;
		case array_enum::boundaryArrays_volumeBoundaryKind: return arrays::boundaryArrays::volumeBoundaryKind::kind;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return arrays::boundaryArrays::volumeBoundaryPosition::kind;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return arrays::boundaryArrays::volumeBoundaryQuaternion::kind;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return arrays::boundaryArrays::volumeBoundaryTransformMatrix::kind;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::kind;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return arrays::boundaryArrays::volumeBoundaryInertiaMatrix::kind;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::kind;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return arrays::boundaryArrays::volumeBoundaryAcceleration::kind;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return arrays::boundaryArrays::volumeBoundaryAngularAcceleration::kind;
		case array_enum::surfaceArrays_decisionBuffer: return arrays::surfaceArrays::decisionBuffer::kind;
		case array_enum::surfaceArrays_surface_idxBuffer: return arrays::surfaceArrays::surface_idxBuffer::kind;
		case array_enum::surfaceArrays_markerBuffer: return arrays::surfaceArrays::markerBuffer::kind;
		case array_enum::surfaceArrays_distanceBuffer: return arrays::surfaceArrays::distanceBuffer::kind;
		case array_enum::surfaceArrays_changeBuffer: return arrays::surfaceArrays::changeBuffer::kind;
		case array_enum::basicArrays_matrixTest: return arrays::basicArrays::matrixTest::kind;
		case array_enum::basicArrays_position: return arrays::basicArrays::position::kind;
		case array_enum::basicArrays_acceleration: return arrays::basicArrays::acceleration::kind;
		case array_enum::basicArrays_velocity: return arrays::basicArrays::velocity::kind;
		case array_enum::basicArrays_particle_type: return arrays::basicArrays::particle_type::kind;
		case array_enum::basicArrays_renderArray: return arrays::basicArrays::renderArray::kind;
		case array_enum::basicArrays_debugArray: return arrays::basicArrays::debugArray::kind;
		case array_enum::basicArrays_volume: return arrays::basicArrays::volume::kind;
		case array_enum::basicArrays_lifetime: return arrays::basicArrays::lifetime::kind;
		case array_enum::basicArrays_pressure: return arrays::basicArrays::pressure::kind;
		case array_enum::basicArrays_density: return arrays::basicArrays::density::kind;
		case array_enum::basicArrays_fluidDensity: return arrays::basicArrays::fluidDensity::kind;
		case array_enum::basicArrays_particleIndex: return arrays::basicArrays::particleIndex::kind;
		case array_enum::basicArrays_particleIndexCompact: return arrays::basicArrays::particleIndexCompact::kind;
		case array_enum::basicArrays_resortArray4: return arrays::basicArrays::resortArray4::kind;
		case array_enum::basicArrays_resortArray: return arrays::basicArrays::resortArray::kind;
		case array_enum::dfsphArrays_dfsphSource: return arrays::dfsphArrays::dfsphSource::kind;
		case array_enum::dfsphArrays_dfsphAlpha: return arrays::dfsphArrays::dfsphAlpha::kind;
		case array_enum::dfsphArrays_dfsphDpDt: return arrays::dfsphArrays::dfsphDpDt::kind;
		case array_enum::dfsphArrays_dfsphRhoStar: return arrays::dfsphArrays::dfsphRhoStar::kind;
		case array_enum::iisphArrays_iisphSum: return arrays::iisphArrays::iisphSum::kind;
		case array_enum::iisphArrays_iisphDii: return arrays::iisphArrays::iisphDii::kind;
		case array_enum::iisphArrays_iisphAiiOld: return arrays::iisphArrays::iisphAiiOld::kind;
		case array_enum::iisphArrays_omega: return arrays::iisphArrays::omega::kind;
		case array_enum::iisphArrays_alpha: return arrays::iisphArrays::alpha::kind;
		case array_enum::iisphArrays_gamma: return arrays::iisphArrays::gamma::kind;
		case array_enum::iisphArrays_sigma: return arrays::iisphArrays::sigma::kind;
		case array_enum::iisphArrays_iisphDensityAdvection: return arrays::iisphArrays::iisphDensityAdvection::kind;
		case array_enum::iisphArrays_iisphDensityIteration: return arrays::iisphArrays::iisphDensityIteration::kind;
		case array_enum::iisphArrays_kernelBuffer: return arrays::iisphArrays::kernelBuffer::kind;
		case array_enum::iisphArrays_velocityAdvection: return arrays::iisphArrays::velocityAdvection::kind;
		case array_enum::iisph17_sourceTerm: return arrays::iisph17::sourceTerm::kind;
		case array_enum::iisph17_boundaryPressure: return arrays::iisph17::boundaryPressure::kind;
		case array_enum::iisph17_omega: return arrays::iisph17::omega::kind;
		case array_enum::iisph17_apparentVolume: return arrays::iisph17::apparentVolume::kind;
		case array_enum::iisph17_Aii: return arrays::iisph17::Aii::kind;
		case array_enum::iisph17_predictedAcceleration: return arrays::iisph17::predictedAcceleration::kind;
		case array_enum::iisph17_volumeError: return arrays::iisph17::volumeError::kind;
		case array_enum::iisph17_band_rb__sourceTerm: return arrays::iisph17_band_rb::_sourceTerm::kind;
		case array_enum::iisph17_band_rb__omega: return arrays::iisph17_band_rb::_omega::kind;
		case array_enum::iisph17_band_rb__apparentVolume: return arrays::iisph17_band_rb::_apparentVolume::kind;
		case array_enum::iisph17_band_rb__Aii: return arrays::iisph17_band_rb::_Aii::kind;
		case array_enum::iisph17_band_rb__predictedAcceleration: return arrays::iisph17_band_rb::_predictedAcceleration::kind;
		case array_enum::iisph17_band_rb__volumeError: return arrays::iisph17_band_rb::_volumeError::kind;
		case array_enum::volumeInletArrays_inletPositions: return arrays::volumeInletArrays::inletPositions::kind;
		case array_enum::volumeInletArrays_inletCounter: return arrays::volumeInletArrays::inletCounter::kind;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return arrays::volumeInletArrays::volumeOutletVolumes::kind;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return arrays::volumeInletArrays::volumeOutletDimensions::kind;
		case array_enum::volumeInletArrays_volumeOutletMin: return arrays::volumeInletArrays::volumeOutletMin::kind;
		case array_enum::volumeInletArrays_volumeOutletMax: return arrays::volumeInletArrays::volumeOutletMax::kind;
		case array_enum::volumeInletArrays_volumeOutletRate: return arrays::volumeInletArrays::volumeOutletRate::kind;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return arrays::volumeInletArrays::volumeOutletRateAccumulator::kind;
		case array_enum::structureArrays_compactCellList: return arrays::structureArrays::compactCellList::kind;
		case array_enum::structureArrays_neighborMask: return arrays::structureArrays::neighborMask::kind;
		case array_enum::structureArrays_compactCellScale: return arrays::structureArrays::compactCellScale::kind;
		case array_enum::structureArrays_spanNeighborList: return arrays::structureArrays::spanNeighborList::kind;
		case array_enum::structureArrays_neighborList: return arrays::structureArrays::neighborList::kind;
		case array_enum::structureArrays_neighborListLength: return arrays::structureArrays::neighborListLength::kind;
		case array_enum::renderArrays_anisotropicMatrices: return arrays::renderArrays::anisotropicMatrices::kind;
		case array_enum::renderArrays_centerPosition: return arrays::renderArrays::centerPosition::kind;
		case array_enum::renderArrays_auxTest: return arrays::renderArrays::auxTest::kind;
		case array_enum::renderArrays_auxIsoDensity: return arrays::renderArrays::auxIsoDensity::kind;
		case array_enum::renderArrays_auxDistance: return arrays::renderArrays::auxDistance::kind;
		case array_enum::renderArrays_auxHashMap: return arrays::renderArrays::auxHashMap::kind;
		case array_enum::renderArrays_auxCellSpan: return arrays::renderArrays::auxCellSpan::kind;
		case array_enum::renderArrays_auxCellInformation: return arrays::renderArrays::auxCellInformation::kind;
		case array_enum::renderArrays_auxCellSurface: return arrays::renderArrays::auxCellSurface::kind;
		case array_enum::structureArrays_classification: return arrays::structureArrays::classification::kind;
		case array_enum::structureArrays_resortIndex: return arrays::structureArrays::resortIndex::kind;
		case array_enum::structureArrays_ZOrder_64: return arrays::structureArrays::ZOrder_64::kind;
		case array_enum::structureArrays_ZOrder_32: return arrays::structureArrays::ZOrder_32::kind;
		case array_enum::structureArrays_cellSpanSwap: return arrays::structureArrays::cellSpanSwap::kind;
		case array_enum::structureArrays_cellSpan: return arrays::structureArrays::cellSpan::kind;
		case array_enum::structureArrays_compactCellSpanSwap: return arrays::structureArrays::compactCellSpanSwap::kind;
		case array_enum::structureArrays_compactCellSpan: return arrays::structureArrays::compactCellSpan::kind;
		case array_enum::structureArrays_MLMResolution: return arrays::structureArrays::MLMResolution::kind;
		case array_enum::structureArrays_cellparticleIndex: return arrays::structureArrays::cellparticleIndex::kind;
		case array_enum::structureArrays_compactparticleIndex: return arrays::structureArrays::compactparticleIndex::kind;
		case array_enum::structureArrays_hashMap: return arrays::structureArrays::hashMap::kind;
		case array_enum::structureArrays_compactHashMap: return arrays::structureArrays::compactHashMap::kind;
		case array_enum::structureArrays_mlmScaling: return arrays::structureArrays::mlmScaling::kind;
		case array_enum::structureArrays_particleparticleIndex: return arrays::structureArrays::particleparticleIndex::kind;
		case array_enum::structureArrays_cellBegin: return arrays::structureArrays::cellBegin::kind;
		case array_enum::structureArrays_cellEnd: return arrays::structureArrays::cellEnd::kind;
		case array_enum::supportArrays_support: return arrays::supportArrays::support::kind;
		case array_enum::supportArrays_supportEstimate: return arrays::supportArrays::supportEstimate::kind;
		case array_enum::supportArrays_neighborCount: return arrays::supportArrays::neighborCount::kind;
		case array_enum::supportArrays_supportMarker: return arrays::supportArrays::supportMarker::kind;
		case array_enum::supportArrays_supportMarkerCompacted: return arrays::supportArrays::supportMarkerCompacted::kind;
		case array_enum::supportArrays_closestNeighbor: return arrays::supportArrays::closestNeighbor::kind;
		case array_enum::supportArrays_closestNeighbor_f: return arrays::supportArrays::closestNeighbor_f::kind;
		case array_enum::supportArrays_neighborOverhead: return arrays::supportArrays::neighborOverhead::kind;
		case array_enum::supportArrays_neighborOverheadCount: return arrays::supportArrays::neighborOverheadCount::kind;
		case array_enum::supportArrays_neighborListSwap: return arrays::supportArrays::neighborListSwap::kind;
		default: return memory_kind::customData;
	}
}
auto inline getArrayDescription(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return arrays::adaptive::mergeable::description;
		case array_enum::adaptive_mergeCounter: return arrays::adaptive::mergeCounter::description;
		case array_enum::adaptive_ptclCounter: return arrays::adaptive::ptclCounter::description;
		case array_enum::adaptive_classification: return arrays::adaptive::classification::description;
		case array_enum::adaptive_splitIndicator: return arrays::adaptive::splitIndicator::description;
		case array_enum::adaptive_adaptivityCounter: return arrays::adaptive::adaptivityCounter::description;
		case array_enum::adaptive_splitIndicatorCompacted: return arrays::adaptive::splitIndicatorCompacted::description;
		case array_enum::adaptive_parentIndex: return arrays::adaptive::parentIndex::description;
		case array_enum::adaptive_parentVolume: return arrays::adaptive::parentVolume::description;
		case array_enum::adaptive_parentPosition: return arrays::adaptive::parentPosition::description;
		case array_enum::advectionArrays_maxVelocity: return arrays::advectionArrays::maxVelocity::description;
		case array_enum::advectionArrays_cflValue: return arrays::advectionArrays::cflValue::description;
		case array_enum::advectionArrays_particleNormal: return arrays::advectionArrays::particleNormal::description;
		case array_enum::advectionArrays_angularVelocity: return arrays::advectionArrays::angularVelocity::description;
		case array_enum::rigidBodyArrays_rigidDensities: return arrays::rigidBodyArrays::rigidDensities::description;
		case array_enum::rigidBodyArrays_rigidVolumes: return arrays::rigidBodyArrays::rigidVolumes::description;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return arrays::rigidBodyArrays::rigidLinearVelocities::description;
		case array_enum::rigidBodyArrays_rigidAVelocities: return arrays::rigidBodyArrays::rigidAVelocities::description;
		case array_enum::rigidBodyArrays_rigidOrigins: return arrays::rigidBodyArrays::rigidOrigins::description;
		case array_enum::rigidBodyArrays_rigidQuaternions: return arrays::rigidBodyArrays::rigidQuaternions::description;
		case array_enum::boundaryArrays_splineLUT: return arrays::boundaryArrays::splineLUT::description;
		case array_enum::boundaryArrays_offsetLUT: return arrays::boundaryArrays::offsetLUT::description;
		case array_enum::boundaryArrays_spikyLUT: return arrays::boundaryArrays::spikyLUT::description;
		case array_enum::boundaryArrays_adhesionLUT: return arrays::boundaryArrays::adhesionLUT::description;
		case array_enum::boundaryArrays_volumeLUT: return arrays::boundaryArrays::volumeLUT::description;
		case array_enum::boundaryArrays_spline2LUT: return arrays::boundaryArrays::spline2LUT::description;
		case array_enum::boundaryArrays_splineGradientLUT: return arrays::boundaryArrays::splineGradientLUT::description;
		case array_enum::boundaryArrays_spikyGradientLUT: return arrays::boundaryArrays::spikyGradientLUT::description;
		case array_enum::boundaryArrays_cohesionLUT: return arrays::boundaryArrays::cohesionLUT::description;
		case array_enum::boundaryArrays_boundaryPlanes: return arrays::boundaryArrays::boundaryPlanes::description;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return arrays::boundaryArrays::boundaryPlaneVelocity::description;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return arrays::boundaryArrays::volumeBoundaryVolumes::description;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return arrays::boundaryArrays::volumeBoundaryDimensions::description;
		case array_enum::boundaryArrays_volumeBoundaryMin: return arrays::boundaryArrays::volumeBoundaryMin::description;
		case array_enum::boundaryArrays_volumeBoundaryMax: return arrays::boundaryArrays::volumeBoundaryMax::description;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return arrays::boundaryArrays::volumeBoundaryDensity::description;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return arrays::boundaryArrays::volumeBoundaryVolume::description;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return arrays::boundaryArrays::volumeBoundaryVelocity::description;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return arrays::boundaryArrays::volumeBoundaryAngularVelocity::description;
		case array_enum::boundaryArrays_volumeBoundaryKind: return arrays::boundaryArrays::volumeBoundaryKind::description;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return arrays::boundaryArrays::volumeBoundaryPosition::description;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return arrays::boundaryArrays::volumeBoundaryQuaternion::description;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return arrays::boundaryArrays::volumeBoundaryTransformMatrix::description;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::description;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return arrays::boundaryArrays::volumeBoundaryInertiaMatrix::description;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::description;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return arrays::boundaryArrays::volumeBoundaryAcceleration::description;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return arrays::boundaryArrays::volumeBoundaryAngularAcceleration::description;
		case array_enum::surfaceArrays_decisionBuffer: return arrays::surfaceArrays::decisionBuffer::description;
		case array_enum::surfaceArrays_surface_idxBuffer: return arrays::surfaceArrays::surface_idxBuffer::description;
		case array_enum::surfaceArrays_markerBuffer: return arrays::surfaceArrays::markerBuffer::description;
		case array_enum::surfaceArrays_distanceBuffer: return arrays::surfaceArrays::distanceBuffer::description;
		case array_enum::surfaceArrays_changeBuffer: return arrays::surfaceArrays::changeBuffer::description;
		case array_enum::basicArrays_matrixTest: return arrays::basicArrays::matrixTest::description;
		case array_enum::basicArrays_position: return arrays::basicArrays::position::description;
		case array_enum::basicArrays_acceleration: return arrays::basicArrays::acceleration::description;
		case array_enum::basicArrays_velocity: return arrays::basicArrays::velocity::description;
		case array_enum::basicArrays_particle_type: return arrays::basicArrays::particle_type::description;
		case array_enum::basicArrays_renderArray: return arrays::basicArrays::renderArray::description;
		case array_enum::basicArrays_debugArray: return arrays::basicArrays::debugArray::description;
		case array_enum::basicArrays_volume: return arrays::basicArrays::volume::description;
		case array_enum::basicArrays_lifetime: return arrays::basicArrays::lifetime::description;
		case array_enum::basicArrays_pressure: return arrays::basicArrays::pressure::description;
		case array_enum::basicArrays_density: return arrays::basicArrays::density::description;
		case array_enum::basicArrays_fluidDensity: return arrays::basicArrays::fluidDensity::description;
		case array_enum::basicArrays_particleIndex: return arrays::basicArrays::particleIndex::description;
		case array_enum::basicArrays_particleIndexCompact: return arrays::basicArrays::particleIndexCompact::description;
		case array_enum::basicArrays_resortArray4: return arrays::basicArrays::resortArray4::description;
		case array_enum::basicArrays_resortArray: return arrays::basicArrays::resortArray::description;
		case array_enum::dfsphArrays_dfsphSource: return arrays::dfsphArrays::dfsphSource::description;
		case array_enum::dfsphArrays_dfsphAlpha: return arrays::dfsphArrays::dfsphAlpha::description;
		case array_enum::dfsphArrays_dfsphDpDt: return arrays::dfsphArrays::dfsphDpDt::description;
		case array_enum::dfsphArrays_dfsphRhoStar: return arrays::dfsphArrays::dfsphRhoStar::description;
		case array_enum::iisphArrays_iisphSum: return arrays::iisphArrays::iisphSum::description;
		case array_enum::iisphArrays_iisphDii: return arrays::iisphArrays::iisphDii::description;
		case array_enum::iisphArrays_iisphAiiOld: return arrays::iisphArrays::iisphAiiOld::description;
		case array_enum::iisphArrays_omega: return arrays::iisphArrays::omega::description;
		case array_enum::iisphArrays_alpha: return arrays::iisphArrays::alpha::description;
		case array_enum::iisphArrays_gamma: return arrays::iisphArrays::gamma::description;
		case array_enum::iisphArrays_sigma: return arrays::iisphArrays::sigma::description;
		case array_enum::iisphArrays_iisphDensityAdvection: return arrays::iisphArrays::iisphDensityAdvection::description;
		case array_enum::iisphArrays_iisphDensityIteration: return arrays::iisphArrays::iisphDensityIteration::description;
		case array_enum::iisphArrays_kernelBuffer: return arrays::iisphArrays::kernelBuffer::description;
		case array_enum::iisphArrays_velocityAdvection: return arrays::iisphArrays::velocityAdvection::description;
		case array_enum::iisph17_sourceTerm: return arrays::iisph17::sourceTerm::description;
		case array_enum::iisph17_boundaryPressure: return arrays::iisph17::boundaryPressure::description;
		case array_enum::iisph17_omega: return arrays::iisph17::omega::description;
		case array_enum::iisph17_apparentVolume: return arrays::iisph17::apparentVolume::description;
		case array_enum::iisph17_Aii: return arrays::iisph17::Aii::description;
		case array_enum::iisph17_predictedAcceleration: return arrays::iisph17::predictedAcceleration::description;
		case array_enum::iisph17_volumeError: return arrays::iisph17::volumeError::description;
		case array_enum::iisph17_band_rb__sourceTerm: return arrays::iisph17_band_rb::_sourceTerm::description;
		case array_enum::iisph17_band_rb__omega: return arrays::iisph17_band_rb::_omega::description;
		case array_enum::iisph17_band_rb__apparentVolume: return arrays::iisph17_band_rb::_apparentVolume::description;
		case array_enum::iisph17_band_rb__Aii: return arrays::iisph17_band_rb::_Aii::description;
		case array_enum::iisph17_band_rb__predictedAcceleration: return arrays::iisph17_band_rb::_predictedAcceleration::description;
		case array_enum::iisph17_band_rb__volumeError: return arrays::iisph17_band_rb::_volumeError::description;
		case array_enum::volumeInletArrays_inletPositions: return arrays::volumeInletArrays::inletPositions::description;
		case array_enum::volumeInletArrays_inletCounter: return arrays::volumeInletArrays::inletCounter::description;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return arrays::volumeInletArrays::volumeOutletVolumes::description;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return arrays::volumeInletArrays::volumeOutletDimensions::description;
		case array_enum::volumeInletArrays_volumeOutletMin: return arrays::volumeInletArrays::volumeOutletMin::description;
		case array_enum::volumeInletArrays_volumeOutletMax: return arrays::volumeInletArrays::volumeOutletMax::description;
		case array_enum::volumeInletArrays_volumeOutletRate: return arrays::volumeInletArrays::volumeOutletRate::description;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return arrays::volumeInletArrays::volumeOutletRateAccumulator::description;
		case array_enum::structureArrays_compactCellList: return arrays::structureArrays::compactCellList::description;
		case array_enum::structureArrays_neighborMask: return arrays::structureArrays::neighborMask::description;
		case array_enum::structureArrays_compactCellScale: return arrays::structureArrays::compactCellScale::description;
		case array_enum::structureArrays_spanNeighborList: return arrays::structureArrays::spanNeighborList::description;
		case array_enum::structureArrays_neighborList: return arrays::structureArrays::neighborList::description;
		case array_enum::structureArrays_neighborListLength: return arrays::structureArrays::neighborListLength::description;
		case array_enum::renderArrays_anisotropicMatrices: return arrays::renderArrays::anisotropicMatrices::description;
		case array_enum::renderArrays_centerPosition: return arrays::renderArrays::centerPosition::description;
		case array_enum::renderArrays_auxTest: return arrays::renderArrays::auxTest::description;
		case array_enum::renderArrays_auxIsoDensity: return arrays::renderArrays::auxIsoDensity::description;
		case array_enum::renderArrays_auxDistance: return arrays::renderArrays::auxDistance::description;
		case array_enum::renderArrays_auxHashMap: return arrays::renderArrays::auxHashMap::description;
		case array_enum::renderArrays_auxCellSpan: return arrays::renderArrays::auxCellSpan::description;
		case array_enum::renderArrays_auxCellInformation: return arrays::renderArrays::auxCellInformation::description;
		case array_enum::renderArrays_auxCellSurface: return arrays::renderArrays::auxCellSurface::description;
		case array_enum::structureArrays_classification: return arrays::structureArrays::classification::description;
		case array_enum::structureArrays_resortIndex: return arrays::structureArrays::resortIndex::description;
		case array_enum::structureArrays_ZOrder_64: return arrays::structureArrays::ZOrder_64::description;
		case array_enum::structureArrays_ZOrder_32: return arrays::structureArrays::ZOrder_32::description;
		case array_enum::structureArrays_cellSpanSwap: return arrays::structureArrays::cellSpanSwap::description;
		case array_enum::structureArrays_cellSpan: return arrays::structureArrays::cellSpan::description;
		case array_enum::structureArrays_compactCellSpanSwap: return arrays::structureArrays::compactCellSpanSwap::description;
		case array_enum::structureArrays_compactCellSpan: return arrays::structureArrays::compactCellSpan::description;
		case array_enum::structureArrays_MLMResolution: return arrays::structureArrays::MLMResolution::description;
		case array_enum::structureArrays_cellparticleIndex: return arrays::structureArrays::cellparticleIndex::description;
		case array_enum::structureArrays_compactparticleIndex: return arrays::structureArrays::compactparticleIndex::description;
		case array_enum::structureArrays_hashMap: return arrays::structureArrays::hashMap::description;
		case array_enum::structureArrays_compactHashMap: return arrays::structureArrays::compactHashMap::description;
		case array_enum::structureArrays_mlmScaling: return arrays::structureArrays::mlmScaling::description;
		case array_enum::structureArrays_particleparticleIndex: return arrays::structureArrays::particleparticleIndex::description;
		case array_enum::structureArrays_cellBegin: return arrays::structureArrays::cellBegin::description;
		case array_enum::structureArrays_cellEnd: return arrays::structureArrays::cellEnd::description;
		case array_enum::supportArrays_support: return arrays::supportArrays::support::description;
		case array_enum::supportArrays_supportEstimate: return arrays::supportArrays::supportEstimate::description;
		case array_enum::supportArrays_neighborCount: return arrays::supportArrays::neighborCount::description;
		case array_enum::supportArrays_supportMarker: return arrays::supportArrays::supportMarker::description;
		case array_enum::supportArrays_supportMarkerCompacted: return arrays::supportArrays::supportMarkerCompacted::description;
		case array_enum::supportArrays_closestNeighbor: return arrays::supportArrays::closestNeighbor::description;
		case array_enum::supportArrays_closestNeighbor_f: return arrays::supportArrays::closestNeighbor_f::description;
		case array_enum::supportArrays_neighborOverhead: return arrays::supportArrays::neighborOverhead::description;
		case array_enum::supportArrays_neighborOverheadCount: return arrays::supportArrays::neighborOverheadCount::description;
		case array_enum::supportArrays_neighborListSwap: return arrays::supportArrays::neighborListSwap::description;
		default: return "invalidEnum";
	}
}
#ifndef __CUDACC__
auto inline getArrayType(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return type_name<arrays::adaptive::mergeable::type>();
		case array_enum::adaptive_mergeCounter: return type_name<arrays::adaptive::mergeCounter::type>();
		case array_enum::adaptive_ptclCounter: return type_name<arrays::adaptive::ptclCounter::type>();
		case array_enum::adaptive_classification: return type_name<arrays::adaptive::classification::type>();
		case array_enum::adaptive_splitIndicator: return type_name<arrays::adaptive::splitIndicator::type>();
		case array_enum::adaptive_adaptivityCounter: return type_name<arrays::adaptive::adaptivityCounter::type>();
		case array_enum::adaptive_splitIndicatorCompacted: return type_name<arrays::adaptive::splitIndicatorCompacted::type>();
		case array_enum::adaptive_parentIndex: return type_name<arrays::adaptive::parentIndex::type>();
		case array_enum::adaptive_parentVolume: return type_name<arrays::adaptive::parentVolume::type>();
		case array_enum::adaptive_parentPosition: return type_name<arrays::adaptive::parentPosition::type>();
		case array_enum::advectionArrays_maxVelocity: return type_name<arrays::advectionArrays::maxVelocity::type>();
		case array_enum::advectionArrays_cflValue: return type_name<arrays::advectionArrays::cflValue::type>();
		case array_enum::advectionArrays_particleNormal: return type_name<arrays::advectionArrays::particleNormal::type>();
		case array_enum::advectionArrays_angularVelocity: return type_name<arrays::advectionArrays::angularVelocity::type>();
		case array_enum::rigidBodyArrays_rigidDensities: return type_name<arrays::rigidBodyArrays::rigidDensities::type>();
		case array_enum::rigidBodyArrays_rigidVolumes: return type_name<arrays::rigidBodyArrays::rigidVolumes::type>();
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return type_name<arrays::rigidBodyArrays::rigidLinearVelocities::type>();
		case array_enum::rigidBodyArrays_rigidAVelocities: return type_name<arrays::rigidBodyArrays::rigidAVelocities::type>();
		case array_enum::rigidBodyArrays_rigidOrigins: return type_name<arrays::rigidBodyArrays::rigidOrigins::type>();
		case array_enum::rigidBodyArrays_rigidQuaternions: return type_name<arrays::rigidBodyArrays::rigidQuaternions::type>();
		case array_enum::boundaryArrays_splineLUT: return type_name<arrays::boundaryArrays::splineLUT::type>();
		case array_enum::boundaryArrays_offsetLUT: return type_name<arrays::boundaryArrays::offsetLUT::type>();
		case array_enum::boundaryArrays_spikyLUT: return type_name<arrays::boundaryArrays::spikyLUT::type>();
		case array_enum::boundaryArrays_adhesionLUT: return type_name<arrays::boundaryArrays::adhesionLUT::type>();
		case array_enum::boundaryArrays_volumeLUT: return type_name<arrays::boundaryArrays::volumeLUT::type>();
		case array_enum::boundaryArrays_spline2LUT: return type_name<arrays::boundaryArrays::spline2LUT::type>();
		case array_enum::boundaryArrays_splineGradientLUT: return type_name<arrays::boundaryArrays::splineGradientLUT::type>();
		case array_enum::boundaryArrays_spikyGradientLUT: return type_name<arrays::boundaryArrays::spikyGradientLUT::type>();
		case array_enum::boundaryArrays_cohesionLUT: return type_name<arrays::boundaryArrays::cohesionLUT::type>();
		case array_enum::boundaryArrays_boundaryPlanes: return type_name<arrays::boundaryArrays::boundaryPlanes::type>();
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return type_name<arrays::boundaryArrays::boundaryPlaneVelocity::type>();
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return type_name<arrays::boundaryArrays::volumeBoundaryVolumes::type>();
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return type_name<arrays::boundaryArrays::volumeBoundaryDimensions::type>();
		case array_enum::boundaryArrays_volumeBoundaryMin: return type_name<arrays::boundaryArrays::volumeBoundaryMin::type>();
		case array_enum::boundaryArrays_volumeBoundaryMax: return type_name<arrays::boundaryArrays::volumeBoundaryMax::type>();
		case array_enum::boundaryArrays_volumeBoundaryDensity: return type_name<arrays::boundaryArrays::volumeBoundaryDensity::type>();
		case array_enum::boundaryArrays_volumeBoundaryVolume: return type_name<arrays::boundaryArrays::volumeBoundaryVolume::type>();
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return type_name<arrays::boundaryArrays::volumeBoundaryVelocity::type>();
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return type_name<arrays::boundaryArrays::volumeBoundaryAngularVelocity::type>();
		case array_enum::boundaryArrays_volumeBoundaryKind: return type_name<arrays::boundaryArrays::volumeBoundaryKind::type>();
		case array_enum::boundaryArrays_volumeBoundaryPosition: return type_name<arrays::boundaryArrays::volumeBoundaryPosition::type>();
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return type_name<arrays::boundaryArrays::volumeBoundaryQuaternion::type>();
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return type_name<arrays::boundaryArrays::volumeBoundaryTransformMatrix::type>();
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return type_name<arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse::type>();
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return type_name<arrays::boundaryArrays::volumeBoundaryInertiaMatrix::type>();
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return type_name<arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse::type>();
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return type_name<arrays::boundaryArrays::volumeBoundaryAcceleration::type>();
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return type_name<arrays::boundaryArrays::volumeBoundaryAngularAcceleration::type>();
		case array_enum::surfaceArrays_decisionBuffer: return type_name<arrays::surfaceArrays::decisionBuffer::type>();
		case array_enum::surfaceArrays_surface_idxBuffer: return type_name<arrays::surfaceArrays::surface_idxBuffer::type>();
		case array_enum::surfaceArrays_markerBuffer: return type_name<arrays::surfaceArrays::markerBuffer::type>();
		case array_enum::surfaceArrays_distanceBuffer: return type_name<arrays::surfaceArrays::distanceBuffer::type>();
		case array_enum::surfaceArrays_changeBuffer: return type_name<arrays::surfaceArrays::changeBuffer::type>();
		case array_enum::basicArrays_matrixTest: return type_name<arrays::basicArrays::matrixTest::type>();
		case array_enum::basicArrays_position: return type_name<arrays::basicArrays::position::type>();
		case array_enum::basicArrays_acceleration: return type_name<arrays::basicArrays::acceleration::type>();
		case array_enum::basicArrays_velocity: return type_name<arrays::basicArrays::velocity::type>();
		case array_enum::basicArrays_particle_type: return type_name<arrays::basicArrays::particle_type::type>();
		case array_enum::basicArrays_renderArray: return type_name<arrays::basicArrays::renderArray::type>();
		case array_enum::basicArrays_debugArray: return type_name<arrays::basicArrays::debugArray::type>();
		case array_enum::basicArrays_volume: return type_name<arrays::basicArrays::volume::type>();
		case array_enum::basicArrays_lifetime: return type_name<arrays::basicArrays::lifetime::type>();
		case array_enum::basicArrays_pressure: return type_name<arrays::basicArrays::pressure::type>();
		case array_enum::basicArrays_density: return type_name<arrays::basicArrays::density::type>();
		case array_enum::basicArrays_fluidDensity: return type_name<arrays::basicArrays::fluidDensity::type>();
		case array_enum::basicArrays_particleIndex: return type_name<arrays::basicArrays::particleIndex::type>();
		case array_enum::basicArrays_particleIndexCompact: return type_name<arrays::basicArrays::particleIndexCompact::type>();
		case array_enum::basicArrays_resortArray4: return type_name<arrays::basicArrays::resortArray4::type>();
		case array_enum::basicArrays_resortArray: return type_name<arrays::basicArrays::resortArray::type>();
		case array_enum::dfsphArrays_dfsphSource: return type_name<arrays::dfsphArrays::dfsphSource::type>();
		case array_enum::dfsphArrays_dfsphAlpha: return type_name<arrays::dfsphArrays::dfsphAlpha::type>();
		case array_enum::dfsphArrays_dfsphDpDt: return type_name<arrays::dfsphArrays::dfsphDpDt::type>();
		case array_enum::dfsphArrays_dfsphRhoStar: return type_name<arrays::dfsphArrays::dfsphRhoStar::type>();
		case array_enum::iisphArrays_iisphSum: return type_name<arrays::iisphArrays::iisphSum::type>();
		case array_enum::iisphArrays_iisphDii: return type_name<arrays::iisphArrays::iisphDii::type>();
		case array_enum::iisphArrays_iisphAiiOld: return type_name<arrays::iisphArrays::iisphAiiOld::type>();
		case array_enum::iisphArrays_omega: return type_name<arrays::iisphArrays::omega::type>();
		case array_enum::iisphArrays_alpha: return type_name<arrays::iisphArrays::alpha::type>();
		case array_enum::iisphArrays_gamma: return type_name<arrays::iisphArrays::gamma::type>();
		case array_enum::iisphArrays_sigma: return type_name<arrays::iisphArrays::sigma::type>();
		case array_enum::iisphArrays_iisphDensityAdvection: return type_name<arrays::iisphArrays::iisphDensityAdvection::type>();
		case array_enum::iisphArrays_iisphDensityIteration: return type_name<arrays::iisphArrays::iisphDensityIteration::type>();
		case array_enum::iisphArrays_kernelBuffer: return type_name<arrays::iisphArrays::kernelBuffer::type>();
		case array_enum::iisphArrays_velocityAdvection: return type_name<arrays::iisphArrays::velocityAdvection::type>();
		case array_enum::iisph17_sourceTerm: return type_name<arrays::iisph17::sourceTerm::type>();
		case array_enum::iisph17_boundaryPressure: return type_name<arrays::iisph17::boundaryPressure::type>();
		case array_enum::iisph17_omega: return type_name<arrays::iisph17::omega::type>();
		case array_enum::iisph17_apparentVolume: return type_name<arrays::iisph17::apparentVolume::type>();
		case array_enum::iisph17_Aii: return type_name<arrays::iisph17::Aii::type>();
		case array_enum::iisph17_predictedAcceleration: return type_name<arrays::iisph17::predictedAcceleration::type>();
		case array_enum::iisph17_volumeError: return type_name<arrays::iisph17::volumeError::type>();
		case array_enum::iisph17_band_rb__sourceTerm: return type_name<arrays::iisph17_band_rb::_sourceTerm::type>();
		case array_enum::iisph17_band_rb__omega: return type_name<arrays::iisph17_band_rb::_omega::type>();
		case array_enum::iisph17_band_rb__apparentVolume: return type_name<arrays::iisph17_band_rb::_apparentVolume::type>();
		case array_enum::iisph17_band_rb__Aii: return type_name<arrays::iisph17_band_rb::_Aii::type>();
		case array_enum::iisph17_band_rb__predictedAcceleration: return type_name<arrays::iisph17_band_rb::_predictedAcceleration::type>();
		case array_enum::iisph17_band_rb__volumeError: return type_name<arrays::iisph17_band_rb::_volumeError::type>();
		case array_enum::volumeInletArrays_inletPositions: return type_name<arrays::volumeInletArrays::inletPositions::type>();
		case array_enum::volumeInletArrays_inletCounter: return type_name<arrays::volumeInletArrays::inletCounter::type>();
		case array_enum::volumeInletArrays_volumeOutletVolumes: return type_name<arrays::volumeInletArrays::volumeOutletVolumes::type>();
		case array_enum::volumeInletArrays_volumeOutletDimensions: return type_name<arrays::volumeInletArrays::volumeOutletDimensions::type>();
		case array_enum::volumeInletArrays_volumeOutletMin: return type_name<arrays::volumeInletArrays::volumeOutletMin::type>();
		case array_enum::volumeInletArrays_volumeOutletMax: return type_name<arrays::volumeInletArrays::volumeOutletMax::type>();
		case array_enum::volumeInletArrays_volumeOutletRate: return type_name<arrays::volumeInletArrays::volumeOutletRate::type>();
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return type_name<arrays::volumeInletArrays::volumeOutletRateAccumulator::type>();
		case array_enum::structureArrays_compactCellList: return type_name<arrays::structureArrays::compactCellList::type>();
		case array_enum::structureArrays_neighborMask: return type_name<arrays::structureArrays::neighborMask::type>();
		case array_enum::structureArrays_compactCellScale: return type_name<arrays::structureArrays::compactCellScale::type>();
		case array_enum::structureArrays_spanNeighborList: return type_name<arrays::structureArrays::spanNeighborList::type>();
		case array_enum::structureArrays_neighborList: return type_name<arrays::structureArrays::neighborList::type>();
		case array_enum::structureArrays_neighborListLength: return type_name<arrays::structureArrays::neighborListLength::type>();
		case array_enum::renderArrays_anisotropicMatrices: return type_name<arrays::renderArrays::anisotropicMatrices::type>();
		case array_enum::renderArrays_centerPosition: return type_name<arrays::renderArrays::centerPosition::type>();
		case array_enum::renderArrays_auxTest: return type_name<arrays::renderArrays::auxTest::type>();
		case array_enum::renderArrays_auxIsoDensity: return type_name<arrays::renderArrays::auxIsoDensity::type>();
		case array_enum::renderArrays_auxDistance: return type_name<arrays::renderArrays::auxDistance::type>();
		case array_enum::renderArrays_auxHashMap: return type_name<arrays::renderArrays::auxHashMap::type>();
		case array_enum::renderArrays_auxCellSpan: return type_name<arrays::renderArrays::auxCellSpan::type>();
		case array_enum::renderArrays_auxCellInformation: return type_name<arrays::renderArrays::auxCellInformation::type>();
		case array_enum::renderArrays_auxCellSurface: return type_name<arrays::renderArrays::auxCellSurface::type>();
		case array_enum::structureArrays_classification: return type_name<arrays::structureArrays::classification::type>();
		case array_enum::structureArrays_resortIndex: return type_name<arrays::structureArrays::resortIndex::type>();
		case array_enum::structureArrays_ZOrder_64: return type_name<arrays::structureArrays::ZOrder_64::type>();
		case array_enum::structureArrays_ZOrder_32: return type_name<arrays::structureArrays::ZOrder_32::type>();
		case array_enum::structureArrays_cellSpanSwap: return type_name<arrays::structureArrays::cellSpanSwap::type>();
		case array_enum::structureArrays_cellSpan: return type_name<arrays::structureArrays::cellSpan::type>();
		case array_enum::structureArrays_compactCellSpanSwap: return type_name<arrays::structureArrays::compactCellSpanSwap::type>();
		case array_enum::structureArrays_compactCellSpan: return type_name<arrays::structureArrays::compactCellSpan::type>();
		case array_enum::structureArrays_MLMResolution: return type_name<arrays::structureArrays::MLMResolution::type>();
		case array_enum::structureArrays_cellparticleIndex: return type_name<arrays::structureArrays::cellparticleIndex::type>();
		case array_enum::structureArrays_compactparticleIndex: return type_name<arrays::structureArrays::compactparticleIndex::type>();
		case array_enum::structureArrays_hashMap: return type_name<arrays::structureArrays::hashMap::type>();
		case array_enum::structureArrays_compactHashMap: return type_name<arrays::structureArrays::compactHashMap::type>();
		case array_enum::structureArrays_mlmScaling: return type_name<arrays::structureArrays::mlmScaling::type>();
		case array_enum::structureArrays_particleparticleIndex: return type_name<arrays::structureArrays::particleparticleIndex::type>();
		case array_enum::structureArrays_cellBegin: return type_name<arrays::structureArrays::cellBegin::type>();
		case array_enum::structureArrays_cellEnd: return type_name<arrays::structureArrays::cellEnd::type>();
		case array_enum::supportArrays_support: return type_name<arrays::supportArrays::support::type>();
		case array_enum::supportArrays_supportEstimate: return type_name<arrays::supportArrays::supportEstimate::type>();
		case array_enum::supportArrays_neighborCount: return type_name<arrays::supportArrays::neighborCount::type>();
		case array_enum::supportArrays_supportMarker: return type_name<arrays::supportArrays::supportMarker::type>();
		case array_enum::supportArrays_supportMarkerCompacted: return type_name<arrays::supportArrays::supportMarkerCompacted::type>();
		case array_enum::supportArrays_closestNeighbor: return type_name<arrays::supportArrays::closestNeighbor::type>();
		case array_enum::supportArrays_closestNeighbor_f: return type_name<arrays::supportArrays::closestNeighbor_f::type>();
		case array_enum::supportArrays_neighborOverhead: return type_name<arrays::supportArrays::neighborOverhead::type>();
		case array_enum::supportArrays_neighborOverheadCount: return type_name<arrays::supportArrays::neighborOverheadCount::type>();
		case array_enum::supportArrays_neighborListSwap: return type_name<arrays::supportArrays::neighborListSwap::type>();
		default: return std::string("invalidEnum");
	}
}
#endif
auto inline getArraySwappable(array_enum e){
	switch(e){
		case array_enum::adaptive_mergeable: return has_rear_ptr<arrays::adaptive::mergeable> ;
		case array_enum::adaptive_mergeCounter: return has_rear_ptr<arrays::adaptive::mergeCounter> ;
		case array_enum::adaptive_ptclCounter: return has_rear_ptr<arrays::adaptive::ptclCounter> ;
		case array_enum::adaptive_classification: return has_rear_ptr<arrays::adaptive::classification> ;
		case array_enum::adaptive_splitIndicator: return has_rear_ptr<arrays::adaptive::splitIndicator> ;
		case array_enum::adaptive_adaptivityCounter: return has_rear_ptr<arrays::adaptive::adaptivityCounter> ;
		case array_enum::adaptive_splitIndicatorCompacted: return has_rear_ptr<arrays::adaptive::splitIndicatorCompacted> ;
		case array_enum::adaptive_parentIndex: return has_rear_ptr<arrays::adaptive::parentIndex> ;
		case array_enum::adaptive_parentVolume: return has_rear_ptr<arrays::adaptive::parentVolume> ;
		case array_enum::adaptive_parentPosition: return has_rear_ptr<arrays::adaptive::parentPosition> ;
		case array_enum::advectionArrays_maxVelocity: return has_rear_ptr<arrays::advectionArrays::maxVelocity> ;
		case array_enum::advectionArrays_cflValue: return has_rear_ptr<arrays::advectionArrays::cflValue> ;
		case array_enum::advectionArrays_particleNormal: return has_rear_ptr<arrays::advectionArrays::particleNormal> ;
		case array_enum::advectionArrays_angularVelocity: return has_rear_ptr<arrays::advectionArrays::angularVelocity> ;
		case array_enum::rigidBodyArrays_rigidDensities: return has_rear_ptr<arrays::rigidBodyArrays::rigidDensities> ;
		case array_enum::rigidBodyArrays_rigidVolumes: return has_rear_ptr<arrays::rigidBodyArrays::rigidVolumes> ;
		case array_enum::rigidBodyArrays_rigidLinearVelocities: return has_rear_ptr<arrays::rigidBodyArrays::rigidLinearVelocities> ;
		case array_enum::rigidBodyArrays_rigidAVelocities: return has_rear_ptr<arrays::rigidBodyArrays::rigidAVelocities> ;
		case array_enum::rigidBodyArrays_rigidOrigins: return has_rear_ptr<arrays::rigidBodyArrays::rigidOrigins> ;
		case array_enum::rigidBodyArrays_rigidQuaternions: return has_rear_ptr<arrays::rigidBodyArrays::rigidQuaternions> ;
		case array_enum::boundaryArrays_splineLUT: return has_rear_ptr<arrays::boundaryArrays::splineLUT> ;
		case array_enum::boundaryArrays_offsetLUT: return has_rear_ptr<arrays::boundaryArrays::offsetLUT> ;
		case array_enum::boundaryArrays_spikyLUT: return has_rear_ptr<arrays::boundaryArrays::spikyLUT> ;
		case array_enum::boundaryArrays_adhesionLUT: return has_rear_ptr<arrays::boundaryArrays::adhesionLUT> ;
		case array_enum::boundaryArrays_volumeLUT: return has_rear_ptr<arrays::boundaryArrays::volumeLUT> ;
		case array_enum::boundaryArrays_spline2LUT: return has_rear_ptr<arrays::boundaryArrays::spline2LUT> ;
		case array_enum::boundaryArrays_splineGradientLUT: return has_rear_ptr<arrays::boundaryArrays::splineGradientLUT> ;
		case array_enum::boundaryArrays_spikyGradientLUT: return has_rear_ptr<arrays::boundaryArrays::spikyGradientLUT> ;
		case array_enum::boundaryArrays_cohesionLUT: return has_rear_ptr<arrays::boundaryArrays::cohesionLUT> ;
		case array_enum::boundaryArrays_boundaryPlanes: return has_rear_ptr<arrays::boundaryArrays::boundaryPlanes> ;
		case array_enum::boundaryArrays_boundaryPlaneVelocity: return has_rear_ptr<arrays::boundaryArrays::boundaryPlaneVelocity> ;
		case array_enum::boundaryArrays_volumeBoundaryVolumes: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryVolumes> ;
		case array_enum::boundaryArrays_volumeBoundaryDimensions: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryDimensions> ;
		case array_enum::boundaryArrays_volumeBoundaryMin: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryMin> ;
		case array_enum::boundaryArrays_volumeBoundaryMax: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryMax> ;
		case array_enum::boundaryArrays_volumeBoundaryDensity: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryDensity> ;
		case array_enum::boundaryArrays_volumeBoundaryVolume: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryVolume> ;
		case array_enum::boundaryArrays_volumeBoundaryVelocity: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryVelocity> ;
		case array_enum::boundaryArrays_volumeBoundaryAngularVelocity: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryAngularVelocity> ;
		case array_enum::boundaryArrays_volumeBoundaryKind: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryKind> ;
		case array_enum::boundaryArrays_volumeBoundaryPosition: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryPosition> ;
		case array_enum::boundaryArrays_volumeBoundaryQuaternion: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryQuaternion> ;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrix: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryTransformMatrix> ;
		case array_enum::boundaryArrays_volumeBoundaryTransformMatrixInverse: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse> ;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrix: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryInertiaMatrix> ;
		case array_enum::boundaryArrays_volumeBoundaryInertiaMatrixInverse: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse> ;
		case array_enum::boundaryArrays_volumeBoundaryAcceleration: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryAcceleration> ;
		case array_enum::boundaryArrays_volumeBoundaryAngularAcceleration: return has_rear_ptr<arrays::boundaryArrays::volumeBoundaryAngularAcceleration> ;
		case array_enum::surfaceArrays_decisionBuffer: return has_rear_ptr<arrays::surfaceArrays::decisionBuffer> ;
		case array_enum::surfaceArrays_surface_idxBuffer: return has_rear_ptr<arrays::surfaceArrays::surface_idxBuffer> ;
		case array_enum::surfaceArrays_markerBuffer: return has_rear_ptr<arrays::surfaceArrays::markerBuffer> ;
		case array_enum::surfaceArrays_distanceBuffer: return has_rear_ptr<arrays::surfaceArrays::distanceBuffer> ;
		case array_enum::surfaceArrays_changeBuffer: return has_rear_ptr<arrays::surfaceArrays::changeBuffer> ;
		case array_enum::basicArrays_matrixTest: return has_rear_ptr<arrays::basicArrays::matrixTest> ;
		case array_enum::basicArrays_position: return has_rear_ptr<arrays::basicArrays::position> ;
		case array_enum::basicArrays_acceleration: return has_rear_ptr<arrays::basicArrays::acceleration> ;
		case array_enum::basicArrays_velocity: return has_rear_ptr<arrays::basicArrays::velocity> ;
		case array_enum::basicArrays_particle_type: return has_rear_ptr<arrays::basicArrays::particle_type> ;
		case array_enum::basicArrays_renderArray: return has_rear_ptr<arrays::basicArrays::renderArray> ;
		case array_enum::basicArrays_debugArray: return has_rear_ptr<arrays::basicArrays::debugArray> ;
		case array_enum::basicArrays_volume: return has_rear_ptr<arrays::basicArrays::volume> ;
		case array_enum::basicArrays_lifetime: return has_rear_ptr<arrays::basicArrays::lifetime> ;
		case array_enum::basicArrays_pressure: return has_rear_ptr<arrays::basicArrays::pressure> ;
		case array_enum::basicArrays_density: return has_rear_ptr<arrays::basicArrays::density> ;
		case array_enum::basicArrays_fluidDensity: return has_rear_ptr<arrays::basicArrays::fluidDensity> ;
		case array_enum::basicArrays_particleIndex: return has_rear_ptr<arrays::basicArrays::particleIndex> ;
		case array_enum::basicArrays_particleIndexCompact: return has_rear_ptr<arrays::basicArrays::particleIndexCompact> ;
		case array_enum::basicArrays_resortArray4: return has_rear_ptr<arrays::basicArrays::resortArray4> ;
		case array_enum::basicArrays_resortArray: return has_rear_ptr<arrays::basicArrays::resortArray> ;
		case array_enum::dfsphArrays_dfsphSource: return has_rear_ptr<arrays::dfsphArrays::dfsphSource> ;
		case array_enum::dfsphArrays_dfsphAlpha: return has_rear_ptr<arrays::dfsphArrays::dfsphAlpha> ;
		case array_enum::dfsphArrays_dfsphDpDt: return has_rear_ptr<arrays::dfsphArrays::dfsphDpDt> ;
		case array_enum::dfsphArrays_dfsphRhoStar: return has_rear_ptr<arrays::dfsphArrays::dfsphRhoStar> ;
		case array_enum::iisphArrays_iisphSum: return has_rear_ptr<arrays::iisphArrays::iisphSum> ;
		case array_enum::iisphArrays_iisphDii: return has_rear_ptr<arrays::iisphArrays::iisphDii> ;
		case array_enum::iisphArrays_iisphAiiOld: return has_rear_ptr<arrays::iisphArrays::iisphAiiOld> ;
		case array_enum::iisphArrays_omega: return has_rear_ptr<arrays::iisphArrays::omega> ;
		case array_enum::iisphArrays_alpha: return has_rear_ptr<arrays::iisphArrays::alpha> ;
		case array_enum::iisphArrays_gamma: return has_rear_ptr<arrays::iisphArrays::gamma> ;
		case array_enum::iisphArrays_sigma: return has_rear_ptr<arrays::iisphArrays::sigma> ;
		case array_enum::iisphArrays_iisphDensityAdvection: return has_rear_ptr<arrays::iisphArrays::iisphDensityAdvection> ;
		case array_enum::iisphArrays_iisphDensityIteration: return has_rear_ptr<arrays::iisphArrays::iisphDensityIteration> ;
		case array_enum::iisphArrays_kernelBuffer: return has_rear_ptr<arrays::iisphArrays::kernelBuffer> ;
		case array_enum::iisphArrays_velocityAdvection: return has_rear_ptr<arrays::iisphArrays::velocityAdvection> ;
		case array_enum::iisph17_sourceTerm: return has_rear_ptr<arrays::iisph17::sourceTerm> ;
		case array_enum::iisph17_boundaryPressure: return has_rear_ptr<arrays::iisph17::boundaryPressure> ;
		case array_enum::iisph17_omega: return has_rear_ptr<arrays::iisph17::omega> ;
		case array_enum::iisph17_apparentVolume: return has_rear_ptr<arrays::iisph17::apparentVolume> ;
		case array_enum::iisph17_Aii: return has_rear_ptr<arrays::iisph17::Aii> ;
		case array_enum::iisph17_predictedAcceleration: return has_rear_ptr<arrays::iisph17::predictedAcceleration> ;
		case array_enum::iisph17_volumeError: return has_rear_ptr<arrays::iisph17::volumeError> ;
		case array_enum::iisph17_band_rb__sourceTerm: return has_rear_ptr<arrays::iisph17_band_rb::_sourceTerm> ;
		case array_enum::iisph17_band_rb__omega: return has_rear_ptr<arrays::iisph17_band_rb::_omega> ;
		case array_enum::iisph17_band_rb__apparentVolume: return has_rear_ptr<arrays::iisph17_band_rb::_apparentVolume> ;
		case array_enum::iisph17_band_rb__Aii: return has_rear_ptr<arrays::iisph17_band_rb::_Aii> ;
		case array_enum::iisph17_band_rb__predictedAcceleration: return has_rear_ptr<arrays::iisph17_band_rb::_predictedAcceleration> ;
		case array_enum::iisph17_band_rb__volumeError: return has_rear_ptr<arrays::iisph17_band_rb::_volumeError> ;
		case array_enum::volumeInletArrays_inletPositions: return has_rear_ptr<arrays::volumeInletArrays::inletPositions> ;
		case array_enum::volumeInletArrays_inletCounter: return has_rear_ptr<arrays::volumeInletArrays::inletCounter> ;
		case array_enum::volumeInletArrays_volumeOutletVolumes: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletVolumes> ;
		case array_enum::volumeInletArrays_volumeOutletDimensions: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletDimensions> ;
		case array_enum::volumeInletArrays_volumeOutletMin: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletMin> ;
		case array_enum::volumeInletArrays_volumeOutletMax: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletMax> ;
		case array_enum::volumeInletArrays_volumeOutletRate: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletRate> ;
		case array_enum::volumeInletArrays_volumeOutletRateAccumulator: return has_rear_ptr<arrays::volumeInletArrays::volumeOutletRateAccumulator> ;
		case array_enum::structureArrays_compactCellList: return has_rear_ptr<arrays::structureArrays::compactCellList> ;
		case array_enum::structureArrays_neighborMask: return has_rear_ptr<arrays::structureArrays::neighborMask> ;
		case array_enum::structureArrays_compactCellScale: return has_rear_ptr<arrays::structureArrays::compactCellScale> ;
		case array_enum::structureArrays_spanNeighborList: return has_rear_ptr<arrays::structureArrays::spanNeighborList> ;
		case array_enum::structureArrays_neighborList: return has_rear_ptr<arrays::structureArrays::neighborList> ;
		case array_enum::structureArrays_neighborListLength: return has_rear_ptr<arrays::structureArrays::neighborListLength> ;
		case array_enum::renderArrays_anisotropicMatrices: return has_rear_ptr<arrays::renderArrays::anisotropicMatrices> ;
		case array_enum::renderArrays_centerPosition: return has_rear_ptr<arrays::renderArrays::centerPosition> ;
		case array_enum::renderArrays_auxTest: return has_rear_ptr<arrays::renderArrays::auxTest> ;
		case array_enum::renderArrays_auxIsoDensity: return has_rear_ptr<arrays::renderArrays::auxIsoDensity> ;
		case array_enum::renderArrays_auxDistance: return has_rear_ptr<arrays::renderArrays::auxDistance> ;
		case array_enum::renderArrays_auxHashMap: return has_rear_ptr<arrays::renderArrays::auxHashMap> ;
		case array_enum::renderArrays_auxCellSpan: return has_rear_ptr<arrays::renderArrays::auxCellSpan> ;
		case array_enum::renderArrays_auxCellInformation: return has_rear_ptr<arrays::renderArrays::auxCellInformation> ;
		case array_enum::renderArrays_auxCellSurface: return has_rear_ptr<arrays::renderArrays::auxCellSurface> ;
		case array_enum::structureArrays_classification: return has_rear_ptr<arrays::structureArrays::classification> ;
		case array_enum::structureArrays_resortIndex: return has_rear_ptr<arrays::structureArrays::resortIndex> ;
		case array_enum::structureArrays_ZOrder_64: return has_rear_ptr<arrays::structureArrays::ZOrder_64> ;
		case array_enum::structureArrays_ZOrder_32: return has_rear_ptr<arrays::structureArrays::ZOrder_32> ;
		case array_enum::structureArrays_cellSpanSwap: return has_rear_ptr<arrays::structureArrays::cellSpanSwap> ;
		case array_enum::structureArrays_cellSpan: return has_rear_ptr<arrays::structureArrays::cellSpan> ;
		case array_enum::structureArrays_compactCellSpanSwap: return has_rear_ptr<arrays::structureArrays::compactCellSpanSwap> ;
		case array_enum::structureArrays_compactCellSpan: return has_rear_ptr<arrays::structureArrays::compactCellSpan> ;
		case array_enum::structureArrays_MLMResolution: return has_rear_ptr<arrays::structureArrays::MLMResolution> ;
		case array_enum::structureArrays_cellparticleIndex: return has_rear_ptr<arrays::structureArrays::cellparticleIndex> ;
		case array_enum::structureArrays_compactparticleIndex: return has_rear_ptr<arrays::structureArrays::compactparticleIndex> ;
		case array_enum::structureArrays_hashMap: return has_rear_ptr<arrays::structureArrays::hashMap> ;
		case array_enum::structureArrays_compactHashMap: return has_rear_ptr<arrays::structureArrays::compactHashMap> ;
		case array_enum::structureArrays_mlmScaling: return has_rear_ptr<arrays::structureArrays::mlmScaling> ;
		case array_enum::structureArrays_particleparticleIndex: return has_rear_ptr<arrays::structureArrays::particleparticleIndex> ;
		case array_enum::structureArrays_cellBegin: return has_rear_ptr<arrays::structureArrays::cellBegin> ;
		case array_enum::structureArrays_cellEnd: return has_rear_ptr<arrays::structureArrays::cellEnd> ;
		case array_enum::supportArrays_support: return has_rear_ptr<arrays::supportArrays::support> ;
		case array_enum::supportArrays_supportEstimate: return has_rear_ptr<arrays::supportArrays::supportEstimate> ;
		case array_enum::supportArrays_neighborCount: return has_rear_ptr<arrays::supportArrays::neighborCount> ;
		case array_enum::supportArrays_supportMarker: return has_rear_ptr<arrays::supportArrays::supportMarker> ;
		case array_enum::supportArrays_supportMarkerCompacted: return has_rear_ptr<arrays::supportArrays::supportMarkerCompacted> ;
		case array_enum::supportArrays_closestNeighbor: return has_rear_ptr<arrays::supportArrays::closestNeighbor> ;
		case array_enum::supportArrays_closestNeighbor_f: return has_rear_ptr<arrays::supportArrays::closestNeighbor_f> ;
		case array_enum::supportArrays_neighborOverhead: return has_rear_ptr<arrays::supportArrays::neighborOverhead> ;
		case array_enum::supportArrays_neighborOverheadCount: return has_rear_ptr<arrays::supportArrays::neighborOverheadCount> ;
		case array_enum::supportArrays_neighborListSwap: return has_rear_ptr<arrays::supportArrays::neighborListSwap> ;
		default: return false;
	}
}
