#pragma once
#include <utility/identifier.h>
/*
Module used to create a Constrained NeighborList based on the original paper. Fairly slow for adaptive simulations.
*/
namespace SPH{
	namespace compactNeighborMask{
		struct Memory{
			// basic information
			parameter<parameters::num_ptcls> num_ptcls;
			parameter<parameters::timestep> timestep;
			parameter<parameters::radius> radius;
			parameter<parameters::rest_density> rest_density;
			parameter<parameters::max_numptcls> max_numptcls;

			write_array<arrays::debugArray> debugArray;

			// parameters
			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array<arrays::volume> volume;
			const_array<arrays::position> position;

			// output resources (mapped as read/write)
			write_array<arrays::neighborMask> neighborMask;
			write_array<arrays::neighborListLength> neighborListLength;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			parameter<parameters::grid_size> grid_size;
			parameter<parameters::min_domain> min_domain;
			parameter<parameters::max_domain> max_domain;
			parameter<parameters::cell_size> cell_size;
			parameter<parameters::hash_entries> hash_entries;
			parameter<parameters::min_coord> min_coord;
			parameter<parameters::mlm_schemes> mlm_schemes;

			const_array<arrays::cellBegin> cellBegin;
			const_array<arrays::cellEnd> cellEnd;
			const_array<arrays::cellSpan> cellSpan;
			const_array<arrays::hashMap> hashMap;
			const_array<arrays::compactHashMap> compactHashMap;
			const_array<arrays::compactCellSpan> compactCellSpan;
			const_array<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			const_array<arrays::neighborList> neighborList;
			const_array<arrays::spanNeighborList> spanNeighborList;
			const_array<arrays::compactCellScale> compactCellScale;
			const_array<arrays::compactCellList> compactCellList;

			// virtual resources (mapped as read only)
			parameter<parameters::boundaryCounter> boundaryCounter;
			parameter<parameters::ptcl_spacing> ptcl_spacing;
			parameter<parameters::boundaryLUTSize> boundaryLUTSize;
			parameter<parameters::LUTOffset> LUTOffset;

			const_array<arrays::boundaryPlanes> boundaryPlanes;
			const_array<arrays::boundaryPlaneVelocity> boundaryPlaneVelocity;
			const_array<arrays::offsetLUT> offsetLUT;
			const_array<arrays::splineLUT> splineLUT;
			const_array<arrays::spline2LUT> spline2LUT;
			const_array<arrays::splineGradientLUT> splineGradientLUT;
			const_array<arrays::spikyLUT> spikyLUT;
			const_array<arrays::spikyGradientLUT> spikyGradientLUT;
			const_array<arrays::cohesionLUT> cohesionLUT;
			const_array<arrays::volumeLUT> volumeLUT;
			const_array<arrays::adhesionLUT> adhesionLUT;

			// volume boundary resources (mapped as read only)
			parameter<parameters::volumeBoundaryCounter> volumeBoundaryCounter;

			const_array<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array<arrays::volumeBoundaryMax> volumeBoundaryMax;
			const_array<arrays::volumeBoundaryDensity> volumeBoundaryDensity;
			const_array<arrays::volumeBoundaryVolume> volumeBoundaryVolume;
			const_array<arrays::volumeBoundaryVelocity> volumeBoundaryVelocity;
			const_array<arrays::volumeBoundaryAngularVelocity> volumeBoundaryAngularVelocity;
			const_array<arrays::volumeBoundaryPosition> volumeBoundaryPosition;
			const_array<arrays::volumeBoundaryQuaternion> volumeBoundaryQuaternion;
			const_array<arrays::volumeBoundaryTransformMatrix> volumeBoundaryTransformMatrix;
			const_array<arrays::volumeBoundaryTransformMatrixInverse> volumeBoundaryTransformMatrixInverse;
			const_array<arrays::volumeBoundaryKind> volumeBoundaryKind;
			const_array<arrays::volumeBoundaryInertiaMatrix> volumeBoundaryInertiaMatrix;
			const_array<arrays::volumeBoundaryInertiaMatrixInverse> volumeBoundaryInertiaMatrixInverse;

			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::volume, arrays::position>;
			using output_arrays = std::tuple<arrays::neighborMask, arrays::neighborListLength>;
			using temporary_arrays = std::tuple<>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize, parameters::LUTOffset>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::boundaryPlaneVelocity, arrays::offsetLUT, arrays::splineLUT, arrays::spline2LUT, arrays::splineGradientLUT, arrays::spikyLUT, arrays::spikyGradientLUT, arrays::cohesionLUT, arrays::volumeLUT, arrays::adhesionLUT>;
			using boundaryInfo_params = std::tuple<parameters::volumeBoundaryCounter>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax, arrays::volumeBoundaryDensity, arrays::volumeBoundaryVolume, arrays::volumeBoundaryVelocity, arrays::volumeBoundaryAngularVelocity, arrays::volumeBoundaryPosition, arrays::volumeBoundaryQuaternion, arrays::volumeBoundaryTransformMatrix, arrays::volumeBoundaryTransformMatrixInverse, arrays::volumeBoundaryKind, arrays::volumeBoundaryInertiaMatrix, arrays::volumeBoundaryInertiaMatrixInverse>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList>;
			using parameters = std::tuple<>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::neighborhood>() == "masked";
			return condition;
		}
		
		void calculate_neighborlist(Memory mem = Memory());
	} // namspace compactNeighborMask
}// namespace SPH
