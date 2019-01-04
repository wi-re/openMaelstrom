#pragma once
#include <utility/identifier.h>
/*
Module used to iteratively created a stable surface distance for every particle in the simulation. Very slow.
*/
namespace SPH{
	namespace distance{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::xsph_viscosity> xsph_viscosity;
			parameter_u<parameters::surface_levelLimit> surface_levelLimit;
			parameter_u<parameters::surface_neighborLimit> surface_neighborLimit;
			parameter_u<parameters::surface_phiMin> surface_phiMin;
			parameter_u<parameters::surface_phiChange> surface_phiChange;
			parameter_u<parameters::surface_distanceFieldDistances> surface_distanceFieldDistances;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::decisionBuffer> decisionBuffer;
			write_array_u<arrays::markerBuffer> markerBuffer;
			write_array_u<arrays::changeBuffer> changeBuffer;

			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;
			const_array_u<arrays::velocity> velocity;

			// output resources (mapped as read/write)
			// swap resources (mapped as read/write)
			swap_array_u<arrays::distanceBuffer> distanceBuffer;
			swap_array_u<arrays::surface_idxBuffer> surface_idxBuffer;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;
			const_array_u<arrays::compactCellScale> compactCellScale;
			const_array_u<arrays::compactCellList> compactCellList;

			// virtual resources (mapped as read only)
			parameter_u<parameters::boundaryCounter> boundaryCounter;
			parameter_u<parameters::ptcl_spacing> ptcl_spacing;
			parameter_u<parameters::boundaryLUTSize> boundaryLUTSize;

			const_array_u<arrays::boundaryPlanes> boundaryPlanes;
			const_array_u<arrays::xbarLUT> xbarLUT;
			const_array_u<arrays::ctrLUT> ctrLUT;
			const_array_u<arrays::boundaryLUT> boundaryLUT;
			const_array_u<arrays::boundaryPressureLUT> boundaryPressureLUT;

			// volume boundary resources (mapped as read only)
			parameter_u<parameters::volumeBoundaryCounter> volumeBoundaryCounter;

			const_array_u<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array_u<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array_u<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array_u<arrays::volumeBoundaryMax> volumeBoundaryMax;

			
			using swap_arrays = std::tuple<arrays::distanceBuffer, arrays::surface_idxBuffer>;
			using input_arrays = std::tuple<arrays::position, arrays::density, arrays::volume, arrays::velocity>;
			using output_arrays = std::tuple<>;
			using temporary_arrays = std::tuple<arrays::decisionBuffer, arrays::markerBuffer, arrays::changeBuffer>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryLUT, arrays::boundaryPressureLUT>;
			using boundaryInfo_params = std::tuple<parameters::volumeBoundaryCounter>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList>;
			using parameters = std::tuple<parameters::xsph_viscosity, parameters::surface_levelLimit, parameters::surface_neighborLimit, parameters::surface_phiMin, parameters::surface_phiChange, parameters::surface_distanceFieldDistances>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::surfaceDistance>() == true;
			return condition;
		}
		
		void distance(Memory mem = Memory());
	} // namspace distance
}// namespace SPH
