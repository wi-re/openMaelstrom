#pragma once
#include <utility/identifier.h>
/*
Module used to implement an Implicit Incompressible SPH (IISPH) Solver modified from the updated IISPH paper.
*/
namespace SPH{
	namespace IISPH17{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::eta> eta;
			parameter_u<parameters::iterations> iterations;
			parameter_u<parameters::density_error> density_error;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::iisphVolume> iisphVolume;
			write_array_u<arrays::iisphSource> iisphSource;
			write_array_u<arrays::iisphAii> iisphAii;
			write_array_u<arrays::iisphVolumeError> iisphVolumeError;
			write_array_u<arrays::iisphOmega> iisphOmega;
			write_array_u<arrays::iisphAcceleration> iisphAcceleration;

			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::acceleration> acceleration;
			write_array_u<arrays::velocity> velocity;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::pressure> pressure;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;

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

			
			using swap_arrays = std::tuple<arrays::pressure>;
			using input_arrays = std::tuple<arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration, arrays::velocity>;
			using temporary_arrays = std::tuple<arrays::iisphVolume, arrays::iisphSource, arrays::iisphAii, arrays::iisphVolumeError, arrays::iisphOmega, arrays::iisphAcceleration>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryLUT, arrays::boundaryPressureLUT>;
			using boundaryInfo_params = std::tuple<parameters::volumeBoundaryCounter>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList>;
			using parameters = std::tuple<parameters::eta, parameters::iterations, parameters::density_error>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::pressure>() == "IISPH17";
			return condition;
		}
		
		void pressure_solve(Memory mem = Memory());
	} // namspace IISPH17
}// namespace SPH
