#pragma once
#include <utility/identifier.h>
/*
Module used to implement an explicit Euler integration with particle shifting and timestep calculation
*/
namespace SPH{
	namespace Integration{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::max_velocity> max_velocity;
			parameter_u<parameters::min_domain> min_domain;
			parameter_u<parameters::max_domain> max_domain;
			parameter_u<parameters::boundaryDampening> boundaryDampening;
			parameter_u<parameters::external_force> external_force;
			parameter_u<parameters::ptcl_support> ptcl_support;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::maxVelocity> maxVelocity;
			write_array_u<arrays::cflValue> cflValue;

			// input resources (mapped as read only)
			const_array_u<arrays::acceleration> acceleration;
			const_array_u<arrays::volume> volume;
			const_array_u<arrays::density> density;

			// output resources (mapped as read/write)
			write_array_u<arrays::lifetime> lifetime;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::position> position;
			swap_array_u<arrays::velocity> velocity;

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

			
			using swap_arrays = std::tuple<arrays::position, arrays::velocity>;
			using input_arrays = std::tuple<arrays::acceleration, arrays::volume, arrays::density>;
			using output_arrays = std::tuple<arrays::lifetime>;
			using temporary_arrays = std::tuple<arrays::maxVelocity, arrays::cflValue>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryLUT, arrays::boundaryPressureLUT>;
			using boundaryInfo_params = std::tuple<parameters::volumeBoundaryCounter>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList>;
			using parameters = std::tuple<parameters::max_velocity, parameters::min_domain, parameters::max_domain, parameters::boundaryDampening, parameters::external_force, parameters::ptcl_support>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = true;
			return condition;
		}
		
		void update_positions(Memory mem = Memory());
		void update_velocities(Memory mem = Memory());
	} // namspace Integration
}// namespace SPH
