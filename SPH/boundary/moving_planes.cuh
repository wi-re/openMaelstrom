#pragma once
#include <utility/identifier.h>
/*
Module used to provide moving boundaries, e.g. wave walls, in simulations. Implicit plane based.
*/
namespace SPH{
	namespace moving_planes{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::boundaryCounter> boundaryCounter;
			parameter_u<parameters::boundaryDampening> boundaryDampening;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			// output resources (mapped as read/write)
			write_array_u<arrays::boundaryPlanes> boundaryPlanes;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::position> position;
			swap_array_u<arrays::velocity> velocity;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::position, arrays::velocity>;
			using input_arrays = std::tuple<>;
			using output_arrays = std::tuple<arrays::boundaryPlanes>;
			using temporary_arrays = std::tuple<>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			using parameters = std::tuple<parameters::boundaryCounter, parameters::boundaryDampening>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::movingBoundaries>() == true;
			return condition;
		}
		
		void correct_position(Memory mem = Memory());
		void correct_velocity(Memory mem = Memory());
		void update_boundaries(Memory mem = Memory());
	} // namspace moving_planes
}// namespace SPH
