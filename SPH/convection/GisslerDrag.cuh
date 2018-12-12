#pragma once
#include <utility/identifier.h>
/*
Module used to provide air drag forces for particles to improve streams and overall behaviour of the simulaltion.
*/
namespace SPH{
	namespace GisslerDrag{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::air_velocity> air_velocity;
			parameter_u<parameters::tension_akinci> tension_akinci;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::velocity> velocity;
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::acceleration> acceleration;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::velocity, arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList>;
			using parameters = std::tuple<parameters::air_velocity, parameters::tension_akinci>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::drag>() == "Gissler17";
			return condition;
		}
		
		void drag(Memory mem = Memory());
	} // namspace GisslerDrag
}// namespace SPH
