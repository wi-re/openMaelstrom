#pragma once
#include <utility/identifier.h>
/*
Module used to implement a simple resorting algorithm that uses a cell entry for every actual cell in the domain. Does not support infinite domains.
*/
namespace SPH{
	namespace Resort{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			write_array_u<arrays::debugArray> debugArray;

			// parameters
			parameter_u<parameters::grid_size> grid_size;
			parameter_u<parameters::min_domain> min_domain;
			parameter_u<parameters::max_domain> max_domain;
			parameter_u<parameters::cell_size> cell_size;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::position> position;

			// output resources (mapped as read/write)
			write_array_u<arrays::cellBegin> cellBegin;
			write_array_u<arrays::cellEnd> cellEnd;
			write_array_u<arrays::resortIndex> resortIndex;
			write_array_u<arrays::particleparticleIndex> particleparticleIndex;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position>;
			using output_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::resortIndex, arrays::particleparticleIndex>;
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
			using parameters = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size>;
			constexpr static const bool resort = true;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::sorting>() == "linear_cell";
			return condition;
		}
		
		void resortParticles(Memory mem = Memory());
	} // namspace Resort
}// namespace SPH
