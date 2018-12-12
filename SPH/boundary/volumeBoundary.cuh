#pragma once
#include <utility/identifier.h>
/*
Module used to implement vdb based solid objects. Maps vdb volumes (signed distance fields) to 3d cuda textures.
*/
namespace SPH{
	namespace volume{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::boundaryDampening> boundaryDampening;
			parameter_u<parameters::volumeBoundaryCounter> volumeBoundaryCounter;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array_u<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array_u<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array_u<arrays::volumeBoundaryMax> volumeBoundaryMax;

			// output resources (mapped as read/write)
			// swap resources (mapped as read/write)
			swap_array_u<arrays::position> position;
			swap_array_u<arrays::velocity> velocity;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::position, arrays::velocity>;
			using input_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using output_arrays = std::tuple<>;
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
			using parameters = std::tuple<parameters::boundaryDampening, parameters::volumeBoundaryCounter>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::volumeBoundary>() == true;
			return condition;
		}
		
		void init_volumes(Memory mem = Memory());
	} // namspace volume
}// namespace SPH
