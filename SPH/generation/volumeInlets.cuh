#pragma once
#include <utility/identifier.h>
/*
Module used for emitters based on openVDB signed distance fields.
*/
namespace SPH{
	namespace streamInlet{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::omega> omega;
			parameter_u<parameters::overhead_size> overhead_size;
			parameter_u<parameters::target_neighbors> target_neighbors;
			parameter_u<parameters::support_leeway> support_leeway;
			parameter_u<parameters::error_factor> error_factor;
			parameter_u<parameters::resort_algorithm> resort_algorithm;
			parameter_u<parameters::ptcl_spacing> ptcl_spacing;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			// output resources (mapped as read/write)
			write_array_u<arrays::position> position;
			write_array_u<arrays::velocity> velocity;
			write_array_u<arrays::lifetime> lifetime;
			write_array_u<arrays::volume> volume;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			parameter_u<parameters::grid_size> grid_size;
			parameter_u<parameters::min_domain> min_domain;
			parameter_u<parameters::max_domain> max_domain;
			parameter_u<parameters::cell_size> cell_size;
			parameter_u<parameters::hash_entries> hash_entries;
			parameter_u<parameters::min_coord> min_coord;
			parameter_u<parameters::mlm_schemes> mlm_schemes;

			const_array_u<arrays::cellBegin> cellBegin;
			const_array_u<arrays::cellEnd> cellEnd;
			const_array_u<arrays::cellSpan> cellSpan;
			const_array_u<arrays::hashMap> hashMap;
			const_array_u<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<>;
			using output_arrays = std::tuple<arrays::position, arrays::velocity, arrays::lifetime, arrays::volume>;
			using temporary_arrays = std::tuple<>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			using parameters = std::tuple<parameters::omega, parameters::overhead_size, parameters::target_neighbors, parameters::support_leeway, parameters::error_factor, parameters::resort_algorithm, parameters::ptcl_spacing>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = true;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::volumeInlets>() == true;
			return condition;
		}
		
		void init(Memory mem = Memory());
		void emit(Memory mem = Memory());
	} // namspace streamInlet
}// namespace SPH
