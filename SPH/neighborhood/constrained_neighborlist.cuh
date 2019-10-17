#pragma once
#include <utility/identifier.h>
/*
Module used to create a Constrained NeighborList based on the original paper. Fairly slow for adaptive simulations.
*/
namespace SPH{
	namespace ConstrainedNeighborList{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			write_array_u<arrays::debugArray> debugArray;

			// parameters
			parameter_u<parameters::omega> omega;
			parameter_u<parameters::overhead_size> overhead_size;
			parameter_u<parameters::target_neighbors> target_neighbors;
			parameter_u<parameters::support_leeway> support_leeway;
			parameter_u<parameters::error_factor> error_factor;
			parameter_u<parameters::resort_algorithm> resort_algorithm;
			parameter_u<parameters::neighborlimit> neighborlimit;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::closestNeighbor> closestNeighbor;
			write_array_u<arrays::supportEstimate> supportEstimate;
			write_array_u<arrays::support> support;
			write_array_u<arrays::supportMarker> supportMarker;
			write_array_u<arrays::supportMarkerCompacted> supportMarkerCompacted;
			write_array_u<arrays::neighborListSwap> neighborListSwap;
			write_array_u<arrays::neighborOverhead> neighborOverhead;
			write_array_u<arrays::neighborOverheadCount> neighborOverheadCount;

			// input resources (mapped as read only)
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::position> position;
			write_array_u<arrays::neighborList> neighborList;
			write_array_u<arrays::neighborListLength> neighborListLength;

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
			const_array_u<arrays::compactHashMap> compactHashMap;
			const_array_u<arrays::compactCellSpan> compactCellSpan;
			const_array_u<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::volume>;
			using output_arrays = std::tuple<arrays::position, arrays::neighborList, arrays::neighborListLength>;
			using temporary_arrays = std::tuple<arrays::closestNeighbor, arrays::supportEstimate, arrays::support, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::neighborListSwap, arrays::neighborOverhead, arrays::neighborOverheadCount>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			using parameters = std::tuple<parameters::omega, parameters::overhead_size, parameters::target_neighbors, parameters::support_leeway, parameters::error_factor, parameters::resort_algorithm, parameters::neighborlimit>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::neighborhood>() == "constrained";
			return condition;
		}
		
		void calculate_neighborlist(Memory mem = Memory());
	} // namspace ConstrainedNeighborList
}// namespace SPH
