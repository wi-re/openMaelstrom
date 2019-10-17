#pragma once
#include <utility/identifier.h>
/*
Module used to implement vdb based solid objects. Maps vdb volumes (signed distance fields) to 3d cuda textures.
*/
namespace SPH{
	namespace Outlet{
		struct Memory{
			// basic information
			parameter<parameters::num_ptcls> num_ptcls;
			parameter<parameters::timestep> timestep;
			parameter<parameters::radius> radius;
			parameter<parameters::rest_density> rest_density;
			parameter<parameters::max_numptcls> max_numptcls;

			write_array<arrays::debugArray> debugArray;

			// parameters
			parameter<parameters::volumeOutletCounter> volumeOutletCounter;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array<arrays::volume> volume;
			const_array<arrays::volumeOutletVolumes> volumeOutletVolumes;
			const_array<arrays::volumeOutletDimensions> volumeOutletDimensions;
			const_array<arrays::volumeOutletMin> volumeOutletMin;
			const_array<arrays::volumeOutletMax> volumeOutletMax;
			const_array<arrays::volumeOutletRate> volumeOutletRate;

			// output resources (mapped as read/write)
			write_array<arrays::position> position;
			write_array<arrays::volumeOutletRateAccumulator> volumeOutletRateAccumulator;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::volume, arrays::volumeOutletVolumes, arrays::volumeOutletDimensions, arrays::volumeOutletMin, arrays::volumeOutletMax, arrays::volumeOutletRate>;
			using output_arrays = std::tuple<arrays::position, arrays::volumeOutletRateAccumulator>;
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
			using parameters = std::tuple<parameters::volumeOutletCounter>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::volumeOutlets>() == true;
			return condition;
		}
		
		void init(Memory mem = Memory());
		void remove(Memory mem = Memory());
	} // namspace Outlet
}// namespace SPH
