#pragma once
#include <utility/identifier.h>
/*
Module used to provide surface tension based on the approaches by Akinci et. al.
*/
namespace SPH{
	namespace AkinciTension{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::tension_akinci> tension_akinci;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::particleNormal> particleNormal;

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
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::velocity, arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<arrays::particleNormal>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryLUT, arrays::boundaryPressureLUT>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList>;
			using parameters = std::tuple<parameters::tension_akinci>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::tension>() == "Akinci";
			return condition;
		}
		
		void tension(Memory mem = Memory());
	} // namspace AkinciTension
}// namespace SPH
