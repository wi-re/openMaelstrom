#pragma once
#include <utility/identifier.h>
/*
Module used to implement a Divergence Free SPH (DFSPH) Solver. This solver is not very stable although it follows the paper as close as reasonably possible.
*/
namespace SPH{
	namespace DFSPH{
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
			parameter_u<parameters::dfsph_iterations> dfsph_iterations;
			parameter_u<parameters::dfsph_density_error> dfsph_density_error;
			parameter_u<parameters::dfsph_density_eta> dfsph_density_eta;
			parameter_u<parameters::dfsph_divergence_eta> dfsph_divergence_eta;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::dfsphDpDt> dfsphDpDt;
			write_array_u<arrays::dfsphAlpha> dfsphAlpha;
			write_array_u<arrays::dfsphRhoStar> dfsphRhoStar;

			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::acceleration> acceleration;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::velocity> velocity;
			swap_array_u<arrays::dfsphKappa> dfsphKappa;
			swap_array_u<arrays::dfsphKappaDivergence> dfsphKappaDivergence;

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
			parameter_u<parameters::volumeBoundaryCounter> volumeBoundaryCounter;

			const_array_u<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array_u<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array_u<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array_u<arrays::volumeBoundaryMax> volumeBoundaryMax;

			
			using swap_arrays = std::tuple<arrays::velocity, arrays::dfsphKappa, arrays::dfsphKappaDivergence>;
			using input_arrays = std::tuple<arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<parameters::boundaryCounter, parameters::ptcl_spacing, parameters::boundaryLUTSize>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryLUT, arrays::boundaryPressureLUT>;
			using boundaryInfo_params = std::tuple<parameters::volumeBoundaryCounter>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList>;
			using parameters = std::tuple<parameters::eta, parameters::iterations, parameters::density_error, parameters::dfsph_iterations, parameters::dfsph_density_error, parameters::dfsph_density_eta, parameters::dfsph_divergence_eta>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::pressure>() == "DFSPH";
			return condition;
		}
		
		void density_solve(Memory mem = Memory());
		void divergence_solve(Memory mem = Memory());
	} // namspace DFSPH
}// namespace SPH
