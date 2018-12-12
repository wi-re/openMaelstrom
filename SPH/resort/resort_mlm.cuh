#pragma once
#include <utility/identifier.h>
/*
Module used to implement a compact hashing style resorting algorithm on GPUs with 64 bit indices, limiting the domain to 2,097,152^3 cells. Additionally implements a multi level scheme that creates multiplle hash tables for a much faster adaptive simulaiton.
*/
namespace SPH{
	namespace resort_mlm{
		struct Memory{
			// basic information
			parameter<parameters::num_ptcls> num_ptcls;
			parameter<parameters::timestep> timestep;
			parameter<parameters::radius> radius;
			parameter<parameters::rest_density> rest_density;
			parameter<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter<parameters::resort_algorithm> resort_algorithm;

			// temporary resources (mapped as read/write)
			write_array<arrays::resortIndex> resortIndex;
			write_array<arrays::particleparticleIndex> particleparticleIndex;
			write_array<arrays::ZOrder_64> ZOrder_64;
			write_array<arrays::ZOrder_32> ZOrder_32;
			write_array<arrays::cellSpanSwap> cellSpanSwap;
			write_array<arrays::cellparticleIndex> cellparticleIndex;
			write_array<arrays::compactparticleIndex> compactparticleIndex;
			write_array<arrays::resortArray> resortArray;
			write_array<arrays::resortArray4> resortArray4;

			// input resources (mapped as read only)
			const_array<arrays::position> position;

			// output resources (mapped as read/write)
			write_array<arrays::cellSpan> cellSpan;
			write_array<arrays::hashMap> hashMap;
			write_array<arrays::MLMResolution> MLMResolution;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			parameter<parameters::grid_size> grid_size;
			parameter<parameters::min_domain> min_domain;
			parameter<parameters::max_domain> max_domain;
			parameter<parameters::cell_size> cell_size;
			parameter<parameters::hash_entries> hash_entries;
			parameter<parameters::min_coord> min_coord;
			parameter<parameters::mlm_schemes> mlm_schemes;

			const_array<arrays::cellBegin> cellBegin;
			const_array<arrays::cellEnd> cellEnd;

			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position>;
			using output_arrays = std::tuple<arrays::cellSpan, arrays::hashMap, arrays::MLMResolution>;
			using temporary_arrays = std::tuple<arrays::resortIndex, arrays::particleparticleIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::resortArray, arrays::resortArray4>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			using parameters = std::tuple<parameters::resort_algorithm>;
			constexpr static const bool resort = true;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::sorting>() == "MLM";
			condition = condition || get<parameters::sorting>() == "hashed_cell";
			return condition;
		}
		
		void resortParticles(Memory mem = Memory());
	} // namspace resort_mlm
}// namespace SPH
