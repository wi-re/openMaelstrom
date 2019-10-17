#pragma once
#include <utility/identifier.h>
/*
Module used to implement a simple resorting algorithm that uses a cell entry for every actual cell in the domain. Does not support infinite domains.
*/
namespace SPH{
	namespace auxilliaryMLM{
		struct Memory{
			// basic information
			parameter<parameters::num_ptcls> num_ptcls;
			parameter<parameters::timestep> timestep;
			parameter<parameters::radius> radius;
			parameter<parameters::rest_density> rest_density;
			parameter<parameters::max_numptcls> max_numptcls;

			write_array<arrays::debugArray> debugArray;

			// parameters
			parameter<parameters::resort_algorithm> resort_algorithm;
			parameter<parameters::auxScale> auxScale;
			parameter<parameters::internalLimit> internalLimit;
			parameter<parameters::vrtxNeighborLimit> vrtxNeighborLimit;
			parameter<parameters::pruneVoxel> pruneVoxel;

			// temporary resources (mapped as read/write)
			write_array<arrays::resortIndex> resortIndex;
			write_array<arrays::auxDistance> auxDistance;
			write_array<arrays::particleparticleIndex> particleparticleIndex;
			write_array<arrays::ZOrder_64> ZOrder_64;
			write_array<arrays::ZOrder_32> ZOrder_32;
			write_array<arrays::compactCellSpanSwap> compactCellSpanSwap;
			write_array<arrays::cellparticleIndex> cellparticleIndex;
			write_array<arrays::compactparticleIndex> compactparticleIndex;

			// input resources (mapped as read only)
			const_array<arrays::position> position;
			const_array<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array<arrays::structureArrays::classification> structureArraysClassification;
			write_array<arrays::auxHashMap> auxHashMap;
			write_array<arrays::auxCellSpan> auxCellSpan;
			write_array<arrays::auxIsoDensity> auxIsoDensity;
			write_array<arrays::auxTest> auxTest;

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
			const_array<arrays::cellSpan> cellSpan;
			const_array<arrays::hashMap> hashMap;
			const_array<arrays::compactHashMap> compactHashMap;
			const_array<arrays::compactCellSpan> compactCellSpan;
			const_array<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			const_array<arrays::neighborList> neighborList;
			const_array<arrays::neighborListLength> neighborListLength;
			const_array<arrays::spanNeighborList> spanNeighborList;
			const_array<arrays::compactCellScale> compactCellScale;
			const_array<arrays::compactCellList> compactCellList;
			const_array<arrays::neighborMask> neighborMask;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position, arrays::volume>;
			using output_arrays = std::tuple<arrays::structureArrays::classification, arrays::auxHashMap, arrays::auxCellSpan, arrays::auxIsoDensity, arrays::auxTest>;
			using temporary_arrays = std::tuple<arrays::resortIndex, arrays::auxDistance, arrays::particleparticleIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::compactCellSpanSwap, arrays::cellparticleIndex, arrays::compactparticleIndex>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList, arrays::neighborMask>;
			using parameters = std::tuple<parameters::resort_algorithm, parameters::auxScale, parameters::internalLimit, parameters::vrtxNeighborLimit, parameters::pruneVoxel>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = true;
			condition = condition && get<parameters::modules::rayTracing>() == true;
			condition = condition && get<parameters::modules::sorting>() == "compactMLM";
			return condition;
		}
		
		void generateAuxilliaryGrid(Memory mem = Memory());
	} // namspace auxilliaryMLM
}// namespace SPH
