#pragma once
#include <utility/identifier.h>
/*
Module used to implement the infinite continuous SPH paper.
*/
namespace SPH{
	namespace adaptive{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::xsph_viscosity> xsph_viscosity;
			parameter_u<parameters::adaptiveRatio> adaptiveRatio;
			parameter_u<parameters::surface_levelLimit> surface_levelLimit;
			parameter_u<parameters::simulationTime> simulationTime;
			parameter_u<parameters::surface_neighborLimit> surface_neighborLimit;
			parameter_u<parameters::surface_phiMin> surface_phiMin;
			parameter_u<parameters::surface_phiChange> surface_phiChange;
			parameter_u<parameters::surface_distanceFieldDistances> surface_distanceFieldDistances;
			parameter_u<parameters::volumeBoundaryCounter> volumeBoundaryCounter;
			parameter_u<parameters::boundaryCounter> boundaryCounter;
			parameter_u<parameters::adaptive_resolution> adaptive_resolution;
			parameter_u<parameters::adaptive_delay> adaptive_delay;
			parameter_u<parameters::blendsteps> blendsteps;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::adaptiveClassification> adaptiveClassification;
			write_array_u<arrays::adaptiveMergeCounter> adaptiveMergeCounter;
			write_array_u<arrays::markerBuffer> markerBuffer;
			write_array_u<arrays::changeBuffer> changeBuffer;
			write_array_u<arrays::adaptiveMergeable> adaptiveMergeable;
			write_array_u<arrays::adaptiveNumPtcls> adaptiveNumPtcls;
			write_array_u<arrays::particleIndex> particleIndex;
			write_array_u<arrays::particleIndexCompact> particleIndexCompact;

			// input resources (mapped as read only)
			// output resources (mapped as read/write)
			write_array_u<arrays::position> position;
			write_array_u<arrays::velocity> velocity;
			write_array_u<arrays::lifetime> lifetime;
			write_array_u<arrays::density> density;
			write_array_u<arrays::volume> volume;
			write_array_u<arrays::adaptiveSplitIndicator> adaptiveSplitIndicator;
			write_array_u<arrays::adaptiveParentVolume> adaptiveParentVolume;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::distanceBuffer> distanceBuffer;
			swap_array_u<arrays::adaptiveParentIndex> adaptiveParentIndex;
			swap_array_u<arrays::adaptiveParentPosition> adaptiveParentPosition;

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
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;
			const_array_u<arrays::compactCellScale> compactCellScale;
			const_array_u<arrays::compactCellList> compactCellList;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::distanceBuffer, arrays::adaptiveParentIndex, arrays::adaptiveParentPosition>;
			using input_arrays = std::tuple<>;
			using output_arrays = std::tuple<arrays::position, arrays::velocity, arrays::lifetime, arrays::density, arrays::volume, arrays::adaptiveSplitIndicator, arrays::adaptiveParentVolume>;
			using temporary_arrays = std::tuple<arrays::adaptiveClassification, arrays::adaptiveMergeCounter, arrays::markerBuffer, arrays::changeBuffer, arrays::adaptiveMergeable, arrays::adaptiveNumPtcls, arrays::particleIndex, arrays::particleIndexCompact>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<parameters::grid_size, parameters::min_domain, parameters::max_domain, parameters::cell_size, parameters::hash_entries, parameters::min_coord, parameters::mlm_schemes>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList>;
			using parameters = std::tuple<parameters::xsph_viscosity, parameters::adaptiveRatio, parameters::surface_levelLimit, parameters::simulationTime, parameters::surface_neighborLimit, parameters::surface_phiMin, parameters::surface_phiChange, parameters::surface_distanceFieldDistances, parameters::volumeBoundaryCounter, parameters::boundaryCounter, parameters::adaptive_resolution, parameters::adaptive_delay, parameters::blendsteps>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = true;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = true;
			condition = condition && get<parameters::surfaceDistance>() == true;
			condition = condition && get<parameters::adaptive>() == true;
			return condition;
		}
		
		void adapt(Memory mem = Memory());
		void blendDensity(Memory mem = Memory());
		void blendVelocity(Memory mem = Memory());
	} // namspace adaptive
}// namespace SPH
