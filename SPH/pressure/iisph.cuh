#pragma once
#include <utility/identifier.h>
/*
Module used to implement an Implicit Incompressible SPH (IISPH) Solver modified from the original IISPH paper.
*/
namespace SPH{
	namespace IISPH{
		struct Memory{
			// basic information
			parameter_u<parameters::num_ptcls> num_ptcls;
			parameter_u<parameters::timestep> timestep;
			parameter_u<parameters::radius> radius;
			parameter_u<parameters::rest_density> rest_density;
			parameter_u<parameters::max_numptcls> max_numptcls;

			// parameters
			parameter_u<parameters::boundaryCounter> boundaryCounter;
			parameter_u<parameters::boundaryDampening> boundaryDampening;
			parameter_u<parameters::eta> eta;
			parameter_u<parameters::iterations> iterations;
			parameter_u<parameters::density_error> density_error;
			parameter_u<parameters::jacobi_omega> jacobi_omega;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::iisphSum> iisphSum;
			write_array_u<arrays::iisphDii> iisphDii;
			write_array_u<arrays::iisphAiiOld> iisphAiiOld;
			write_array_u<arrays::kernelBuffer> kernelBuffer;
			write_array_u<arrays::iisphDensityAdvection> iisphDensityAdvection;
			write_array_u<arrays::iisphDensityIteration> iisphDensityIteration;

			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;
			const_array_u<arrays::boundaryPlanes> boundaryPlanes;
			const_array_u<arrays::lifetime> lifetime;

			// output resources (mapped as read/write)
			write_array_u<arrays::acceleration> acceleration;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::pressure> pressure;
			swap_array_u<arrays::velocity> velocity;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::pressure, arrays::velocity>;
			using input_arrays = std::tuple<arrays::position, arrays::density, arrays::volume, arrays::boundaryPlanes, arrays::lifetime>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::kernelBuffer, arrays::iisphDensityAdvection, arrays::iisphDensityIteration>;
			using basic_info_params = std::tuple<parameters::num_ptcls, parameters::timestep, parameters::radius, parameters::rest_density, parameters::max_numptcls>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList>;
			using parameters = std::tuple<parameters::boundaryCounter, parameters::boundaryDampening, parameters::eta, parameters::iterations, parameters::density_error, parameters::jacobi_omega>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::pressure>() == "IISPH";
			return condition;
		}
		
		void pressure_solve(Memory mem = Memory());
	} // namspace IISPH
}// namespace SPH
