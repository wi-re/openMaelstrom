#pragma once
#include <utility/cuda.h>
#include <utility/helpers.h>
#include <iostream>

enum struct error_level { last_error, device_synchronize, thread_synchronize };

namespace cuda {
	template <typename T, typename U>
	void checkMessages(T errorMessage, U file = std::string(""), int32_t line = 0,
		error_level err_level = error_level::thread_synchronize) {
		auto err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::stringstream sstream;
			sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
				<< " ) ";
			std::cerr << sstream.str() << std::endl;
			//logger(log_level::error) << sstream.str() << std::endl;
			throw std::runtime_error(sstream.str().c_str());
		}
		if (err_level == error_level::device_synchronize ||
			err_level == error_level::thread_synchronize) {
			auto err = cuda::sync_quiet();
			if (err != cudaSuccess) {
				std::stringstream sstream;
				sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
					<< " ) ";
				std::cerr << sstream.str() << std::endl;
				//logger(log_level::error) << sstream.str() << std::endl;
				throw std::runtime_error(sstream.str().c_str());
			}
		}
		//if (err_level == error_level::thread_synchronize) {
		//	auto err = cudaThreadSynchronize();
		//	if (err != cudaSuccess) {
		//		std::stringstream sstream;
		//		sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
		//			<< " ) ";
		//		std::cerr << sstream.str() << std::endl;
		//		//logger(log_level::error) << sstream.str() << std::endl;
		//		throw std::runtime_error(sstream.str().c_str());
		//	}
		//}
	}

	cudaError_t error_check(std::string message = "");

	cudaError_t error_check(cudaError_t err, std::string message = "");

	cudaError_t error_check_quiet(std::string message = "");

	cudaError_t error_check_quiet(cudaError_t err, std::string message = "");

}

#define CHECK_CUDA(x) cuda::checkMessages(x, __FILE__, __LINE__);
#define CHECK_ERROR cuda::checkMessages("", __FILE__, __LINE__);
