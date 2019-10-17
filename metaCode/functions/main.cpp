#include <iostream>
#include <vector>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sstream>
#include <map>
#include <iterator>
#include <regex>
#include <string>

using node_t = std::pair<const std::string, boost::property_tree::ptree>;

std::map<std::string, std::vector<std::string>> global_map;

struct transformation{
	std::vector<std::shared_ptr<transformation>> children;
	transformation* parent = nullptr;

	node_t* node;

	std::stringstream source;
	std::stringstream header;

	std::function<void(transformation&, transformation*, node_t&)> transfer_fn =
		[](auto& node, auto, node_t& tree) {
		for (auto& t_fn : node.children)
			for (auto& children : tree.second)
				t_fn->transform(children);
	};

	transformation(transformation* p = nullptr) :parent(p){}
	transformation(decltype(transfer_fn) t_fn, transformation* p = nullptr) :parent(p), transfer_fn(t_fn) {}

	std::shared_ptr<transformation> addChild() {
		children.push_back(std::make_shared<transformation>(transfer_fn, this));
		return children[children.size() - 1];
	}

	std::shared_ptr<transformation> addChild(decltype(transfer_fn) t_fn){
		children.push_back(std::make_shared<transformation>(t_fn, this));
		return children[children.size() - 1];
	}

	void transform(node_t& t_node) {
		node = &t_node;
		transfer_fn(*this, parent, t_node);
	}
	std::string tuple_h() {
		std::stringstream sstream;
		sstream << header.str() << std::endl;
		for (auto& child : children)
			sstream << child->tuple_h();
		return sstream.str();
	}
	std::string tuple_s() {
		std::stringstream sstream;
		sstream << source.str();
		for (auto& child : children)
			sstream << child->tuple_s();
		return sstream.str();
	}

};
#if defined(_MSC_VER) && !defined(__CLANG__)
#define template_flag
#else
#define template_flag template
#endif
#define TYPE node.second.template_flag get<std::string>("type")
#define NAME parent->node->first

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include <config/config.h>
fs::file_time_type newestFile;

auto resolveIncludes(boost::property_tree::ptree &pt) {
	fs::path source(sourceDirectory);
	auto folder = source / "jsonFiles";
    for(auto& fileName : fs::directory_iterator(folder)){
		auto t = fs::last_write_time(fileName);
		if(t > newestFile)
			newestFile = t;
	  std::stringstream ss;
      std::ifstream file(fileName.path().string());
      ss << file.rdbuf();
      file.close();

      boost::property_tree::ptree pt2;
      boost::property_tree::read_json(ss, pt2);
      auto arrs = pt2.get_child_optional("functions");
      if (arrs) {
        for (auto it : arrs.get()) {
          //std::cout << it.first << " : " << it.second.data() << std::endl;
          pt.add_child(it.first, it.second);
        }
      }
    }  
	auto arrays = fs::path(sourceDirectory) / "functions.json";
	// if(fs::exists(arrays) && fs::last_write_time(arrays) > newestFile)
	// 	return;
	// std::cout << "Writing " << arrays << std::endl;
    // std::ofstream file(arrays.string());
    // boost::property_tree::write_json(file, pt);
}


int main(int, char** argv) try {
	std::cout << "Running function meta-code generation" << std::endl;
	fs::path input(R"()");
	fs::path output(R"()");
	fs::path output_folder(R"()");
	input = argv[1];
	output = argv[2];
	output_folder = argv[3];
	std::vector<std::string> headers;
	std::stringstream ss;

	boost::property_tree::ptree pt;
	//boost::property_tree::read_json(ss, pt);

	resolveIncludes(pt);

	transformation base_transformation;
	auto function_transformation = base_transformation.addChild([&](auto&, auto, auto& node) {
		std::string function_str = R"(#pragma once
#include <utility/identifier.h>
/*
$description
*/
namespace SPH{
	namespace $name{
		struct Memory{
			// basic information$basic_info
			// parameters$parameter
			// temporary resources (mapped as read/write)$temporary
			// input resources (mapped as read only)$input
			// output resources (mapped as read/write)$output
			// swap resources (mapped as read/write)$swap
			// cell resources (mapped as read only)$cell_info
			// neighborhood resources (mapped as read only)$neighbor_info
			// virtual resources (mapped as read only)$virtual_info
			// volume boundary resources (mapped as read only)$boundaryInfo
			$using
			$properties
		};
		//valid checking function
		inline bool valid(Memory){
			$valid_str
			return condition;
		}
		$functions
	} // namspace $name
}// namespace SPH
)";
		
		//std::cout << parent->node->first << " -> " << node.first << " : " << node.second.template get<std::string>("description") << std::endl;

		auto description = node.second.template_flag get("description", std::string("no description"));
		auto name = node.second.template get<std::string>("name");
		auto units = node.second.template_flag get("units", false);

		std::stringstream usings;
		std::vector<std::function<std::string(std::string)>> text_fn;

		auto arr_fn = [&](auto text, std::string prefix, std::string arr_t) {
			std::stringstream sstream;
			std::vector<std::string> tuple_elems;

			auto swaps = node.second.template_flag get_child_optional(prefix);
			if (swaps) for (auto swap : swaps.get()) {
				std::string fn_name = swap.second.data();
				std::string copy = boost::replace_all_copy(fn_name, "::", "_");
				std::vector<std::string> splitVec;
				boost::split(splitVec, fn_name, boost::is_any_of("::"), boost::token_compress_on);
				for(int32_t i = 1; i < splitVec.size(); ++i)
					splitVec[i][0] = ::toupper(splitVec[i][0]);
	
				copy = boost::join(splitVec, "");
				sstream << R"(
			)" << arr_t << R"()" << (units ? "_u" : "") << R"(<arrays::)" << fn_name << "> " << copy << ";";
				tuple_elems.push_back("arrays::" + fn_name);
			}
			if (swaps) {
				usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<)";
				for (int i = 0; i < (int32_t)tuple_elems.size(); ++i) {
					usings << tuple_elems[i] << (i != (int32_t)tuple_elems.size() - 1 ? std::string(", ") : std::string(""));
				}
				usings << ">;";
			}
			else {
				usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<>;)";
			}
			sstream << "\n";
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), swaps ? sstream.str() : std::string(""));
		};

		text_fn.push_back([&](auto text) {
			auto dependency_any = node.second.template_flag get_child_optional("depends_any");
			std::string valid_str = R"(bool condition = true;)";
			if (dependency_any) {
				std::stringstream d_ss;
				d_ss << R"(bool condition = false;)";
				for (auto d : dependency_any.get()) {
					for (auto de : d.second) {
						d_ss << R"(
			condition = condition || get<parameters::)" << de.first << R"(>() == )";
						if (de.second.data() == "true") d_ss << "true;";
						else if (de.second.data() == "false") d_ss << "false;";
						else d_ss << R"(")" << de.second.data() << R"(";)";
					}
				}
				valid_str = d_ss.str();
			}
			auto dependency_all = node.second.template_flag get_child_optional("depends_all");
			if (dependency_all) {
				std::stringstream d_ss;
				d_ss << R"(bool condition = true;)";
				for (auto d : dependency_all.get()) {
					for (auto de : d.second) {
						d_ss << R"(
			condition = condition && get<parameters::)" << de.first << R"(>() == )";
						if (de.second.data() == "true") d_ss << "true;";
						else if (de.second.data() == "false") d_ss << "false;";
						else d_ss << R"(")" << de.second.data() << R"(";)";
					}
				}
				valid_str = d_ss.str();
			}
			return std::regex_replace(text, std::regex(R"(\$valid_str)"), valid_str);
		});
		text_fn.push_back([&](auto text) {
			auto functions = node.second.template_flag get_child_optional("functions");
			std::stringstream sstream;
			if (functions) for (auto fn : functions.get()) {
				std::string fn_name = fn.second.data();
				sstream << R"(
		void )" << fn_name << R"((Memory mem = Memory());)";
			}
			return std::regex_replace(text, std::regex(R"(\$functions)"), sstream.str());
		});
		text_fn.push_back([&](auto text) {return arr_fn(text, "swap", "swap_array"); });
		text_fn.push_back([&](auto text) {return arr_fn(text, "input","const_array"); });
		text_fn.push_back([&](auto text) {return arr_fn(text, "output","write_array"); });
		text_fn.push_back([&](auto text) {return arr_fn(text, "temporary","write_array"); });
		text_fn.push_back([&](auto text) {
			auto resort = node.second.template get_optional<bool>("resort");
			auto inlet = node.second.template get_optional<bool>("inlet");

			std::stringstream sstream;
			if (resort && resort.get())
				sstream << "constexpr static const bool resort = true;\n";
			else
				sstream << "constexpr static const bool resort = false;\n";
			if (inlet && inlet.get())
				sstream << "constexpr static const bool inlet = true;";
			else
				sstream << "constexpr static const bool inlet = false;";

			return std::regex_replace(text, std::regex(R"(\$properties)"), sstream.str());
		});
		text_fn.push_back([&](auto text) {
			std::string prefix = "basic_info";
			std::stringstream sstream;
			std::vector<std::string> tuple_params;
			std::vector<std::string> tuple_arrays;

			std::vector<std::string> cell_params = { "num_ptcls", "timestep", "radius", "rest_density","max_numptcls" };
			std::vector<std::string> cell_arrays = {  "debugArray"};

			auto cell_info = node.second.template_flag get("basic_info", true);
			if (cell_info) {
				for (auto param : cell_params) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "parameter" << R"()" << (units ? "_u" : "") << R"(<parameters::)" << param << "> " << param << ";";
					tuple_params.push_back("parameters::" + param);
				}
				sstream << std::endl;

				for (auto param : cell_arrays) {
					sstream << R"(
			)" << "write_array" << R"()" << (units ? "_u" : "") << R"(<arrays::)" << param << "> " << param << ";";
					tuple_arrays.push_back("arrays::" + param);
				}
				sstream << std::endl;

				if (cell_info) {
					usings << R"(
			using )" << prefix << R"(_params = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_params.size(); ++i) {
						usings << tuple_params[i] << (i != (int32_t)tuple_params.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";

			//		usings << R"(
			//using )" << prefix << R"(_params = std::tuple<)";
			//		for (int i = 0; i < tuple_arrays.size(); ++i) {
			//			usings << tuple_arrays[i] << (i != tuple_arrays.size() - 1 ? std::string(", ") : std::string(""));
			//		}
			//		usings << ">;";
				}

			}
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), cell_info ? sstream.str() : std::string("")); });

		text_fn.push_back([&](auto text) {
			std::string prefix = "cell_info";
			std::stringstream sstream;
			std::vector<std::string> tuple_params;
			std::vector<std::string> tuple_arrays;

			std::vector<std::string> cell_params = { "grid_size", "min_domain", "max_domain", "cell_size", "hash_entries", "min_coord","mlm_schemes" };
			std::vector<std::string> cell_arrays = { "cellBegin", "cellEnd", "cellSpan", "hashMap", "compactHashMap", "compactCellSpan", "MLMResolution"  };

			auto cell_info = node.second.template_flag get("cell_info", false);
			if (cell_info) {
				for (auto param : cell_params) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "parameter" << R"()" << (units ? "_u" : "") << R"(<parameters::)" << param << "> " << param << ";";
					tuple_params.push_back("parameters::" + param);
				}
				sstream << std::endl;

				for (auto param : cell_arrays) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "const_array" << R"()" << (units ? "_u" : "") << R"(<arrays::)" << param << "> " << param << ";";
					tuple_arrays.push_back("arrays::" + param);
				}
				sstream << std::endl;

				if (cell_info) {
						usings << R"(
			using )" << prefix << R"(_params = std::tuple<)";
						for (int i = 0; i < (int32_t)tuple_params.size(); ++i) {
							usings << tuple_params[i] << (i != (int32_t)tuple_params.size() - 1 ? std::string(", ") : std::string(""));
						}
						usings << ">;";

						usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<)";
						for (int i = 0; i < (int32_t)tuple_arrays.size(); ++i) {
							usings << tuple_arrays[i] << (i != (int32_t)tuple_arrays.size() - 1 ? std::string(", ") : std::string(""));
						}
						usings << ">;";
					}

				}
			else {
				usings << R"(
			using )" << prefix << R"(_params = std::tuple<>;
			using )" << prefix << R"(_arrays = std::tuple<>;)";
			}
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), cell_info ? sstream.str() : std::string("")); });

		text_fn.push_back([&](auto text) {
			std::string prefix = "virtual_info";
			std::stringstream sstream;
			std::vector<std::string> tuple_params;
			std::vector<std::string> tuple_arrays;

			std::vector<std::string> cell_params = { "boundaryCounter", "ptcl_spacing", "radius", "boundaryLUTSize", "LUTOffset"};
			std::vector<std::string> cell_arrays = { "boundaryPlanes", "boundaryPlaneVelocity","offsetLUT", "splineLUT", "spline2LUT", "splineGradientLUT", "spikyLUT", "spikyGradientLUT","cohesionLUT", "volumeLUT", "adhesionLUT" };

			auto cell_info = node.second.template_flag get("virtual_info", false);
			if (cell_info) {
				for (auto param : cell_params) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "parameter" << R"()" << (units ? "_u" : "") << R"(<parameters::)" << param << "> " << param << ";";
					tuple_params.push_back("parameters::" + param);
				}
				sstream << std::endl;

				for (auto param : cell_arrays) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "const_array" << R"()" << (units ? "_u" : "") << R"(<arrays::)" << param << "> " << param << ";";
					tuple_arrays.push_back("arrays::" + param);
				}
				sstream << std::endl;

				if (cell_info) {
					usings << R"(
			using )" << prefix << R"(_params = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_params.size(); ++i) {
						usings << tuple_params[i] << (i != (int32_t)tuple_params.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";

					usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_arrays.size(); ++i) {
						usings << tuple_arrays[i] << (i != (int32_t)tuple_arrays.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";
				}
			}
			else {
				usings << R"(
			using )" << prefix << R"(_params = std::tuple<>;
			using )" << prefix << R"(_arrays = std::tuple<>;)";
			}
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), cell_info ? sstream.str() : std::string("")); });

		text_fn.push_back([&](auto text) {
			std::string prefix = "boundaryInfo";
			std::stringstream sstream;
			std::vector<std::string> tuple_params;
			std::vector<std::string> tuple_arrays;

			std::vector<std::string> cell_params = { "volumeBoundaryCounter" };
			std::vector<std::string> cell_arrays = { "volumeBoundaryVolumes", "volumeBoundaryDimensions", "volumeBoundaryMin", "volumeBoundaryMax",
			"volumeBoundaryDensity","volumeBoundaryVolume","volumeBoundaryVelocity","volumeBoundaryAngularVelocity","volumeBoundaryPosition","volumeBoundaryQuaternion",
			"volumeBoundaryTransformMatrix", "volumeBoundaryTransformMatrixInverse","volumeBoundaryKind","volumeBoundaryInertiaMatrix","volumeBoundaryInertiaMatrixInverse" };

			auto cell_info = node.second.template_flag get("boundaryInfo", false);
			if (cell_info) {
				for (auto param : cell_params) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "parameter" << R"()" << (units ? "_u" : "") << R"(<parameters::)" << param << "> " << param << ";";
					tuple_params.push_back("parameters::" + param);
				}
				sstream << std::endl;

				for (auto param : cell_arrays) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "const_array" << R"()" << (units ? "_u" : "") << R"(<arrays::)" << param << "> " << param << ";";
					tuple_arrays.push_back("arrays::" + param);
				}
				sstream << std::endl;

				if (cell_info) {
					usings << R"(
			using )" << prefix << R"(_params = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_params.size(); ++i) {
						usings << tuple_params[i] << (i != (int32_t)tuple_params.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";

					usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_arrays.size(); ++i) {
						usings << tuple_arrays[i] << (i != (int32_t)tuple_arrays.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";
				}
			}
			else {
				usings << R"(
			using )" << prefix << R"(_params = std::tuple<>;
			using )" << prefix << R"(_arrays = std::tuple<>;)";
			}
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), cell_info ? sstream.str() : std::string("")); });
		text_fn.push_back([&](auto text) {
			std::string prefix = "neighbor_info";
			std::stringstream sstream;
			std::vector<std::string> tuple_params;
			std::vector<std::string> tuple_arrays;

			std::vector<std::string> cell_params = {  };
			std::vector<std::string> cell_arrays = { "neighborList", "neighborListLength","spanNeighborList", "compactCellScale", "compactCellList", "neighborMask" };

			auto cell_info = node.second.template_flag get("neighbor_info", false);
			if (cell_info) {
			//	for (auto param : cell_params) {
			//	if (sstream.str().find(param) != std::string::npos)
			//		continue;
			//		sstream << R"(
			//)" << "parameter" << R"()" << (units ? "_u" : "") << R"(<parameters::)" << param << "> " << param << ";";
			//		tuple_params.push_back("parameters::" + param);
			//	}
			//	sstream << std::endl;
				for (auto param : cell_arrays) {
					if (text.find(param + '>') != std::string::npos)
						continue;
					sstream << R"(
			)" << "const_array" << R"()" << (units ? "_u" : "") << R"(<arrays::)" << param << "> " << param << ";";
					tuple_arrays.push_back("arrays::" + param);
				}
				sstream << std::endl;

				if (cell_info) {
			//		usings << R"(
			//using )" << prefix << R"(_params = std::tuple<)";
			//		for (int i = 0; i < tuple_params.size(); ++i) {
			//			usings << tuple_params[i] << (i != tuple_params.size() - 1 ? std::string(", ") : std::string(""));
			//		}
			//		usings << ">;";

					usings << R"(
			using )" << prefix << R"(_arrays = std::tuple<)";
					for (int i = 0; i < (int32_t)tuple_arrays.size(); ++i) {
						usings << tuple_arrays[i] << (i != (int32_t)tuple_arrays.size() - 1 ? std::string(", ") : std::string(""));
					}
					usings << ">;";
				}

			}
			else {
				usings << R"(
			using )" << prefix << R"(_params = std::tuple<>;
			using )" << prefix << R"(_arrays = std::tuple<>;)";
			}
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), cell_info ? sstream.str() : std::string("")); });
		text_fn.push_back([&](auto text) {
			std::string prefix = "parameter";
			std::stringstream sstream;
			std::vector<std::string> tuple_elems;

			auto swaps = node.second.template_flag get_child_optional(prefix + "s");
			if (swaps) for (auto swap : swaps.get()) {
				std::string fn_name = swap.second.data();
				if (text.find(fn_name + '>') != std::string::npos)
					continue;
				std::string copy = boost::replace_all_copy(fn_name, "::", "_");
				sstream << R"(
			)" << prefix << R"()" << (units ? "_u" : "") << R"(<parameters::)" << fn_name << "> " << copy << ";";
				tuple_elems.push_back("parameters::" + fn_name);
			}
			if (swaps) {
				usings << R"(
			using )" << prefix << R"(s = std::tuple<)";
				for (int i = 0; i < (int32_t)tuple_elems.size(); ++i) {
					usings << tuple_elems[i] << (i != (int32_t)tuple_elems.size() - 1 ? std::string(", ") : std::string(""));
				}
				usings << ">;";
			}
			else {
				usings << R"(
			using )" << prefix << R"(s = std::tuple<>;)";
			}
			sstream << "\n";
			return std::regex_replace(text, std::regex(std::string(R"(\$)" + prefix)), swaps ? sstream.str() : std::string("")); });

		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$name)"), node.second.template get<std::string>("name")); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$using)"), usings.str()); }); 
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$description)"), node.second.template_flag get("description",std::string("No Description"))); });

		for (auto f : text_fn)
			function_str = f(function_str);

		fs::path folder = output_folder / "SPH" /  node.second.template get<std::string>("folder");
		fs::create_directories(folder);
		folder /= node.first;

		fs::path header = folder.replace_extension(".cuh");
		fs::path source = folder.replace_extension(".cu");
		std::stringstream header_str;
		header_str << R"(#include <SPH/)" << node.second.template get<std::string>("folder") << "/" << node.first << R"(.cuh>)";


		headers.push_back(header_str.str());

		if (fs::exists(header)) {
			std::stringstream ss;
			std::ifstream file(header.string());
			ss << file.rdbuf();
			file.close();
			if (ss.str() == function_str)
				return;
		}
		std::cout << "Writing header file " << header << std::endl;
		std::ofstream header_out(header.string());
		header_out << function_str;
		header_out.close();
		if (!fs::exists(source)) {
			std::ofstream source_out(source.string());
			source_out << header_str.str() << R"(
#include <utility/include_all.h>
)";
			auto functions = node.second.template_flag get_child_optional("functions");
			if (functions) for (auto fn : functions.get()) {
				std::string fn_name = fn.second.data();
				source_out << R"(
void SPH::)" << name << "::" << fn_name << R"((Memory mem){})";
			}
			source_out.close();
		}
	});
	node_t tree{".",pt };
	base_transformation.transform(tree);


	if (fs::exists(output)) {
		auto input_ts = newestFile;
		auto output_ts = fs::last_write_time(output);
		if (input_ts <= output_ts)
			return 0;		
	}

	fs::create_directories(output.parent_path());
	std::ofstream header_out(output.string());
	header_out << "#pragma once\n";
	for (auto header : headers) {
		header_out << header << std::endl;
	}
	header_out.close();
}
catch (std::exception e) {
	std::cerr << "Caught exception: " << e.what() << std::endl;
	throw;
}