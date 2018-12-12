#include <iostream>
#include <vector>
//#define BOOST_MSVC
//using std::is_assignable;
//using std::is_volatile;
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

#if defined(_MSC_VER) && !defined(__clang__)
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
      auto arrs = pt2.get_child_optional("arrays");
      if (arrs) {
        for (auto it : arrs.get()) {
          //std::cout << it.first << " : " << it.second.data() << std::endl;
          pt.add_child(it.first, it.second);
        }
      }
    }  
	auto arrays = fs::path(sourceDirectory) / "arrays.json";
	// if(fs::exists(arrays) && fs::last_write_time(arrays) > newestFile)
	// 	return;
	// std::cout << "Writing " << arrays << std::endl;
    // std::ofstream file(arrays.string());
    // boost::property_tree::write_json(file, pt);
}

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "Wrong number of arguments provided" << std::endl;
		return 1;
	}
	std::cout << "Running array meta-code generation" << std::endl;
	fs::path output(R"()");

	output = argv[1];
	
	boost::property_tree::ptree pt;
	//boost::property_tree::read_json(ss, pt);

	resolveIncludes(pt);

	transformation base_transformation;
	auto category_transformation = base_transformation.addChild();
	auto array_transformation = category_transformation->addChild([](auto& curr, auto, auto& node) {
		//std::cout << parent->parent->node->first << " -> " << parent->node->first << " -> " << node.first << " : " << node.second.template get<std::string>("type")<< std::endl;

		std::stringstream sstream;
		for (auto t_fn : curr.children)
			t_fn->transform(node);

	});
	
	array_transformation->addChild([](auto&, auto parent, auto& node) {
		std::string header_text = R"(
struct $identifier{
	using type = $type;
	using unit_type = $unit;
	static constexpr const array_enum identifier = array_enum::$identifier;
	static constexpr const auto variableName = "$identifier";
	static $type* ptr;
	static size_t alloc_size;
	static constexpr const auto description = "$description";
$swap_h
	static constexpr const memory_kind kind = memory_kind::$kind;
$default_alloc_h
	static void allocate(size_t size);
	static void free();
	operator type*();
	type& operator[](size_t idx);
	static bool valid();
	template<class T> static inline auto& get_member(T& var) { return var.$identifier;}
};
)";

		std::string src_text = R"(
$type* arrays::$identifier::ptr = nullptr;
size_t arrays::$identifier::alloc_size = 0;
$swap_src
$default_alloc_src
$functions_src)";


		std::vector<std::function<std::string(std::string)>> text_fn;
		text_fn.push_back([&](auto text) {
			auto swap = node.second.template_flag get("swap", false);
			std::string swap_str = "";
			if ( swap) {
				swap_str =  R"(
	static void swap();
	static $type* rear_ptr;)";
			}
			return std::regex_replace(text, std::regex(R"(\$swap_h)"), swap_str);
		});
		text_fn.push_back([&](auto text) {
			auto swap = node.second.template_flag get("swap", false);
			std::string swap_str = "";
			if (swap) {
				swap_str = R"(
$type* arrays::$identifier::rear_ptr = nullptr;
void arrays::$identifier::swap() { std::swap(ptr, rear_ptr); })";
			}
			return std::regex_replace(text, std::regex(R"(\$swap_src)"), swap_str);
		});
		text_fn.push_back([&](auto text) {
			//auto swap = node.second.template get("swap", false);
			auto kind = node.second.template_flag get("kind", std::string("customData"));

			std::string swap_str = "";
			if (kind != "customData") {
				swap_str = R"(
	static void defaultAllocate();
	static void leanAllocate();)";
			}
			return std::regex_replace(text, std::regex(R"(\$default_alloc_h)"), swap_str);
		});
		text_fn.push_back([&](auto text) {
			auto swap = node.second.template_flag get("swap", false);
			auto size = node.second.template_flag get_optional<int32_t>("size");
			auto kind = node.second.template_flag get("kind", std::string("customData"));

			std::string swap_str = "";
			if (kind != "customData") {
				swap_str = R"()";
				std::string size_str = "1";
				if (kind == "singleData") size_str = "(1)";
				if (kind == "particleData") size_str = "(get<parameters::max_numptcls>() + 1)";
				if (kind == "diffuseData") size_str = "get<parameters::max_diffuse_numptcls>()";
				if (kind == "cellData") size_str = "get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z";
				swap_str = R"(
void arrays::$identifier::defaultAllocate(){
	auto elems = $size_str;
	alloc_size = $size * elems * sizeof($type);
	cudaAllocateMemory(&ptr, alloc_size);$malloc_rear
}
void arrays::$identifier::leanAllocate(){
	auto elems = $size_str;
	alloc_size = $size * elems * sizeof($type);
	$malloc_lean
})";
				swap_str = std::regex_replace(swap_str, std::regex(R"(\$size_str)"), size_str);
				if (size)
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), std::to_string(size.get()));
				else if (auto sz = node.second.template_flag get_optional<std::string>("size"))
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), sz.get());
				else
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), "1");
				swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_rear)"), (swap ? R"(
	cudaAllocateMemory(&rear_ptr, alloc_size);)" : ""));
				swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_lean)"), (swap ? R"(
	cudaAllocateMemory(&ptr, alloc_size);)" : ""));
			}
			return std::regex_replace(text, std::regex(R"(\$default_alloc_src)"), swap_str);
		});
		text_fn.push_back([&](auto text) {
			auto swap = node.second.template_flag get("swap", false);
			auto dependency_any = node.second.template_flag get_child_optional("depends_any");
			auto dependency_all = node.second.template_flag get_child_optional("depends_all");

			std::string swap_str = R"(
void arrays::$identifier::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);$malloc_rear
}
void arrays::$identifier::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;$free_rear
}
arrays::$identifier::operator type*(){ return ptr;}
$type& arrays::$identifier::operator[](size_t idx){ return ptr[idx];}
bool arrays::$identifier::valid(){
	$valid_str
	return condition;
})";
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

swap_str = std::regex_replace(swap_str, std::regex(R"(\$valid_str)"), valid_str);
			swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_rear)"), (swap ? R"(
	cudaAllocateMemory(&rear_ptr, alloc_size);)" : ""));
			swap_str = std::regex_replace(swap_str, std::regex(R"(\$free_rear)"), (swap ? R"(
	cudaFree(rear_ptr);
	rear_ptr = nullptr;)" : ""));
			
			return std::regex_replace(text, std::regex(R"(\$functions_src)"), swap_str);
		});
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$name)"), node.second.template_flag get("identifier", std::string(parent->node->first))); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$description)"), node.second.template_flag get("description", std::string(""))); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$identifier)"), node.second.template_flag get("identifier", std::string(parent->node->first))); });
		
		text_fn.push_back([&](auto text) {
			auto unit = node.second.template_flag get("unit", std::string("none"));
			if (unit == "void")
				unit = "void_unit_ty";
			std::string unit_text = "value_unit<$type, " + unit + ">";
			if (unit == "none")
				unit_text = "$type";
			return std::regex_replace(text, std::regex(R"(\$unit)"), unit_text); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$type)"), node.second.template_flag get<std::string>("type")); });
		//text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$unit)"), node.second.template get("unit", std::string("void"))); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$kind)"), node.second.template_flag get("kind", std::string("customData"))); });

		if (auto resort = node.second.template_flag get("resort", false); resort) {
			global_map["sorting_list"].push_back(NAME);
		};
		if (auto kind = node.second.template_flag get_optional<std::string>("kind"); kind) {
			if (kind.get() == "particleData")
				global_map["swapping_list"].push_back(NAME);
			else if (kind.get() == "diffuseData")
				global_map["diffuse_swapping_list"].push_back(NAME);
			if (kind.get() != "customData") {
				global_map["allocations_list"].push_back(NAME);
			}
		}

		for (auto f : text_fn)
			header_text = f(header_text);
		for (auto f : text_fn)
			src_text = f(src_text);

		global_map["arrays_list"].push_back(NAME);

		parent->header << header_text;
		parent->source << src_text;

	});

	node_t tree{".",pt };
	base_transformation.transform(tree);

	std::stringstream header;
	header << R"(#pragma once
#include <type_traits>
#include <utility>
#include <tuple>
#include <vector>
#include <array>
#include <cstring>
#include <string>
#include <array>
#include <utility/template/nonesuch.h>
#include <utility/unit_math.h>
#include <texture_types.h>

)";
	header << R"(
enum struct array_enum{)";
	auto arrs = global_map["arrays_list"];
	for (int i = 0; i < (int32_t)arrs.size(); ++i) {
		header << arrs[i];
		if (i != (int32_t)arrs.size() - 1) header << ", ";
	}
	header << R"(};
)";

	header << R"(
#include <utility/identifier/resource_helper.h>
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
constexpr T info() { return T(); }
template<typename T, typename = std::enable_if_t<std::is_same<array_enum, std::decay_t<decltype(T::identifier)>>::value>>
typename T::type* get() { return T().ptr; }

namespace arrays{
)";

	header << base_transformation.tuple_h() << std::endl;

	header << R"(
})";
	for (auto tup : global_map) {
		header << R"(
extern std::tuple<)";
		for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i) {
			header << "arrays::" << tup.second[i] << "";
			if (i != (int32_t)tup.second.size() - 1) header << ", ";
		}
		header << "> " << tup.first << ";";
	}

	header << R"(
template<typename T, bool b>
struct arrays_add_const;

template <typename T >
struct arrays_add_const<T, false> {
	using type = T;
};

template < typename T >
struct arrays_add_const<T, true> {
	using type = const T;
};

template< typename T>
using const_array = typename arrays_add_const<typename T::type, true>::type* __restrict;
template< typename T>
using write_array = typename arrays_add_const<typename T::type, false>::type* __restrict;
template< typename T, bool b = false>
using swap_array = std::pair<typename arrays_add_const<typename T::type, b>::type* __restrict__, typename arrays_add_const<typename T::type, b>::type* __restrict__>;

template< typename T>
using const_array_u = typename arrays_add_const<typename T::unit_type, true>::type* __restrict;
template< typename T>
using write_array_u = typename arrays_add_const<typename T::unit_type, false>::type* __restrict;
template< typename T, bool b = false>
using swap_array_u = std::pair<typename arrays_add_const<typename T::unit_type, b>::type* __restrict__, typename arrays_add_const<typename T::unit_type, b>::type* __restrict__>;
)";

	std::stringstream source;
	source << R"(
#include "utility/identifier/arrays.h"
#include "utility/identifier/uniform.h"
#include <cuda.h>
#include <cuda_runtime.h>
)";
	source << base_transformation.tuple_s() << std::endl;

	for (auto tup : global_map) {
		source << R"(
std::tuple<)";
		for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i){
			source << "arrays::" << tup.second[i] << "";
			if (i != (int32_t)tup.second.size() - 1) source << ", ";		
		}
		source << "> " << tup.first << ";";
	}

	fs::create_directories(output.parent_path());
	auto source_file = output;
	source_file+=".cpp";
	auto header_file = output;
	header_file+=".h";
	output = source_file;
	//std::cout << output.extension() << std::endl;
	if (output.extension() == ".cpp") {
		if (fs::exists(output)) {
			auto input_ts = newestFile;
			auto output_ts = fs::last_write_time(output);
			if (input_ts <= output_ts)
				return 0;
		}
		std::cout << "Writing source file " << output << std::endl;
		std::ofstream source_out(output.string());
		source_out << source.str() << std::endl;
		source_out.close();
	}
	output = header_file;
	if (output.extension() == ".h") {
		if (fs::exists(output)) {
			auto input_ts = newestFile;
			auto output_ts = fs::last_write_time(output);
			if (input_ts <= output_ts)
				return 0;
		}
		std::cout << "Writing header file " << output << std::endl;
		std::ofstream header_out(output.string());
		header_out << header.str() << std::endl;
		header_out.close();
	}
}
 