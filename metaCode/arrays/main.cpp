#include <iostream>
#include <vector>
//#define BOOST_MSVC
//using std::is_assignable;
//using std::is_volatile; 
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <boost/algorithm/string/case_conv.hpp>
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

	std::vector<std::tuple<std::string, std::string, std::string>> source;
	std::vector<std::tuple<std::string, std::string, std::string>> header;

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
		std::sort(header.begin(), header.end(), [](const auto& lhs, const auto& rhs) {return std::get<0>(lhs) < std::get<0>(rhs); });
		std::string ns = "";
		for (auto [nameSpace, identifier, text] : header) {
			if(ns != nameSpace && ns != "")
				sstream << "}\n";
			if (ns != nameSpace) {
				sstream << "namespace " << nameSpace << "{\n";
				ns = nameSpace;
			}
			auto id = identifier;
			auto ambiF = std::count_if(header.begin(), header.end(), [=](auto&& val) {
				return std::get<1>(val) == id;
			});
			auto textCopy = text;
			if (ambiF != 1) {
				identifier[0] = ::toupper(identifier[0]);
				textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi_identifier)"), nameSpace + identifier);
				textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi)"), "true");
			}
			else {
				textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi_identifier)"), identifier);
				textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi)"), "false");

			}
			sstream << textCopy;
		}
		if(header.size() != 0)
		sstream << "}\n";
		//sstream << boost::join(header,"\n") << std::endl;
		for (auto& child : children)
			sstream << child->tuple_h();
		return sstream.str();
	}
	std::string tuple_s() {
		std::stringstream sstream;
		std::sort(source.begin(), source.end(), [](const auto& lhs, const auto& rhs) {return std::get<0>(lhs) < std::get<0>(rhs); });
		for (auto&[nameSpace, identifier, text] : source) {
			sstream << text;
		}
		//sstream << boost::join(source, "\n");
		for (auto& child : children)
			sstream << child->tuple_s();
		return sstream.str();
	}

};

#if defined(_MSC_VER)
#define template_flag
#else
#define template_flag template
#endif
#define TYPE node.second.template_flag get<std::string>("type")
#define NAME (parent->parent->node->first + "::" + parent->node->first)

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

std::vector<std::string> namespaces;
std::vector<std::tuple<std::string, std::string, std::string>> identifierTuples;
int main(int argc, char** argv) try {
	std::cout << "Running array meta-code generation" << std::endl;
	if (argc != 2) {
		std::cerr << "Wrong number of arguments provided" << std::endl;
		for (int32_t i = 0; i < argc; ++i) {
			std::cout << i << ": " << argv[i] << std::endl;
		}
		return 1;
	}
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
		std::string header_text = R"(struct $identifier{
	using type = $type;
	using unit_type = $unit;
	static constexpr const array_enum identifier = array_enum::$ns_$identifier;
	static constexpr const auto variableName = "$identifier";
	static constexpr const auto qualifiedName = "$qualified";
	static constexpr const auto ambiguous = $ambi;
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
	template<class T> static inline auto& get_member(T& var) { return var.$ambi_identifier;}
};
)";

		std::string src_text = R"(
namespace $ns{
$type* $identifier::ptr = nullptr;
size_t $identifier::alloc_size = 0;
$swap_src
$default_alloc_src
$functions_src
})";


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
$type* $identifier::rear_ptr = nullptr;
void $identifier::swap() { std::swap(ptr, rear_ptr); })";
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
			auto size = node.second.template get_optional<int32_t>("size");
			auto kind = node.second.template_flag get("kind", std::string("customData"));

			std::string swap_str = "";
			if (kind != "customData") {
				swap_str = R"()";
				std::string size_str = "1";
				if (kind == "individualData") size_str = "(1)";
				if (kind == "singleData") size_str = "(1)";
				if (kind == "particleData") size_str = "(get<parameters::max_numptcls>() + 1)";
				if (kind == "diffuseData") size_str = "get<parameters::max_diffuse_numptcls>()";
				if (kind == "cellData") size_str = "get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z";
				swap_str = R"(
void $identifier::defaultAllocate(){
	auto elems = $size_str;
	alloc_size = $size * elems * sizeof($type);
	cudaAllocateMemory(&ptr, alloc_size);$malloc_rear
}
void $identifier::leanAllocate(){
	auto elems = $size_str;
	alloc_size = $size * elems * sizeof($type);
	$malloc_lean
})";
				swap_str = std::regex_replace(swap_str, std::regex(R"(\$size_str)"), size_str);
				if (size)
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), std::to_string(size.get()));
				else if (auto sz = node.second.template get_optional<std::string>("size"))
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), sz.get());
				else
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$size)"), "1");
				if (kind == "individualData") {
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_rear)"), "");
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_lean)"), R"(
	cudaAllocateMemory(&ptr, alloc_size);)");
				}
				else {
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_rear)"), (swap ? R"(
	cudaAllocateMemory(&rear_ptr, alloc_size);)" : ""));
					swap_str = std::regex_replace(swap_str, std::regex(R"(\$malloc_lean)"), (swap ? R"(
	cudaAllocateMemory(&ptr, alloc_size);)" : ""));
				}
			}
			return std::regex_replace(text, std::regex(R"(\$default_alloc_src)"), swap_str);
		});
		text_fn.push_back([&](auto text) {
			auto swap = node.second.template_flag get("swap", false);
			auto dependency_any = node.second.template_flag get_child_optional("depends_any");
			auto dependency_all = node.second.template_flag get_child_optional("depends_all");

			std::string swap_str = R"(
void $identifier::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);$malloc_rear
}
void $identifier::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;$free_rear
}
$identifier::operator type*(){ return ptr;}
$type& $identifier::operator[](size_t idx){ return ptr[idx];}
bool $identifier::valid(){
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
			namespaces.push_back(std::string(parent->parent->node->first));
			return std::regex_replace(text, std::regex(R"(\$ns)"), std::string(parent->parent->node->first)); 
		});
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$qualified)"), std::string(parent->parent->node->first) + "." + std::string(parent->node->first)); });
		
		text_fn.push_back([&](auto text) {
			auto unit = node.second.template_flag get("unit", std::string("none"));
			if (unit == "void")
				unit = "void_unit_ty";
			std::string unit_text = "value_unit<$type, " + unit + ">";
			if (unit == "none")
				unit_text = "$type";
			return std::regex_replace(text, std::regex(R"(\$unit)"), unit_text); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$type)"), node.second.template get<std::string>("type")); });
		//text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$unit)"), node.second.template get("unit", std::string("void"))); });
		text_fn.push_back([&](auto text) {return std::regex_replace(text, std::regex(R"(\$kind)"), node.second.template_flag get("kind", std::string("customData"))); });

		if (auto resort = node.second.template_flag get("resort", false); resort) {
			global_map["sorting_list"].push_back(NAME);
		};
		if (auto kind = node.second.template get_optional<std::string>("kind"); kind) {
			if (kind.get() == "particleData")
				global_map["swapping_list"].push_back(NAME);
			else if (kind.get() == "diffuseData")
				global_map["diffuse_swapping_list"].push_back(NAME);
			if (kind.get() != "customData") {
				global_map["allocations_list"].push_back(NAME);
			}
			if (kind.get() == "individualData") {
				global_map["individual_list"].push_back(NAME);
			}
		}

		for (auto f : text_fn)
			header_text = f(header_text);
		for (auto f : text_fn)
			src_text = f(src_text);

		global_map["arrays_list"].push_back(NAME);

		auto id = node.second.template_flag get("identifier", std::string(parent->node->first));
		auto ns = std::string(parent->parent->node->first);
		auto enu = std::string("array_enum::") + ns + std::string("_") + id;
		identifierTuples.push_back(std::tuple<std::string, std::string, std::string>(id, ns, enu));

		parent->header.push_back(std::make_tuple(parent->parent->node->first,parent->node->first, header_text));
		parent->source.push_back(std::make_tuple(parent->parent->node->first, parent->node->first, src_text));

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
		std::string copy = boost::replace_all_copy(arrs[i], "::", "_");
		header << copy;
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
	std::sort(namespaces.begin(), namespaces.end());
	namespaces.erase(std::unique(namespaces.begin(), namespaces.end()), namespaces.end());
	header << R"(
namespace arrays{
)";
	for (auto ns : namespaces) {
		header << R"(	using namespace )" << ns << ";\n";
	}
	header << R"(}
)";
	for (auto tup : global_map) {
		header << R"(
extern std::tuple<)";
		for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i) {
			header << "arrays::" << tup.second[i] << "";
			if (i != (int32_t)tup.second.size() - 1) header << ", ";
		}
		header << "> " << tup.first << ";";
	}
	for (auto tup : global_map) {
		std::string copy = boost::replace_all_copy(tup.first, "::", "_");
		std::vector<std::string> splitVec;
		boost::split(splitVec, copy, boost::is_any_of("_"), boost::token_compress_on);
		for (auto& s : splitVec)
			s[0] = ::toupper(s[0]);
		auto joined = boost::join(splitVec, "");
		header << R"(
template<typename C, typename... Ts>
auto iterate)" << joined << R"((C&& fn, Ts&&... args){
)";
		for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i) {
			header << "\tfn(arrays::" << tup.second[i] << "(), args...);\n";
		}
		header << "}\n";
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
namespace arrays{
)";
	source << base_transformation.tuple_s() << std::endl;
	source << "}\n";
	for (auto tup : global_map) {
		source << R"(
std::tuple<)";
		for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i){
			source << "arrays::" << tup.second[i] << "";
			if (i != (int32_t)tup.second.size() - 1) source << ", ";		
		}
		source << "> " << tup.first << ";";
	}
	{
		auto sorting_list = global_map["sorting_list"];
		header << "struct sortingArray{\n";
		for (auto v : sorting_list) {
			std::string copy = boost::replace_all_copy(v, "::", "_");
			std::vector<std::string> splitVec;
			boost::split(splitVec, copy, boost::is_any_of("_"), boost::token_compress_on);
			for (auto& s : splitVec)
				s[0] = ::toupper(s[0]);
			auto joined = boost::join(splitVec, "");
			header << "\tstd::pair<arrays::" << v << "::type*, arrays::" << v << "::type*> " << joined << ";\n";
		}
		header << "\n\thostInline void fillArray(){\n";
		for (auto v : sorting_list) {
			std::string copy = boost::replace_all_copy(v, "::", "_");
			std::vector<std::string> splitVec;
			boost::split(splitVec, copy, boost::is_any_of("_"), boost::token_compress_on);
			for (auto& s : splitVec)
				s[0] = ::toupper(s[0]);
			auto joined = boost::join(splitVec, "");
			header << "\t\t" << joined << " = std::make_pair( arrays::" << v << "::ptr, arrays::" << v << "::rear_ptr);\n";
		}
		header << "\t}\n";
		header << "\n\ttemplate<typename C> hostDeviceInline void callOnArray(C func){\n";
		for (auto v : sorting_list) {
			std::string copy = boost::replace_all_copy(v, "::", "_");
			std::vector<std::string> splitVec;
			boost::split(splitVec, copy, boost::is_any_of("_"), boost::token_compress_on);
			for (auto& s : splitVec)
				s[0] = ::toupper(s[0]);
			auto joined = boost::join(splitVec, "");
			header << "\t\tfunc(" << joined << ".first, " << joined << ".second);\n";
		}
		header << "\t}\n";
		header << "};\n";
	}
	auto helperFn = [&](std::string fnName, auto func, std::string def) {
		header << std::string(R"(auto inline getArray)") + fnName + std::string(R"((array_enum e){
	switch(e){
)");
		for (const auto& elem : identifierTuples) {
			const auto&[i, n, e] = elem;
			header << R"(		case )" << e << R"(: return )" << func(i,n,e) << ";" << std::endl;
		}
		header << R"(		default: return )" << def << ";" << std::endl;
		header << R"(	}
}
)";
	};
	helperFn("QualifiedName", [](auto i, auto n, auto e) {
		return std::string("arrays::" + n + "::" + i + "::qualifiedName"); }, R"("invalidEnum")");
	helperFn("VariableName", [](auto i, auto n, auto e) {
		return std::string("arrays::" + n + "::" + i + "::variableName"); }, R"("invalidEnum")");
	helperFn("Ptr", [](auto i, auto n, auto e) {
		return std::string("(void*) arrays::" + n + "::" + i + "::ptr"); }, R"((void*) nullptr)");
	helperFn("AllocationSize", [](auto i, auto n, auto e) {
		return std::string("arrays::" + n + "::" + i + "::alloc_size"); }, R"((size_t)0u)");
	helperFn("Kind", [](auto i, auto n, auto e) {
		return std::string("arrays::" + n + "::" + i + "::kind"); }, R"(memory_kind::customData)");
	helperFn("Description", [](auto i, auto n, auto e) {
		return std::string("arrays::" + n + "::" + i + "::description"); }, R"("invalidEnum")");
	header << "#ifndef __CUDACC__\n";
	helperFn("Type", [](auto i, auto n, auto e) {
		return std::string("type_name<arrays::" + n + "::" + i + "::type>()"); }, R"(std::string("invalidEnum"))");
	header << "#endif\n";
	helperFn("Swappable", [](auto i, auto n, auto e) {
		return std::string("has_rear_ptr<arrays::" + n + "::" + i + "> "); }, R"(false)");

	//for (const auto & elem : identifierTuples) {
	//	const auto&[i, n, e] = elem;
	//	std::cout << i << " @ " << n << " -> " << e << std::endl;
	//}

	fs::create_directories(output.parent_path());
	auto source_file = output;
	source_file+=".cpp";
	auto header_file = output;
	header_file+=".h";
	output = source_file;
	//std::cout << output.extension() << std::endl;
	do{
	if (output.extension() == ".cpp") {
		if (fs::exists(output)) {
			std::ifstream t(output.string());
			std::string str;
			t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);

			str.assign((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());

			if (str == source.str())
				continue;
			//auto input_ts = newestFile;
			//auto output_ts = fs::last_write_time(output);
			//if (input_ts <= output_ts)
			//	return 0;
		}
		std::cout << "Writing source file " << output << std::endl;
		std::ofstream source_out(output.string());
		source_out << source.str();
		source_out.close();
	}
	} while (!true);
	output = header_file;
	do{
		if (output.extension() == ".h") {
		if (fs::exists(output)) {
			std::ifstream t(output.string());
			std::string str;
			t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);

			str.assign((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
			if (str == header.str())
				continue;
			//auto input_ts = newestFile;
			//auto output_ts = fs::last_write_time(output);
			//if (input_ts <= output_ts)
			//	return 0;
		}
		std::cout << "Writing header file " << output << std::endl;
		std::ofstream header_out(output.string());
		header_out << header.str();
		header_out.close();
	}
	} while (!true);
} catch (std::exception e) {
	std::cerr << "Caught exception: " << e.what() << std::endl;
	throw;
}