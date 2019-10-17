#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using node_t = std::pair<const std::string, boost::property_tree::ptree>;

std::map<std::string, std::vector<std::string>> global_map;
std::vector<std::string> complex_types;

struct transformation {
  std::vector<std::shared_ptr<transformation>> children;
  transformation *parent = nullptr;

  node_t *node;

  std::vector<std::tuple<std::string, std::string, std::string>> source;
  std::vector<std::tuple<std::string, std::string, std::string>> header;

  std::function<void(transformation &, transformation *, node_t &)> transfer_fn = [](auto &node, auto, node_t &tree) {
    for (auto &t_fn : node.children)
      for (auto &children : tree.second)
        t_fn->transform(children);
  };

  transformation(transformation *p = nullptr) : parent(p) {}
  transformation(decltype(transfer_fn) t_fn, transformation *p = nullptr) : parent(p), transfer_fn(t_fn) {}

  std::shared_ptr<transformation> addChild() {
    children.push_back(std::make_shared<transformation>(transfer_fn, this));
    return children[children.size() - 1];
  }

  std::shared_ptr<transformation> addChild(decltype(transfer_fn) t_fn) {
    children.push_back(std::make_shared<transformation>(t_fn, this));
    return children[children.size() - 1];
  }

  void transform(node_t &t_node) {
    node = &t_node;
    transfer_fn(*this, parent, t_node);
  }
  std::string tuple_h() {
	  std::stringstream sstream;
	  std::sort(header.begin(), header.end(), [](const auto& lhs, const auto& rhs) {return std::get<0>(lhs) < std::get<0>(rhs); });
	  std::string ns = "";
	  for (auto&[nameSpace, identifier, text] : header) {
		  if (ns != nameSpace && ns != "")
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
	  if (header.size() != 0)
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
  for (auto &fileName : fs::directory_iterator(folder)) {
    auto t = fs::last_write_time(fileName);
    if (t > newestFile)
      newestFile = t;
    std::stringstream ss;
    std::ifstream file(fileName.path().string());
    ss << file.rdbuf();
    file.close();

    boost::property_tree::ptree pt2;
    boost::property_tree::read_json(ss, pt2);
    auto arrs = pt2.get_child_optional("uniforms");
    if (arrs) {
      for (auto it : arrs.get()) {
        // std::cout << it.first << " : " << it.second.data() << std::endl;
        pt.add_child(it.first, it.second);
      }
    }
  }
  auto arrays = fs::path(sourceDirectory) / "parameters.json";
  // if(fs::exists(arrays) && fs::last_write_time(arrays) > newestFile)
  // 	return;
  // std::cout << "Writing " << arrays << std::endl;
  // std::ofstream file(arrays.string());
  // boost::property_tree::write_json(file, pt);
}
std::vector<std::string> namespaces;

int main(int argc, char **argv) try {
	std::cout << "Running uniform meta-code generation" << std::endl;
  if (argc != 2) {
	  std::cerr << "Wrong number of arguments provided" << std::endl;
	  for (int32_t i = 0; i < argc; ++i) {
		  std::cout << i << ": " << argv[i] << std::endl;
	  }
	  return 1;
  }
  fs::path output(R"()");

  output = argv[1];

  std::stringstream ss;

  boost::property_tree::ptree pt;
  resolveIncludes(pt);
  // boost::property_tree::read_json(ss, pt);

  transformation base_transformation;
  auto category_transformation = base_transformation.addChild();
  auto array_transformation =
      category_transformation->addChild([](auto &curr, [[maybe_unused]] auto parent, auto &node) {
        for (auto t_fn : curr.children)
          t_fn->transform(node);
      });

  array_transformation->addChild([]([[maybe_unused]] auto &curr, [[maybe_unused]] auto parent, auto &node) {
    if (auto complex_type = node.second.template_flag get_child_optional("complex_type"); complex_type) {
      auto ct = complex_type.get();
      auto name = ct.template get<std::string>("name");

      std::string str = R"(
struct $identifier{
	using uniform_type = complex_uniform;
	$members
};
)";
      std::vector<std::function<std::string(std::string)>> text_fn;
      text_fn.push_back([&](auto text) {
        return std::regex_replace(text, std::regex(R"(\$name)"),
                                  node.second.template_flag get("identifier", std::string(parent->node->first)));
      });
      text_fn.push_back([&](auto text) {
        return std::regex_replace(text, std::regex(R"(\$identifier)"), ct.template get<std::string>("name"));
      });
      text_fn.push_back([&](auto text) {
        std::stringstream ss;
        for (auto mem : ct.get_child("description")) {
          std::string member = R"(
	complex_type<$type> $identifier{ "$name", $def};)";
          auto def = mem.second.template get<std::string>("default");
          member =
              std::regex_replace(member, std::regex(R"(\$type)"), mem.second.template get<std::string>("type"));
          member = std::regex_replace(member, std::regex(R"(\$name)"), mem.first);
          member = std::regex_replace(member, std::regex(R"(\$identifier)"),
                                      mem.second.template get<std::string>("identifier"));
          member = std::regex_replace(member, std::regex(R"(\$def)"), (def == "" ? R"("")" : def));
          ss << member;
        }
        return std::regex_replace(text, std::regex(R"(\$members)"), ss.str());
      });
      for (auto f : text_fn)
        str = f(str);
      complex_types.push_back(str);
    }
  });

  array_transformation->addChild([]([[maybe_unused]] auto &curr, auto parent, auto &node) {
    std::string header_text = R"header(
	struct $identifier{
		using type = $type;
		using unit_type = $unit;
		static constexpr const uniforms identifier = uniforms::$ns_$identifier;
		$identifier(const type& val){*ptr = val;}
		$identifier() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "$identifier";
		static constexpr const auto description = R"($description)";
		static $type* ptr;
		static $unit* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "$json";
		static constexpr const auto ambiguous = $ambi;
		static constexpr const bool modifiable = $constant;
		static constexpr const bool visible = $visible;

		template<class T> static inline auto& get_member(T& var) { return var.$ambi_identifier;}$range_statement		
	};)header";
    std::string src_text = R"(
namespace $ns{
static $type HELPER_VARIABLE$identifier{$default_value};
$type* $identifier::ptr = &HELPER_VARIABLE$identifier;
$unit* $identifier::unit_ptr = ($unit*) &HELPER_VARIABLE$identifier;
std::vector<std::string>  $identifier::getPresets(){
	std::vector<std::string> presets $prStatement;
	return presets;
}
}
)";

    std::vector<std::function<std::string(std::string)>> text_fn;

    text_fn.push_back([&](auto text) {
      std::string range_text = "";
      if (auto range = node.second.template_flag get_child_optional("range"); range) {
        std::stringstream rss;
        for (auto &v : range.get()) {
          rss << R"(
		static constexpr const $type )"
              << v.first << '{' << v.second.data() << "};";
        }
        range_text = rss.str();
      }
      return std::regex_replace(text, std::regex(R"(\$range_statement)"), range_text);
    });
	text_fn.push_back([&](auto text) {
		std::string prText = "";
		std::string jText = node.second.template_flag get("presets", std::string(""));
		if (jText != "") {
			prText += std::string("= {");
			std::vector<std::string> presets;
			std::stringstream ss(jText);
			while (ss.good()){
				std::string substr;
				std::getline(ss, substr, ',');
				presets.push_back(substr);
			}
			for (int32_t i = 0; i < presets.size(); ++i) {
				prText += (i == 0 ? std::string("") : std::string(",")) + std::string("\"") + presets[i] + std::string("\"");
			}
			prText += std::string("}");
		}
		if (jText.find("(") != std::string::npos) {
			prText = std::string("= ") + jText;
		}
		return std::regex_replace(text, std::regex(R"(\$prStatement)"),prText);		
	});
    text_fn.push_back([&](auto text) {
      return std::regex_replace(text, std::regex(R"(\$identifier)"),
                                node.second.template_flag get("identifier", std::string(parent->node->first)));
    });
    text_fn.push_back([&](auto text) {
      auto unit = node.second.template_flag get("unit", std::string("none"));
      if (unit == "void") {
        unit = "void_unit_ty";
      }
      std::string unit_text = "value_unit<$type, " + unit + ">";
      if (unit == "none")
        unit_text = "$type";
      return std::regex_replace(text, std::regex(R"(\$unit)"), unit_text);
    });
	text_fn.push_back([&](auto text) {
		auto descr = node.second.template_flag get("description", std::string(""));
		return std::regex_replace(text, std::regex(R"(\$description)"), descr);
	});
    text_fn.push_back([&](auto text) {
      return std::regex_replace(text, std::regex(R"(\$type)"), node.second.template get<std::string>("type"));
    });
    text_fn.push_back([&](auto text) {
      return std::regex_replace(text, std::regex(R"(\$visible)"),
                                (node.second.template_flag get("visible", true) ? "true" : "false"));
    });
    text_fn.push_back([&](auto text) {
      return std::regex_replace(text, std::regex(R"(\$constant)"),
                                (node.second.template_flag get("const", true) ? "false" : "true"));
    });
    text_fn.push_back([&](auto text) {
      return std::regex_replace(text, std::regex(R"(\$json)"), (parent->parent->node->first + "." + node.first));
    });
	text_fn.push_back([&](auto text) {
		namespaces.push_back(parent->parent->node->first);
		return std::regex_replace(text, std::regex(R"(\$ns)"), parent->parent->node->first);
	});
    text_fn.push_back([&](auto text) {
      std::string default_text = node.second.template_flag get("default", std::string(""));
      if (node.second.template get<std::string>("type") == "std::string")
        default_text = "\"" + default_text + "\"";
      return std::regex_replace(text, std::regex(R"(\$default_value)"), default_text);
    });

    for (auto f : text_fn)
      header_text = f(header_text);
    for (auto f : text_fn)
      src_text = f(src_text);

	parent->header.push_back(std::make_tuple(parent->parent->node->first, node.second.template_flag get("identifier", std::string(parent->node->first)), header_text));
	parent->source.push_back(std::make_tuple(parent->parent->node->first, node.second.template_flag get("identifier", std::string(parent->node->first)), src_text));
    //parent->header << header_text;
    //parent->source << src_text;
    global_map["uniforms_list"].push_back( parent->parent->node->first + "::" +
        node.second.template_flag get("identifier", std::string(parent->node->first)));
  });
  node_t tree{".", pt};
  base_transformation.transform(tree);

  std::string header_file =
      R"(#pragma once
#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility/template/nonesuch.h>
#include <utility/unit_math.h>
#include <utility>
#include <vector>
$enum
$enum_size
#include <utility/identifier/resource_helper.h>
$get_fns
$complex_types
namespace parameters{
$uniforms_h
}
$namespaces
$tuples_h
$usings
$loop_function
)";

  std::string src_file =
      R"(#include <utility/identifier/uniform.h>
#include <utility/identifier/arrays.h>
#include <utility/helpers/pathfinder.h>
namespace parameters{
$uniforms_src
}
$tuples_src
)";

  std::vector<std::function<std::string(std::string)>> text_fn;

  text_fn.push_back([&](auto text) { 
    std::stringstream enum_ss;
    enum_ss << "\nenum struct uniforms{";
    auto arrs = global_map["uniforms_list"];
    for (int i = 0; i < (int32_t)arrs.size(); ++i) {
			std::string copy = boost::replace_all_copy(arrs[i], "::", "_");
      enum_ss << copy;
      if (i != (int32_t)arrs.size() - 1)
        enum_ss << ", ";
    }
    enum_ss << "};\n";
	std::stringstream enum_sizess;
	enum_sizess << R"(#define PARAMETER_COUNT )" << arrs.size() << "\n";
	auto imm = std::regex_replace(text, std::regex(R"(\$enum_size)"), enum_sizess.str());
    return std::regex_replace(imm, std::regex(R"(\$enum)"), enum_ss.str());
  });
  text_fn.push_back([&](auto text) {
    return std::regex_replace(text, std::regex(R"(\$get_fns)"),
                              R"(
template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get(T&& = T{}) {
	return *T::ptr;
}
template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get_u(T&& = T{}) {
	return *T::unit_ptr;
})");
  });
  text_fn.push_back([&](auto text) {
    std::stringstream complex_ss;
    for (auto c : complex_types)
      complex_ss << c << std::endl;
    return std::regex_replace(text, std::regex(R"(\$complex_types)"), complex_ss.str());
  });
  text_fn.push_back([&](auto text) {
    return std::regex_replace(text, std::regex(R"(\$uniforms_h)"), base_transformation.tuple_h());
  });
  text_fn.push_back([&](auto text) {
    return std::regex_replace(text, std::regex(R"(\$uniforms_src)"), base_transformation.tuple_s());
  });
  text_fn.push_back([&](auto text) {
    std::stringstream complex_ss;
    for (auto tup : global_map) {
      complex_ss << R"(
extern std::tuple<)";
      for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i) {
        complex_ss << "parameters::" << tup.second[i];
        if (i != (int32_t)tup.second.size() - 1)
          complex_ss << ", ";
      }
      complex_ss << "> " << tup.first << ";";
    }
    return std::regex_replace(text, std::regex(R"(\$tuples_h)"), complex_ss.str());
  });
  text_fn.push_back([&](auto text) {
	  std::stringstream complex_ss;
	  complex_ss << R"(template<typename Func, typename... Ts>
auto iterateParameters(Func&& fn, Ts&&... args){
)";
	  for (auto tup : global_map["uniforms_list"]) {
		  complex_ss << "\tfn(parameters::" << tup << "(), args...);\n";
	  }
	  complex_ss << "}\n";
	  return std::regex_replace(text, std::regex(R"(\$loop_function)"), complex_ss.str());
  });
  text_fn.push_back([&](auto text) {
    std::stringstream complex_ss;
    for (auto tup : global_map) {
      complex_ss << R"(
std::tuple<)";
      for (int32_t i = 0; i < (int32_t)tup.second.size(); ++i) {
        complex_ss << "parameters::" << tup.second[i];
        if (i != (int32_t)tup.second.size() - 1)
          complex_ss << ", ";
      }
      complex_ss << "> " << tup.first << ";";
    }
    return std::regex_replace(text, std::regex(R"(\$tuples_src)"), complex_ss.str());
  });
  text_fn.push_back([&](auto text) {
    return std::regex_replace(text, std::regex(R"(\$usings)"),
                              R"(
template<typename T>
using parameter = typename T::type;
template<typename T>
using parameter_u = typename T::unit_type;
)");
  });

  text_fn.push_back([&](auto text) {
	  std::stringstream complex_ss; 
	  std::sort(namespaces.begin(), namespaces.end());
	  namespaces.erase(std::unique(namespaces.begin(), namespaces.end()), namespaces.end());
	  complex_ss << R"(namespace parameters{
)";
	  for (auto ns : namespaces) {
		  complex_ss << R"(using namespace )" << ns << ";\n";
	  }
	  complex_ss << R"(}
)";
	  return std::regex_replace(text, std::regex(R"(\$namespaces)"), complex_ss.str());
  });  
  for (auto f : text_fn)
    header_file = f(header_file);
  for (auto f : text_fn)
    src_file = f(src_file);

  fs::create_directories(output.parent_path());
	auto source_file = output;
	source_file+=".cpp";
	auto header_file_path = output;
	header_file_path+=".h";
	output = source_file;
  // std::cout << output.extension() << std::endl;
	do {
		if (output.extension() == ".cpp") {
			if (fs::exists(output)) {
				std::ifstream t(output.string());
				std::string str;
				t.seekg(0, std::ios::end);
				str.reserve(t.tellg());
				t.seekg(0, std::ios::beg);

				str.assign((std::istreambuf_iterator<char>(t)),
					std::istreambuf_iterator<char>());
				if (str == src_file)
					continue;
			}
			std::cout << "Writing source file " << output << std::endl;
			std::ofstream source_out(output.string());
			source_out << src_file;
			source_out.close();
		}
	} while (!true);
	output = header_file_path;
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
		if (header_file == str) {
			continue;
		}
    }
    std::cout << "Writing header file " << output << std::endl;
    std::ofstream header_out(output.string());
    header_out << header_file;
    header_out.close();
  }
	} while (!true);
  // std::cout << output.parent_path() << std::endl;
}
catch (std::exception e) {
	std::cerr << "Caught exception: " << e.what() << std::endl;
	throw;
}