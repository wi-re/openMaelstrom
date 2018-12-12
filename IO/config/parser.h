#pragma once
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>
#include <utility/template/for_struct.h>
#include <vector>

template <typename Out> void split(const std::string &s, char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}
template <typename T> std::vector<std::string> split(const std::string &s, T delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

namespace IO::config {

template <class T> using min_t = decltype(T::min);
template <class Ptr> using min_type = detected_or_t<std::ptrdiff_t, min_t, Ptr>;

template <class T> using max_t = decltype(T::max);
template <class Ptr> using max_type = detected_or_t<std::ptrdiff_t, max_t, Ptr>;
template <class T> using step_t = decltype(T::step);
template <class Ptr> using step_type = detected_or_t<std::ptrdiff_t, step_t, Ptr>;
template <class T> using presets_t = decltype(T::presets);
template <class Ptr> using presets_type = detected_or_t<std::ptrdiff_t, presets_t, Ptr>;

template <typename T>
constexpr bool min_type_v = !std::is_same<std::decay_t<min_type<T>>, std::ptrdiff_t>::value;
template <typename T>
constexpr bool max_type_v = !std::is_same<std::decay_t<max_type<T>>, std::ptrdiff_t>::value;
template <typename T>
constexpr bool step_type_v = !std::is_same<std::decay_t<step_type<T>>, std::ptrdiff_t>::value;
template <typename T>
constexpr bool presets_type_v = !std::is_same<std::decay_t<presets_type<T>>, std::ptrdiff_t>::value;

template <class T> using uniform_t = typename T::uniform_type;

template <class Ptr> using uniform_type_template = detected_or_t<std::ptrdiff_t, uniform_t, Ptr>;

template <typename T>
constexpr bool uniform_type_template_v =
    !std::is_same<std::decay_t<uniform_type_template<T>>, std::ptrdiff_t>::value;

template <typename T, typename _ = void> struct is_vector : std::false_type {};
template <typename T>
struct is_vector<
    T, typename std::enable_if<std::is_same<
           T, std::vector<typename T::value_type, typename T::allocator_type>>::value>::type>
    : std::true_type {};

template <typename T> constexpr bool is_vector_v = is_vector<T>::value;

template <typename T, typename _ = void> struct is_aggregate : std::false_type {};
template <typename T>
struct is_aggregate<T, typename std::enable_if<T::type == resource_t::aggregate_uniform_type>>
    : std::true_type {};

template <typename T> constexpr bool is_aggregate_v = is_aggregate<T>::value;

template <typename T, typename _ = void> struct is_std_array : std::false_type {};
template <typename T>
struct is_std_array<T,
                    typename std::enable_if_t<
                        !std::is_same<void, typename std::tuple_size<T>::value_type>::value, void>>
    : std::true_type {};

template <typename T> constexpr bool is_std_array_v = is_std_array<T>::value;

template <typename T> auto convertString(std::string argument, std::string = "") {
  if constexpr (std::is_same<T, std::string>::value)
    return argument;
  else if constexpr (std::is_same<T, bool>::value) {
    return boost::iequals(argument, "true") ? true : false;
  } else if constexpr (math::dimension<T>::value != 0xDEADBEEF && math::dimension<T>::value > 0) {
    constexpr uint32_t dim = math::dimension<T>::value;
    using base_type = decltype(std::declval<T>().x);
    std::istringstream iss(argument);
    std::vector<base_type> tokens;
    std::copy(std::istream_iterator<base_type>(iss), std::istream_iterator<base_type>(),
              std::back_inserter(tokens));
    if constexpr (dim == 0)
      return tokens[0];
    if constexpr (dim == 1)
      return T{tokens[0]};
    if constexpr (dim == 2)
      return T{tokens[0], tokens[1]};
    if constexpr (dim == 3)
      return T{tokens[0], tokens[1], tokens[2]};
    if constexpr (dim == 4)
      return T{tokens[0], tokens[1], tokens[2], tokens[3]};
  } else if constexpr (math::dimension<T>::value == 0) {
    using base_type = T;
    std::istringstream iss(argument);
    std::vector<base_type> tokens;
    std::copy(std::istream_iterator<base_type>(iss), std::istream_iterator<base_type>(),
              std::back_inserter(tokens));
    return tokens[0];
  } else if constexpr (std::is_same<uniform_type_template<T>, complex_uniform>::value) {
    return T();
  } else if constexpr (is_vector<T>::value) {
    struct csv_reader : std::ctype<char> {
      csv_reader() : std::ctype<char>(get_table()) {}
      static std::ctype_base::mask const *get_table() {
        static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());
        rc[','] = std::ctype_base::space;
        return &rc[0];
      }
    };
    std::istringstream iss = std::istringstream(argument);
    iss.imbue(std::locale(std::locale(), new csv_reader()));
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>{}};
    T parsed;
    for (auto x : tokens)
      parsed.push_back(convertString<typename T::value_type>(x));
    return parsed;
  } 
}

template <typename T>
std::pair<bool, T> convertStringChecked(std::string argument, std::string = "") {
  if constexpr (std::is_same<T, std::string>::value)
    return std::make_pair(true, argument);
  else if constexpr (math::dimension<T>::value != 0xDEADBEEF && math::dimension<T>::value > 0) {
    constexpr uint32_t dim = math::dimension<T>::value;
    using base_type = decltype(std::declval<T>().x);
    std::istringstream iss(argument);
    std::vector<base_type> tokens;
    std::copy(std::istream_iterator<base_type>(iss), std::istream_iterator<base_type>(),
              std::back_inserter(tokens));
    if constexpr (dim == 0)
      return tokens[0];
    if constexpr (dim == 1)
      return tokens.size() > 0 ? std::make_pair(true, T{tokens[0]}) : std::make_pair(false, T());
    if constexpr (dim == 2)
      return tokens.size() > 1 ? std::make_pair(true, T{tokens[0], tokens[1]})
                               : std::make_pair(false, T());
    if constexpr (dim == 3)
      return tokens.size() > 2 ? std::make_pair(true, T{tokens[0], tokens[1], tokens[2]})
                               : std::make_pair(false, T());
    if constexpr (dim == 4)
      return tokens.size() > 3 ? std::make_pair(true, T{tokens[0], tokens[1], tokens[2], tokens[3]})
                               : std::make_pair(false, T());
  } else if constexpr (math::dimension<T>::value == 0) {
    using base_type = T;
    std::istringstream iss(argument);
    std::vector<base_type> tokens;
    std::copy(std::istream_iterator<base_type>(iss), std::istream_iterator<base_type>(),
              std::back_inserter(tokens));
    return tokens.size() > 0 ? std::make_pair(true, T{tokens[0]}) : std::make_pair(false, T());
  } else if constexpr (std::is_same<uniform_type_template<T>, complex_uniform>::value) {
    return T();
  } else if constexpr (is_vector<T>::value) {
    struct csv_reader : std::ctype<char> {
      csv_reader() : std::ctype<char>(get_table()) {}
      static std::ctype_base::mask const *get_table() {
        static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());
        rc[','] = std::ctype_base::space;
        return &rc[0];
      }
    };
    std::istringstream iss = std::istringstream(argument);
    iss.imbue(std::locale(std::locale(), new csv_reader()));
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>{}};
    T parsed;
    for (auto x : tokens) {
      auto conversion = convertString<typename T::value_type>(x);
      if (conversion.first)
        parsed.push_back(conversion.second);
    }
    return std::make_pair(true, parsed);
  } 
  return std::make_pair(false, T{});
}

template <typename T> std::string to_string(T arg) {
	std::stringstream sstream;
#ifndef __CUDACC__
  if constexpr (math::dimension<T>::value == 0)
    return std::to_string(arg);

  sstream << "";
  sstream << math::get<1>(arg);
  if constexpr (math::dimension<T>::value > 1) {
    sstream << " " << math::get<2>(arg);
  }
  if constexpr (math::dimension<T>::value > 2) {
    sstream << " " << math::get<3>(arg);
  }
  if constexpr (math::dimension<T>::value > 3) {
    sstream << " " << math::get<4>(arg);
  }
  sstream << "";
  return sstream.str();
#endif
}

template <typename T> std::string convertToString([[maybe_unused]] T argument, std::string = "") {
  if constexpr (std::is_same<T, std::string>::value)
    return argument;
  else if constexpr (std::is_same<T, bool>::value) {
    return argument == true ? "true" : "false";
  } else if constexpr (math::dimension<T>::value != 0xDEADBEEF && math::dimension<T>::value > 0) {
    return to_string(argument);
  } else if constexpr (math::dimension<T>::value == 0) {
    return std::to_string(argument);
  } else if constexpr (std::is_same<uniform_type_template<T>, complex_uniform>::value) {
    return "";
  } else if constexpr (is_vector<T>::value) {
    std::vector<std::string> temporary;
    for (auto val : argument)
      temporary.push_back(convertToString(val));
    std::ostringstream ss;
    if (temporary.size() > 0) {
      std::copy(temporary.begin(), temporary.end() - 1,
                std::ostream_iterator<std::string>(ss, ", "));
      ss << temporary.back();
    }

    return ss.str();
  } else if constexpr (is_std_array<T>::value) {
    std::ostringstream ss;
    for_each_i(argument, [&ss](auto &x, auto) { ss << to_string(x) << " "; });
    return ss.str();
  }
  return "";
}

template <typename T>
T parseComplex(T &value, std::string jsonName, boost::property_tree::ptree &pt);
template <typename T>
std::vector<T> parseWildcard(std::vector<T> &value, std::string jsonName,
                             boost::property_tree::ptree &pt);
template <typename T> T parse(T &value, std::string jsonName, boost::property_tree::ptree &pt);
template <typename T> auto parse(boost::property_tree::ptree &pt);
template <typename T>
void parseComplexStore(T &value, std::string jsonName, boost::property_tree::ptree &pt);
template <typename T>
void parseWildcardStore(std::vector<T> &value, std::string jsonName,
                        boost::property_tree::ptree &pt);
template <typename T>
void parseStore(T &value, std::string jsonName, boost::property_tree::ptree &pt);
template <typename T> void parseStore(boost::property_tree::ptree &pt);



template <typename T>
T parseComplex(T &value, std::string jsonName, boost::property_tree::ptree &pt) {
  for_struct(value, [jsonName, &pt](auto &x) {
    if (auto argument = pt.get_optional<std::string>(jsonName + "." + x.jsonName))
      x.value =
          convertString<typename std::remove_reference<decltype(x)>::type::type>(argument.get());
  });
  return value;
}
template <typename T>
std::vector<T> parseWildcard(std::vector<T> &value, std::string jsonName,
                             boost::property_tree::ptree &pt) {
  for (uint32_t i = 1; i < 10; ++i) {
    std::string current = jsonName;
    char id = '0' + i;
    std::replace(current.begin(), current.end(), '$', id);
    if (auto argument = pt.get_optional<std::string>(current)) {
      T val;
      parse(val, current, pt);
      value.push_back(val);
    } else
      break;
  }
  return value;
}

template <typename T> T parse(T &value, std::string jsonName, boost::property_tree::ptree &pt) {
  if constexpr (is_vector<T>::value) {
    if (jsonName.find("$") != std::string::npos)
      return parseWildcard(value, jsonName, pt);
    else if (auto argument = pt.get_optional<std::string>(jsonName))
      value = convertString<T>(argument.get());
  } else if constexpr (std::is_same<uniform_type_template<T>, complex_uniform>::value)
    return parseComplex(value, jsonName, pt);
  else {
    if (auto argument = pt.get_optional<std::string>(jsonName))
      value = convertString<T>(argument.get());
  }
  return value;
}

template <typename T> auto parse(boost::property_tree::ptree &pt) {
  parse(*T::ptr, T::jsonName, pt);
  return *T::ptr;
}

template <typename T>
void parseComplexStore(T &value, std::string jsonName, boost::property_tree::ptree &pt) {
  for_struct(value, [jsonName, &pt](auto &x) {
    pt.put(jsonName + "." + x.jsonName, convertToString(x.value));
  });
}

template <typename T>
void parseWildcardStore(std::vector<T> &value, std::string jsonName,
                        boost::property_tree::ptree &pt) {
  for (uint32_t i = 1; i < 10 && i <= value.size(); ++i) {
    std::string current = jsonName;
    char id = '0' + i;
    std::replace(current.begin(), current.end(), '$', id);
    parseStore(value[i - 1], current, pt);
  }
}

template <typename T>
void parseStore(T &value, std::string jsonName, boost::property_tree::ptree &pt) {
  if constexpr (is_vector<T>::value) {
    if (jsonName.find("$") != std::string::npos)
      parseWildcardStore(value, jsonName, pt);
    else
      pt.put(jsonName, convertToString(value));
  } else if constexpr (std::is_same<uniform_type_template<T>, complex_uniform>::value)
    parseComplexStore(value, jsonName, pt);
  else
    pt.put(jsonName, convertToString(value));
}

template <typename T> void parseStore(boost::property_tree::ptree &pt) {
  parseStore(*T::ptr, T::jsonName, pt);
}


} // namespace IO::config
