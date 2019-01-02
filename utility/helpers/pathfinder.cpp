#include <utility/identifier.h>
#include <sstream>
#include <iostream>
#include <utility/helpers/log.h>
#include <experimental/filesystem>

std::experimental::filesystem::path expand(std::experimental::filesystem::path in) {
	namespace fs = std::experimental::filesystem;
#ifndef _WIN32
	if (in.string().size() < 1) return in;

	const char * home = getenv("HOME");
	if (home == NULL) {
		std::cerr << "error: HOME variable not set." << std::endl;
		throw std::invalid_argument("error: HOME environment variable not set.");
	}

	std::string s = in.string();
	if (s[0] == '~') {
		s = std::string(home) + s.substr(1, s.size() - 1);
		return fs::path(s);
	}
	else {
		return in;
	}
#else
	if (in.string().size() < 1) return in;

	const char * home = getenv("USERPROFILE");
	if (home == NULL) {
		std::cerr << "error: USERPROFILE variable not set." << std::endl;
		throw std::invalid_argument("error: USERPROFILE environment variable not set.");
	}

	std::string s = in.string();
	if (s[0] == '~') {
		s = std::string(home) + s.substr(1, s.size() - 1);
		return fs::path(s);
	}
	else {
		return in;
	}
#endif 

}

std::experimental::filesystem::path resolveFile(std::string fileName, std::vector<std::string> search_paths) {
	
	namespace fs = std::experimental::filesystem;
	fs::path working_dir = *parameters::working_directory::ptr;
	fs::path binary_dir = *parameters::binary_directory::ptr;
	fs::path source_dir = *parameters::source_directory::ptr;
	fs::path build_dir = *parameters::build_directory::ptr;
	fs::path expanded = expand(fs::path(fileName));

	fs::path base_path = "";

	for (const auto& path : search_paths){
		auto p = expand(fs::path(path));
		if (fs::exists(p / fileName))
			return p.string() + std::string("/") + fileName;
}

	if (fs::exists(fileName)) return fs::path(fileName);
        if (fs::exists(expanded))
          return expanded;

        for (const auto &pathi : search_paths) {
                        auto path = expand(fs::path(pathi));
          if (fs::exists(working_dir / path / fileName))
            return (working_dir / path / fileName).string();
          if (fs::exists(binary_dir / path / fileName))
            return (binary_dir / path / fileName).string();
          if (fs::exists(source_dir / path / fileName))
            return (source_dir / path / fileName).string();
          if (fs::exists(build_dir / path / fileName))
            return (build_dir / path / fileName).string();
        }

	if (fs::exists(working_dir / fileName))
          return (working_dir / fileName);
        if (fs::exists(binary_dir / fileName))
          return (binary_dir / fileName);
        if (fs::exists(source_dir / fileName))
          return (source_dir / fileName);
        if (fs::exists(build_dir / fileName))
          return (build_dir / fileName);

	std::stringstream sstream;
	sstream << "File '" << fileName << "' could not be found in any provided search path" << std::endl;
	LOG_ERROR << sstream.str();
	throw std::runtime_error(sstream.str().c_str());
}
