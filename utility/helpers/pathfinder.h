#pragma once
#include <sstream>
#include <experimental/filesystem>

std::experimental::filesystem::path expand(std::experimental::filesystem::path in);
std::experimental::filesystem::path resolveFile(std::string fileName, std::vector<std::string> search_paths = {});
