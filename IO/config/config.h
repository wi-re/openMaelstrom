#pragma once
#include <string>

namespace IO::config {
void load_config(std::string fileName);
void save_config(std::string fileName); 
void show_config();

void take_snapshot();
void load_snapshot();
void clear_snapshot();
bool has_snapshot();

std::string bytesToString(size_t bytes);
std::string numberToString(double bytes);

} // namespace IO::config
