#pragma once
#include <cstdint>
#include <string>

namespace IO {
namespace alembic {
void load_particles(std::string filename, uint32_t frame_idx);
void save_particles();
} // namespace alembic
} // namespace IO
