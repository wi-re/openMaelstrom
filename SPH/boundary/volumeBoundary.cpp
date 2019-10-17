#define BOOST_USE_WINDOWS_H
#include <SPH/boundary/volumeBoundary.cuh>
#include <utility/generation.h>
#include <utility/include_all.h>
#include <utility/volumeBullet.h>
// At some point this will have to be replaced with <filesystem>
#include <fstream>



// This function is used to load the vdb files from disk and transforms them into cuda 3d textures.
void SPH::volume::update(Memory mem) {
	return;
}

void SPH::volume::init_volumes(Memory mem) {
}