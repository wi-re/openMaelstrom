#pragma once
#include <functional>
#include <iostream>
#include <utility/identifier/uniform.h>
#include <utility/math.h> 
namespace IO::config {

void initParameters();

void initDomain();

void defaultAllocate();

void defaultRigidAllocate();

void initSnapshot();

void initKernel();

void initBoundary();

void initVolumeBoundary();
} // namespace IO::config
