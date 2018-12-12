#pragma once
#ifndef OLD_STYLE
#define void_unit_ty SI::unit_ty<>
#else
#include <sstream>
#include <tuple>
#define void_unit_ty void
#include <utility/unitmath/SI_Unit.h>
#include <utility/unitmath/math.h>
#include <utility/unitmath/operators.h>
#include <utility/unitmath/ratio.h>
#include <utility/unitmath/typedefs.h>
#include <utility>
#endif