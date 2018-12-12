#pragma once
#include <utility/generation.h>

namespace generation {
// Loads a level set from a *.vdb file and converts it into a mesh, also calculates normals
render_tuple ObjFromVDBWithNormals(fs::path path);
// Loads a mesh from an *.obj file and calculates normals for all vertices
render_tuple ObjFromObjWithNormals(fs::path path);
// Loads a mesh from either a *.vdb and *.obj file and calculates normals for all vertices
render_tuple ObjWithNormals(fs::path path);
} // namespace generation  