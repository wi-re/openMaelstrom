#pragma once
#include <utility/mathv2.h>
#include <utility/generation/base_generation.h>
#include <utility/generation.h>

namespace generation {
// Grid helper functions
// Loads a .vdb file and returns the parsed information as a float grid
// loads the "surface" attribute from the *.vdb file which should be a signed level set
grid_tuple loadVDBFile(fs::path fileName);
// Loads an .obj file and uses the mesh to create a level set grid which works the same as
// the level set grids expected by the other methods
grid_tuple VDBfromOBJ(fs::path fileName);
// Loads a .ply file and uses the mesh to create a level set grid which works the same as
// the level set grids expected by the other methods
grid_tuple VDBfromPly(fs::path fileName);

// Helper function which reads either an obj or vdb file into a level set
grid_tuple fileToVDB(fs::path path);

// Mesh helper functions
// Loads a mesh from a *.ply file
obj_tuple ObjFromPly(fs::path path);
// Loads a mesh from an *.obj file
obj_tuple ObjFromObj(fs::path path);
// Loads a level set from a *.vdb file and converts it into a mesh
obj_tuple ObjFromVDB(fs::path path);
//Helper function which reads either an obj or vdb file into a mesh
obj_tuple fileToObj(fs::path path);

// Wrapper function which can be configured using a genTechnique argument. Works with *.obj and *.vdb files.
// The fileName argument is parsed through resolveFiles with the configuration directory added.
std::vector<vdb::Vec4f> generateParticles(std::string fileName, float r, genTechnique kind = genTechnique::hex_grid,
                                              bool clampToDomain = false);
std::vector<vdb::Vec4f> generateParticlesRigid(std::string fileName, float r, genTechnique kind = genTechnique::hex_grid,
                                              bool clampToDomain = false, std::map<std::string, float3> maxmin = {});

} // namespace generation 