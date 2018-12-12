#include <SPH/neighborhood/basic_neighborlist.cuh>
#include <utility/include_all.h>

// This function creates a very basic neighborlist that does not check for lenght requirements as it
// cannot constrain the list anyways. Due to this this method might crash the simulation if used
// with a too small neighborhood size. Use with caution.
cellFunctionType basic_neighborhood(SPH::BasicNeighborList::Memory arrays) {
  checkedParticleIdx(i);
  auto idx = [=](auto id) {return arrays.max_numptcls * id + i; };
  auto x_i = arrays.position[i];
  auto h_i = support_H(x_i);

  int32_t curNeighborIdx = 0;

  iterateCells(x_i,j) {
    auto x_j = arrays.position[j];
    auto h_j = support_H(x_j);
    auto x_ij = math::distance3(x_i, x_j);
    auto h_ij = 0.5f * (h_i + h_j);
    if (x_ij <= h_ij)
      arrays.neighborList[idx(curNeighborIdx++)] = j;
  }

  arrays.neighborListLength[i] = curNeighborIdx;
}
cellFunction(basicNeighborlist, basic_neighborhood, "Neighborhood: Basic");

void SPH::BasicNeighborList::calculate_neighborlist(Memory mem) {
  launch<basicNeighborlist>(mem.num_ptcls, mem);
}
