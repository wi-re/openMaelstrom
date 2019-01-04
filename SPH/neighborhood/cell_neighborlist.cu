#include <SPH/neighborhood/cell_neighborlist.cuh>
#include <utility/include_all.h>
// Due to the nature of the cell based neighborhood the neighborhood search based on cell spans has
// to be directly interwoven into the cell iteration and as such the function needs to be manually
// overloaded on a template level. This set of functions creates the cell based neighborlist and
// everytime a new cell based system is added it needs to be represented here.
cellFunctionType createCellNeighborList(SPH::spanNeighborList::Memory arrays) {
  checkedParticleIdx(i);
  auto x_i = arrays.position[i];
  auto h_i = support_H(x_i);
  auto f_i = structure == cell_structuring::MLM ? powf(0.5f, (float)arrays.MLMResolution[i]) : 1.f;
  //f_i = 1.f;
  auto code_equal = [=](auto x_i, auto x_j) {
    if (structure == cell_structuring::MLM || (structure == cell_structuring::hashed && hash_width == hash_length::bit_64))
      return position_to_morton(x_i, arrays, f_i) == position_to_morton(x_j, arrays, f_i);
    else
      return position_to_morton_32(x_i, arrays, f_i) == position_to_morton_32(x_j, arrays, f_i);
  }; 
  auto x_p = float4_u<SI::m>{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
  neigh_span current{BITFIELD_MAX_VALUE, 0};
  int32_t valid_idx = 0;
  int32_t counter = 0;
  iterateCells(x_i, j) {
    auto x_j = arrays.position[j];
    auto h_j = support_H(x_j);
    auto x_ij = math::distance3(x_i, x_j);
    auto h_ij = 0.5f * (h_i + h_j);
    if (x_ij <= h_ij) {
      counter++;
      if (!code_equal(x_p, x_j)) {
        if (current.beginning != BITFIELD_MAX_VALUE)
          arrays.spanNeighborList[i + (valid_idx++) * arrays.num_ptcls] = current;
        current = neigh_span{(uint32_t)j, (uint32_t)0};
        x_p = x_j;
      } else
        current.length = (uint32_t)(j)-current.beginning;
    }
  }
  if (current.beginning != BITFIELD_MAX_VALUE)
    arrays.spanNeighborList[i + (valid_idx++) * arrays.num_ptcls] = current;
  arrays.spanNeighborList[i + (valid_idx)*arrays.num_ptcls] = neigh_span{BITFIELD_MAX_VALUE, 0};

  arrays.neighborListLength[i] = counter;
}

cellFunction(cellNeighborList2, createCellNeighborList, "Neighborhood: Cell Based");

void SPH::spanNeighborList::calculate_neighborlist(Memory arrays) {
  launch<cellNeighborList2>(arrays.num_ptcls, arrays);
}
