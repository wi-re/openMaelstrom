#include <utility/include_all.h>
#include <utility/generation.h>

namespace generation {
sort_Edge::sort_Edge(const float3 v1, const float3 v2) {
  start = v1;
  end = v2;
}

bool sort_Edge::operator<(const sort_Edge &rhs) const {
  return math::length(end - start) < math::length(rhs.end - rhs.start);
}
} // namespace generation 