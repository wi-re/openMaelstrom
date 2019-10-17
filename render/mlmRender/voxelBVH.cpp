#define QT_NO_KEYWORDS
#include <render/mlmRender/voxelBVH.h>
#include <simulation/particleSystem.h>
#include <bitset>

BVHManager::BVHManager() {}
std::pair<std::vector<BVHManager::Primitive>, std::vector<BVHManager::Primitive>>
BVHManager::makePartition(const std::vector<Primitive> &P, int32_t bit) {
	if (bit == 64)
		return std::make_pair(std::vector<Primitive>(P.begin(), P.begin() + P.size() / 2), std::vector<Primitive>(P.begin() + P.size() / 2, P.end()));
	auto val = binarySearch(P, [&](auto v) { return std::bitset<64>(v)[63 - bit]; });
	if (val == -1)
		return std::make_pair(P, std::vector<Primitive>{});
	return std::make_pair(std::vector<Primitive>(P.begin(), P.begin() + val), std::vector<Primitive>(P.begin() + val, P.end()));
}
BVHManager::BVHNode* BVHManager::mergeClusters(BVHNode* a, BVHNode* b) {
	auto mi = math::min(a->center - a->extent * 0.5f, b->center - b->extent * 0.5f);
	auto ma = math::max(a->center + a->extent * 0.5f, b->center + b->extent * 0.5f);
	auto inner = new BVHInner;
	inner->center = (mi + ma) * 0.5f;
	inner->extent = (ma - mi);
	inner->left = a;
	inner->right = b;

	globalAllocations.push_back(inner);
	return inner;
}
std::vector<BVHManager::BVHNode*> BVHManager::combineBVHNodes(const std::vector<BVHNode*>& C, int32_t n) {
	auto getSurfaceArea = [](auto v) {
		auto a = v.max.x - v.min.x;
		auto b = v.max.y - v.min.y;
		auto c = v.max.z - v.min.z;
		return 2 * a * b + 2 * b * c + 2 * a * c;
	};
	auto mergeAABB = [](auto lhs, auto rhs) {
		return AABB{ math::min(lhs.min, rhs.min), math::max(lhs.max, rhs.max) };
	};
	auto getAABB = [](BVHNode* C) {
		return AABB{ C->center - C->extent*0.5f, C->center + C->extent*0.5f };
	};
	auto distance = [=](auto Ci, auto Cj) {
		return getSurfaceArea(mergeAABB(getAABB(Ci), getAABB(Cj)));
	};
	struct wrappedBVHNode {
		BVHNode* node;
		BVHNode* closest = nullptr;
	};
	std::vector<wrappedBVHNode> Cs;
	for (auto Ci : C) {
		Cs.push_back(wrappedBVHNode{ Ci, nullptr });
	}
	for (int32_t i = 0; i < Cs.size(); ++i) {
		auto& Ci = Cs[i];
		Ci.closest = nullptr;
		auto d = FLT_MAX;
		for (int32_t j = 0; j < Cs.size(); ++j) {
			if (i == j) continue;
			auto& Cj = Cs[j];
			auto SA = distance(Ci.node, Cj.node);
			if (SA < d) {
				d = SA;
				Ci.closest = Cj.node;
			}
		}
	}

	while (Cs.size() > n) {
		auto best = FLT_MAX;
		BVHNode* left = nullptr, *right = nullptr;
		for (auto& Ci : Cs) {
			auto dist = distance(Ci.node, Ci.closest);
			if (dist < best) {
				best = dist;
				left = Ci.node;
				right = Ci.closest;
			}
		}
		auto Cnew = mergeClusters(left, right);
		std::vector<wrappedBVHNode> newNodes;
		for (auto& Ci : Cs) {
			if (Ci.node != left && Ci.node != right) {
				if (Ci.closest == left || Ci.closest == right)
					Ci.closest = nullptr;
				newNodes.push_back(Ci);
			}
		}
		newNodes.push_back(wrappedBVHNode{ Cnew, nullptr });
		for (int32_t i = 0; i < newNodes.size(); ++i) {
			auto& Ci = newNodes[i];
			if (Ci.closest == nullptr) {
				auto d = FLT_MAX;
				for (int32_t j = 0; j < newNodes.size(); ++j) {
					if (i == j) continue;
					auto& Cj = newNodes[j];
					auto SA = distance(Ci.node, Cj.node);
					if (SA < d) {
						d = SA;
						Ci.closest = Cj.node;
					}
				}
			}
		}
		Cs = newNodes;
	}
	std::vector<BVHNode*> result;
	for (auto& C : Cs)
		result.push_back(C.node);
	return result;
}
std::vector<BVHManager::BVHNode*> BVHManager::buildBVH(const std::vector<Primitive>& P, int32_t bit){
	constexpr auto alpha = 0.2;
	constexpr auto delta = 4;
	constexpr auto minSize = delta / 2;
	auto f = [=](auto v) {return int(minSize*pow(float(v) / minSize / 2, 0.5 - alpha) - 1e-8) + 1; };

	if (P.size() < delta) {
		std::vector<BVHNode*> localNodes;
		for (const auto& prim : P) {
			auto new_node = new BVHLeaf(prim);
			globalAllocations.push_back(new_node);
			localNodes.push_back(new_node);
		}
		return combineBVHNodes(localNodes, f(delta));
	}
	auto[Pl, Pr] = makePartition(P, bit);
	auto Cl = Pl.size() == 0 ? std::vector<BVHNode*>{} : buildBVH(Pl, bit + 1);
	auto Cr = Pr.size() == 0 ? std::vector<BVHNode*>{} : buildBVH(Pr, bit + 1);
	Cl.insert(Cl.begin(), Cr.begin(), Cr.end());
	return combineBVHNodes(Cl, f(P.size()));
}
void BVHManager::buildCFBVH_impl(CFBVHNode* CFBVHroot, BVHNode *node, int32_t &idxBoxes) {
	uint32_t currIdxBoxes = idxBoxes;
	CFBVHroot[currIdxBoxes].min = node->center - node->extent * 0.5f;
	CFBVHroot[currIdxBoxes].max = node->center + node->extent * 0.5f;

	if (!node->leafNode()) {
		BVHInner *p = dynamic_cast<BVHInner *>(node);
		int32_t idxLeft = ++idxBoxes;
		buildCFBVH_impl(CFBVHroot, p->left, idxBoxes);
		int32_t idxRight = ++idxBoxes;
		buildCFBVH_impl(CFBVHroot, p->right, idxBoxes);
		CFBVHroot[currIdxBoxes].u.inner.idxLeft = idxLeft;
		CFBVHroot[currIdxBoxes].u.inner.idxRight = idxRight;
	}
	else {
		BVHLeaf *p = dynamic_cast<BVHLeaf *>(node);
		CFBVHroot[currIdxBoxes].u.leaf.count = 0x80000000 | p->beginning;
		CFBVHroot[currIdxBoxes].u.leaf.startIdx = p->length;
	}
}
int32_t BVHManager::countBoxes(BVHNode* node) {
	if (node->leafNode()) return 1;
	return 1 + countBoxes(dynamic_cast<BVHInner*>(node)->left) + countBoxes(dynamic_cast<BVHInner*>(node)->right);
}
// Inverse of Part1By1 - "delete" all odd-indexed bits
uint32_t Compact1By1(uint32_t x)
{
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
uint32_t Compact1By2(uint32_t x)
{
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

uint32_t DecodeMorton2X(uint32_t code)
{
	return Compact1By1(code >> 0);
}

uint32_t DecodeMorton2Y(uint32_t code)
{
	return Compact1By1(code >> 1);
}

uint32_t DecodeMorton3X(uint32_t code)
{
	return Compact1By2(code >> 0);
}

uint32_t DecodeMorton3Y(uint32_t code)
{
	return Compact1By2(code >> 1);
}

uint32_t DecodeMorton3Z(uint32_t code)
{
	return Compact1By2(code >> 2);
}

void BVHManager::buildCFBVH() {
	if (frame == get<parameters::frame>()) return;
	frame = get<parameters::frame>();

	for (auto alloc : globalAllocations)
		free(alloc);
	globalAllocations.clear();

	std::unique_ptr<float4[]> positions;
	positions = std::make_unique <float4[]>(get<parameters::num_ptcls>());
	cudaMemcpy(positions.get(), arrays::position::ptr, sizeof(float4) * get<parameters::num_ptcls>(), cudaMemcpyDeviceToHost);

	auto nodes = 0;
	int32_t numCells = get<parameters::valid_cells>();
	std::vector<compact_cellSpan> Cells;
	Cells.resize(numCells);
	cudaMemcpy(Cells.data(), arrays::compactCellSpan::ptr, sizeof(compact_cellSpan) * numCells, cudaMemcpyDeviceToHost);


	std::vector<Primitive> P;
	auto min_coord = get<parameters::min_coord>();
	auto cell_size = get<parameters::cell_size>();
	for (const auto& cell : Cells) {
		auto idx = position_to_idx3D_i(positions[cell.beginning], min_coord, cell_size.x);
		//int3 idx{ DecodeMorton3X(morton), DecodeMorton3Y(morton), DecodeMorton3Z(morton) };
		auto morton = idx3D_to_morton(idx);
		//auto firstPosition = positions[cell.beginning];
		//auto idx = position_to_idx3D(firstPosition, min_coord, cell_size.x * get<parameters::auxScale>());
		auto lb = (math::castTo<float3>(idx))* cell_size.x * get<parameters::auxScale>() + min_coord;
		auto center = lb + cell_size * get<parameters::auxScale>() * 0.5f;
		//auto morton = idx3D_to_morton(position_to_idx3D(center, min_coord, math::unit_get<1>(cell_size) *  get<parameters::auxScale>()));
		//std::cout << std::bitset<64>(morton) << std::endl;
		P.push_back(Primitive{ center, cell_size * get<parameters::auxScale>(), (uint64_t) morton });
	}
	std::sort(P.begin(), P.end(), [](const auto& lhs, const auto& rhs) { return lhs.zIndex < rhs.zIndex; });
	//getchar();

	auto C = buildBVH(P);
	bvhRoot = combineBVHNodes(C, 1)[0];
	boxCount = countBoxes(bvhRoot);
	cfbvhHost = std::make_unique<CFBVHNode[]>(boxCount);
	int32_t boxCounter = 0;
	buildCFBVH_impl(cfbvhHost.get(), bvhRoot, boxCounter);
	//std::cout << C.size() << std::endl;
	//std::cout << "box counter: " << countBoxes(bvhRoot) << std::endl;
	//std::cout << "allocation counter: " << globalAllocations.size() << std::endl;
	//getchar();

}
std::pair<CFBVHNode*, int32_t> BVHManager::getGPUData() {
	buildCFBVH();
	if (cfbvhDevice != nullptr)
		cudaFree(cfbvhDevice);
	cudaMalloc(&cfbvhDevice, sizeof(CFBVHNode) * boxCount);
	cudaMemcpy(cfbvhDevice, cfbvhHost.get(), sizeof(CFBVHNode) * boxCount, cudaMemcpyHostToDevice);
	return std::make_pair(cfbvhDevice, boxCount);
}