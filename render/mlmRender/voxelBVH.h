#pragma once
#include <utility/include_all.h>     

struct CFBVHNode {
	float3 min;
	float3 max;
	union {
		struct {
			int32_t idxLeft;
			int32_t idxRight;
		} inner;
		struct {
			int32_t count;
			int32_t startIdx;
		} leaf;
	} u;
};

class BVHManager {
	struct AABB {
		float3 min, max;
	};
	struct Primitive {
		float3 center;
		float3 extent;
		uint64_t zIndex;
		cell_span span;
	};
	struct BVHNode {
		float3 center;
		float3 extent;
		virtual bool leafNode() = 0;
	};
	struct BVHLeaf : BVHNode {
		int32_t beginning;
		int32_t length;
		BVHLeaf(Primitive prim) {
			center = prim.center;
			extent = prim.extent;
			beginning = prim.span.beginning;
			length = prim.span.length;
		}
		virtual bool leafNode() {
			return true;
		}
	};
	struct BVHInner : BVHNode {
		BVHNode* left;
		BVHNode* right;
		virtual bool leafNode() {
			return false;
		}
	};
public:
	BVHManager();
	
	void buildCFBVH();
	std::pair<CFBVHNode*, int32_t> getGPUData();
private:
	template<typename U>
	auto binarySearch(const std::vector<Primitive>& data, U&& fn) {
		int32_t low = 0;
		int32_t high = (int32_t)data.size() - 1;

		while (low <= high) {
			int32_t mid = (low + high) >> 1;
			auto midVal = fn(data[mid].zIndex);
			if (midVal != 1)
				low = mid + 1;
			else if (low != mid) //Equal but range is not fully scanned
				high = mid; //Set upper bound to current number and rescan
			else //Equal and full range is scanned
				return mid;
		}
		return -1; // key not found. return insertion point
	}

	std::pair< std::vector<Primitive>, std::vector<Primitive>> makePartition(const std::vector<Primitive>& P, int32_t bit);
	BVHNode* mergeClusters(BVHNode* a, BVHNode* b);
	std::vector<BVHNode*> combineBVHNodes(const std::vector<BVHNode*>& C, int32_t n);
	std::vector<BVHNode*> buildBVH(const std::vector<Primitive>& P, int32_t bit = 0);
	void buildCFBVH_impl(CFBVHNode* CFBVHroot, BVHNode *node, int32_t &idxBoxes);
	int32_t countBoxes(BVHNode* node);
	   
	std::unique_ptr<CFBVHNode[]> cfbvhHost;
	CFBVHNode* cfbvhDevice = nullptr;

	std::vector<BVHNode*> globalAllocations;
	BVHNode* bvhRoot;

	int32_t boxCount;
	int32_t frame = -1;
};



