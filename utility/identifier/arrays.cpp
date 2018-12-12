
#include "utility/identifier/arrays.h"
#include "utility/identifier/uniform.h"
#include <cuda.h>
#include <cuda_runtime.h>

int32_t* arrays::adaptiveMergeable::ptr = nullptr;
size_t arrays::adaptiveMergeable::alloc_size = 0;


void arrays::adaptiveMergeable::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveMergeable::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::adaptiveMergeable::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveMergeable::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::adaptiveMergeable::operator type*(){ return ptr;}
int32_t& arrays::adaptiveMergeable::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveMergeable::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
int32_t* arrays::adaptiveMergeCounter::ptr = nullptr;
size_t arrays::adaptiveMergeCounter::alloc_size = 0;


void arrays::adaptiveMergeCounter::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveMergeCounter::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::adaptiveMergeCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveMergeCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::adaptiveMergeCounter::operator type*(){ return ptr;}
int32_t& arrays::adaptiveMergeCounter::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveMergeCounter::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
int32_t* arrays::adaptiveNumPtcls::ptr = nullptr;
size_t arrays::adaptiveNumPtcls::alloc_size = 0;


void arrays::adaptiveNumPtcls::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveNumPtcls::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::adaptiveNumPtcls::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveNumPtcls::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::adaptiveNumPtcls::operator type*(){ return ptr;}
int32_t& arrays::adaptiveNumPtcls::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveNumPtcls::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
float* arrays::adaptiveClassification::ptr = nullptr;
size_t arrays::adaptiveClassification::alloc_size = 0;


void arrays::adaptiveClassification::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveClassification::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::adaptiveClassification::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveClassification::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::adaptiveClassification::operator type*(){ return ptr;}
float& arrays::adaptiveClassification::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveClassification::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
int32_t* arrays::adaptiveSplitIndicator::ptr = nullptr;
size_t arrays::adaptiveSplitIndicator::alloc_size = 0;

int32_t* arrays::adaptiveSplitIndicator::rear_ptr = nullptr;
void arrays::adaptiveSplitIndicator::swap() { std::swap(ptr, rear_ptr); }

void arrays::adaptiveSplitIndicator::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveSplitIndicator::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::adaptiveSplitIndicator::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveSplitIndicator::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::adaptiveSplitIndicator::operator type*(){ return ptr;}
int32_t& arrays::adaptiveSplitIndicator::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveSplitIndicator::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
int32_t* arrays::adaptiveSplitIndicatorCompacted::ptr = nullptr;
size_t arrays::adaptiveSplitIndicatorCompacted::alloc_size = 0;


void arrays::adaptiveSplitIndicatorCompacted::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveSplitIndicatorCompacted::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::adaptiveSplitIndicatorCompacted::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::adaptiveSplitIndicatorCompacted::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::adaptiveSplitIndicatorCompacted::operator type*(){ return ptr;}
int32_t& arrays::adaptiveSplitIndicatorCompacted::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveSplitIndicatorCompacted::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
int32_t* arrays::adaptiveParentIndex::ptr = nullptr;
size_t arrays::adaptiveParentIndex::alloc_size = 0;

int32_t* arrays::adaptiveParentIndex::rear_ptr = nullptr;
void arrays::adaptiveParentIndex::swap() { std::swap(ptr, rear_ptr); }

void arrays::adaptiveParentIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::adaptiveParentIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::adaptiveParentIndex::operator type*(){ return ptr;}
int32_t& arrays::adaptiveParentIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveParentIndex::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
float* arrays::adaptiveParentVolume::ptr = nullptr;
size_t arrays::adaptiveParentVolume::alloc_size = 0;

float* arrays::adaptiveParentVolume::rear_ptr = nullptr;
void arrays::adaptiveParentVolume::swap() { std::swap(ptr, rear_ptr); }

void arrays::adaptiveParentVolume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentVolume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::adaptiveParentVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::adaptiveParentVolume::operator type*(){ return ptr;}
float& arrays::adaptiveParentVolume::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveParentVolume::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
float4* arrays::adaptiveParentPosition::ptr = nullptr;
size_t arrays::adaptiveParentPosition::alloc_size = 0;

float4* arrays::adaptiveParentPosition::rear_ptr = nullptr;
void arrays::adaptiveParentPosition::swap() { std::swap(ptr, rear_ptr); }

void arrays::adaptiveParentPosition::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentPosition::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::adaptiveParentPosition::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::adaptiveParentPosition::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::adaptiveParentPosition::operator type*(){ return ptr;}
float4& arrays::adaptiveParentPosition::operator[](size_t idx){ return ptr[idx];}
bool arrays::adaptiveParentPosition::valid(){
	bool condition = true;
	condition = condition && get<parameters::surfaceDistance>() == true;
	condition = condition && get<parameters::adaptive>() == true;
	return condition;
}
float* arrays::maxVelocity::ptr = nullptr;
size_t arrays::maxVelocity::alloc_size = 0;


void arrays::maxVelocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::maxVelocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::maxVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::maxVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::maxVelocity::operator type*(){ return ptr;}
float& arrays::maxVelocity::operator[](size_t idx){ return ptr[idx];}
bool arrays::maxVelocity::valid(){
	bool condition = true;
	return condition;
}
float* arrays::cflValue::ptr = nullptr;
size_t arrays::cflValue::alloc_size = 0;


void arrays::cflValue::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cflValue::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::cflValue::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cflValue::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cflValue::operator type*(){ return ptr;}
float& arrays::cflValue::operator[](size_t idx){ return ptr[idx];}
bool arrays::cflValue::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::particleNormal::ptr = nullptr;
size_t arrays::particleNormal::alloc_size = 0;


void arrays::particleNormal::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleNormal::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::particleNormal::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleNormal::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::particleNormal::operator type*(){ return ptr;}
float4& arrays::particleNormal::operator[](size_t idx){ return ptr[idx];}
bool arrays::particleNormal::valid(){
	bool condition = false;
	condition = condition || get<parameters::tension>() == "Akinci";
	return condition;
}
float4* arrays::angularVelocity::ptr = nullptr;
size_t arrays::angularVelocity::alloc_size = 0;

float4* arrays::angularVelocity::rear_ptr = nullptr;
void arrays::angularVelocity::swap() { std::swap(ptr, rear_ptr); }

void arrays::angularVelocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::angularVelocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::angularVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::angularVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::angularVelocity::operator type*(){ return ptr;}
float4& arrays::angularVelocity::operator[](size_t idx){ return ptr[idx];}
bool arrays::angularVelocity::valid(){
	bool condition = false;
	condition = condition || get<parameters::vorticity>() == "Bender17";
	return condition;
}
float* arrays::boundaryLUT::ptr = nullptr;
size_t arrays::boundaryLUT::alloc_size = 0;



void arrays::boundaryLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::boundaryLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::boundaryLUT::operator type*(){ return ptr;}
float& arrays::boundaryLUT::operator[](size_t idx){ return ptr[idx];}
bool arrays::boundaryLUT::valid(){
	bool condition = true;
	return condition;
}
float* arrays::boundaryPressureLUT::ptr = nullptr;
size_t arrays::boundaryPressureLUT::alloc_size = 0;



void arrays::boundaryPressureLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::boundaryPressureLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::boundaryPressureLUT::operator type*(){ return ptr;}
float& arrays::boundaryPressureLUT::operator[](size_t idx){ return ptr[idx];}
bool arrays::boundaryPressureLUT::valid(){
	bool condition = true;
	return condition;
}
float* arrays::xbarLUT::ptr = nullptr;
size_t arrays::xbarLUT::alloc_size = 0;



void arrays::xbarLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::xbarLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::xbarLUT::operator type*(){ return ptr;}
float& arrays::xbarLUT::operator[](size_t idx){ return ptr[idx];}
bool arrays::xbarLUT::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::ctrLUT::ptr = nullptr;
size_t arrays::ctrLUT::alloc_size = 0;



void arrays::ctrLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::ctrLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::ctrLUT::operator type*(){ return ptr;}
int32_t& arrays::ctrLUT::operator[](size_t idx){ return ptr[idx];}
bool arrays::ctrLUT::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::boundaryPlanes::ptr = nullptr;
size_t arrays::boundaryPlanes::alloc_size = 0;



void arrays::boundaryPlanes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::boundaryPlanes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::boundaryPlanes::operator type*(){ return ptr;}
float4& arrays::boundaryPlanes::operator[](size_t idx){ return ptr[idx];}
bool arrays::boundaryPlanes::valid(){
	bool condition = true;
	return condition;
}
cudaTextureObject_t* arrays::volumeBoundaryVolumes::ptr = nullptr;
size_t arrays::volumeBoundaryVolumes::alloc_size = 0;



void arrays::volumeBoundaryVolumes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeBoundaryVolumes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeBoundaryVolumes::operator type*(){ return ptr;}
cudaTextureObject_t& arrays::volumeBoundaryVolumes::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeBoundaryVolumes::valid(){
	bool condition = true;
	return condition;
}
int4* arrays::volumeBoundaryDimensions::ptr = nullptr;
size_t arrays::volumeBoundaryDimensions::alloc_size = 0;



void arrays::volumeBoundaryDimensions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeBoundaryDimensions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeBoundaryDimensions::operator type*(){ return ptr;}
int4& arrays::volumeBoundaryDimensions::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeBoundaryDimensions::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::volumeBoundaryMin::ptr = nullptr;
size_t arrays::volumeBoundaryMin::alloc_size = 0;



void arrays::volumeBoundaryMin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeBoundaryMin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeBoundaryMin::operator type*(){ return ptr;}
float4& arrays::volumeBoundaryMin::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeBoundaryMin::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::volumeBoundaryMax::ptr = nullptr;
size_t arrays::volumeBoundaryMax::alloc_size = 0;



void arrays::volumeBoundaryMax::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeBoundaryMax::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeBoundaryMax::operator type*(){ return ptr;}
float4& arrays::volumeBoundaryMax::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeBoundaryMax::valid(){
	bool condition = true;
	return condition;
}
float* arrays::decisionBuffer::ptr = nullptr;
size_t arrays::decisionBuffer::alloc_size = 0;


void arrays::decisionBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::decisionBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::decisionBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::decisionBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::decisionBuffer::operator type*(){ return ptr;}
float& arrays::decisionBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::decisionBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::surfaceDistance>() == true;
	return condition;
}
int32_t* arrays::surface_idxBuffer::ptr = nullptr;
size_t arrays::surface_idxBuffer::alloc_size = 0;

int32_t* arrays::surface_idxBuffer::rear_ptr = nullptr;
void arrays::surface_idxBuffer::swap() { std::swap(ptr, rear_ptr); }

void arrays::surface_idxBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::surface_idxBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::surface_idxBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::surface_idxBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::surface_idxBuffer::operator type*(){ return ptr;}
int32_t& arrays::surface_idxBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::surface_idxBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::surfaceDistance>() == true;
	return condition;
}
float* arrays::markerBuffer::ptr = nullptr;
size_t arrays::markerBuffer::alloc_size = 0;


void arrays::markerBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::markerBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::markerBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::markerBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::markerBuffer::operator type*(){ return ptr;}
float& arrays::markerBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::markerBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::surfaceDistance>() == true;
	return condition;
}
float* arrays::distanceBuffer::ptr = nullptr;
size_t arrays::distanceBuffer::alloc_size = 0;

float* arrays::distanceBuffer::rear_ptr = nullptr;
void arrays::distanceBuffer::swap() { std::swap(ptr, rear_ptr); }

void arrays::distanceBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::distanceBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::distanceBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::distanceBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::distanceBuffer::operator type*(){ return ptr;}
float& arrays::distanceBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::distanceBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::surfaceDistance>() == true;
	return condition;
}
float* arrays::changeBuffer::ptr = nullptr;
size_t arrays::changeBuffer::alloc_size = 0;


void arrays::changeBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::changeBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::changeBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::changeBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::changeBuffer::operator type*(){ return ptr;}
float& arrays::changeBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::changeBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::surfaceDistance>() == true;
	return condition;
}
float4* arrays::position::ptr = nullptr;
size_t arrays::position::alloc_size = 0;

float4* arrays::position::rear_ptr = nullptr;
void arrays::position::swap() { std::swap(ptr, rear_ptr); }

void arrays::position::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::position::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::position::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::position::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::position::operator type*(){ return ptr;}
float4& arrays::position::operator[](size_t idx){ return ptr[idx];}
bool arrays::position::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::acceleration::ptr = nullptr;
size_t arrays::acceleration::alloc_size = 0;


void arrays::acceleration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::acceleration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::acceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::acceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::acceleration::operator type*(){ return ptr;}
float4& arrays::acceleration::operator[](size_t idx){ return ptr[idx];}
bool arrays::acceleration::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::velocity::ptr = nullptr;
size_t arrays::velocity::alloc_size = 0;

float4* arrays::velocity::rear_ptr = nullptr;
void arrays::velocity::swap() { std::swap(ptr, rear_ptr); }

void arrays::velocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::velocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::velocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::velocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::velocity::operator type*(){ return ptr;}
float4& arrays::velocity::operator[](size_t idx){ return ptr[idx];}
bool arrays::velocity::valid(){
	bool condition = true;
	return condition;
}
float* arrays::renderIntensity::ptr = nullptr;
size_t arrays::renderIntensity::alloc_size = 0;

float* arrays::renderIntensity::rear_ptr = nullptr;
void arrays::renderIntensity::swap() { std::swap(ptr, rear_ptr); }

void arrays::renderIntensity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::renderIntensity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::renderIntensity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::renderIntensity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::renderIntensity::operator type*(){ return ptr;}
float& arrays::renderIntensity::operator[](size_t idx){ return ptr[idx];}
bool arrays::renderIntensity::valid(){
	bool condition = true;
	return condition;
}
float* arrays::volume::ptr = nullptr;
size_t arrays::volume::alloc_size = 0;

float* arrays::volume::rear_ptr = nullptr;
void arrays::volume::swap() { std::swap(ptr, rear_ptr); }

void arrays::volume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::volume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::volume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::volume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::volume::operator type*(){ return ptr;}
float& arrays::volume::operator[](size_t idx){ return ptr[idx];}
bool arrays::volume::valid(){
	bool condition = true;
	return condition;
}
float* arrays::lifetime::ptr = nullptr;
size_t arrays::lifetime::alloc_size = 0;

float* arrays::lifetime::rear_ptr = nullptr;
void arrays::lifetime::swap() { std::swap(ptr, rear_ptr); }

void arrays::lifetime::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::lifetime::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::lifetime::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::lifetime::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::lifetime::operator type*(){ return ptr;}
float& arrays::lifetime::operator[](size_t idx){ return ptr[idx];}
bool arrays::lifetime::valid(){
	bool condition = true;
	return condition;
}
float* arrays::pressure::ptr = nullptr;
size_t arrays::pressure::alloc_size = 0;

float* arrays::pressure::rear_ptr = nullptr;
void arrays::pressure::swap() { std::swap(ptr, rear_ptr); }

void arrays::pressure::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::pressure::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::pressure::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::pressure::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::pressure::operator type*(){ return ptr;}
float& arrays::pressure::operator[](size_t idx){ return ptr[idx];}
bool arrays::pressure::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	condition = condition || get<parameters::pressure>() == "IISPH17";
	return condition;
}
float* arrays::density::ptr = nullptr;
size_t arrays::density::alloc_size = 0;

float* arrays::density::rear_ptr = nullptr;
void arrays::density::swap() { std::swap(ptr, rear_ptr); }

void arrays::density::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::density::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::density::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::density::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::density::operator type*(){ return ptr;}
float& arrays::density::operator[](size_t idx){ return ptr[idx];}
bool arrays::density::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::particleIndex::ptr = nullptr;
size_t arrays::particleIndex::alloc_size = 0;


void arrays::particleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::particleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::particleIndex::operator type*(){ return ptr;}
int32_t& arrays::particleIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::particleIndex::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::particleIndexCompact::ptr = nullptr;
size_t arrays::particleIndexCompact::alloc_size = 0;


void arrays::particleIndexCompact::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleIndexCompact::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::particleIndexCompact::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleIndexCompact::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::particleIndexCompact::operator type*(){ return ptr;}
int32_t& arrays::particleIndexCompact::operator[](size_t idx){ return ptr[idx];}
bool arrays::particleIndexCompact::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::resortArray4::ptr = nullptr;
size_t arrays::resortArray4::alloc_size = 0;


void arrays::resortArray4::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortArray4::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::resortArray4::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortArray4::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::resortArray4::operator type*(){ return ptr;}
float4& arrays::resortArray4::operator[](size_t idx){ return ptr[idx];}
bool arrays::resortArray4::valid(){
	bool condition = true;
	return condition;
}
float* arrays::resortArray::ptr = nullptr;
size_t arrays::resortArray::alloc_size = 0;


void arrays::resortArray::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortArray::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::resortArray::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortArray::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::resortArray::operator type*(){ return ptr;}
float& arrays::resortArray::operator[](size_t idx){ return ptr[idx];}
bool arrays::resortArray::valid(){
	bool condition = true;
	return condition;
}
float* arrays::dfsphDpDt::ptr = nullptr;
size_t arrays::dfsphDpDt::alloc_size = 0;


void arrays::dfsphDpDt::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphDpDt::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::dfsphDpDt::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphDpDt::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::dfsphDpDt::operator type*(){ return ptr;}
float& arrays::dfsphDpDt::operator[](size_t idx){ return ptr[idx];}
bool arrays::dfsphDpDt::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "DFSPH";
	return condition;
}
float* arrays::dfsphAlpha::ptr = nullptr;
size_t arrays::dfsphAlpha::alloc_size = 0;


void arrays::dfsphAlpha::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphAlpha::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::dfsphAlpha::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphAlpha::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::dfsphAlpha::operator type*(){ return ptr;}
float& arrays::dfsphAlpha::operator[](size_t idx){ return ptr[idx];}
bool arrays::dfsphAlpha::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "DFSPH";
	return condition;
}
float* arrays::dfsphRhoStar::ptr = nullptr;
size_t arrays::dfsphRhoStar::alloc_size = 0;


void arrays::dfsphRhoStar::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphRhoStar::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::dfsphRhoStar::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::dfsphRhoStar::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::dfsphRhoStar::operator type*(){ return ptr;}
float& arrays::dfsphRhoStar::operator[](size_t idx){ return ptr[idx];}
bool arrays::dfsphRhoStar::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "DFSPH";
	return condition;
}
float* arrays::dfsphKappa::ptr = nullptr;
size_t arrays::dfsphKappa::alloc_size = 0;

float* arrays::dfsphKappa::rear_ptr = nullptr;
void arrays::dfsphKappa::swap() { std::swap(ptr, rear_ptr); }

void arrays::dfsphKappa::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::dfsphKappa::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::dfsphKappa::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::dfsphKappa::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::dfsphKappa::operator type*(){ return ptr;}
float& arrays::dfsphKappa::operator[](size_t idx){ return ptr[idx];}
bool arrays::dfsphKappa::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "DFSPH";
	return condition;
}
float* arrays::dfsphKappaDivergence::ptr = nullptr;
size_t arrays::dfsphKappaDivergence::alloc_size = 0;

float* arrays::dfsphKappaDivergence::rear_ptr = nullptr;
void arrays::dfsphKappaDivergence::swap() { std::swap(ptr, rear_ptr); }

void arrays::dfsphKappaDivergence::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::dfsphKappaDivergence::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::dfsphKappaDivergence::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::dfsphKappaDivergence::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::dfsphKappaDivergence::operator type*(){ return ptr;}
float& arrays::dfsphKappaDivergence::operator[](size_t idx){ return ptr[idx];}
bool arrays::dfsphKappaDivergence::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "DFSPH";
	return condition;
}
float4* arrays::iisphSum::ptr = nullptr;
size_t arrays::iisphSum::alloc_size = 0;


void arrays::iisphSum::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphSum::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::iisphSum::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphSum::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphSum::operator type*(){ return ptr;}
float4& arrays::iisphSum::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphSum::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float4* arrays::iisphDii::ptr = nullptr;
size_t arrays::iisphDii::alloc_size = 0;


void arrays::iisphDii::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDii::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::iisphDii::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDii::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphDii::operator type*(){ return ptr;}
float4& arrays::iisphDii::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphDii::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::iisphAiiOld::ptr = nullptr;
size_t arrays::iisphAiiOld::alloc_size = 0;


void arrays::iisphAiiOld::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAiiOld::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphAiiOld::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAiiOld::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphAiiOld::operator type*(){ return ptr;}
float& arrays::iisphAiiOld::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphAiiOld::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::omega::ptr = nullptr;
size_t arrays::omega::alloc_size = 0;


void arrays::omega::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::omega::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::omega::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::omega::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::omega::operator type*(){ return ptr;}
float& arrays::omega::operator[](size_t idx){ return ptr[idx];}
bool arrays::omega::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::alpha::ptr = nullptr;
size_t arrays::alpha::alloc_size = 0;


void arrays::alpha::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::alpha::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::alpha::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::alpha::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::alpha::operator type*(){ return ptr;}
float& arrays::alpha::operator[](size_t idx){ return ptr[idx];}
bool arrays::alpha::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::gamma::ptr = nullptr;
size_t arrays::gamma::alloc_size = 0;


void arrays::gamma::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::gamma::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::gamma::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::gamma::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::gamma::operator type*(){ return ptr;}
float& arrays::gamma::operator[](size_t idx){ return ptr[idx];}
bool arrays::gamma::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::sigma::ptr = nullptr;
size_t arrays::sigma::alloc_size = 0;


void arrays::sigma::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::sigma::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::sigma::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::sigma::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::sigma::operator type*(){ return ptr;}
float& arrays::sigma::operator[](size_t idx){ return ptr[idx];}
bool arrays::sigma::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::iisphDensityAdvection::ptr = nullptr;
size_t arrays::iisphDensityAdvection::alloc_size = 0;


void arrays::iisphDensityAdvection::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDensityAdvection::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphDensityAdvection::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDensityAdvection::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphDensityAdvection::operator type*(){ return ptr;}
float& arrays::iisphDensityAdvection::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphDensityAdvection::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::iisphDensityIteration::ptr = nullptr;
size_t arrays::iisphDensityIteration::alloc_size = 0;


void arrays::iisphDensityIteration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDensityIteration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphDensityIteration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphDensityIteration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphDensityIteration::operator type*(){ return ptr;}
float& arrays::iisphDensityIteration::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphDensityIteration::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float4* arrays::kernelBuffer::ptr = nullptr;
size_t arrays::kernelBuffer::alloc_size = 0;


void arrays::kernelBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::kernelBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::kernelBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::kernelBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::kernelBuffer::operator type*(){ return ptr;}
float4& arrays::kernelBuffer::operator[](size_t idx){ return ptr[idx];}
bool arrays::kernelBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float4* arrays::velocityAdvection::ptr = nullptr;
size_t arrays::velocityAdvection::alloc_size = 0;


void arrays::velocityAdvection::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::velocityAdvection::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::velocityAdvection::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::velocityAdvection::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::velocityAdvection::operator type*(){ return ptr;}
float4& arrays::velocityAdvection::operator[](size_t idx){ return ptr[idx];}
bool arrays::velocityAdvection::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH";
	return condition;
}
float* arrays::iisphSource::ptr = nullptr;
size_t arrays::iisphSource::alloc_size = 0;


void arrays::iisphSource::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphSource::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphSource::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphSource::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphSource::operator type*(){ return ptr;}
float& arrays::iisphSource::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphSource::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float* arrays::iisphOmega::ptr = nullptr;
size_t arrays::iisphOmega::alloc_size = 0;


void arrays::iisphOmega::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphOmega::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphOmega::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphOmega::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphOmega::operator type*(){ return ptr;}
float& arrays::iisphOmega::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphOmega::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float* arrays::iisphVolume::ptr = nullptr;
size_t arrays::iisphVolume::alloc_size = 0;


void arrays::iisphVolume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphVolume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphVolume::operator type*(){ return ptr;}
float& arrays::iisphVolume::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphVolume::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float* arrays::iisphAii::ptr = nullptr;
size_t arrays::iisphAii::alloc_size = 0;


void arrays::iisphAii::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAii::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphAii::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAii::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphAii::operator type*(){ return ptr;}
float& arrays::iisphAii::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphAii::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float4* arrays::iisphAcceleration::ptr = nullptr;
size_t arrays::iisphAcceleration::alloc_size = 0;


void arrays::iisphAcceleration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAcceleration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void arrays::iisphAcceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphAcceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphAcceleration::operator type*(){ return ptr;}
float4& arrays::iisphAcceleration::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphAcceleration::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float* arrays::iisphVolumeError::ptr = nullptr;
size_t arrays::iisphVolumeError::alloc_size = 0;


void arrays::iisphVolumeError::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphVolumeError::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::iisphVolumeError::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::iisphVolumeError::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::iisphVolumeError::operator type*(){ return ptr;}
float& arrays::iisphVolumeError::operator[](size_t idx){ return ptr[idx];}
bool arrays::iisphVolumeError::valid(){
	bool condition = false;
	condition = condition || get<parameters::pressure>() == "IISPH17";
	condition = condition || get<parameters::pressure>() == "densityMapIISPH";
	return condition;
}
float4* arrays::inletPositions::ptr = nullptr;
size_t arrays::inletPositions::alloc_size = 0;



void arrays::inletPositions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::inletPositions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::inletPositions::operator type*(){ return ptr;}
float4& arrays::inletPositions::operator[](size_t idx){ return ptr[idx];}
bool arrays::inletPositions::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::inletCounter::ptr = nullptr;
size_t arrays::inletCounter::alloc_size = 0;



void arrays::inletCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::inletCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::inletCounter::operator type*(){ return ptr;}
int32_t& arrays::inletCounter::operator[](size_t idx){ return ptr[idx];}
bool arrays::inletCounter::valid(){
	bool condition = true;
	return condition;
}
cudaTextureObject_t* arrays::volumeOutletVolumes::ptr = nullptr;
size_t arrays::volumeOutletVolumes::alloc_size = 0;



void arrays::volumeOutletVolumes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletVolumes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletVolumes::operator type*(){ return ptr;}
cudaTextureObject_t& arrays::volumeOutletVolumes::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletVolumes::valid(){
	bool condition = true;
	return condition;
}
int4* arrays::volumeOutletDimensions::ptr = nullptr;
size_t arrays::volumeOutletDimensions::alloc_size = 0;



void arrays::volumeOutletDimensions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletDimensions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletDimensions::operator type*(){ return ptr;}
int4& arrays::volumeOutletDimensions::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletDimensions::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::volumeOutletMin::ptr = nullptr;
size_t arrays::volumeOutletMin::alloc_size = 0;



void arrays::volumeOutletMin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletMin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletMin::operator type*(){ return ptr;}
float4& arrays::volumeOutletMin::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletMin::valid(){
	bool condition = true;
	return condition;
}
float4* arrays::volumeOutletMax::ptr = nullptr;
size_t arrays::volumeOutletMax::alloc_size = 0;



void arrays::volumeOutletMax::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletMax::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletMax::operator type*(){ return ptr;}
float4& arrays::volumeOutletMax::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletMax::valid(){
	bool condition = true;
	return condition;
}
float* arrays::volumeOutletRate::ptr = nullptr;
size_t arrays::volumeOutletRate::alloc_size = 0;



void arrays::volumeOutletRate::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletRate::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletRate::operator type*(){ return ptr;}
float& arrays::volumeOutletRate::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletRate::valid(){
	bool condition = true;
	return condition;
}
float* arrays::volumeOutletRateAccumulator::ptr = nullptr;
size_t arrays::volumeOutletRateAccumulator::alloc_size = 0;



void arrays::volumeOutletRateAccumulator::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::volumeOutletRateAccumulator::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::volumeOutletRateAccumulator::operator type*(){ return ptr;}
float& arrays::volumeOutletRateAccumulator::operator[](size_t idx){ return ptr[idx];}
bool arrays::volumeOutletRateAccumulator::valid(){
	bool condition = true;
	return condition;
}
neigh_span* arrays::spanNeighborList::ptr = nullptr;
size_t arrays::spanNeighborList::alloc_size = 0;


void arrays::spanNeighborList::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(neigh_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::spanNeighborList::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(neigh_span);
	
}

void arrays::spanNeighborList::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::spanNeighborList::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::spanNeighborList::operator type*(){ return ptr;}
neigh_span& arrays::spanNeighborList::operator[](size_t idx){ return ptr[idx];}
bool arrays::spanNeighborList::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "cell_based";
	return condition;
}
int32_t* arrays::neighborList::ptr = nullptr;
size_t arrays::neighborList::alloc_size = 0;


void arrays::neighborList::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::neighborlimit>() * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborList::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::neighborlimit>() * elems * sizeof(int32_t);
	
}

void arrays::neighborList::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborList::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::neighborList::operator type*(){ return ptr;}
int32_t& arrays::neighborList::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborList::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	condition = condition || get<parameters::neighborhood>() == "basic";
	return condition;
}
int32_t* arrays::neighborListLength::ptr = nullptr;
size_t arrays::neighborListLength::alloc_size = 0;

int32_t* arrays::neighborListLength::rear_ptr = nullptr;
void arrays::neighborListLength::swap() { std::swap(ptr, rear_ptr); }

void arrays::neighborListLength::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::neighborListLength::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void arrays::neighborListLength::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void arrays::neighborListLength::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
arrays::neighborListLength::operator type*(){ return ptr;}
int32_t& arrays::neighborListLength::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborListLength::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::classification::ptr = nullptr;
size_t arrays::classification::alloc_size = 0;


void arrays::classification::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::classification::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::classification::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::classification::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::classification::operator type*(){ return ptr;}
int32_t& arrays::classification::operator[](size_t idx){ return ptr[idx];}
bool arrays::classification::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::resortIndex::ptr = nullptr;
size_t arrays::resortIndex::alloc_size = 0;


void arrays::resortIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::resortIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::resortIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::resortIndex::operator type*(){ return ptr;}
int32_t& arrays::resortIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::resortIndex::valid(){
	bool condition = true;
	return condition;
}
int64_t* arrays::ZOrder_64::ptr = nullptr;
size_t arrays::ZOrder_64::alloc_size = 0;


void arrays::ZOrder_64::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int64_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::ZOrder_64::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int64_t);
	
}

void arrays::ZOrder_64::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::ZOrder_64::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::ZOrder_64::operator type*(){ return ptr;}
int64_t& arrays::ZOrder_64::operator[](size_t idx){ return ptr[idx];}
bool arrays::ZOrder_64::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
int32_t* arrays::ZOrder_32::ptr = nullptr;
size_t arrays::ZOrder_32::alloc_size = 0;


void arrays::ZOrder_32::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::ZOrder_32::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::ZOrder_32::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::ZOrder_32::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::ZOrder_32::operator type*(){ return ptr;}
int32_t& arrays::ZOrder_32::operator[](size_t idx){ return ptr[idx];}
bool arrays::ZOrder_32::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
cell_span* arrays::cellSpanSwap::ptr = nullptr;
size_t arrays::cellSpanSwap::alloc_size = 0;


void arrays::cellSpanSwap::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cell_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellSpanSwap::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cell_span);
	
}

void arrays::cellSpanSwap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellSpanSwap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cellSpanSwap::operator type*(){ return ptr;}
cell_span& arrays::cellSpanSwap::operator[](size_t idx){ return ptr[idx];}
bool arrays::cellSpanSwap::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
cell_span* arrays::cellSpan::ptr = nullptr;
size_t arrays::cellSpan::alloc_size = 0;


void arrays::cellSpan::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(cell_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellSpan::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(cell_span);
	
}

void arrays::cellSpan::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellSpan::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cellSpan::operator type*(){ return ptr;}
cell_span& arrays::cellSpan::operator[](size_t idx){ return ptr[idx];}
bool arrays::cellSpan::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
int32_t* arrays::MLMResolution::ptr = nullptr;
size_t arrays::MLMResolution::alloc_size = 0;


void arrays::MLMResolution::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::MLMResolution::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::MLMResolution::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::MLMResolution::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::MLMResolution::operator type*(){ return ptr;}
int32_t& arrays::MLMResolution::operator[](size_t idx){ return ptr[idx];}
bool arrays::MLMResolution::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
int32_t* arrays::cellparticleIndex::ptr = nullptr;
size_t arrays::cellparticleIndex::alloc_size = 0;


void arrays::cellparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::cellparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cellparticleIndex::operator type*(){ return ptr;}
int32_t& arrays::cellparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::cellparticleIndex::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
int32_t* arrays::compactparticleIndex::ptr = nullptr;
size_t arrays::compactparticleIndex::alloc_size = 0;


void arrays::compactparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::compactparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::compactparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::compactparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::compactparticleIndex::operator type*(){ return ptr;}
int32_t& arrays::compactparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::compactparticleIndex::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	condition = condition || get<parameters::sorting>() == "MLM";
	return condition;
}
hash_span* arrays::hashMap::ptr = nullptr;
size_t arrays::hashMap::alloc_size = 0;


void arrays::hashMap::defaultAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(hash_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::hashMap::leanAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(hash_span);
	
}

void arrays::hashMap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::hashMap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::hashMap::operator type*(){ return ptr;}
hash_span& arrays::hashMap::operator[](size_t idx){ return ptr[idx];}
bool arrays::hashMap::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "MLM";
	condition = condition || get<parameters::sorting>() == "hashed_cell";
	return condition;
}
int32_t* arrays::particleparticleIndex::ptr = nullptr;
size_t arrays::particleparticleIndex::alloc_size = 0;


void arrays::particleparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::particleparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::particleparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::particleparticleIndex::operator type*(){ return ptr;}
int32_t& arrays::particleparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool arrays::particleparticleIndex::valid(){
	bool condition = true;
	return condition;
}
int32_t* arrays::cellBegin::ptr = nullptr;
size_t arrays::cellBegin::alloc_size = 0;


void arrays::cellBegin::defaultAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellBegin::leanAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::cellBegin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellBegin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cellBegin::operator type*(){ return ptr;}
int32_t& arrays::cellBegin::operator[](size_t idx){ return ptr[idx];}
bool arrays::cellBegin::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "linear_cell";
	return condition;
}
int32_t* arrays::cellEnd::ptr = nullptr;
size_t arrays::cellEnd::alloc_size = 0;


void arrays::cellEnd::defaultAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellEnd::leanAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::cellEnd::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::cellEnd::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::cellEnd::operator type*(){ return ptr;}
int32_t& arrays::cellEnd::operator[](size_t idx){ return ptr[idx];}
bool arrays::cellEnd::valid(){
	bool condition = false;
	condition = condition || get<parameters::sorting>() == "linear_cell";
	return condition;
}
float* arrays::support::ptr = nullptr;
size_t arrays::support::alloc_size = 0;


void arrays::support::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::support::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::support::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::support::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::support::operator type*(){ return ptr;}
float& arrays::support::operator[](size_t idx){ return ptr[idx];}
bool arrays::support::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
float* arrays::supportEstimate::ptr = nullptr;
size_t arrays::supportEstimate::alloc_size = 0;


void arrays::supportEstimate::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportEstimate::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void arrays::supportEstimate::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportEstimate::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::supportEstimate::operator type*(){ return ptr;}
float& arrays::supportEstimate::operator[](size_t idx){ return ptr[idx];}
bool arrays::supportEstimate::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::neighborCount::ptr = nullptr;
size_t arrays::neighborCount::alloc_size = 0;


void arrays::neighborCount::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborCount::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::neighborCount::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborCount::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::neighborCount::operator type*(){ return ptr;}
int32_t& arrays::neighborCount::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborCount::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::supportMarker::ptr = nullptr;
size_t arrays::supportMarker::alloc_size = 0;


void arrays::supportMarker::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportMarker::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::supportMarker::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportMarker::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::supportMarker::operator type*(){ return ptr;}
int32_t& arrays::supportMarker::operator[](size_t idx){ return ptr[idx];}
bool arrays::supportMarker::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::supportMarkerCompacted::ptr = nullptr;
size_t arrays::supportMarkerCompacted::alloc_size = 0;


void arrays::supportMarkerCompacted::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportMarkerCompacted::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::supportMarkerCompacted::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::supportMarkerCompacted::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::supportMarkerCompacted::operator type*(){ return ptr;}
int32_t& arrays::supportMarkerCompacted::operator[](size_t idx){ return ptr[idx];}
bool arrays::supportMarkerCompacted::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::closestNeighbor::ptr = nullptr;
size_t arrays::closestNeighbor::alloc_size = 0;


void arrays::closestNeighbor::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::closestNeighbor::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::closestNeighbor::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::closestNeighbor::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::closestNeighbor::operator type*(){ return ptr;}
int32_t& arrays::closestNeighbor::operator[](size_t idx){ return ptr[idx];}
bool arrays::closestNeighbor::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::closestNeighbor_f::ptr = nullptr;
size_t arrays::closestNeighbor_f::alloc_size = 0;


void arrays::closestNeighbor_f::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::closestNeighbor_f::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::closestNeighbor_f::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::closestNeighbor_f::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::closestNeighbor_f::operator type*(){ return ptr;}
int32_t& arrays::closestNeighbor_f::operator[](size_t idx){ return ptr[idx];}
bool arrays::closestNeighbor_f::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::neighborOverhead::ptr = nullptr;
size_t arrays::neighborOverhead::alloc_size = 0;


void arrays::neighborOverhead::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborOverhead::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::neighborOverhead::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborOverhead::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::neighborOverhead::operator type*(){ return ptr;}
int32_t& arrays::neighborOverhead::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborOverhead::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::neighborOverheadCount::ptr = nullptr;
size_t arrays::neighborOverheadCount::alloc_size = 0;


void arrays::neighborOverheadCount::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborOverheadCount::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::neighborOverheadCount::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborOverheadCount::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::neighborOverheadCount::operator type*(){ return ptr;}
int32_t& arrays::neighborOverheadCount::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborOverheadCount::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}
int32_t* arrays::neighborListSwap::ptr = nullptr;
size_t arrays::neighborListSwap::alloc_size = 0;


void arrays::neighborListSwap::defaultAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborListSwap::leanAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void arrays::neighborListSwap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void arrays::neighborListSwap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
arrays::neighborListSwap::operator type*(){ return ptr;}
int32_t& arrays::neighborListSwap::operator[](size_t idx){ return ptr[idx];}
bool arrays::neighborListSwap::valid(){
	bool condition = false;
	condition = condition || get<parameters::neighborhood>() == "constrained";
	return condition;
}

std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::hashMap, arrays::particleparticleIndex, arrays::cellBegin, arrays::cellEnd, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount, arrays::neighborListSwap> allocations_list;
std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::boundaryLUT, arrays::boundaryPressureLUT, arrays::xbarLUT, arrays::ctrLUT, arrays::boundaryPlanes, arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::inletPositions, arrays::inletCounter, arrays::volumeOutletVolumes, arrays::volumeOutletDimensions, arrays::volumeOutletMin, arrays::volumeOutletMax, arrays::volumeOutletRate, arrays::volumeOutletRateAccumulator, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::hashMap, arrays::particleparticleIndex, arrays::cellBegin, arrays::cellEnd, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount, arrays::neighborListSwap> arrays_list;
std::tuple<arrays::adaptiveSplitIndicator, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::angularVelocity, arrays::distanceBuffer, arrays::position, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::neighborListLength> sorting_list;
std::tuple<arrays::adaptiveMergeable, arrays::adaptiveMergeCounter, arrays::adaptiveNumPtcls, arrays::adaptiveClassification, arrays::adaptiveSplitIndicator, arrays::adaptiveSplitIndicatorCompacted, arrays::adaptiveParentIndex, arrays::adaptiveParentVolume, arrays::adaptiveParentPosition, arrays::maxVelocity, arrays::cflValue, arrays::particleNormal, arrays::angularVelocity, arrays::decisionBuffer, arrays::surface_idxBuffer, arrays::markerBuffer, arrays::distanceBuffer, arrays::changeBuffer, arrays::position, arrays::acceleration, arrays::velocity, arrays::renderIntensity, arrays::volume, arrays::lifetime, arrays::pressure, arrays::density, arrays::particleIndex, arrays::particleIndexCompact, arrays::resortArray4, arrays::resortArray, arrays::dfsphDpDt, arrays::dfsphAlpha, arrays::dfsphRhoStar, arrays::dfsphKappa, arrays::dfsphKappaDivergence, arrays::iisphSum, arrays::iisphDii, arrays::iisphAiiOld, arrays::omega, arrays::alpha, arrays::gamma, arrays::sigma, arrays::iisphDensityAdvection, arrays::iisphDensityIteration, arrays::kernelBuffer, arrays::velocityAdvection, arrays::iisphSource, arrays::iisphOmega, arrays::iisphVolume, arrays::iisphAii, arrays::iisphAcceleration, arrays::iisphVolumeError, arrays::spanNeighborList, arrays::neighborList, arrays::neighborListLength, arrays::classification, arrays::resortIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellSpan, arrays::MLMResolution, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::particleparticleIndex, arrays::support, arrays::supportEstimate, arrays::neighborCount, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::closestNeighbor, arrays::closestNeighbor_f, arrays::neighborOverhead, arrays::neighborOverheadCount> swapping_list;
