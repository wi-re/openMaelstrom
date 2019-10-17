
#include "utility/identifier/arrays.h"
#include "utility/identifier/uniform.h"
#include <cuda.h>
#include <cuda_runtime.h>
namespace arrays{

namespace adaptive{
int32_t* mergeable::ptr = nullptr;
size_t mergeable::alloc_size = 0;


void mergeable::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void mergeable::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void mergeable::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void mergeable::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
mergeable::operator type*(){ return ptr;}
int32_t& mergeable::operator[](size_t idx){ return ptr[idx];}
bool mergeable::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* mergeCounter::ptr = nullptr;
size_t mergeCounter::alloc_size = 0;


void mergeCounter::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void mergeCounter::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void mergeCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void mergeCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
mergeCounter::operator type*(){ return ptr;}
int32_t& mergeCounter::operator[](size_t idx){ return ptr[idx];}
bool mergeCounter::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* ptclCounter::ptr = nullptr;
size_t ptclCounter::alloc_size = 0;


void ptclCounter::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void ptclCounter::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void ptclCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void ptclCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
ptclCounter::operator type*(){ return ptr;}
int32_t& ptclCounter::operator[](size_t idx){ return ptr[idx];}
bool ptclCounter::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
float* classification::ptr = nullptr;
size_t classification::alloc_size = 0;


void classification::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void classification::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void classification::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void classification::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
classification::operator type*(){ return ptr;}
float& classification::operator[](size_t idx){ return ptr[idx];}
bool classification::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* splitIndicator::ptr = nullptr;
size_t splitIndicator::alloc_size = 0;

int32_t* splitIndicator::rear_ptr = nullptr;
void splitIndicator::swap() { std::swap(ptr, rear_ptr); }

void splitIndicator::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void splitIndicator::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void splitIndicator::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void splitIndicator::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
splitIndicator::operator type*(){ return ptr;}
int32_t& splitIndicator::operator[](size_t idx){ return ptr[idx];}
bool splitIndicator::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* adaptivityCounter::ptr = nullptr;
size_t adaptivityCounter::alloc_size = 0;


void adaptivityCounter::defaultAllocate(){
	auto elems = (1);
	alloc_size = 16 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void adaptivityCounter::leanAllocate(){
	auto elems = (1);
	alloc_size = 16 * elems * sizeof(int32_t);
	
}

void adaptivityCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void adaptivityCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
adaptivityCounter::operator type*(){ return ptr;}
int32_t& adaptivityCounter::operator[](size_t idx){ return ptr[idx];}
bool adaptivityCounter::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* splitIndicatorCompacted::ptr = nullptr;
size_t splitIndicatorCompacted::alloc_size = 0;


void splitIndicatorCompacted::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void splitIndicatorCompacted::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void splitIndicatorCompacted::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void splitIndicatorCompacted::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
splitIndicatorCompacted::operator type*(){ return ptr;}
int32_t& splitIndicatorCompacted::operator[](size_t idx){ return ptr[idx];}
bool splitIndicatorCompacted::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
int32_t* parentIndex::ptr = nullptr;
size_t parentIndex::alloc_size = 0;

int32_t* parentIndex::rear_ptr = nullptr;
void parentIndex::swap() { std::swap(ptr, rear_ptr); }

void parentIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void parentIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
parentIndex::operator type*(){ return ptr;}
int32_t& parentIndex::operator[](size_t idx){ return ptr[idx];}
bool parentIndex::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
float* parentVolume::ptr = nullptr;
size_t parentVolume::alloc_size = 0;

float* parentVolume::rear_ptr = nullptr;
void parentVolume::swap() { std::swap(ptr, rear_ptr); }

void parentVolume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentVolume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void parentVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
parentVolume::operator type*(){ return ptr;}
float& parentVolume::operator[](size_t idx){ return ptr[idx];}
bool parentVolume::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace adaptive{
float4* parentPosition::ptr = nullptr;
size_t parentPosition::alloc_size = 0;

float4* parentPosition::rear_ptr = nullptr;
void parentPosition::swap() { std::swap(ptr, rear_ptr); }

void parentPosition::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentPosition::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void parentPosition::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void parentPosition::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
parentPosition::operator type*(){ return ptr;}
float4& parentPosition::operator[](size_t idx){ return ptr[idx];}
bool parentPosition::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::adaptive>() == true;
	return condition;
}
}
namespace advectionArrays{
float* maxVelocity::ptr = nullptr;
size_t maxVelocity::alloc_size = 0;


void maxVelocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void maxVelocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void maxVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void maxVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
maxVelocity::operator type*(){ return ptr;}
float& maxVelocity::operator[](size_t idx){ return ptr[idx];}
bool maxVelocity::valid(){
	bool condition = true;
	return condition;
}
}
namespace advectionArrays{
float* cflValue::ptr = nullptr;
size_t cflValue::alloc_size = 0;


void cflValue::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cflValue::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void cflValue::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cflValue::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cflValue::operator type*(){ return ptr;}
float& cflValue::operator[](size_t idx){ return ptr[idx];}
bool cflValue::valid(){
	bool condition = true;
	return condition;
}
}
namespace advectionArrays{
float4* particleNormal::ptr = nullptr;
size_t particleNormal::alloc_size = 0;


void particleNormal::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleNormal::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void particleNormal::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleNormal::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
particleNormal::operator type*(){ return ptr;}
float4& particleNormal::operator[](size_t idx){ return ptr[idx];}
bool particleNormal::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::tension>() == "Akinci";
	return condition;
}
}
namespace advectionArrays{
float4* angularVelocity::ptr = nullptr;
size_t angularVelocity::alloc_size = 0;

float4* angularVelocity::rear_ptr = nullptr;
void angularVelocity::swap() { std::swap(ptr, rear_ptr); }

void angularVelocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void angularVelocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void angularVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void angularVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
angularVelocity::operator type*(){ return ptr;}
float4& angularVelocity::operator[](size_t idx){ return ptr[idx];}
bool angularVelocity::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::vorticity>() == "Bender17";
	return condition;
}
}
namespace basicArrays{
float* fluidDensity::ptr = nullptr;
size_t fluidDensity::alloc_size = 0;


void fluidDensity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void fluidDensity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void fluidDensity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void fluidDensity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
fluidDensity::operator type*(){ return ptr;}
float& fluidDensity::operator[](size_t idx){ return ptr[idx];}
bool fluidDensity::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
int32_t* particleIndex::ptr = nullptr;
size_t particleIndex::alloc_size = 0;


void particleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void particleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
particleIndex::operator type*(){ return ptr;}
int32_t& particleIndex::operator[](size_t idx){ return ptr[idx];}
bool particleIndex::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
int32_t* particleIndexCompact::ptr = nullptr;
size_t particleIndexCompact::alloc_size = 0;


void particleIndexCompact::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleIndexCompact::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void particleIndexCompact::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleIndexCompact::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
particleIndexCompact::operator type*(){ return ptr;}
int32_t& particleIndexCompact::operator[](size_t idx){ return ptr[idx];}
bool particleIndexCompact::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* resortArray4::ptr = nullptr;
size_t resortArray4::alloc_size = 0;


void resortArray4::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortArray4::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void resortArray4::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortArray4::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
resortArray4::operator type*(){ return ptr;}
float4& resortArray4::operator[](size_t idx){ return ptr[idx];}
bool resortArray4::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float* resortArray::ptr = nullptr;
size_t resortArray::alloc_size = 0;


void resortArray::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortArray::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void resortArray::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortArray::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
resortArray::operator type*(){ return ptr;}
float& resortArray::operator[](size_t idx){ return ptr[idx];}
bool resortArray::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float* volume::ptr = nullptr;
size_t volume::alloc_size = 0;

float* volume::rear_ptr = nullptr;
void volume::swap() { std::swap(ptr, rear_ptr); }

void volume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void volume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void volume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
volume::operator type*(){ return ptr;}
float& volume::operator[](size_t idx){ return ptr[idx];}
bool volume::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
Matrix4x4* matrixTest::ptr = nullptr;
size_t matrixTest::alloc_size = 0;


void matrixTest::defaultAllocate(){
	auto elems = (1);
	alloc_size = (math::max(1,(int32_t) get<parameters::rigidVolumes>().size())) * elems * sizeof(Matrix4x4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void matrixTest::leanAllocate(){
	auto elems = (1);
	alloc_size = (math::max(1,(int32_t) get<parameters::rigidVolumes>().size())) * elems * sizeof(Matrix4x4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void matrixTest::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void matrixTest::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
matrixTest::operator type*(){ return ptr;}
Matrix4x4& matrixTest::operator[](size_t idx){ return ptr[idx];}
bool matrixTest::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* position::ptr = nullptr;
size_t position::alloc_size = 0;

float4* position::rear_ptr = nullptr;
void position::swap() { std::swap(ptr, rear_ptr); }

void position::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void position::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void position::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void position::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
position::operator type*(){ return ptr;}
float4& position::operator[](size_t idx){ return ptr[idx];}
bool position::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* acceleration::ptr = nullptr;
size_t acceleration::alloc_size = 0;

float4* acceleration::rear_ptr = nullptr;
void acceleration::swap() { std::swap(ptr, rear_ptr); }

void acceleration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void acceleration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void acceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void acceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
acceleration::operator type*(){ return ptr;}
float4& acceleration::operator[](size_t idx){ return ptr[idx];}
bool acceleration::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* velocity::ptr = nullptr;
size_t velocity::alloc_size = 0;

float4* velocity::rear_ptr = nullptr;
void velocity::swap() { std::swap(ptr, rear_ptr); }

void velocity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void velocity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void velocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void velocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
velocity::operator type*(){ return ptr;}
float4& velocity::operator[](size_t idx){ return ptr[idx];}
bool velocity::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
int* particle_type::ptr = nullptr;
size_t particle_type::alloc_size = 0;

int* particle_type::rear_ptr = nullptr;
void particle_type::swap() { std::swap(ptr, rear_ptr); }

void particle_type::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void particle_type::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void particle_type::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void particle_type::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
particle_type::operator type*(){ return ptr;}
int& particle_type::operator[](size_t idx){ return ptr[idx];}
bool particle_type::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* renderArray::ptr = nullptr;
size_t renderArray::alloc_size = 0;

float4* renderArray::rear_ptr = nullptr;
void renderArray::swap() { std::swap(ptr, rear_ptr); }

void renderArray::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void renderArray::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void renderArray::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void renderArray::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
renderArray::operator type*(){ return ptr;}
float4& renderArray::operator[](size_t idx){ return ptr[idx];}
bool renderArray::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float4* debugArray::ptr = nullptr;
size_t debugArray::alloc_size = 0;

float4* debugArray::rear_ptr = nullptr;
void debugArray::swap() { std::swap(ptr, rear_ptr); }

void debugArray::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void debugArray::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void debugArray::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void debugArray::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
debugArray::operator type*(){ return ptr;}
float4& debugArray::operator[](size_t idx){ return ptr[idx];}
bool debugArray::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float* lifetime::ptr = nullptr;
size_t lifetime::alloc_size = 0;

float* lifetime::rear_ptr = nullptr;
void lifetime::swap() { std::swap(ptr, rear_ptr); }

void lifetime::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void lifetime::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void lifetime::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void lifetime::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
lifetime::operator type*(){ return ptr;}
float& lifetime::operator[](size_t idx){ return ptr[idx];}
bool lifetime::valid(){
	bool condition = true;
	return condition;
}
}
namespace basicArrays{
float* pressure::ptr = nullptr;
size_t pressure::alloc_size = 0;

float* pressure::rear_ptr = nullptr;
void pressure::swap() { std::swap(ptr, rear_ptr); }

void pressure::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void pressure::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void pressure::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void pressure::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
pressure::operator type*(){ return ptr;}
float& pressure::operator[](size_t idx){ return ptr[idx];}
bool pressure::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace basicArrays{
float* density::ptr = nullptr;
size_t density::alloc_size = 0;

float* density::rear_ptr = nullptr;
void density::swap() { std::swap(ptr, rear_ptr); }

void density::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void density::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void density::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void density::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
density::operator type*(){ return ptr;}
float& density::operator[](size_t idx){ return ptr[idx];}
bool density::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryMin::ptr = nullptr;
size_t volumeBoundaryMin::alloc_size = 0;


void volumeBoundaryMin::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryMin::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryMin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryMin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryMin::operator type*(){ return ptr;}
float4& volumeBoundaryMin::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryMin::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* boundaryPlaneVelocity::ptr = nullptr;
size_t boundaryPlaneVelocity::alloc_size = 0;



void boundaryPlaneVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void boundaryPlaneVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
boundaryPlaneVelocity::operator type*(){ return ptr;}
float4& boundaryPlaneVelocity::operator[](size_t idx){ return ptr[idx];}
bool boundaryPlaneVelocity::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
cudaTextureObject_t* volumeBoundaryVolumes::ptr = nullptr;
size_t volumeBoundaryVolumes::alloc_size = 0;


void volumeBoundaryVolumes::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(cudaTextureObject_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVolumes::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(cudaTextureObject_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryVolumes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVolumes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryVolumes::operator type*(){ return ptr;}
cudaTextureObject_t& volumeBoundaryVolumes::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryVolumes::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
int4* volumeBoundaryDimensions::ptr = nullptr;
size_t volumeBoundaryDimensions::alloc_size = 0;


void volumeBoundaryDimensions::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(int4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryDimensions::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(int4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryDimensions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryDimensions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryDimensions::operator type*(){ return ptr;}
int4& volumeBoundaryDimensions::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryDimensions::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* volumeBoundaryVolume::ptr = nullptr;
size_t volumeBoundaryVolume::alloc_size = 0;


void volumeBoundaryVolume::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVolume::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryVolume::operator type*(){ return ptr;}
float& volumeBoundaryVolume::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryVolume::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* volumeBoundaryDensity::ptr = nullptr;
size_t volumeBoundaryDensity::alloc_size = 0;


void volumeBoundaryDensity::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryDensity::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryDensity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryDensity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryDensity::operator type*(){ return ptr;}
float& volumeBoundaryDensity::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryDensity::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* splineLUT::ptr = nullptr;
size_t splineLUT::alloc_size = 0;



void splineLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void splineLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
splineLUT::operator type*(){ return ptr;}
float& splineLUT::operator[](size_t idx){ return ptr[idx];}
bool splineLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* offsetLUT::ptr = nullptr;
size_t offsetLUT::alloc_size = 0;



void offsetLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void offsetLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
offsetLUT::operator type*(){ return ptr;}
float& offsetLUT::operator[](size_t idx){ return ptr[idx];}
bool offsetLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* spikyLUT::ptr = nullptr;
size_t spikyLUT::alloc_size = 0;



void spikyLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void spikyLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
spikyLUT::operator type*(){ return ptr;}
float& spikyLUT::operator[](size_t idx){ return ptr[idx];}
bool spikyLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* adhesionLUT::ptr = nullptr;
size_t adhesionLUT::alloc_size = 0;



void adhesionLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void adhesionLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
adhesionLUT::operator type*(){ return ptr;}
float& adhesionLUT::operator[](size_t idx){ return ptr[idx];}
bool adhesionLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* volumeLUT::ptr = nullptr;
size_t volumeLUT::alloc_size = 0;



void volumeLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeLUT::operator type*(){ return ptr;}
float& volumeLUT::operator[](size_t idx){ return ptr[idx];}
bool volumeLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* spline2LUT::ptr = nullptr;
size_t spline2LUT::alloc_size = 0;



void spline2LUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void spline2LUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
spline2LUT::operator type*(){ return ptr;}
float& spline2LUT::operator[](size_t idx){ return ptr[idx];}
bool spline2LUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* splineGradientLUT::ptr = nullptr;
size_t splineGradientLUT::alloc_size = 0;



void splineGradientLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void splineGradientLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
splineGradientLUT::operator type*(){ return ptr;}
float& splineGradientLUT::operator[](size_t idx){ return ptr[idx];}
bool splineGradientLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* spikyGradientLUT::ptr = nullptr;
size_t spikyGradientLUT::alloc_size = 0;



void spikyGradientLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void spikyGradientLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
spikyGradientLUT::operator type*(){ return ptr;}
float& spikyGradientLUT::operator[](size_t idx){ return ptr[idx];}
bool spikyGradientLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float* cohesionLUT::ptr = nullptr;
size_t cohesionLUT::alloc_size = 0;



void cohesionLUT::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cohesionLUT::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cohesionLUT::operator type*(){ return ptr;}
float& cohesionLUT::operator[](size_t idx){ return ptr[idx];}
bool cohesionLUT::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* boundaryPlanes::ptr = nullptr;
size_t boundaryPlanes::alloc_size = 0;



void boundaryPlanes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void boundaryPlanes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
boundaryPlanes::operator type*(){ return ptr;}
float4& boundaryPlanes::operator[](size_t idx){ return ptr[idx];}
bool boundaryPlanes::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryVelocity::ptr = nullptr;
size_t volumeBoundaryVelocity::alloc_size = 0;


void volumeBoundaryVelocity::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVelocity::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryVelocity::operator type*(){ return ptr;}
float4& volumeBoundaryVelocity::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryVelocity::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryAngularVelocity::ptr = nullptr;
size_t volumeBoundaryAngularVelocity::alloc_size = 0;


void volumeBoundaryAngularVelocity::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAngularVelocity::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryAngularVelocity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAngularVelocity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryAngularVelocity::operator type*(){ return ptr;}
float4& volumeBoundaryAngularVelocity::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryAngularVelocity::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
int32_t* volumeBoundaryKind::ptr = nullptr;
size_t volumeBoundaryKind::alloc_size = 0;


void volumeBoundaryKind::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryKind::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryKind::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryKind::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryKind::operator type*(){ return ptr;}
int32_t& volumeBoundaryKind::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryKind::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryPosition::ptr = nullptr;
size_t volumeBoundaryPosition::alloc_size = 0;


void volumeBoundaryPosition::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryPosition::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryPosition::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryPosition::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryPosition::operator type*(){ return ptr;}
float4& volumeBoundaryPosition::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryPosition::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryQuaternion::ptr = nullptr;
size_t volumeBoundaryQuaternion::alloc_size = 0;


void volumeBoundaryQuaternion::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryQuaternion::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryQuaternion::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryQuaternion::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryQuaternion::operator type*(){ return ptr;}
float4& volumeBoundaryQuaternion::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryQuaternion::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
Matrix4x4* volumeBoundaryTransformMatrix::ptr = nullptr;
size_t volumeBoundaryTransformMatrix::alloc_size = 0;


void volumeBoundaryTransformMatrix::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryTransformMatrix::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryTransformMatrix::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryTransformMatrix::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryTransformMatrix::operator type*(){ return ptr;}
Matrix4x4& volumeBoundaryTransformMatrix::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryTransformMatrix::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryMax::ptr = nullptr;
size_t volumeBoundaryMax::alloc_size = 0;


void volumeBoundaryMax::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryMax::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryMax::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryMax::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryMax::operator type*(){ return ptr;}
float4& volumeBoundaryMax::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryMax::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
Matrix4x4* volumeBoundaryTransformMatrixInverse::ptr = nullptr;
size_t volumeBoundaryTransformMatrixInverse::alloc_size = 0;


void volumeBoundaryTransformMatrixInverse::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryTransformMatrixInverse::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryTransformMatrixInverse::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryTransformMatrixInverse::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryTransformMatrixInverse::operator type*(){ return ptr;}
Matrix4x4& volumeBoundaryTransformMatrixInverse::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryTransformMatrixInverse::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
Matrix4x4* volumeBoundaryInertiaMatrix::ptr = nullptr;
size_t volumeBoundaryInertiaMatrix::alloc_size = 0;


void volumeBoundaryInertiaMatrix::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryInertiaMatrix::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryInertiaMatrix::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryInertiaMatrix::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryInertiaMatrix::operator type*(){ return ptr;}
Matrix4x4& volumeBoundaryInertiaMatrix::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryInertiaMatrix::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
Matrix4x4* volumeBoundaryInertiaMatrixInverse::ptr = nullptr;
size_t volumeBoundaryInertiaMatrixInverse::alloc_size = 0;


void volumeBoundaryInertiaMatrixInverse::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryInertiaMatrixInverse::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(Matrix4x4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryInertiaMatrixInverse::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryInertiaMatrixInverse::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryInertiaMatrixInverse::operator type*(){ return ptr;}
Matrix4x4& volumeBoundaryInertiaMatrixInverse::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryInertiaMatrixInverse::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryAcceleration::ptr = nullptr;
size_t volumeBoundaryAcceleration::alloc_size = 0;


void volumeBoundaryAcceleration::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAcceleration::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryAcceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAcceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryAcceleration::operator type*(){ return ptr;}
float4& volumeBoundaryAcceleration::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryAcceleration::valid(){
	bool condition = true;
	return condition;
}
}
namespace boundaryArrays{
float4* volumeBoundaryAngularAcceleration::ptr = nullptr;
size_t volumeBoundaryAngularAcceleration::alloc_size = 0;


void volumeBoundaryAngularAcceleration::defaultAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAngularAcceleration::leanAllocate(){
	auto elems = (1);
	alloc_size = math::max(1u,(uint32_t)get<parameters::boundaryVolumes>().size()) * elems * sizeof(float4);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void volumeBoundaryAngularAcceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeBoundaryAngularAcceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeBoundaryAngularAcceleration::operator type*(){ return ptr;}
float4& volumeBoundaryAngularAcceleration::operator[](size_t idx){ return ptr[idx];}
bool volumeBoundaryAngularAcceleration::valid(){
	bool condition = true;
	return condition;
}
}
namespace dfsphArrays{
float* dfsphAlpha::ptr = nullptr;
size_t dfsphAlpha::alloc_size = 0;


void dfsphAlpha::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphAlpha::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void dfsphAlpha::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphAlpha::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
dfsphAlpha::operator type*(){ return ptr;}
float& dfsphAlpha::operator[](size_t idx){ return ptr[idx];}
bool dfsphAlpha::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace dfsphArrays{
float* dfsphDpDt::ptr = nullptr;
size_t dfsphDpDt::alloc_size = 0;


void dfsphDpDt::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphDpDt::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void dfsphDpDt::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphDpDt::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
dfsphDpDt::operator type*(){ return ptr;}
float& dfsphDpDt::operator[](size_t idx){ return ptr[idx];}
bool dfsphDpDt::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace dfsphArrays{
float* dfsphRhoStar::ptr = nullptr;
size_t dfsphRhoStar::alloc_size = 0;


void dfsphRhoStar::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphRhoStar::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void dfsphRhoStar::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphRhoStar::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
dfsphRhoStar::operator type*(){ return ptr;}
float& dfsphRhoStar::operator[](size_t idx){ return ptr[idx];}
bool dfsphRhoStar::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace dfsphArrays{
float* dfsphSource::ptr = nullptr;
size_t dfsphSource::alloc_size = 0;


void dfsphSource::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphSource::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void dfsphSource::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void dfsphSource::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
dfsphSource::operator type*(){ return ptr;}
float& dfsphSource::operator[](size_t idx){ return ptr[idx];}
bool dfsphSource::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace iisph17{
float* omega::ptr = nullptr;
size_t omega::alloc_size = 0;


void omega::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void omega::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void omega::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void omega::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
omega::operator type*(){ return ptr;}
float& omega::operator[](size_t idx){ return ptr[idx];}
bool omega::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17{
float* boundaryPressure::ptr = nullptr;
size_t boundaryPressure::alloc_size = 0;


void boundaryPressure::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void boundaryPressure::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void boundaryPressure::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void boundaryPressure::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
boundaryPressure::operator type*(){ return ptr;}
float& boundaryPressure::operator[](size_t idx){ return ptr[idx];}
bool boundaryPressure::valid(){
	bool condition = true;
	return condition;
}
}
namespace iisph17{
float* volumeError::ptr = nullptr;
size_t volumeError::alloc_size = 0;


void volumeError::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeError::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void volumeError::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeError::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeError::operator type*(){ return ptr;}
float& volumeError::operator[](size_t idx){ return ptr[idx];}
bool volumeError::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17{
float4* predictedAcceleration::ptr = nullptr;
size_t predictedAcceleration::alloc_size = 0;


void predictedAcceleration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void predictedAcceleration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void predictedAcceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void predictedAcceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
predictedAcceleration::operator type*(){ return ptr;}
float4& predictedAcceleration::operator[](size_t idx){ return ptr[idx];}
bool predictedAcceleration::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace iisph17{
float* Aii::ptr = nullptr;
size_t Aii::alloc_size = 0;


void Aii::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void Aii::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void Aii::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void Aii::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
Aii::operator type*(){ return ptr;}
float& Aii::operator[](size_t idx){ return ptr[idx];}
bool Aii::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17{
float* apparentVolume::ptr = nullptr;
size_t apparentVolume::alloc_size = 0;


void apparentVolume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void apparentVolume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void apparentVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void apparentVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
apparentVolume::operator type*(){ return ptr;}
float& apparentVolume::operator[](size_t idx){ return ptr[idx];}
bool apparentVolume::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "DFSPH";
	return condition;
}
}
namespace iisph17{
float* sourceTerm::ptr = nullptr;
size_t sourceTerm::alloc_size = 0;


void sourceTerm::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void sourceTerm::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void sourceTerm::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void sourceTerm::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
sourceTerm::operator type*(){ return ptr;}
float& sourceTerm::operator[](size_t idx){ return ptr[idx];}
bool sourceTerm::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float* _apparentVolume::ptr = nullptr;
size_t _apparentVolume::alloc_size = 0;


void _apparentVolume::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _apparentVolume::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void _apparentVolume::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _apparentVolume::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_apparentVolume::operator type*(){ return ptr;}
float& _apparentVolume::operator[](size_t idx){ return ptr[idx];}
bool _apparentVolume::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float* _Aii::ptr = nullptr;
size_t _Aii::alloc_size = 0;


void _Aii::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _Aii::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void _Aii::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _Aii::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_Aii::operator type*(){ return ptr;}
float& _Aii::operator[](size_t idx){ return ptr[idx];}
bool _Aii::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float4* _predictedAcceleration::ptr = nullptr;
size_t _predictedAcceleration::alloc_size = 0;


void _predictedAcceleration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _predictedAcceleration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void _predictedAcceleration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _predictedAcceleration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_predictedAcceleration::operator type*(){ return ptr;}
float4& _predictedAcceleration::operator[](size_t idx){ return ptr[idx];}
bool _predictedAcceleration::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float* _volumeError::ptr = nullptr;
size_t _volumeError::alloc_size = 0;


void _volumeError::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _volumeError::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void _volumeError::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _volumeError::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_volumeError::operator type*(){ return ptr;}
float& _volumeError::operator[](size_t idx){ return ptr[idx];}
bool _volumeError::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float* _sourceTerm::ptr = nullptr;
size_t _sourceTerm::alloc_size = 0;


void _sourceTerm::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _sourceTerm::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void _sourceTerm::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _sourceTerm::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_sourceTerm::operator type*(){ return ptr;}
float& _sourceTerm::operator[](size_t idx){ return ptr[idx];}
bool _sourceTerm::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisph17_band_rb{
float* _omega::ptr = nullptr;
size_t _omega::alloc_size = 0;


void _omega::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void _omega::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void _omega::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void _omega::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
_omega::operator type*(){ return ptr;}
float& _omega::operator[](size_t idx){ return ptr[idx];}
bool _omega::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH17_BAND_RB";
	condition = condition || get<parameters::modules::pressure>() == "densityMapIISPH";
	return condition;
}
}
namespace iisphArrays{
float* iisphAiiOld::ptr = nullptr;
size_t iisphAiiOld::alloc_size = 0;


void iisphAiiOld::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphAiiOld::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void iisphAiiOld::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphAiiOld::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
iisphAiiOld::operator type*(){ return ptr;}
float& iisphAiiOld::operator[](size_t idx){ return ptr[idx];}
bool iisphAiiOld::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* sigma::ptr = nullptr;
size_t sigma::alloc_size = 0;


void sigma::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void sigma::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void sigma::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void sigma::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
sigma::operator type*(){ return ptr;}
float& sigma::operator[](size_t idx){ return ptr[idx];}
bool sigma::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* iisphDensityAdvection::ptr = nullptr;
size_t iisphDensityAdvection::alloc_size = 0;


void iisphDensityAdvection::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDensityAdvection::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void iisphDensityAdvection::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDensityAdvection::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
iisphDensityAdvection::operator type*(){ return ptr;}
float& iisphDensityAdvection::operator[](size_t idx){ return ptr[idx];}
bool iisphDensityAdvection::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* iisphDensityIteration::ptr = nullptr;
size_t iisphDensityIteration::alloc_size = 0;


void iisphDensityIteration::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDensityIteration::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void iisphDensityIteration::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDensityIteration::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
iisphDensityIteration::operator type*(){ return ptr;}
float& iisphDensityIteration::operator[](size_t idx){ return ptr[idx];}
bool iisphDensityIteration::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float4* kernelBuffer::ptr = nullptr;
size_t kernelBuffer::alloc_size = 0;


void kernelBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void kernelBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void kernelBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void kernelBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
kernelBuffer::operator type*(){ return ptr;}
float4& kernelBuffer::operator[](size_t idx){ return ptr[idx];}
bool kernelBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float4* velocityAdvection::ptr = nullptr;
size_t velocityAdvection::alloc_size = 0;


void velocityAdvection::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void velocityAdvection::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void velocityAdvection::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void velocityAdvection::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
velocityAdvection::operator type*(){ return ptr;}
float4& velocityAdvection::operator[](size_t idx){ return ptr[idx];}
bool velocityAdvection::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float4* iisphSum::ptr = nullptr;
size_t iisphSum::alloc_size = 0;


void iisphSum::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphSum::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void iisphSum::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphSum::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
iisphSum::operator type*(){ return ptr;}
float4& iisphSum::operator[](size_t idx){ return ptr[idx];}
bool iisphSum::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float4* iisphDii::ptr = nullptr;
size_t iisphDii::alloc_size = 0;


void iisphDii::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDii::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void iisphDii::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void iisphDii::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
iisphDii::operator type*(){ return ptr;}
float4& iisphDii::operator[](size_t idx){ return ptr[idx];}
bool iisphDii::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* omega::ptr = nullptr;
size_t omega::alloc_size = 0;


void omega::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void omega::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void omega::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void omega::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
omega::operator type*(){ return ptr;}
float& omega::operator[](size_t idx){ return ptr[idx];}
bool omega::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* alpha::ptr = nullptr;
size_t alpha::alloc_size = 0;


void alpha::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void alpha::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void alpha::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void alpha::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
alpha::operator type*(){ return ptr;}
float& alpha::operator[](size_t idx){ return ptr[idx];}
bool alpha::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace iisphArrays{
float* gamma::ptr = nullptr;
size_t gamma::alloc_size = 0;


void gamma::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void gamma::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void gamma::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void gamma::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
gamma::operator type*(){ return ptr;}
float& gamma::operator[](size_t idx){ return ptr[idx];}
bool gamma::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::pressure>() == "IISPH";
	return condition;
}
}
namespace renderArrays{
float* auxIsoDensity::ptr = nullptr;
size_t auxIsoDensity::alloc_size = 0;


void auxIsoDensity::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxIsoDensity::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void auxIsoDensity::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxIsoDensity::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxIsoDensity::operator type*(){ return ptr;}
float& auxIsoDensity::operator[](size_t idx){ return ptr[idx];}
bool auxIsoDensity::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::rayTracing>() == true;
	return condition;
}
}
namespace renderArrays{
float* anisotropicMatrices::ptr = nullptr;
size_t anisotropicMatrices::alloc_size = 0;


void anisotropicMatrices::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 9 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void anisotropicMatrices::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 9 * elems * sizeof(float);
	
}

void anisotropicMatrices::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void anisotropicMatrices::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
anisotropicMatrices::operator type*(){ return ptr;}
float& anisotropicMatrices::operator[](size_t idx){ return ptr[idx];}
bool anisotropicMatrices::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::rayTracing>() == true;
	condition = condition && get<parameters::modules::anisotropicSurface>() == true;
	return condition;
}
}
namespace renderArrays{
float* auxTest::ptr = nullptr;
size_t auxTest::alloc_size = 0;


void auxTest::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxTest::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void auxTest::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxTest::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxTest::operator type*(){ return ptr;}
float& auxTest::operator[](size_t idx){ return ptr[idx];}
bool auxTest::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::rayTracing>() == true;
	return condition;
}
}
namespace renderArrays{
float4* centerPosition::ptr = nullptr;
size_t centerPosition::alloc_size = 0;


void centerPosition::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void centerPosition::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float4);
	
}

void centerPosition::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void centerPosition::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
centerPosition::operator type*(){ return ptr;}
float4& centerPosition::operator[](size_t idx){ return ptr[idx];}
bool centerPosition::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::rayTracing>() == true;
	condition = condition && get<parameters::modules::anisotropicSurface>() == true;
	return condition;
}
}
namespace renderArrays{
cellSurface* auxCellSurface::ptr = nullptr;
size_t auxCellSurface::alloc_size = 0;


void auxCellSurface::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cellSurface);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellSurface::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cellSurface);
	
}

void auxCellSurface::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellSurface::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxCellSurface::operator type*(){ return ptr;}
cellSurface& auxCellSurface::operator[](size_t idx){ return ptr[idx];}
bool auxCellSurface::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::rayTracing>() == true;
	condition = condition && get<parameters::modules::anisotropicSurface>() == true;
	return condition;
}
}
namespace renderArrays{
cellInformation* auxCellInformation::ptr = nullptr;
size_t auxCellInformation::alloc_size = 0;


void auxCellInformation::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cellInformation);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellInformation::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cellInformation);
	
}

void auxCellInformation::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellInformation::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxCellInformation::operator type*(){ return ptr;}
cellInformation& auxCellInformation::operator[](size_t idx){ return ptr[idx];}
bool auxCellInformation::valid(){
	bool condition = true;
	condition = condition && get<parameters::modules::rayTracing>() == true;
	condition = condition && get<parameters::modules::anisotropicSurface>() == true;
	return condition;
}
}
namespace renderArrays{
compactListEntry* auxCellSpan::ptr = nullptr;
size_t auxCellSpan::alloc_size = 0;


void auxCellSpan::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compactListEntry);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellSpan::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compactListEntry);
	
}

void auxCellSpan::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxCellSpan::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxCellSpan::operator type*(){ return ptr;}
compactListEntry& auxCellSpan::operator[](size_t idx){ return ptr[idx];}
bool auxCellSpan::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::rayTracing>() == true;
	return condition;
}
}
namespace renderArrays{
compactListEntry* auxHashMap::ptr = nullptr;
size_t auxHashMap::alloc_size = 0;


void auxHashMap::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compactListEntry);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxHashMap::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compactListEntry);
	
}

void auxHashMap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxHashMap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxHashMap::operator type*(){ return ptr;}
compactListEntry& auxHashMap::operator[](size_t idx){ return ptr[idx];}
bool auxHashMap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::rayTracing>() == true;
	return condition;
}
}
namespace renderArrays{
float* auxDistance::ptr = nullptr;
size_t auxDistance::alloc_size = 0;


void auxDistance::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxDistance::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void auxDistance::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void auxDistance::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
auxDistance::operator type*(){ return ptr;}
float& auxDistance::operator[](size_t idx){ return ptr[idx];}
bool auxDistance::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::rayTracing>() == true;
	return condition;
}
}
namespace rigidBodyArrays{
float* rigidVolumes::ptr = nullptr;
size_t rigidVolumes::alloc_size = 0;


void rigidVolumes::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidVolumes::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float);
	
}

void rigidVolumes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidVolumes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidVolumes::operator type*(){ return ptr;}
float& rigidVolumes::operator[](size_t idx){ return ptr[idx];}
bool rigidVolumes::valid(){
	bool condition = true;
	return condition;
}
}
namespace rigidBodyArrays{
float4* rigidLinearVelocities::ptr = nullptr;
size_t rigidLinearVelocities::alloc_size = 0;


void rigidLinearVelocities::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidLinearVelocities::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float4);
	
}

void rigidLinearVelocities::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidLinearVelocities::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidLinearVelocities::operator type*(){ return ptr;}
float4& rigidLinearVelocities::operator[](size_t idx){ return ptr[idx];}
bool rigidLinearVelocities::valid(){
	bool condition = true;
	return condition;
}
}
namespace rigidBodyArrays{
float3* rigidAVelocities::ptr = nullptr;
size_t rigidAVelocities::alloc_size = 0;


void rigidAVelocities::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float3);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidAVelocities::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float3);
	
}

void rigidAVelocities::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidAVelocities::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidAVelocities::operator type*(){ return ptr;}
float3& rigidAVelocities::operator[](size_t idx){ return ptr[idx];}
bool rigidAVelocities::valid(){
	bool condition = true;
	return condition;
}
}
namespace rigidBodyArrays{
float* rigidDensities::ptr = nullptr;
size_t rigidDensities::alloc_size = 0;


void rigidDensities::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidDensities::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float);
	
}

void rigidDensities::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidDensities::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidDensities::operator type*(){ return ptr;}
float& rigidDensities::operator[](size_t idx){ return ptr[idx];}
bool rigidDensities::valid(){
	bool condition = true;
	return condition;
}
}
namespace rigidBodyArrays{
float3* rigidOrigins::ptr = nullptr;
size_t rigidOrigins::alloc_size = 0;


void rigidOrigins::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float3);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidOrigins::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float3);
	
}

void rigidOrigins::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidOrigins::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidOrigins::operator type*(){ return ptr;}
float3& rigidOrigins::operator[](size_t idx){ return ptr[idx];}
bool rigidOrigins::valid(){
	bool condition = true;
	return condition;
}
}
namespace rigidBodyArrays{
float4* rigidQuaternions::ptr = nullptr;
size_t rigidQuaternions::alloc_size = 0;


void rigidQuaternions::defaultAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float4);
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidQuaternions::leanAllocate(){
	auto elems = 1;
	alloc_size = 1 * elems * sizeof(float4);
	
}

void rigidQuaternions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void rigidQuaternions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
rigidQuaternions::operator type*(){ return ptr;}
float4& rigidQuaternions::operator[](size_t idx){ return ptr[idx];}
bool rigidQuaternions::valid(){
	bool condition = true;
	return condition;
}
}
namespace structureArrays{
int32_t* mlmScaling::ptr = nullptr;
size_t mlmScaling::alloc_size = 0;


void mlmScaling::defaultAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void mlmScaling::leanAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void mlmScaling::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void mlmScaling::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
mlmScaling::operator type*(){ return ptr;}
int32_t& mlmScaling::operator[](size_t idx){ return ptr[idx];}
bool mlmScaling::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
int32_t* particleparticleIndex::ptr = nullptr;
size_t particleparticleIndex::alloc_size = 0;


void particleparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void particleparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void particleparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
particleparticleIndex::operator type*(){ return ptr;}
int32_t& particleparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool particleparticleIndex::valid(){
	bool condition = true;
	return condition;
}
}
namespace structureArrays{
int32_t* cellBegin::ptr = nullptr;
size_t cellBegin::alloc_size = 0;


void cellBegin::defaultAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellBegin::leanAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void cellBegin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellBegin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cellBegin::operator type*(){ return ptr;}
int32_t& cellBegin::operator[](size_t idx){ return ptr[idx];}
bool cellBegin::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "linear_cell";
	return condition;
}
}
namespace structureArrays{
int32_t* cellEnd::ptr = nullptr;
size_t cellEnd::alloc_size = 0;


void cellEnd::defaultAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellEnd::leanAllocate(){
	auto elems = get<parameters::grid_size>().x * get<parameters::grid_size>().y * get<parameters::grid_size>().z;
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void cellEnd::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellEnd::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cellEnd::operator type*(){ return ptr;}
int32_t& cellEnd::operator[](size_t idx){ return ptr[idx];}
bool cellEnd::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "linear_cell";
	return condition;
}
}
namespace structureArrays{
compactSpan* compactCellList::ptr = nullptr;
size_t compactCellList::alloc_size = 0;


void compactCellList::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(compactSpan);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellList::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(compactSpan);
	
}

void compactCellList::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellList::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactCellList::operator type*(){ return ptr;}
compactSpan& compactCellList::operator[](size_t idx){ return ptr[idx];}
bool compactCellList::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "compactCell";
	return condition;
}
}
namespace structureArrays{
compactCellNeighbors* neighborMask::ptr = nullptr;
size_t neighborMask::alloc_size = 0;


void neighborMask::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(compactCellNeighbors);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborMask::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(compactCellNeighbors);
	
}

void neighborMask::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborMask::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborMask::operator type*(){ return ptr;}
compactCellNeighbors& neighborMask::operator[](size_t idx){ return ptr[idx];}
bool neighborMask::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "masked";
	return condition;
}
}
namespace structureArrays{
int8_t* compactCellScale::ptr = nullptr;
size_t compactCellScale::alloc_size = 0;


void compactCellScale::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int8_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellScale::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int8_t);
	
}

void compactCellScale::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellScale::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactCellScale::operator type*(){ return ptr;}
int8_t& compactCellScale::operator[](size_t idx){ return ptr[idx];}
bool compactCellScale::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "compactCell";
	return condition;
}
}
namespace structureArrays{
neigh_span* spanNeighborList::ptr = nullptr;
size_t spanNeighborList::alloc_size = 0;


void spanNeighborList::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(neigh_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void spanNeighborList::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 28 * elems * sizeof(neigh_span);
	
}

void spanNeighborList::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void spanNeighborList::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
spanNeighborList::operator type*(){ return ptr;}
neigh_span& spanNeighborList::operator[](size_t idx){ return ptr[idx];}
bool spanNeighborList::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "cell_based";
	return condition;
}
}
namespace structureArrays{
int32_t* neighborList::ptr = nullptr;
size_t neighborList::alloc_size = 0;


void neighborList::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::neighborlimit>() * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborList::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::neighborlimit>() * elems * sizeof(int32_t);
	
}

void neighborList::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborList::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborList::operator type*(){ return ptr;}
int32_t& neighborList::operator[](size_t idx){ return ptr[idx];}
bool neighborList::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	condition = condition || get<parameters::modules::neighborhood>() == "basic";
	return condition;
}
}
namespace structureArrays{
int32_t* neighborListLength::ptr = nullptr;
size_t neighborListLength::alloc_size = 0;

int32_t* neighborListLength::rear_ptr = nullptr;
void neighborListLength::swap() { std::swap(ptr, rear_ptr); }

void neighborListLength::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void neighborListLength::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void neighborListLength::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void neighborListLength::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
neighborListLength::operator type*(){ return ptr;}
int32_t& neighborListLength::operator[](size_t idx){ return ptr[idx];}
bool neighborListLength::valid(){
	bool condition = true;
	return condition;
}
}
namespace structureArrays{
int32_t* classification::ptr = nullptr;
size_t classification::alloc_size = 0;


void classification::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void classification::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void classification::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void classification::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
classification::operator type*(){ return ptr;}
int32_t& classification::operator[](size_t idx){ return ptr[idx];}
bool classification::valid(){
	bool condition = true;
	return condition;
}
}
namespace structureArrays{
int32_t* resortIndex::ptr = nullptr;
size_t resortIndex::alloc_size = 0;


void resortIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void resortIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void resortIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
resortIndex::operator type*(){ return ptr;}
int32_t& resortIndex::operator[](size_t idx){ return ptr[idx];}
bool resortIndex::valid(){
	bool condition = true;
	return condition;
}
}
namespace structureArrays{
int64_t* ZOrder_64::ptr = nullptr;
size_t ZOrder_64::alloc_size = 0;


void ZOrder_64::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int64_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void ZOrder_64::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int64_t);
	
}

void ZOrder_64::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void ZOrder_64::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
ZOrder_64::operator type*(){ return ptr;}
int64_t& ZOrder_64::operator[](size_t idx){ return ptr[idx];}
bool ZOrder_64::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
int32_t* ZOrder_32::ptr = nullptr;
size_t ZOrder_32::alloc_size = 0;


void ZOrder_32::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void ZOrder_32::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void ZOrder_32::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void ZOrder_32::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
ZOrder_32::operator type*(){ return ptr;}
int32_t& ZOrder_32::operator[](size_t idx){ return ptr[idx];}
bool ZOrder_32::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
cell_span* cellSpanSwap::ptr = nullptr;
size_t cellSpanSwap::alloc_size = 0;


void cellSpanSwap::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cell_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellSpanSwap::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(cell_span);
	
}

void cellSpanSwap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellSpanSwap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cellSpanSwap::operator type*(){ return ptr;}
cell_span& cellSpanSwap::operator[](size_t idx){ return ptr[idx];}
bool cellSpanSwap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	return condition;
}
}
namespace structureArrays{
cell_span* cellSpan::ptr = nullptr;
size_t cellSpan::alloc_size = 0;


void cellSpan::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(cell_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellSpan::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(cell_span);
	
}

void cellSpan::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellSpan::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cellSpan::operator type*(){ return ptr;}
cell_span& cellSpan::operator[](size_t idx){ return ptr[idx];}
bool cellSpan::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	return condition;
}
}
namespace structureArrays{
compact_cellSpan* compactCellSpanSwap::ptr = nullptr;
size_t compactCellSpanSwap::alloc_size = 0;


void compactCellSpanSwap::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compact_cellSpan);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellSpanSwap::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(compact_cellSpan);
	
}

void compactCellSpanSwap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellSpanSwap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactCellSpanSwap::operator type*(){ return ptr;}
compact_cellSpan& compactCellSpanSwap::operator[](size_t idx){ return ptr[idx];}
bool compactCellSpanSwap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
compact_cellSpan* compactCellSpan::ptr = nullptr;
size_t compactCellSpan::alloc_size = 0;


void compactCellSpan::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(compact_cellSpan);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellSpan::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = get<parameters::mlm_schemes>() * elems * sizeof(compact_cellSpan);
	
}

void compactCellSpan::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactCellSpan::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactCellSpan::operator type*(){ return ptr;}
compact_cellSpan& compactCellSpan::operator[](size_t idx){ return ptr[idx];}
bool compactCellSpan::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
int32_t* MLMResolution::ptr = nullptr;
size_t MLMResolution::alloc_size = 0;


void MLMResolution::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void MLMResolution::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void MLMResolution::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void MLMResolution::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
MLMResolution::operator type*(){ return ptr;}
int32_t& MLMResolution::operator[](size_t idx){ return ptr[idx];}
bool MLMResolution::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
int32_t* cellparticleIndex::ptr = nullptr;
size_t cellparticleIndex::alloc_size = 0;


void cellparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void cellparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void cellparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
cellparticleIndex::operator type*(){ return ptr;}
int32_t& cellparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool cellparticleIndex::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
int32_t* compactparticleIndex::ptr = nullptr;
size_t compactparticleIndex::alloc_size = 0;


void compactparticleIndex::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactparticleIndex::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void compactparticleIndex::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactparticleIndex::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactparticleIndex::operator type*(){ return ptr;}
int32_t& compactparticleIndex::operator[](size_t idx){ return ptr[idx];}
bool compactparticleIndex::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace structureArrays{
hash_span* hashMap::ptr = nullptr;
size_t hashMap::alloc_size = 0;


void hashMap::defaultAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(hash_span);
	cudaAllocateMemory(&ptr, alloc_size);
}
void hashMap::leanAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(hash_span);
	
}

void hashMap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void hashMap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
hashMap::operator type*(){ return ptr;}
hash_span& hashMap::operator[](size_t idx){ return ptr[idx];}
bool hashMap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "MLM";
	condition = condition || get<parameters::modules::sorting>() == "hashed_cell";
	return condition;
}
}
namespace structureArrays{
compactHashSpan* compactHashMap::ptr = nullptr;
size_t compactHashMap::alloc_size = 0;


void compactHashMap::defaultAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(compactHashSpan);
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactHashMap::leanAllocate(){
	auto elems = (1);
	alloc_size = get<parameters::mlm_schemes>() * get<parameters::hash_entries>() * elems * sizeof(compactHashSpan);
	
}

void compactHashMap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void compactHashMap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
compactHashMap::operator type*(){ return ptr;}
compactHashSpan& compactHashMap::operator[](size_t idx){ return ptr[idx];}
bool compactHashMap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::sorting>() == "compactMLM";
	return condition;
}
}
namespace supportArrays{
float* support::ptr = nullptr;
size_t support::alloc_size = 0;


void support::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void support::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void support::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void support::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
support::operator type*(){ return ptr;}
float& support::operator[](size_t idx){ return ptr[idx];}
bool support::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
float* supportEstimate::ptr = nullptr;
size_t supportEstimate::alloc_size = 0;


void supportEstimate::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportEstimate::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void supportEstimate::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportEstimate::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
supportEstimate::operator type*(){ return ptr;}
float& supportEstimate::operator[](size_t idx){ return ptr[idx];}
bool supportEstimate::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* neighborCount::ptr = nullptr;
size_t neighborCount::alloc_size = 0;


void neighborCount::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborCount::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void neighborCount::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborCount::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborCount::operator type*(){ return ptr;}
int32_t& neighborCount::operator[](size_t idx){ return ptr[idx];}
bool neighborCount::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* supportMarker::ptr = nullptr;
size_t supportMarker::alloc_size = 0;


void supportMarker::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportMarker::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void supportMarker::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportMarker::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
supportMarker::operator type*(){ return ptr;}
int32_t& supportMarker::operator[](size_t idx){ return ptr[idx];}
bool supportMarker::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* supportMarkerCompacted::ptr = nullptr;
size_t supportMarkerCompacted::alloc_size = 0;


void supportMarkerCompacted::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportMarkerCompacted::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void supportMarkerCompacted::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void supportMarkerCompacted::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
supportMarkerCompacted::operator type*(){ return ptr;}
int32_t& supportMarkerCompacted::operator[](size_t idx){ return ptr[idx];}
bool supportMarkerCompacted::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* closestNeighbor::ptr = nullptr;
size_t closestNeighbor::alloc_size = 0;


void closestNeighbor::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void closestNeighbor::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void closestNeighbor::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void closestNeighbor::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
closestNeighbor::operator type*(){ return ptr;}
int32_t& closestNeighbor::operator[](size_t idx){ return ptr[idx];}
bool closestNeighbor::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* closestNeighbor_f::ptr = nullptr;
size_t closestNeighbor_f::alloc_size = 0;


void closestNeighbor_f::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void closestNeighbor_f::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void closestNeighbor_f::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void closestNeighbor_f::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
closestNeighbor_f::operator type*(){ return ptr;}
int32_t& closestNeighbor_f::operator[](size_t idx){ return ptr[idx];}
bool closestNeighbor_f::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* neighborOverhead::ptr = nullptr;
size_t neighborOverhead::alloc_size = 0;


void neighborOverhead::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborOverhead::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void neighborOverhead::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborOverhead::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborOverhead::operator type*(){ return ptr;}
int32_t& neighborOverhead::operator[](size_t idx){ return ptr[idx];}
bool neighborOverhead::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* neighborOverheadCount::ptr = nullptr;
size_t neighborOverheadCount::alloc_size = 0;


void neighborOverheadCount::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborOverheadCount::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void neighborOverheadCount::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborOverheadCount::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborOverheadCount::operator type*(){ return ptr;}
int32_t& neighborOverheadCount::operator[](size_t idx){ return ptr[idx];}
bool neighborOverheadCount::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace supportArrays{
int32_t* neighborListSwap::ptr = nullptr;
size_t neighborListSwap::alloc_size = 0;


void neighborListSwap::defaultAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborListSwap::leanAllocate(){
	auto elems = (1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
}

void neighborListSwap::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void neighborListSwap::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
neighborListSwap::operator type*(){ return ptr;}
int32_t& neighborListSwap::operator[](size_t idx){ return ptr[idx];}
bool neighborListSwap::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::neighborhood>() == "constrained";
	return condition;
}
}
namespace surfaceArrays{
float* distanceBuffer::ptr = nullptr;
size_t distanceBuffer::alloc_size = 0;

float* distanceBuffer::rear_ptr = nullptr;
void distanceBuffer::swap() { std::swap(ptr, rear_ptr); }

void distanceBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void distanceBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void distanceBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void distanceBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
distanceBuffer::operator type*(){ return ptr;}
float& distanceBuffer::operator[](size_t idx){ return ptr[idx];}
bool distanceBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::surfaceDistance>() == true;
	condition = condition || get<parameters::modules::surfaceDetection>() == true;
	return condition;
}
}
namespace surfaceArrays{
float* markerBuffer::ptr = nullptr;
size_t markerBuffer::alloc_size = 0;


void markerBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void markerBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void markerBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void markerBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
markerBuffer::operator type*(){ return ptr;}
float& markerBuffer::operator[](size_t idx){ return ptr[idx];}
bool markerBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::surfaceDistance>() == true;
	condition = condition || get<parameters::modules::surfaceDetection>() == true;
	return condition;
}
}
namespace surfaceArrays{
int32_t* surface_idxBuffer::ptr = nullptr;
size_t surface_idxBuffer::alloc_size = 0;

int32_t* surface_idxBuffer::rear_ptr = nullptr;
void surface_idxBuffer::swap() { std::swap(ptr, rear_ptr); }

void surface_idxBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void surface_idxBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(int32_t);
	
	cudaAllocateMemory(&ptr, alloc_size);
}

void surface_idxBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
	cudaAllocateMemory(&rear_ptr, alloc_size);
}
void surface_idxBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
	cudaFree(rear_ptr);
	rear_ptr = nullptr;
}
surface_idxBuffer::operator type*(){ return ptr;}
int32_t& surface_idxBuffer::operator[](size_t idx){ return ptr[idx];}
bool surface_idxBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::surfaceDistance>() == true;
	condition = condition || get<parameters::modules::surfaceDetection>() == true;
	return condition;
}
}
namespace surfaceArrays{
float* decisionBuffer::ptr = nullptr;
size_t decisionBuffer::alloc_size = 0;


void decisionBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void decisionBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void decisionBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void decisionBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
decisionBuffer::operator type*(){ return ptr;}
float& decisionBuffer::operator[](size_t idx){ return ptr[idx];}
bool decisionBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::surfaceDistance>() == true;
	condition = condition || get<parameters::modules::surfaceDetection>() == true;
	return condition;
}
}
namespace surfaceArrays{
float* changeBuffer::ptr = nullptr;
size_t changeBuffer::alloc_size = 0;


void changeBuffer::defaultAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	cudaAllocateMemory(&ptr, alloc_size);
}
void changeBuffer::leanAllocate(){
	auto elems = (get<parameters::max_numptcls>() + 1);
	alloc_size = 1 * elems * sizeof(float);
	
}

void changeBuffer::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void changeBuffer::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
changeBuffer::operator type*(){ return ptr;}
float& changeBuffer::operator[](size_t idx){ return ptr[idx];}
bool changeBuffer::valid(){
	bool condition = false;
	condition = condition || get<parameters::modules::surfaceDistance>() == true;
	condition = condition || get<parameters::modules::surfaceDetection>() == true;
	return condition;
}
}
namespace volumeInletArrays{
float4* volumeOutletMax::ptr = nullptr;
size_t volumeOutletMax::alloc_size = 0;



void volumeOutletMax::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletMax::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletMax::operator type*(){ return ptr;}
float4& volumeOutletMax::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletMax::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
float* volumeOutletRateAccumulator::ptr = nullptr;
size_t volumeOutletRateAccumulator::alloc_size = 0;



void volumeOutletRateAccumulator::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletRateAccumulator::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletRateAccumulator::operator type*(){ return ptr;}
float& volumeOutletRateAccumulator::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletRateAccumulator::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
float* volumeOutletRate::ptr = nullptr;
size_t volumeOutletRate::alloc_size = 0;



void volumeOutletRate::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletRate::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletRate::operator type*(){ return ptr;}
float& volumeOutletRate::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletRate::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
float4* volumeOutletMin::ptr = nullptr;
size_t volumeOutletMin::alloc_size = 0;



void volumeOutletMin::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletMin::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletMin::operator type*(){ return ptr;}
float4& volumeOutletMin::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletMin::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
int4* volumeOutletDimensions::ptr = nullptr;
size_t volumeOutletDimensions::alloc_size = 0;



void volumeOutletDimensions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletDimensions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletDimensions::operator type*(){ return ptr;}
int4& volumeOutletDimensions::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletDimensions::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
cudaTextureObject_t* volumeOutletVolumes::ptr = nullptr;
size_t volumeOutletVolumes::alloc_size = 0;



void volumeOutletVolumes::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void volumeOutletVolumes::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
volumeOutletVolumes::operator type*(){ return ptr;}
cudaTextureObject_t& volumeOutletVolumes::operator[](size_t idx){ return ptr[idx];}
bool volumeOutletVolumes::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
int32_t* inletCounter::ptr = nullptr;
size_t inletCounter::alloc_size = 0;



void inletCounter::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void inletCounter::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
inletCounter::operator type*(){ return ptr;}
int32_t& inletCounter::operator[](size_t idx){ return ptr[idx];}
bool inletCounter::valid(){
	bool condition = true;
	return condition;
}
}
namespace volumeInletArrays{
float4* inletPositions::ptr = nullptr;
size_t inletPositions::alloc_size = 0;



void inletPositions::allocate(size_t size){
	alloc_size = size;
	cudaAllocateMemory(&ptr, alloc_size);
}
void inletPositions::free(){
	alloc_size = 0;
	cudaFree(ptr);
	ptr = nullptr;
}
inletPositions::operator type*(){ return ptr;}
float4& inletPositions::operator[](size_t idx){ return ptr[idx];}
bool inletPositions::valid(){
	bool condition = true;
	return condition;
}
}
}

std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::adaptivityCounter, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::rigidBodyArrays::rigidDensities, arrays::rigidBodyArrays::rigidVolumes, arrays::rigidBodyArrays::rigidLinearVelocities, arrays::rigidBodyArrays::rigidAVelocities, arrays::rigidBodyArrays::rigidOrigins, arrays::rigidBodyArrays::rigidQuaternions, arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::matrixTest, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::hashMap, arrays::structureArrays::compactHashMap, arrays::structureArrays::mlmScaling, arrays::structureArrays::particleparticleIndex, arrays::structureArrays::cellBegin, arrays::structureArrays::cellEnd, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount, arrays::supportArrays::neighborListSwap> allocations_list;
std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::adaptivityCounter, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::rigidBodyArrays::rigidDensities, arrays::rigidBodyArrays::rigidVolumes, arrays::rigidBodyArrays::rigidLinearVelocities, arrays::rigidBodyArrays::rigidAVelocities, arrays::rigidBodyArrays::rigidOrigins, arrays::rigidBodyArrays::rigidQuaternions, arrays::boundaryArrays::splineLUT, arrays::boundaryArrays::offsetLUT, arrays::boundaryArrays::spikyLUT, arrays::boundaryArrays::adhesionLUT, arrays::boundaryArrays::volumeLUT, arrays::boundaryArrays::spline2LUT, arrays::boundaryArrays::splineGradientLUT, arrays::boundaryArrays::spikyGradientLUT, arrays::boundaryArrays::cohesionLUT, arrays::boundaryArrays::boundaryPlanes, arrays::boundaryArrays::boundaryPlaneVelocity, arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::matrixTest, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::volumeInletArrays::inletPositions, arrays::volumeInletArrays::inletCounter, arrays::volumeInletArrays::volumeOutletVolumes, arrays::volumeInletArrays::volumeOutletDimensions, arrays::volumeInletArrays::volumeOutletMin, arrays::volumeInletArrays::volumeOutletMax, arrays::volumeInletArrays::volumeOutletRate, arrays::volumeInletArrays::volumeOutletRateAccumulator, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::hashMap, arrays::structureArrays::compactHashMap, arrays::structureArrays::mlmScaling, arrays::structureArrays::particleparticleIndex, arrays::structureArrays::cellBegin, arrays::structureArrays::cellEnd, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount, arrays::supportArrays::neighborListSwap> arrays_list;
std::tuple<arrays::boundaryArrays::volumeBoundaryVolumes, arrays::boundaryArrays::volumeBoundaryDimensions, arrays::boundaryArrays::volumeBoundaryMin, arrays::boundaryArrays::volumeBoundaryMax, arrays::boundaryArrays::volumeBoundaryDensity, arrays::boundaryArrays::volumeBoundaryVolume, arrays::boundaryArrays::volumeBoundaryVelocity, arrays::boundaryArrays::volumeBoundaryAngularVelocity, arrays::boundaryArrays::volumeBoundaryKind, arrays::boundaryArrays::volumeBoundaryPosition, arrays::boundaryArrays::volumeBoundaryQuaternion, arrays::boundaryArrays::volumeBoundaryTransformMatrix, arrays::boundaryArrays::volumeBoundaryTransformMatrixInverse, arrays::boundaryArrays::volumeBoundaryInertiaMatrix, arrays::boundaryArrays::volumeBoundaryInertiaMatrixInverse, arrays::boundaryArrays::volumeBoundaryAcceleration, arrays::boundaryArrays::volumeBoundaryAngularAcceleration, arrays::basicArrays::matrixTest> individual_list;
std::tuple<arrays::adaptive::splitIndicator, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::angularVelocity, arrays::surfaceArrays::distanceBuffer, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::structureArrays::neighborListLength> sorting_list;
std::tuple<arrays::adaptive::mergeable, arrays::adaptive::mergeCounter, arrays::adaptive::ptclCounter, arrays::adaptive::classification, arrays::adaptive::splitIndicator, arrays::adaptive::splitIndicatorCompacted, arrays::adaptive::parentIndex, arrays::adaptive::parentVolume, arrays::adaptive::parentPosition, arrays::advectionArrays::maxVelocity, arrays::advectionArrays::cflValue, arrays::advectionArrays::particleNormal, arrays::advectionArrays::angularVelocity, arrays::surfaceArrays::decisionBuffer, arrays::surfaceArrays::surface_idxBuffer, arrays::surfaceArrays::markerBuffer, arrays::surfaceArrays::distanceBuffer, arrays::surfaceArrays::changeBuffer, arrays::basicArrays::position, arrays::basicArrays::acceleration, arrays::basicArrays::velocity, arrays::basicArrays::particle_type, arrays::basicArrays::renderArray, arrays::basicArrays::debugArray, arrays::basicArrays::volume, arrays::basicArrays::lifetime, arrays::basicArrays::pressure, arrays::basicArrays::density, arrays::basicArrays::fluidDensity, arrays::basicArrays::particleIndex, arrays::basicArrays::particleIndexCompact, arrays::basicArrays::resortArray4, arrays::basicArrays::resortArray, arrays::dfsphArrays::dfsphSource, arrays::dfsphArrays::dfsphAlpha, arrays::dfsphArrays::dfsphDpDt, arrays::dfsphArrays::dfsphRhoStar, arrays::iisphArrays::iisphSum, arrays::iisphArrays::iisphDii, arrays::iisphArrays::iisphAiiOld, arrays::iisphArrays::omega, arrays::iisphArrays::alpha, arrays::iisphArrays::gamma, arrays::iisphArrays::sigma, arrays::iisphArrays::iisphDensityAdvection, arrays::iisphArrays::iisphDensityIteration, arrays::iisphArrays::kernelBuffer, arrays::iisphArrays::velocityAdvection, arrays::iisph17::sourceTerm, arrays::iisph17::boundaryPressure, arrays::iisph17::omega, arrays::iisph17::apparentVolume, arrays::iisph17::Aii, arrays::iisph17::predictedAcceleration, arrays::iisph17::volumeError, arrays::iisph17_band_rb::_sourceTerm, arrays::iisph17_band_rb::_omega, arrays::iisph17_band_rb::_apparentVolume, arrays::iisph17_band_rb::_Aii, arrays::iisph17_band_rb::_predictedAcceleration, arrays::iisph17_band_rb::_volumeError, arrays::structureArrays::compactCellList, arrays::structureArrays::neighborMask, arrays::structureArrays::compactCellScale, arrays::structureArrays::spanNeighborList, arrays::structureArrays::neighborList, arrays::structureArrays::neighborListLength, arrays::renderArrays::anisotropicMatrices, arrays::renderArrays::centerPosition, arrays::renderArrays::auxTest, arrays::renderArrays::auxIsoDensity, arrays::renderArrays::auxDistance, arrays::renderArrays::auxHashMap, arrays::renderArrays::auxCellSpan, arrays::renderArrays::auxCellInformation, arrays::renderArrays::auxCellSurface, arrays::structureArrays::classification, arrays::structureArrays::resortIndex, arrays::structureArrays::ZOrder_64, arrays::structureArrays::ZOrder_32, arrays::structureArrays::cellSpanSwap, arrays::structureArrays::cellSpan, arrays::structureArrays::compactCellSpanSwap, arrays::structureArrays::compactCellSpan, arrays::structureArrays::MLMResolution, arrays::structureArrays::cellparticleIndex, arrays::structureArrays::compactparticleIndex, arrays::structureArrays::particleparticleIndex, arrays::supportArrays::support, arrays::supportArrays::supportEstimate, arrays::supportArrays::neighborCount, arrays::supportArrays::supportMarker, arrays::supportArrays::supportMarkerCompacted, arrays::supportArrays::closestNeighbor, arrays::supportArrays::closestNeighbor_f, arrays::supportArrays::neighborOverhead, arrays::supportArrays::neighborOverheadCount> swapping_list;