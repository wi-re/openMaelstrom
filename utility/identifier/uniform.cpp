#include <utility/identifier/uniform.h>
#include <utility/identifier/arrays.h>
#include <utility/helpers/pathfinder.h>
namespace parameters{

namespace adaptive{
static std::vector<int32_t> HELPER_VARIABLEsharedPtcls{0};
std::vector<int32_t>* sharedPtcls::ptr = &HELPER_VARIABLEsharedPtcls;
std::vector<int32_t>* sharedPtcls::unit_ptr = (std::vector<int32_t>*) &HELPER_VARIABLEsharedPtcls;
std::vector<std::string>  sharedPtcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEadaptivityScaling{1.f};
float* adaptivityScaling::ptr = &HELPER_VARIABLEadaptivityScaling;
float* adaptivityScaling::unit_ptr = (float*) &HELPER_VARIABLEadaptivityScaling;
std::vector<std::string>  adaptivityScaling::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEadaptivityThreshold{1.0f};
float* adaptivityThreshold::ptr = &HELPER_VARIABLEadaptivityThreshold;
float* adaptivityThreshold::unit_ptr = (float*) &HELPER_VARIABLEadaptivityThreshold;
std::vector<std::string>  adaptivityThreshold::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEadaptivityGamma{0.1f};
float* adaptivityGamma::ptr = &HELPER_VARIABLEadaptivityGamma;
float* adaptivityGamma::unit_ptr = (float*) &HELPER_VARIABLEadaptivityGamma;
std::vector<std::string>  adaptivityGamma::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEresolution{32.f};
float* resolution::ptr = &HELPER_VARIABLEresolution;
float* resolution::unit_ptr = (float*) &HELPER_VARIABLEresolution;
std::vector<std::string>  resolution::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static int32_t HELPER_VARIABLEuseVolume{1};
int32_t* useVolume::ptr = &HELPER_VARIABLEuseVolume;
int32_t* useVolume::unit_ptr = (int32_t*) &HELPER_VARIABLEuseVolume;
std::vector<std::string>  useVolume::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEminVolume{1.f};
float* minVolume::ptr = &HELPER_VARIABLEminVolume;
value_unit<float, SI::volume>* minVolume::unit_ptr = (value_unit<float, SI::volume>*) &HELPER_VARIABLEminVolume;
std::vector<std::string>  minVolume::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static int32_t HELPER_VARIABLEdetailedAdaptiveStatistics{1};
int32_t* detailedAdaptiveStatistics::ptr = &HELPER_VARIABLEdetailedAdaptiveStatistics;
int32_t* detailedAdaptiveStatistics::unit_ptr = (int32_t*) &HELPER_VARIABLEdetailedAdaptiveStatistics;
std::vector<std::string>  detailedAdaptiveStatistics::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEratio{1.f};
float* ratio::ptr = &HELPER_VARIABLEratio;
float* ratio::unit_ptr = (float*) &HELPER_VARIABLEratio;
std::vector<std::string>  ratio::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEblendSteps{10.f};
float* blendSteps::ptr = &HELPER_VARIABLEblendSteps;
float* blendSteps::unit_ptr = (float*) &HELPER_VARIABLEblendSteps;
std::vector<std::string>  blendSteps::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static float HELPER_VARIABLEdelay{1.f};
float* delay::ptr = &HELPER_VARIABLEdelay;
value_unit<float, SI::s>* delay::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEdelay;
std::vector<std::string>  delay::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static std::vector<int32_t> HELPER_VARIABLEsplitPtcls{0};
std::vector<int32_t>* splitPtcls::ptr = &HELPER_VARIABLEsplitPtcls;
std::vector<int32_t>* splitPtcls::unit_ptr = (std::vector<int32_t>*) &HELPER_VARIABLEsplitPtcls;
std::vector<std::string>  splitPtcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static int32_t HELPER_VARIABLEblendedPtcls{0};
int32_t* blendedPtcls::ptr = &HELPER_VARIABLEblendedPtcls;
int32_t* blendedPtcls::unit_ptr = (int32_t*) &HELPER_VARIABLEblendedPtcls;
std::vector<std::string>  blendedPtcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace adaptive{
static std::vector<int32_t> HELPER_VARIABLEmergedPtcls{0};
std::vector<int32_t>* mergedPtcls::ptr = &HELPER_VARIABLEmergedPtcls;
std::vector<int32_t>* mergedPtcls::unit_ptr = (std::vector<int32_t>*) &HELPER_VARIABLEmergedPtcls;
std::vector<std::string>  mergedPtcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace alembic{
static int32_t HELPER_VARIABLEalembic_fps{24};
int32_t* alembic_fps::ptr = &HELPER_VARIABLEalembic_fps;
int32_t* alembic_fps::unit_ptr = (int32_t*) &HELPER_VARIABLEalembic_fps;
std::vector<std::string>  alembic_fps::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace alembic{
static std::string HELPER_VARIABLEalembic_file_name{"export/alembic_$f.abc"};
std::string* alembic_file_name::ptr = &HELPER_VARIABLEalembic_file_name;
std::string* alembic_file_name::unit_ptr = (std::string*) &HELPER_VARIABLEalembic_file_name;
std::vector<std::string>  alembic_file_name::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace boundary_volumes{
static int32_t HELPER_VARIABLEvolumeBoundaryCounter{0};
int32_t* volumeBoundaryCounter::ptr = &HELPER_VARIABLEvolumeBoundaryCounter;
int32_t* volumeBoundaryCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEvolumeBoundaryCounter;
std::vector<std::string>  volumeBoundaryCounter::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace boundary_volumes{
static std::vector<boundaryVolume> HELPER_VARIABLEboundaryVolumes{};
std::vector<boundaryVolume>* boundaryVolumes::ptr = &HELPER_VARIABLEboundaryVolumes;
std::vector<boundaryVolume>* boundaryVolumes::unit_ptr = (std::vector<boundaryVolume>*) &HELPER_VARIABLEboundaryVolumes;
std::vector<std::string>  boundaryVolumes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static std::string HELPER_VARIABLEvectorMode{"length"};
std::string* vectorMode::ptr = &HELPER_VARIABLEvectorMode;
std::string* vectorMode::unit_ptr = (std::string*) &HELPER_VARIABLEvectorMode;
std::vector<std::string>  vectorMode::getPresets(){
	std::vector<std::string> presets = {"length","x","y","z","w"};
	return presets;
}
}

namespace color_map{
static std::string HELPER_VARIABLErender_colormap{"inferno"};
std::string* render_colormap::ptr = &HELPER_VARIABLErender_colormap;
std::string* render_colormap::unit_ptr = (std::string*) &HELPER_VARIABLErender_colormap;
std::vector<std::string>  render_colormap::getPresets(){
	std::vector<std::string> presets = []() {std::vector <std::string> colorMaps;auto f = resolveFile("cfg/style.css");auto p = f.parent_path().string();if (*(p.end() - 1) == '/' || *(p.end() - 1) == '\\')p = p.substr(0, p.length() - 1);std::replace(p.begin(), p.end(), '\\', '/');for (auto& p : std::experimental::filesystem::directory_iterator(p))if (p.path().extension().string().find(".png") != std::string::npos)colorMaps.push_back(p.path().filename().replace_extension("").string());return colorMaps; }();
	return presets;
}
}

namespace color_map{
static std::string HELPER_VARIABLEmapping_mode{"linear"};
std::string* mapping_mode::ptr = &HELPER_VARIABLEmapping_mode;
std::string* mapping_mode::unit_ptr = (std::string*) &HELPER_VARIABLEmapping_mode;
std::vector<std::string>  mapping_mode::getPresets(){
	std::vector<std::string> presets = {"linear","cubicRoot","cubic","squareRoot","square","log"};
	return presets;
}
}

namespace color_map{
static int32_t HELPER_VARIABLEvectorScaling{0};
int32_t* vectorScaling::ptr = &HELPER_VARIABLEvectorScaling;
int32_t* vectorScaling::unit_ptr = (int32_t*) &HELPER_VARIABLEvectorScaling;
std::vector<std::string>  vectorScaling::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static int32_t HELPER_VARIABLEpruneVoxel{0};
int32_t* pruneVoxel::ptr = &HELPER_VARIABLEpruneVoxel;
int32_t* pruneVoxel::unit_ptr = (int32_t*) &HELPER_VARIABLEpruneVoxel;
std::vector<std::string>  pruneVoxel::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static std::string HELPER_VARIABLErender_buffer{"density"};
std::string* render_buffer::ptr = &HELPER_VARIABLErender_buffer;
std::string* render_buffer::unit_ptr = (std::string*) &HELPER_VARIABLErender_buffer;
std::vector<std::string>  render_buffer::getPresets(){
	std::vector<std::string> presets = [](){std::vector<std::string> names;iterateArraysList([&](auto x){using Ty = std::decay_t<decltype(x)>;if (Ty::valid() && Ty::kind == memory_kind::particleData)names.push_back(Ty::qualifiedName);});return names;}();
	return presets;
}
}

namespace color_map{
static int HELPER_VARIABLEmap_flipped{0};
int* map_flipped::ptr = &HELPER_VARIABLEmap_flipped;
int* map_flipped::unit_ptr = (int*) &HELPER_VARIABLEmap_flipped;
std::vector<std::string>  map_flipped::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static int HELPER_VARIABLErender_auto{1};
int* render_auto::ptr = &HELPER_VARIABLErender_auto;
int* render_auto::unit_ptr = (int*) &HELPER_VARIABLErender_auto;
std::vector<std::string>  render_auto::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static int32_t HELPER_VARIABLEmapping_fn{0};
int32_t* mapping_fn::ptr = &HELPER_VARIABLEmapping_fn;
int32_t* mapping_fn::unit_ptr = (int32_t*) &HELPER_VARIABLEmapping_fn;
std::vector<std::string>  mapping_fn::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static int32_t HELPER_VARIABLEtransfer_fn{0};
int32_t* transfer_fn::ptr = &HELPER_VARIABLEtransfer_fn;
int32_t* transfer_fn::unit_ptr = (int32_t*) &HELPER_VARIABLEtransfer_fn;
std::vector<std::string>  transfer_fn::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static std::string HELPER_VARIABLEtransfer_mode{"linear"};
std::string* transfer_mode::ptr = &HELPER_VARIABLEtransfer_mode;
std::string* transfer_mode::unit_ptr = (std::string*) &HELPER_VARIABLEtransfer_mode;
std::vector<std::string>  transfer_mode::getPresets(){
	std::vector<std::string> presets = {"linear","cubicRoot","cubic","squareRoot","square","log"};
	return presets;
}
}

namespace color_map{
static float HELPER_VARIABLErender_max{1.f};
float* render_max::ptr = &HELPER_VARIABLErender_max;
float* render_max::unit_ptr = (float*) &HELPER_VARIABLErender_max;
std::vector<std::string>  render_max::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static float HELPER_VARIABLErender_min{0.f};
float* render_min::ptr = &HELPER_VARIABLErender_min;
float* render_min::unit_ptr = (float*) &HELPER_VARIABLErender_min;
std::vector<std::string>  render_min::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static float HELPER_VARIABLEvectorScale{1};
float* vectorScale::ptr = &HELPER_VARIABLEvectorScale;
float* vectorScale::unit_ptr = (float*) &HELPER_VARIABLEvectorScale;
std::vector<std::string>  vectorScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace color_map{
static int32_t HELPER_VARIABLEvisualizeDirection{0};
int32_t* visualizeDirection::ptr = &HELPER_VARIABLEvisualizeDirection;
int32_t* visualizeDirection::unit_ptr = (int32_t*) &HELPER_VARIABLEvisualizeDirection;
std::vector<std::string>  visualizeDirection::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static float HELPER_VARIABLEdensityError{0.f};
float* densityError::ptr = &HELPER_VARIABLEdensityError;
float* densityError::unit_ptr = (float*) &HELPER_VARIABLEdensityError;
std::vector<std::string>  densityError::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static float HELPER_VARIABLEdivergenceEta{0.001f};
float* divergenceEta::ptr = &HELPER_VARIABLEdivergenceEta;
value_unit<float, SI::Hz>* divergenceEta::unit_ptr = (value_unit<float, SI::Hz>*) &HELPER_VARIABLEdivergenceEta;
std::vector<std::string>  divergenceEta::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static float HELPER_VARIABLEdensityEta{0.0001f};
float* densityEta::ptr = &HELPER_VARIABLEdensityEta;
float* densityEta::unit_ptr = (float*) &HELPER_VARIABLEdensityEta;
std::vector<std::string>  densityEta::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static float HELPER_VARIABLEdivergenceError{0.f};
float* divergenceError::ptr = &HELPER_VARIABLEdivergenceError;
value_unit<float, SI::Hz>* divergenceError::unit_ptr = (value_unit<float, SI::Hz>*) &HELPER_VARIABLEdivergenceError;
std::vector<std::string>  divergenceError::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static int32_t HELPER_VARIABLEdivergenceSolverIterations{0};
int32_t* divergenceSolverIterations::ptr = &HELPER_VARIABLEdivergenceSolverIterations;
int32_t* divergenceSolverIterations::unit_ptr = (int32_t*) &HELPER_VARIABLEdivergenceSolverIterations;
std::vector<std::string>  divergenceSolverIterations::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace dfsph_settings{
static int32_t HELPER_VARIABLEdensitySolverIterations{0};
int32_t* densitySolverIterations::ptr = &HELPER_VARIABLEdensitySolverIterations;
int32_t* densitySolverIterations::unit_ptr = (int32_t*) &HELPER_VARIABLEdensitySolverIterations;
std::vector<std::string>  densitySolverIterations::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace iisph_settings{
static float HELPER_VARIABLEjacobi_omega{0.2f};
float* jacobi_omega::ptr = &HELPER_VARIABLEjacobi_omega;
float* jacobi_omega::unit_ptr = (float*) &HELPER_VARIABLEjacobi_omega;
std::vector<std::string>  jacobi_omega::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace iisph_settings{
static int32_t HELPER_VARIABLEiterations{0};
int32_t* iterations::ptr = &HELPER_VARIABLEiterations;
int32_t* iterations::unit_ptr = (int32_t*) &HELPER_VARIABLEiterations;
std::vector<std::string>  iterations::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace iisph_settings{
static float HELPER_VARIABLEeta{0.1f};
float* eta::ptr = &HELPER_VARIABLEeta;
float* eta::unit_ptr = (float*) &HELPER_VARIABLEeta;
std::vector<std::string>  eta::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace iisph_settings{
static float HELPER_VARIABLEdensity_error{0.f};
float* density_error::ptr = &HELPER_VARIABLEdensity_error;
float* density_error::unit_ptr = (float*) &HELPER_VARIABLEdensity_error;
std::vector<std::string>  density_error::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace inlet_volumes{
static std::vector<inletVolume> HELPER_VARIABLEinletVolumes{};
std::vector<inletVolume>* inletVolumes::ptr = &HELPER_VARIABLEinletVolumes;
std::vector<inletVolume>* inletVolumes::unit_ptr = (std::vector<inletVolume>*) &HELPER_VARIABLEinletVolumes;
std::vector<std::string>  inletVolumes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEnum_ptcls{0u};
int32_t* num_ptcls::ptr = &HELPER_VARIABLEnum_ptcls;
int32_t* num_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEnum_ptcls;
std::vector<std::string>  num_ptcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float3 HELPER_VARIABLEmax_domain{0.f,0.f,0.f};
float3* max_domain::ptr = &HELPER_VARIABLEmax_domain;
value_unit<float3, SI::m>* max_domain::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmax_domain;
std::vector<std::string>  max_domain::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static neighbor_list HELPER_VARIABLEneighborhood_kind{neighbor_list::constrained};
neighbor_list* neighborhood_kind::ptr = &HELPER_VARIABLEneighborhood_kind;
neighbor_list* neighborhood_kind::unit_ptr = (neighbor_list*) &HELPER_VARIABLEneighborhood_kind;
std::vector<std::string>  neighborhood_kind::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEdumpNextframe{0};
int32_t* dumpNextframe::ptr = &HELPER_VARIABLEdumpNextframe;
int32_t* dumpNextframe::unit_ptr = (int32_t*) &HELPER_VARIABLEdumpNextframe;
std::vector<std::string>  dumpNextframe::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEdumpForSSSPH{0};
int32_t* dumpForSSSPH::ptr = &HELPER_VARIABLEdumpForSSSPH;
int32_t* dumpForSSSPH::unit_ptr = (int32_t*) &HELPER_VARIABLEdumpForSSSPH;
std::vector<std::string>  dumpForSSSPH::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static launch_config HELPER_VARIABLEtarget{launch_config::device};
launch_config* target::ptr = &HELPER_VARIABLEtarget;
launch_config* target::unit_ptr = (launch_config*) &HELPER_VARIABLEtarget;
std::vector<std::string>  target::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static hash_length HELPER_VARIABLEhash_size{hash_length::bit_64};
hash_length* hash_size::ptr = &HELPER_VARIABLEhash_size;
hash_length* hash_size::unit_ptr = (hash_length*) &HELPER_VARIABLEhash_size;
std::vector<std::string>  hash_size::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static cell_ordering HELPER_VARIABLEcell_order{cell_ordering::z_order};
cell_ordering* cell_order::ptr = &HELPER_VARIABLEcell_order;
cell_ordering* cell_order::unit_ptr = (cell_ordering*) &HELPER_VARIABLEcell_order;
std::vector<std::string>  cell_order::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static cell_structuring HELPER_VARIABLEcell_structure{cell_structuring::hashed};
cell_structuring* cell_structure::ptr = &HELPER_VARIABLEcell_structure;
cell_structuring* cell_structure::unit_ptr = (cell_structuring*) &HELPER_VARIABLEcell_structure;
std::vector<std::string>  cell_structure::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEboundaryCounter{0u};
int32_t* boundaryCounter::ptr = &HELPER_VARIABLEboundaryCounter;
int32_t* boundaryCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEboundaryCounter;
std::vector<std::string>  boundaryCounter::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEboundaryLUTSize{0u};
int32_t* boundaryLUTSize::ptr = &HELPER_VARIABLEboundaryLUTSize;
int32_t* boundaryLUTSize::unit_ptr = (int32_t*) &HELPER_VARIABLEboundaryLUTSize;
std::vector<std::string>  boundaryLUTSize::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int32_t HELPER_VARIABLEframe{0u};
int32_t* frame::ptr = &HELPER_VARIABLEframe;
int32_t* frame::unit_ptr = (int32_t*) &HELPER_VARIABLEframe;
std::vector<std::string>  frame::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float HELPER_VARIABLEmax_velocity{1.f};
float* max_velocity::ptr = &HELPER_VARIABLEmax_velocity;
value_unit<float, SI::velocity>* max_velocity::unit_ptr = (value_unit<float, SI::velocity>*) &HELPER_VARIABLEmax_velocity;
std::vector<std::string>  max_velocity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float3 HELPER_VARIABLEmin_domain{0.f,0.f,0.f};
float3* min_domain::ptr = &HELPER_VARIABLEmin_domain;
value_unit<float3, SI::m>* min_domain::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmin_domain;
std::vector<std::string>  min_domain::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float HELPER_VARIABLEsimulationTime{0.f};
float* simulationTime::ptr = &HELPER_VARIABLEsimulationTime;
value_unit<float, SI::s>* simulationTime::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEsimulationTime;
std::vector<std::string>  simulationTime::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float3 HELPER_VARIABLEmin_coord{0.f,0.f,0.f};
float3* min_coord::ptr = &HELPER_VARIABLEmin_coord;
value_unit<float3, SI::m>* min_coord::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmin_coord;
std::vector<std::string>  min_coord::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float3 HELPER_VARIABLEmax_coord{0.f,0.f,0.f};
float3* max_coord::ptr = &HELPER_VARIABLEmax_coord;
value_unit<float3, SI::m>* max_coord::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmax_coord;
std::vector<std::string>  max_coord::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float3 HELPER_VARIABLEcell_size{0.f,0.f,0.f};
float3* cell_size::ptr = &HELPER_VARIABLEcell_size;
value_unit<float3, SI::m>* cell_size::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEcell_size;
std::vector<std::string>  cell_size::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static int3 HELPER_VARIABLEgrid_size{0u,0u,0u};
int3* grid_size::ptr = &HELPER_VARIABLEgrid_size;
int3* grid_size::unit_ptr = (int3*) &HELPER_VARIABLEgrid_size;
std::vector<std::string>  grid_size::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float HELPER_VARIABLEptcl_spacing{0.f};
float* ptcl_spacing::ptr = &HELPER_VARIABLEptcl_spacing;
value_unit<float, SI::m>* ptcl_spacing::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEptcl_spacing;
std::vector<std::string>  ptcl_spacing::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float HELPER_VARIABLEptcl_support{0.f};
float* ptcl_support::ptr = &HELPER_VARIABLEptcl_support;
value_unit<float, SI::m>* ptcl_support::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEptcl_support;
std::vector<std::string>  ptcl_support::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEconfig_file{"DamBreakObstacle.json"};
std::string* config_file::ptr = &HELPER_VARIABLEconfig_file;
std::string* config_file::unit_ptr = (std::string*) &HELPER_VARIABLEconfig_file;
std::vector<std::string>  config_file::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEconfig_folder{"D:/DamBreak"};
std::string* config_folder::ptr = &HELPER_VARIABLEconfig_folder;
std::string* config_folder::unit_ptr = (std::string*) &HELPER_VARIABLEconfig_folder;
std::vector<std::string>  config_folder::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEworking_directory{"D:/DamBreak"};
std::string* working_directory::ptr = &HELPER_VARIABLEworking_directory;
std::string* working_directory::unit_ptr = (std::string*) &HELPER_VARIABLEworking_directory;
std::vector<std::string>  working_directory::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEbuild_directory{"D:/DamBreak"};
std::string* build_directory::ptr = &HELPER_VARIABLEbuild_directory;
std::string* build_directory::unit_ptr = (std::string*) &HELPER_VARIABLEbuild_directory;
std::vector<std::string>  build_directory::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEsource_directory{"D:/DamBreak"};
std::string* source_directory::ptr = &HELPER_VARIABLEsource_directory;
std::string* source_directory::unit_ptr = (std::string*) &HELPER_VARIABLEsource_directory;
std::vector<std::string>  source_directory::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static std::string HELPER_VARIABLEbinary_directory{"D:/DamBreak"};
std::string* binary_directory::ptr = &HELPER_VARIABLEbinary_directory;
std::string* binary_directory::unit_ptr = (std::string*) &HELPER_VARIABLEbinary_directory;
std::vector<std::string>  binary_directory::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace internal{
static float HELPER_VARIABLEtimestep{0.f};
float* timestep::ptr = &HELPER_VARIABLEtimestep;
value_unit<float, SI::s>* timestep::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep;
std::vector<std::string>  timestep::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEdebug{false};
bool* debug::ptr = &HELPER_VARIABLEdebug;
bool* debug::unit_ptr = (bool*) &HELPER_VARIABLEdebug;
std::vector<std::string>  debug::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEsorting{"hashed_cell"};
std::string* sorting::ptr = &HELPER_VARIABLEsorting;
std::string* sorting::unit_ptr = (std::string*) &HELPER_VARIABLEsorting;
std::vector<std::string>  sorting::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEhash_width{"64bit"};
std::string* hash_width::ptr = &HELPER_VARIABLEhash_width;
std::string* hash_width::unit_ptr = (std::string*) &HELPER_VARIABLEhash_width;
std::vector<std::string>  hash_width::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEalembic_export{false};
bool* alembic_export::ptr = &HELPER_VARIABLEalembic_export;
bool* alembic_export::unit_ptr = (bool*) &HELPER_VARIABLEalembic_export;
std::vector<std::string>  alembic_export::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEmovingBoundaries{false};
bool* movingBoundaries::ptr = &HELPER_VARIABLEmovingBoundaries;
bool* movingBoundaries::unit_ptr = (bool*) &HELPER_VARIABLEmovingBoundaries;
std::vector<std::string>  movingBoundaries::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLElaunch_cfg{"device"};
std::string* launch_cfg::ptr = &HELPER_VARIABLElaunch_cfg;
std::string* launch_cfg::unit_ptr = (std::string*) &HELPER_VARIABLElaunch_cfg;
std::vector<std::string>  launch_cfg::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEerror_checking{false};
bool* error_checking::ptr = &HELPER_VARIABLEerror_checking;
bool* error_checking::unit_ptr = (bool*) &HELPER_VARIABLEerror_checking;
std::vector<std::string>  error_checking::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static int32_t HELPER_VARIABLErenderMode{0};
int32_t* renderMode::ptr = &HELPER_VARIABLErenderMode;
int32_t* renderMode::unit_ptr = (int32_t*) &HELPER_VARIABLErenderMode;
std::vector<std::string>  renderMode::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEanisotropicSurface{true};
bool* anisotropicSurface::ptr = &HELPER_VARIABLEanisotropicSurface;
bool* anisotropicSurface::unit_ptr = (bool*) &HELPER_VARIABLEanisotropicSurface;
std::vector<std::string>  anisotropicSurface::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEdensity{"standard"};
std::string* density::ptr = &HELPER_VARIABLEdensity;
std::string* density::unit_ptr = (std::string*) &HELPER_VARIABLEdensity;
std::vector<std::string>  density::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEgl_record{false};
bool* gl_record::ptr = &HELPER_VARIABLEgl_record;
bool* gl_record::unit_ptr = (bool*) &HELPER_VARIABLEgl_record;
std::vector<std::string>  gl_record::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLErayTracing{true};
bool* rayTracing::ptr = &HELPER_VARIABLErayTracing;
bool* rayTracing::unit_ptr = (bool*) &HELPER_VARIABLErayTracing;
std::vector<std::string>  rayTracing::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEregex_cfg{false};
bool* regex_cfg::ptr = &HELPER_VARIABLEregex_cfg;
bool* regex_cfg::unit_ptr = (bool*) &HELPER_VARIABLEregex_cfg;
std::vector<std::string>  regex_cfg::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEvolumeBoundary{true};
bool* volumeBoundary::ptr = &HELPER_VARIABLEvolumeBoundary;
bool* volumeBoundary::unit_ptr = (bool*) &HELPER_VARIABLEvolumeBoundary;
std::vector<std::string>  volumeBoundary::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEadaptive{true};
bool* adaptive::ptr = &HELPER_VARIABLEadaptive;
bool* adaptive::unit_ptr = (bool*) &HELPER_VARIABLEadaptive;
std::vector<std::string>  adaptive::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEpressure{"DFSPH"};
std::string* pressure::ptr = &HELPER_VARIABLEpressure;
std::string* pressure::unit_ptr = (std::string*) &HELPER_VARIABLEpressure;
std::vector<std::string>  pressure::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEsupport{"constrained"};
std::string* support::ptr = &HELPER_VARIABLEsupport;
std::string* support::unit_ptr = (std::string*) &HELPER_VARIABLEsupport;
std::vector<std::string>  support::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEparticleCleanUp{true};
bool* particleCleanUp::ptr = &HELPER_VARIABLEparticleCleanUp;
bool* particleCleanUp::unit_ptr = (bool*) &HELPER_VARIABLEparticleCleanUp;
std::vector<std::string>  particleCleanUp::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEsurfaceDistance{false};
bool* surfaceDistance::ptr = &HELPER_VARIABLEsurfaceDistance;
bool* surfaceDistance::unit_ptr = (bool*) &HELPER_VARIABLEsurfaceDistance;
std::vector<std::string>  surfaceDistance::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEvorticity{"Bender17"};
std::string* vorticity::ptr = &HELPER_VARIABLEvorticity;
std::string* vorticity::unit_ptr = (std::string*) &HELPER_VARIABLEvorticity;
std::vector<std::string>  vorticity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEtension{"Akinci"};
std::string* tension::ptr = &HELPER_VARIABLEtension;
std::string* tension::unit_ptr = (std::string*) &HELPER_VARIABLEtension;
std::vector<std::string>  tension::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEviscosity{false};
bool* viscosity::ptr = &HELPER_VARIABLEviscosity;
bool* viscosity::unit_ptr = (bool*) &HELPER_VARIABLEviscosity;
std::vector<std::string>  viscosity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEneighborhood{"constrained"};
std::string* neighborhood::ptr = &HELPER_VARIABLEneighborhood;
std::string* neighborhood::unit_ptr = (std::string*) &HELPER_VARIABLEneighborhood;
std::vector<std::string>  neighborhood::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEvolumeInlets{false};
bool* volumeInlets::ptr = &HELPER_VARIABLEvolumeInlets;
bool* volumeInlets::unit_ptr = (bool*) &HELPER_VARIABLEvolumeInlets;
std::vector<std::string>  volumeInlets::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEvolumeOutlets{false};
bool* volumeOutlets::ptr = &HELPER_VARIABLEvolumeOutlets;
bool* volumeOutlets::unit_ptr = (bool*) &HELPER_VARIABLEvolumeOutlets;
std::vector<std::string>  volumeOutlets::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static int32_t HELPER_VARIABLEneighborSorting{1};
int32_t* neighborSorting::ptr = &HELPER_VARIABLEneighborSorting;
int32_t* neighborSorting::unit_ptr = (int32_t*) &HELPER_VARIABLEneighborSorting;
std::vector<std::string>  neighborSorting::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLEsurfaceDetection{true};
bool* surfaceDetection::ptr = &HELPER_VARIABLEsurfaceDetection;
bool* surfaceDetection::unit_ptr = (bool*) &HELPER_VARIABLEsurfaceDetection;
std::vector<std::string>  surfaceDetection::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static std::string HELPER_VARIABLEdrag{"Gissler17"};
std::string* drag::ptr = &HELPER_VARIABLEdrag;
std::string* drag::unit_ptr = (std::string*) &HELPER_VARIABLEdrag;
std::vector<std::string>  drag::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace modules{
static bool HELPER_VARIABLExsph{true};
bool* xsph::ptr = &HELPER_VARIABLExsph;
bool* xsph::unit_ptr = (bool*) &HELPER_VARIABLExsph;
std::vector<std::string>  xsph::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace moving_plane{
static std::vector<movingPlane> HELPER_VARIABLEmovingPlanes{};
std::vector<movingPlane>* movingPlanes::ptr = &HELPER_VARIABLEmovingPlanes;
std::vector<movingPlane>* movingPlanes::unit_ptr = (std::vector<movingPlane>*) &HELPER_VARIABLEmovingPlanes;
std::vector<std::string>  movingPlanes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace outlet_volumes{
static float HELPER_VARIABLEvolumeOutletTime{-1.f};
float* volumeOutletTime::ptr = &HELPER_VARIABLEvolumeOutletTime;
value_unit<float, SI::s>* volumeOutletTime::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEvolumeOutletTime;
std::vector<std::string>  volumeOutletTime::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace outlet_volumes{
static int32_t HELPER_VARIABLEvolumeOutletCounter{0};
int32_t* volumeOutletCounter::ptr = &HELPER_VARIABLEvolumeOutletCounter;
int32_t* volumeOutletCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEvolumeOutletCounter;
std::vector<std::string>  volumeOutletCounter::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace outlet_volumes{
static std::vector<outletVolume> HELPER_VARIABLEoutletVolumes{};
std::vector<outletVolume>* outletVolumes::ptr = &HELPER_VARIABLEoutletVolumes;
std::vector<outletVolume>* outletVolumes::unit_ptr = (std::vector<outletVolume>*) &HELPER_VARIABLEoutletVolumes;
std::vector<std::string>  outletVolumes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particleSets{
static std::vector<std::string> HELPER_VARIABLEparticleSets{};
std::vector<std::string>* particleSets::ptr = &HELPER_VARIABLEparticleSets;
std::vector<std::string>* particleSets::unit_ptr = (std::vector<std::string>*) &HELPER_VARIABLEparticleSets;
std::vector<std::string>  particleSets::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLEradius{0.5f};
float* radius::ptr = &HELPER_VARIABLEradius;
value_unit<float, SI::m>* radius::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEradius;
std::vector<std::string>  radius::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLEboundaryAdhesion_akinci{0.f};
float* boundaryAdhesion_akinci::ptr = &HELPER_VARIABLEboundaryAdhesion_akinci;
value_unit<float, SI::acceleration>* boundaryAdhesion_akinci::unit_ptr = (value_unit<float, SI::acceleration>*) &HELPER_VARIABLEboundaryAdhesion_akinci;
std::vector<std::string>  boundaryAdhesion_akinci::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLErest_density{998.f};
float* rest_density::ptr = &HELPER_VARIABLErest_density;
value_unit<float, SI::density>* rest_density::unit_ptr = (value_unit<float, SI::density>*) &HELPER_VARIABLErest_density;
std::vector<std::string>  rest_density::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float4 HELPER_VARIABLEair_velocity{0.f,0.f,0.f,0.f};
float4* air_velocity::ptr = &HELPER_VARIABLEair_velocity;
value_unit<float4, SI::velocity>* air_velocity::unit_ptr = (value_unit<float4, SI::velocity>*) &HELPER_VARIABLEair_velocity;
std::vector<std::string>  air_velocity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLEtension_akinci{0.15f};
float* tension_akinci::ptr = &HELPER_VARIABLEtension_akinci;
value_unit<float, SI::acceleration>* tension_akinci::unit_ptr = (value_unit<float, SI::acceleration>*) &HELPER_VARIABLEtension_akinci;
std::vector<std::string>  tension_akinci::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLErigidAdhesion_akinci{0.f};
float* rigidAdhesion_akinci::ptr = &HELPER_VARIABLErigidAdhesion_akinci;
value_unit<float, SI::acceleration>* rigidAdhesion_akinci::unit_ptr = (value_unit<float, SI::acceleration>*) &HELPER_VARIABLErigidAdhesion_akinci;
std::vector<std::string>  rigidAdhesion_akinci::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLExsph_viscosity{0.05f};
float* xsph_viscosity::ptr = &HELPER_VARIABLExsph_viscosity;
float* xsph_viscosity::unit_ptr = (float*) &HELPER_VARIABLExsph_viscosity;
std::vector<std::string>  xsph_viscosity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLEmonaghan_viscosity{5.f};
float* monaghan_viscosity::ptr = &HELPER_VARIABLEmonaghan_viscosity;
value_unit<float, SI::velocity>* monaghan_viscosity::unit_ptr = (value_unit<float, SI::velocity>*) &HELPER_VARIABLEmonaghan_viscosity;
std::vector<std::string>  monaghan_viscosity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_settings{
static float HELPER_VARIABLEboundaryViscosity{0.0375f};
float* boundaryViscosity::ptr = &HELPER_VARIABLEboundaryViscosity;
value_unit<float, SI::velocity>* boundaryViscosity::unit_ptr = (value_unit<float, SI::velocity>*) &HELPER_VARIABLEboundaryViscosity;
std::vector<std::string>  boundaryViscosity::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace particle_volumes{
static std::vector<particleVolume> HELPER_VARIABLEparticleVolumes{};
std::vector<particleVolume>* particleVolumes::ptr = &HELPER_VARIABLEparticleVolumes;
std::vector<particleVolume>* particleVolumes::unit_ptr = (std::vector<particleVolume>*) &HELPER_VARIABLEparticleVolumes;
std::vector<std::string>  particleVolumes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderGrid{0};
int32_t* vrtxRenderGrid::ptr = &HELPER_VARIABLEvrtxRenderGrid;
int32_t* vrtxRenderGrid::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderGrid;
std::vector<std::string>  vrtxRenderGrid::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderFluid{1};
int32_t* vrtxRenderFluid::ptr = &HELPER_VARIABLEvrtxRenderFluid;
int32_t* vrtxRenderFluid::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderFluid;
std::vector<std::string>  vrtxRenderFluid::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderSurface{1};
int32_t* vrtxRenderSurface::ptr = &HELPER_VARIABLEvrtxRenderSurface;
int32_t* vrtxRenderSurface::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderSurface;
std::vector<std::string>  vrtxRenderSurface::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxDisplayStats{1};
int32_t* vrtxDisplayStats::ptr = &HELPER_VARIABLEvrtxDisplayStats;
int32_t* vrtxDisplayStats::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxDisplayStats;
std::vector<std::string>  vrtxDisplayStats::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderBVH{1};
int32_t* vrtxRenderBVH::ptr = &HELPER_VARIABLEvrtxRenderBVH;
int32_t* vrtxRenderBVH::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderBVH;
std::vector<std::string>  vrtxRenderBVH::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxBVHMaterial{1};
int32_t* vrtxBVHMaterial::ptr = &HELPER_VARIABLEvrtxBVHMaterial;
int32_t* vrtxBVHMaterial::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxBVHMaterial;
std::vector<std::string>  vrtxBVHMaterial::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxMaterial{0};
int32_t* vrtxMaterial::ptr = &HELPER_VARIABLEvrtxMaterial;
int32_t* vrtxMaterial::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxMaterial;
std::vector<std::string>  vrtxMaterial::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxDomainEpsilon{-1.762063f};
float* vrtxDomainEpsilon::ptr = &HELPER_VARIABLEvrtxDomainEpsilon;
float* vrtxDomainEpsilon::unit_ptr = (float*) &HELPER_VARIABLEvrtxDomainEpsilon;
std::vector<std::string>  vrtxDomainEpsilon::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEvrtxDomainMin{-1.f, -1.f, -1.f};
float3* vrtxDomainMin::ptr = &HELPER_VARIABLEvrtxDomainMin;
float3* vrtxDomainMin::unit_ptr = (float3*) &HELPER_VARIABLEvrtxDomainMin;
std::vector<std::string>  vrtxDomainMin::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEvrtxDomainMax{1.f, 1.f, 1.f};
float3* vrtxDomainMax::ptr = &HELPER_VARIABLEvrtxDomainMax;
float3* vrtxDomainMax::unit_ptr = (float3*) &HELPER_VARIABLEvrtxDomainMax;
std::vector<std::string>  vrtxDomainMax::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxDebeerScale{0.056f};
float* vrtxDebeerScale::ptr = &HELPER_VARIABLEvrtxDebeerScale;
float* vrtxDebeerScale::unit_ptr = (float*) &HELPER_VARIABLEvrtxDebeerScale;
std::vector<std::string>  vrtxDebeerScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEvrtxDebeer{0.94902f, 0.76863f, 0.505823f};
float3* vrtxDebeer::ptr = &HELPER_VARIABLEvrtxDebeer;
float3* vrtxDebeer::unit_ptr = (float3*) &HELPER_VARIABLEvrtxDebeer;
std::vector<std::string>  vrtxDebeer::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEbvhColor{0.566f, 0.621f, 0.641f};
float3* bvhColor::ptr = &HELPER_VARIABLEbvhColor;
float3* bvhColor::unit_ptr = (float3*) &HELPER_VARIABLEbvhColor;
std::vector<std::string>  bvhColor::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEvrtxFluidColor{0.897f, 0.917f, 1.f};
float3* vrtxFluidColor::ptr = &HELPER_VARIABLEvrtxFluidColor;
float3* vrtxFluidColor::unit_ptr = (float3*) &HELPER_VARIABLEvrtxFluidColor;
std::vector<std::string>  vrtxFluidColor::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxDepth{0};
int32_t* vrtxDepth::ptr = &HELPER_VARIABLEvrtxDepth;
int32_t* vrtxDepth::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxDepth;
std::vector<std::string>  vrtxDepth::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxDepthScale{0.1f};
float* vrtxDepthScale::ptr = &HELPER_VARIABLEvrtxDepthScale;
float* vrtxDepthScale::unit_ptr = (float*) &HELPER_VARIABLEvrtxDepthScale;
std::vector<std::string>  vrtxDepthScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxWMin{0.4f};
float* vrtxWMin::ptr = &HELPER_VARIABLEvrtxWMin;
float* vrtxWMin::unit_ptr = (float*) &HELPER_VARIABLEvrtxWMin;
std::vector<std::string>  vrtxWMin::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxR{0.586f};
float* vrtxR::ptr = &HELPER_VARIABLEvrtxR;
float* vrtxR::unit_ptr = (float*) &HELPER_VARIABLEvrtxR;
std::vector<std::string>  vrtxR::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEcamera_fov{96.f};
float* camera_fov::ptr = &HELPER_VARIABLEcamera_fov;
float* camera_fov::unit_ptr = (float*) &HELPER_VARIABLEcamera_fov;
std::vector<std::string>  camera_fov::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxWMax{2.f};
float* vrtxWMax::ptr = &HELPER_VARIABLEvrtxWMax;
float* vrtxWMax::unit_ptr = (float*) &HELPER_VARIABLEvrtxWMax;
std::vector<std::string>  vrtxWMax::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxBounces{5};
int32_t* vrtxBounces::ptr = &HELPER_VARIABLEvrtxBounces;
int32_t* vrtxBounces::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxBounces;
std::vector<std::string>  vrtxBounces::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEauxScale{1.f};
float* auxScale::ptr = &HELPER_VARIABLEauxScale;
float* auxScale::unit_ptr = (float*) &HELPER_VARIABLEauxScale;
std::vector<std::string>  auxScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxIOR{1.3f};
float* vrtxIOR::ptr = &HELPER_VARIABLEvrtxIOR;
float* vrtxIOR::unit_ptr = (float*) &HELPER_VARIABLEvrtxIOR;
std::vector<std::string>  vrtxIOR::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLErenderSteps{25};
int32_t* renderSteps::ptr = &HELPER_VARIABLErenderSteps;
int32_t* renderSteps::unit_ptr = (int32_t*) &HELPER_VARIABLErenderSteps;
std::vector<std::string>  renderSteps::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEinternalLimit{40.f};
float* internalLimit::ptr = &HELPER_VARIABLEinternalLimit;
float* internalLimit::unit_ptr = (float*) &HELPER_VARIABLEinternalLimit;
std::vector<std::string>  internalLimit::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEauxCellCount{};
int32_t* auxCellCount::ptr = &HELPER_VARIABLEauxCellCount;
int32_t* auxCellCount::unit_ptr = (int32_t*) &HELPER_VARIABLEauxCellCount;
std::vector<std::string>  auxCellCount::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEaxesRender{1};
int32_t* axesRender::ptr = &HELPER_VARIABLEaxesRender;
int32_t* axesRender::unit_ptr = (int32_t*) &HELPER_VARIABLEaxesRender;
std::vector<std::string>  axesRender::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEboundsRender{1};
int32_t* boundsRender::ptr = &HELPER_VARIABLEboundsRender;
int32_t* boundsRender::unit_ptr = (int32_t*) &HELPER_VARIABLEboundsRender;
std::vector<std::string>  boundsRender::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEfloorRender{0};
int32_t* floorRender::ptr = &HELPER_VARIABLEfloorRender;
int32_t* floorRender::unit_ptr = (int32_t*) &HELPER_VARIABLEfloorRender;
std::vector<std::string>  floorRender::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEaxesScale{1.f};
float* axesScale::ptr = &HELPER_VARIABLEaxesScale;
float* axesScale::unit_ptr = (float*) &HELPER_VARIABLEaxesScale;
std::vector<std::string>  axesScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEcamera_position{125, 0, -50};
float3* camera_position::ptr = &HELPER_VARIABLEcamera_position;
value_unit<float3, SI::m>* camera_position::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEcamera_position;
std::vector<std::string>  camera_position::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLEcamera_angle{-90, 0, 90};
float3* camera_angle::ptr = &HELPER_VARIABLEcamera_angle;
float3* camera_angle::unit_ptr = (float3*) &HELPER_VARIABLEcamera_angle;
std::vector<std::string>  camera_angle::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float2 HELPER_VARIABLEcamera_resolution{1920, 1080};
float2* camera_resolution::ptr = &HELPER_VARIABLEcamera_resolution;
float2* camera_resolution::unit_ptr = (float2*) &HELPER_VARIABLEcamera_resolution;
std::vector<std::string>  camera_resolution::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEcamera_fps{60.f};
float* camera_fps::ptr = &HELPER_VARIABLEcamera_fps;
float* camera_fps::unit_ptr = (float*) &HELPER_VARIABLEcamera_fps;
std::vector<std::string>  camera_fps::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static std::string HELPER_VARIABLEgl_file{"gl.mp4"};
std::string* gl_file::ptr = &HELPER_VARIABLEgl_file;
std::string* gl_file::unit_ptr = (std::string*) &HELPER_VARIABLEgl_file;
std::vector<std::string>  gl_file::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderNormals{0};
int32_t* vrtxRenderNormals::ptr = &HELPER_VARIABLEvrtxRenderNormals;
int32_t* vrtxRenderNormals::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderNormals;
std::vector<std::string>  vrtxRenderNormals::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEapertureRadius{0.15f};
float* apertureRadius::ptr = &HELPER_VARIABLEapertureRadius;
float* apertureRadius::unit_ptr = (float*) &HELPER_VARIABLEapertureRadius;
std::vector<std::string>  apertureRadius::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEanisotropicLambda{0.980198f};
float* anisotropicLambda::ptr = &HELPER_VARIABLEanisotropicLambda;
float* anisotropicLambda::unit_ptr = (float*) &HELPER_VARIABLEanisotropicLambda;
std::vector<std::string>  anisotropicLambda::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEanisotropicNepsilon{40};
int32_t* anisotropicNepsilon::ptr = &HELPER_VARIABLEanisotropicNepsilon;
int32_t* anisotropicNepsilon::unit_ptr = (int32_t*) &HELPER_VARIABLEanisotropicNepsilon;
std::vector<std::string>  anisotropicNepsilon::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEanisotropicKs{1.f};
float* anisotropicKs::ptr = &HELPER_VARIABLEanisotropicKs;
float* anisotropicKs::unit_ptr = (float*) &HELPER_VARIABLEanisotropicKs;
std::vector<std::string>  anisotropicKs::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEanisotropicKr{3.0f};
float* anisotropicKr::ptr = &HELPER_VARIABLEanisotropicKr;
float* anisotropicKr::unit_ptr = (float*) &HELPER_VARIABLEanisotropicKr;
std::vector<std::string>  anisotropicKr::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float3 HELPER_VARIABLErender_clamp{0.f,0.f,0.f};
float3* render_clamp::ptr = &HELPER_VARIABLErender_clamp;
float3* render_clamp::unit_ptr = (float3*) &HELPER_VARIABLErender_clamp;
std::vector<std::string>  render_clamp::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEanisotropicKn{0.188806f};
float* anisotropicKn::ptr = &HELPER_VARIABLEanisotropicKn;
float* anisotropicKn::unit_ptr = (float*) &HELPER_VARIABLEanisotropicKn;
std::vector<std::string>  anisotropicKn::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEfocalDistance{100.f};
float* focalDistance::ptr = &HELPER_VARIABLEfocalDistance;
float* focalDistance::unit_ptr = (float*) &HELPER_VARIABLEfocalDistance;
std::vector<std::string>  focalDistance::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxNeighborLimit{0};
int32_t* vrtxNeighborLimit::ptr = &HELPER_VARIABLEvrtxNeighborLimit;
int32_t* vrtxNeighborLimit::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxNeighborLimit;
std::vector<std::string>  vrtxNeighborLimit::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static float HELPER_VARIABLEvrtxFluidBias{0.05f};
float* vrtxFluidBias::ptr = &HELPER_VARIABLEvrtxFluidBias;
float* vrtxFluidBias::unit_ptr = (float*) &HELPER_VARIABLEvrtxFluidBias;
std::vector<std::string>  vrtxFluidBias::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxSurfaceExtraction{0};
int32_t* vrtxSurfaceExtraction::ptr = &HELPER_VARIABLEvrtxSurfaceExtraction;
int32_t* vrtxSurfaceExtraction::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxSurfaceExtraction;
std::vector<std::string>  vrtxSurfaceExtraction::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace render_settings{
static int32_t HELPER_VARIABLEvrtxRenderMode{0};
int32_t* vrtxRenderMode::ptr = &HELPER_VARIABLEvrtxRenderMode;
int32_t* vrtxRenderMode::unit_ptr = (int32_t*) &HELPER_VARIABLEvrtxRenderMode;
std::vector<std::string>  vrtxRenderMode::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static float HELPER_VARIABLEzOrderScale{1.f};
float* zOrderScale::ptr = &HELPER_VARIABLEzOrderScale;
float* zOrderScale::unit_ptr = (float*) &HELPER_VARIABLEzOrderScale;
std::vector<std::string>  zOrderScale::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static int HELPER_VARIABLEvalid_cells{0};
int* valid_cells::ptr = &HELPER_VARIABLEvalid_cells;
int* valid_cells::unit_ptr = (int*) &HELPER_VARIABLEvalid_cells;
std::vector<std::string>  valid_cells::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static int HELPER_VARIABLEresort_algorithm{0};
int* resort_algorithm::ptr = &HELPER_VARIABLEresort_algorithm;
int* resort_algorithm::unit_ptr = (int*) &HELPER_VARIABLEresort_algorithm;
std::vector<std::string>  resort_algorithm::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static int HELPER_VARIABLEauxCells{0};
int* auxCells::ptr = &HELPER_VARIABLEauxCells;
int* auxCells::unit_ptr = (int*) &HELPER_VARIABLEauxCells;
std::vector<std::string>  auxCells::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static int HELPER_VARIABLEauxCollisions{0};
int* auxCollisions::ptr = &HELPER_VARIABLEauxCollisions;
int* auxCollisions::unit_ptr = (int*) &HELPER_VARIABLEauxCollisions;
std::vector<std::string>  auxCollisions::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static std::vector<int32_t> HELPER_VARIABLEoccupiedCells{};
std::vector<int32_t>* occupiedCells::ptr = &HELPER_VARIABLEoccupiedCells;
std::vector<int32_t>* occupiedCells::unit_ptr = (std::vector<int32_t>*) &HELPER_VARIABLEoccupiedCells;
std::vector<std::string>  occupiedCells::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace resort{
static int HELPER_VARIABLEcollision_cells{0};
int* collision_cells::ptr = &HELPER_VARIABLEcollision_cells;
int* collision_cells::unit_ptr = (int*) &HELPER_VARIABLEcollision_cells;
std::vector<std::string>  collision_cells::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace rigid_volumes{
static std::vector<rigidVolume> HELPER_VARIABLErigidVolumes{};
std::vector<rigidVolume>* rigidVolumes::ptr = &HELPER_VARIABLErigidVolumes;
std::vector<rigidVolume>* rigidVolumes::unit_ptr = (std::vector<rigidVolume>*) &HELPER_VARIABLErigidVolumes;
std::vector<std::string>  rigidVolumes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace rigid_volumes{
static float HELPER_VARIABLEbeta{0.1f};
float* beta::ptr = &HELPER_VARIABLEbeta;
float* beta::unit_ptr = (float*) &HELPER_VARIABLEbeta;
std::vector<std::string>  beta::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace rigid_volumes{
static float HELPER_VARIABLEgamma{0.7f};
float* gamma::ptr = &HELPER_VARIABLEgamma;
float* gamma::unit_ptr = (float*) &HELPER_VARIABLEgamma;
std::vector<std::string>  gamma::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace rtxScene{
static std::vector<rtxSphere> HELPER_VARIABLErtxSpheres{};
std::vector<rtxSphere>* rtxSpheres::ptr = &HELPER_VARIABLErtxSpheres;
std::vector<rtxSphere>* rtxSpheres::unit_ptr = (std::vector<rtxSphere>*) &HELPER_VARIABLErtxSpheres;
std::vector<std::string>  rtxSpheres::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace rtxScene{
static std::vector<rtxBox> HELPER_VARIABLErtxBoxes{};
std::vector<rtxBox>* rtxBoxes::ptr = &HELPER_VARIABLErtxBoxes;
std::vector<rtxBox>* rtxBoxes::unit_ptr = (std::vector<rtxBox>*) &HELPER_VARIABLErtxBoxes;
std::vector<std::string>  rtxBoxes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEboundaryObject{""};
std::string* boundaryObject::ptr = &HELPER_VARIABLEboundaryObject;
std::string* boundaryObject::unit_ptr = (std::string*) &HELPER_VARIABLEboundaryObject;
std::vector<std::string>  boundaryObject::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static float HELPER_VARIABLELUTOffset{0.f};
float* LUTOffset::ptr = &HELPER_VARIABLELUTOffset;
float* LUTOffset::unit_ptr = (float*) &HELPER_VARIABLELUTOffset;
std::vector<std::string>  LUTOffset::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static float HELPER_VARIABLEboundaryDampening{0.97f};
float* boundaryDampening::ptr = &HELPER_VARIABLEboundaryDampening;
float* boundaryDampening::unit_ptr = (float*) &HELPER_VARIABLEboundaryDampening;
std::vector<std::string>  boundaryDampening::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static float HELPER_VARIABLEtimestep_min{0.001f};
float* timestep_min::ptr = &HELPER_VARIABLEtimestep_min;
value_unit<float, SI::s>* timestep_min::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep_min;
std::vector<std::string>  timestep_min::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static float4 HELPER_VARIABLEexternal_force{0.f,0.f,-9.81f,0.f};
float4* external_force::ptr = &HELPER_VARIABLEexternal_force;
value_unit<float4, SI::acceleration>* external_force::unit_ptr = (value_unit<float4, SI::acceleration>*) &HELPER_VARIABLEexternal_force;
std::vector<std::string>  external_force::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static int32_t HELPER_VARIABLEdensitySteps{10u};
int32_t* densitySteps::ptr = &HELPER_VARIABLEdensitySteps;
int32_t* densitySteps::unit_ptr = (int32_t*) &HELPER_VARIABLEdensitySteps;
std::vector<std::string>  densitySteps::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEdebugRegex{""};
std::string* debugRegex::ptr = &HELPER_VARIABLEdebugRegex;
std::string* debugRegex::unit_ptr = (std::string*) &HELPER_VARIABLEdebugRegex;
std::vector<std::string>  debugRegex::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static uint32_t HELPER_VARIABLEmlm_schemes{UINT_MAX};
uint32_t* mlm_schemes::ptr = &HELPER_VARIABLEmlm_schemes;
uint32_t* mlm_schemes::unit_ptr = (uint32_t*) &HELPER_VARIABLEmlm_schemes;
std::vector<std::string>  mlm_schemes::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEhostRegex{""};
std::string* hostRegex::ptr = &HELPER_VARIABLEhostRegex;
std::string* hostRegex::unit_ptr = (std::string*) &HELPER_VARIABLEhostRegex;
std::vector<std::string>  hostRegex::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEdeviceRegex{""};
std::string* deviceRegex::ptr = &HELPER_VARIABLEdeviceRegex;
std::string* deviceRegex::unit_ptr = (std::string*) &HELPER_VARIABLEdeviceRegex;
std::vector<std::string>  deviceRegex::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static float HELPER_VARIABLEtimestep_max{0.01f};
float* timestep_max::ptr = &HELPER_VARIABLEtimestep_max;
value_unit<float, SI::s>* timestep_max::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep_max;
std::vector<std::string>  timestep_max::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static int32_t HELPER_VARIABLEneighborlimit{150u};
int32_t* neighborlimit::ptr = &HELPER_VARIABLEneighborlimit;
int32_t* neighborlimit::unit_ptr = (int32_t*) &HELPER_VARIABLEneighborlimit;
std::vector<std::string>  neighborlimit::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEdomainWalls{"x+-y+-z+-"};
std::string* domainWalls::ptr = &HELPER_VARIABLEdomainWalls;
std::string* domainWalls::unit_ptr = (std::string*) &HELPER_VARIABLEdomainWalls;
std::vector<std::string>  domainWalls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static uint32_t HELPER_VARIABLEhash_entries{UINT_MAX};
uint32_t* hash_entries::ptr = &HELPER_VARIABLEhash_entries;
uint32_t* hash_entries::unit_ptr = (uint32_t*) &HELPER_VARIABLEhash_entries;
std::vector<std::string>  hash_entries::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static int32_t HELPER_VARIABLEmax_numptcls{1000000u};
int32_t* max_numptcls::ptr = &HELPER_VARIABLEmax_numptcls;
int32_t* max_numptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEmax_numptcls;
std::vector<std::string>  max_numptcls::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace simulation_settings{
static std::string HELPER_VARIABLEdumpFile{"simulation.dump"};
std::string* dumpFile::ptr = &HELPER_VARIABLEdumpFile;
std::string* dumpFile::unit_ptr = (std::string*) &HELPER_VARIABLEdumpFile;
std::vector<std::string>  dumpFile::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static uint32_t HELPER_VARIABLEsupport_current_iteration{0};
uint32_t* support_current_iteration::ptr = &HELPER_VARIABLEsupport_current_iteration;
uint32_t* support_current_iteration::unit_ptr = (uint32_t*) &HELPER_VARIABLEsupport_current_iteration;
std::vector<std::string>  support_current_iteration::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static int32_t HELPER_VARIABLEadjusted_particles{0};
int32_t* adjusted_particles::ptr = &HELPER_VARIABLEadjusted_particles;
int32_t* adjusted_particles::unit_ptr = (int32_t*) &HELPER_VARIABLEadjusted_particles;
std::vector<std::string>  adjusted_particles::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static float HELPER_VARIABLEomega{0.97f};
float* omega::ptr = &HELPER_VARIABLEomega;
float* omega::unit_ptr = (float*) &HELPER_VARIABLEomega;
std::vector<std::string>  omega::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static int32_t HELPER_VARIABLEtarget_neighbors{0};
int32_t* target_neighbors::ptr = &HELPER_VARIABLEtarget_neighbors;
int32_t* target_neighbors::unit_ptr = (int32_t*) &HELPER_VARIABLEtarget_neighbors;
std::vector<std::string>  target_neighbors::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static int32_t HELPER_VARIABLEsupport_leeway{0};
int32_t* support_leeway::ptr = &HELPER_VARIABLEsupport_leeway;
int32_t* support_leeway::unit_ptr = (int32_t*) &HELPER_VARIABLEsupport_leeway;
std::vector<std::string>  support_leeway::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static int32_t HELPER_VARIABLEoverhead_size{0};
int32_t* overhead_size::ptr = &HELPER_VARIABLEoverhead_size;
int32_t* overhead_size::unit_ptr = (int32_t*) &HELPER_VARIABLEoverhead_size;
std::vector<std::string>  overhead_size::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace support{
static int32_t HELPER_VARIABLEerror_factor{3};
int32_t* error_factor::ptr = &HELPER_VARIABLEerror_factor;
int32_t* error_factor::unit_ptr = (int32_t*) &HELPER_VARIABLEerror_factor;
std::vector<std::string>  error_factor::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static float HELPER_VARIABLEsurface_levelLimit{-20.f};
float* surface_levelLimit::ptr = &HELPER_VARIABLEsurface_levelLimit;
value_unit<float, SI::m>* surface_levelLimit::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEsurface_levelLimit;
std::vector<std::string>  surface_levelLimit::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static int32_t HELPER_VARIABLEsurface_neighborLimit{40};
int32_t* surface_neighborLimit::ptr = &HELPER_VARIABLEsurface_neighborLimit;
int32_t* surface_neighborLimit::unit_ptr = (int32_t*) &HELPER_VARIABLEsurface_neighborLimit;
std::vector<std::string>  surface_neighborLimit::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static float HELPER_VARIABLEsurface_phiMin{0.f};
float* surface_phiMin::ptr = &HELPER_VARIABLEsurface_phiMin;
value_unit<float, SI::m>* surface_phiMin::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEsurface_phiMin;
std::vector<std::string>  surface_phiMin::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static float HELPER_VARIABLEsurface_phiChange{0.f};
float* surface_phiChange::ptr = &HELPER_VARIABLEsurface_phiChange;
float* surface_phiChange::unit_ptr = (float*) &HELPER_VARIABLEsurface_phiChange;
std::vector<std::string>  surface_phiChange::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static float3 HELPER_VARIABLEsurface_distanceFieldDistances{0.f,0.f,1.5f};
float3* surface_distanceFieldDistances::ptr = &HELPER_VARIABLEsurface_distanceFieldDistances;
value_unit<float3, SI::m>* surface_distanceFieldDistances::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEsurface_distanceFieldDistances;
std::vector<std::string>  surface_distanceFieldDistances::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace surfaceDistance{
static int32_t HELPER_VARIABLEsurface_iterations{0};
int32_t* surface_iterations::ptr = &HELPER_VARIABLEsurface_iterations;
int32_t* surface_iterations::unit_ptr = (int32_t*) &HELPER_VARIABLEsurface_iterations;
std::vector<std::string>  surface_iterations::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace vorticitySettings{
static float HELPER_VARIABLEvorticityCoeff{0.05f};
float* vorticityCoeff::ptr = &HELPER_VARIABLEvorticityCoeff;
float* vorticityCoeff::unit_ptr = (float*) &HELPER_VARIABLEvorticityCoeff;
std::vector<std::string>  vorticityCoeff::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace vorticitySettings{
static float HELPER_VARIABLEintertiaInverse{0.5f};
float* intertiaInverse::ptr = &HELPER_VARIABLEintertiaInverse;
float* intertiaInverse::unit_ptr = (float*) &HELPER_VARIABLEintertiaInverse;
std::vector<std::string>  intertiaInverse::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

namespace vorticitySettings{
static float HELPER_VARIABLEviscosityOmega{0.1f};
float* viscosityOmega::ptr = &HELPER_VARIABLEviscosityOmega;
float* viscosityOmega::unit_ptr = (float*) &HELPER_VARIABLEviscosityOmega;
std::vector<std::string>  viscosityOmega::getPresets(){
	std::vector<std::string> presets ;
	return presets;
}
}

}

std::tuple<parameters::modules::adaptive, parameters::adaptive::adaptivityScaling, parameters::adaptive::adaptivityThreshold, parameters::adaptive::adaptivityGamma, parameters::adaptive::resolution, parameters::adaptive::useVolume, parameters::adaptive::minVolume, parameters::adaptive::detailedAdaptiveStatistics, parameters::adaptive::ratio, parameters::adaptive::blendSteps, parameters::adaptive::delay, parameters::adaptive::splitPtcls, parameters::adaptive::blendedPtcls, parameters::adaptive::mergedPtcls, parameters::adaptive::sharedPtcls, parameters::modules::pressure, parameters::modules::volumeBoundary, parameters::modules::xsph, parameters::modules::drag, parameters::modules::viscosity, parameters::modules::tension, parameters::modules::vorticity, parameters::particle_settings::monaghan_viscosity, parameters::particle_settings::boundaryViscosity, parameters::particle_settings::xsph_viscosity, parameters::particle_settings::rigidAdhesion_akinci, parameters::particle_settings::boundaryAdhesion_akinci, parameters::particle_settings::tension_akinci, parameters::particle_settings::air_velocity, parameters::vorticitySettings::intertiaInverse, parameters::vorticitySettings::viscosityOmega, parameters::vorticitySettings::vorticityCoeff, parameters::simulation_settings::external_force, parameters::simulation_settings::timestep_min, parameters::simulation_settings::timestep_max, parameters::modules::movingBoundaries, parameters::simulation_settings::boundaryDampening, parameters::simulation_settings::LUTOffset, parameters::simulation_settings::boundaryObject, parameters::simulation_settings::domainWalls, parameters::simulation_settings::neighborlimit, parameters::moving_plane::movingPlanes, parameters::boundary_volumes::volumeBoundaryCounter, parameters::boundary_volumes::boundaryVolumes, parameters::modules::debug, parameters::internal::neighborhood_kind, parameters::internal::dumpNextframe, parameters::internal::dumpForSSSPH, parameters::internal::target, parameters::internal::hash_size, parameters::internal::cell_order, parameters::internal::cell_structure, parameters::internal::num_ptcls, parameters::internal::boundaryCounter, parameters::internal::boundaryLUTSize, parameters::internal::frame, parameters::internal::max_velocity, parameters::internal::min_domain, parameters::internal::max_domain, parameters::internal::min_coord, parameters::internal::max_coord, parameters::internal::cell_size, parameters::internal::grid_size, parameters::internal::ptcl_spacing, parameters::internal::ptcl_support, parameters::internal::config_file, parameters::internal::config_folder, parameters::internal::working_directory, parameters::internal::build_directory, parameters::internal::source_directory, parameters::internal::binary_directory, parameters::internal::timestep, parameters::internal::simulationTime, parameters::particle_settings::radius, parameters::simulation_settings::dumpFile, parameters::simulation_settings::max_numptcls, parameters::simulation_settings::hash_entries, parameters::simulation_settings::mlm_schemes, parameters::simulation_settings::deviceRegex, parameters::simulation_settings::hostRegex, parameters::simulation_settings::debugRegex, parameters::modules::density, parameters::particle_settings::rest_density, parameters::simulation_settings::densitySteps, parameters::dfsph_settings::densityError, parameters::dfsph_settings::divergenceError, parameters::dfsph_settings::densitySolverIterations, parameters::dfsph_settings::divergenceSolverIterations, parameters::dfsph_settings::densityEta, parameters::dfsph_settings::divergenceEta, parameters::iisph_settings::density_error, parameters::iisph_settings::iterations, parameters::iisph_settings::eta, parameters::iisph_settings::jacobi_omega, parameters::modules::particleCleanUp, parameters::modules::volumeInlets, parameters::modules::volumeOutlets, parameters::particle_volumes::particleVolumes, parameters::rigid_volumes::gamma, parameters::rigid_volumes::beta, parameters::rigid_volumes::rigidVolumes, parameters::particleSets::particleSets, parameters::inlet_volumes::inletVolumes, parameters::outlet_volumes::volumeOutletCounter, parameters::outlet_volumes::volumeOutletTime, parameters::outlet_volumes::outletVolumes, parameters::modules::neighborhood, parameters::modules::neighborSorting, parameters::rtxScene::rtxSpheres, parameters::rtxScene::rtxBoxes, parameters::modules::rayTracing, parameters::modules::anisotropicSurface, parameters::modules::renderMode, parameters::render_settings::apertureRadius, parameters::render_settings::anisotropicLambda, parameters::render_settings::anisotropicNepsilon, parameters::render_settings::anisotropicKs, parameters::render_settings::anisotropicKr, parameters::render_settings::anisotropicKn, parameters::render_settings::focalDistance, parameters::render_settings::vrtxNeighborLimit, parameters::render_settings::vrtxFluidBias, parameters::render_settings::vrtxSurfaceExtraction, parameters::render_settings::vrtxRenderMode, parameters::render_settings::vrtxRenderGrid, parameters::render_settings::vrtxRenderFluid, parameters::render_settings::vrtxRenderSurface, parameters::render_settings::vrtxDisplayStats, parameters::render_settings::vrtxRenderBVH, parameters::render_settings::vrtxBVHMaterial, parameters::render_settings::vrtxRenderNormals, parameters::render_settings::vrtxMaterial, parameters::render_settings::vrtxDomainEpsilon, parameters::render_settings::vrtxDomainMin, parameters::render_settings::vrtxDomainMax, parameters::render_settings::vrtxDebeerScale, parameters::render_settings::vrtxDebeer, parameters::render_settings::bvhColor, parameters::render_settings::vrtxFluidColor, parameters::render_settings::vrtxDepth, parameters::render_settings::vrtxDepthScale, parameters::render_settings::vrtxWMin, parameters::render_settings::vrtxR, parameters::render_settings::camera_fov, parameters::render_settings::vrtxWMax, parameters::render_settings::vrtxBounces, parameters::render_settings::auxScale, parameters::render_settings::vrtxIOR, parameters::render_settings::renderSteps, parameters::render_settings::internalLimit, parameters::render_settings::auxCellCount, parameters::render_settings::axesRender, parameters::render_settings::boundsRender, parameters::render_settings::floorRender, parameters::render_settings::axesScale, parameters::render_settings::render_clamp, parameters::render_settings::camera_position, parameters::render_settings::camera_angle, parameters::render_settings::camera_resolution, parameters::render_settings::camera_fps, parameters::render_settings::gl_file, parameters::color_map::transfer_mode, parameters::color_map::mapping_mode, parameters::color_map::vectorMode, parameters::color_map::visualizeDirection, parameters::color_map::vectorScale, parameters::color_map::vectorScaling, parameters::color_map::render_min, parameters::color_map::render_max, parameters::color_map::transfer_fn, parameters::color_map::pruneVoxel, parameters::color_map::mapping_fn, parameters::color_map::render_auto, parameters::color_map::map_flipped, parameters::color_map::render_buffer, parameters::color_map::render_colormap, parameters::modules::sorting, parameters::modules::hash_width, parameters::modules::alembic_export, parameters::modules::error_checking, parameters::modules::gl_record, parameters::modules::launch_cfg, parameters::modules::regex_cfg, parameters::resort::auxCells, parameters::resort::auxCollisions, parameters::resort::resort_algorithm, parameters::resort::valid_cells, parameters::resort::zOrderScale, parameters::resort::collision_cells, parameters::resort::occupiedCells, parameters::alembic::alembic_file_name, parameters::alembic::alembic_fps, parameters::modules::support, parameters::support::support_current_iteration, parameters::support::adjusted_particles, parameters::support::omega, parameters::support::target_neighbors, parameters::support::support_leeway, parameters::support::overhead_size, parameters::support::error_factor, parameters::modules::surfaceDistance, parameters::modules::surfaceDetection, parameters::surfaceDistance::surface_levelLimit, parameters::surfaceDistance::surface_neighborLimit, parameters::surfaceDistance::surface_phiMin, parameters::surfaceDistance::surface_phiChange, parameters::surfaceDistance::surface_distanceFieldDistances, parameters::surfaceDistance::surface_iterations> uniforms_list;
