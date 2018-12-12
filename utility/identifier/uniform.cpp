#include <utility/identifier/uniform.h>

static bool HELPER_VARIABLEadaptive{true};
bool* parameters::adaptive::ptr = &HELPER_VARIABLEadaptive;
bool* parameters::adaptive::unit_ptr = (bool*) &HELPER_VARIABLEadaptive;

static float HELPER_VARIABLEadaptive_resolution{16.f};
float* parameters::adaptive_resolution::ptr = &HELPER_VARIABLEadaptive_resolution;
float* parameters::adaptive_resolution::unit_ptr = (float*) &HELPER_VARIABLEadaptive_resolution;

static float HELPER_VARIABLEadaptiveRatio{1.f};
float* parameters::adaptiveRatio::ptr = &HELPER_VARIABLEadaptiveRatio;
float* parameters::adaptiveRatio::unit_ptr = (float*) &HELPER_VARIABLEadaptiveRatio;

static float HELPER_VARIABLEblendsteps{10.f};
float* parameters::blendsteps::ptr = &HELPER_VARIABLEblendsteps;
float* parameters::blendsteps::unit_ptr = (float*) &HELPER_VARIABLEblendsteps;

static float HELPER_VARIABLEadaptive_delay{1.f};
float* parameters::adaptive_delay::ptr = &HELPER_VARIABLEadaptive_delay;
value_unit<float, SI::s>* parameters::adaptive_delay::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEadaptive_delay;

static int32_t HELPER_VARIABLEsplit_ptcls{0};
int32_t* parameters::split_ptcls::ptr = &HELPER_VARIABLEsplit_ptcls;
int32_t* parameters::split_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEsplit_ptcls;

static int32_t HELPER_VARIABLEblend_ptcls{0};
int32_t* parameters::blend_ptcls::ptr = &HELPER_VARIABLEblend_ptcls;
int32_t* parameters::blend_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEblend_ptcls;

static int32_t HELPER_VARIABLEmerged_ptcls{0};
int32_t* parameters::merged_ptcls::ptr = &HELPER_VARIABLEmerged_ptcls;
int32_t* parameters::merged_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEmerged_ptcls;

static int32_t HELPER_VARIABLEshared_ptcls{0};
int32_t* parameters::shared_ptcls::ptr = &HELPER_VARIABLEshared_ptcls;
int32_t* parameters::shared_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEshared_ptcls;

static std::string HELPER_VARIABLEpressure{"IISPH"};
std::string* parameters::pressure::ptr = &HELPER_VARIABLEpressure;
std::string* parameters::pressure::unit_ptr = (std::string*) &HELPER_VARIABLEpressure;

static bool HELPER_VARIABLEvolumeBoundary{true};
bool* parameters::volumeBoundary::ptr = &HELPER_VARIABLEvolumeBoundary;
bool* parameters::volumeBoundary::unit_ptr = (bool*) &HELPER_VARIABLEvolumeBoundary;

static bool HELPER_VARIABLExsph{true};
bool* parameters::xsph::ptr = &HELPER_VARIABLExsph;
bool* parameters::xsph::unit_ptr = (bool*) &HELPER_VARIABLExsph;

static std::string HELPER_VARIABLEdrag{"Gissler17"};
std::string* parameters::drag::ptr = &HELPER_VARIABLEdrag;
std::string* parameters::drag::unit_ptr = (std::string*) &HELPER_VARIABLEdrag;

static bool HELPER_VARIABLEviscosity{false};
bool* parameters::viscosity::ptr = &HELPER_VARIABLEviscosity;
bool* parameters::viscosity::unit_ptr = (bool*) &HELPER_VARIABLEviscosity;

static std::string HELPER_VARIABLEtension{"Akinci"};
std::string* parameters::tension::ptr = &HELPER_VARIABLEtension;
std::string* parameters::tension::unit_ptr = (std::string*) &HELPER_VARIABLEtension;

static std::string HELPER_VARIABLEvorticity{"Bender17"};
std::string* parameters::vorticity::ptr = &HELPER_VARIABLEvorticity;
std::string* parameters::vorticity::unit_ptr = (std::string*) &HELPER_VARIABLEvorticity;

static float HELPER_VARIABLEmonaghan_viscosity{15.f};
float* parameters::monaghan_viscosity::ptr = &HELPER_VARIABLEmonaghan_viscosity;
value_unit<float, SI::velocity>* parameters::monaghan_viscosity::unit_ptr = (value_unit<float, SI::velocity>*) &HELPER_VARIABLEmonaghan_viscosity;

static float HELPER_VARIABLEintertia_inverse{0.5f};
float* parameters::intertia_inverse::ptr = &HELPER_VARIABLEintertia_inverse;
float* parameters::intertia_inverse::unit_ptr = (float*) &HELPER_VARIABLEintertia_inverse;

static float HELPER_VARIABLEvorticity_viscosity{0.1f};
float* parameters::vorticity_viscosity::ptr = &HELPER_VARIABLEvorticity_viscosity;
float* parameters::vorticity_viscosity::unit_ptr = (float*) &HELPER_VARIABLEvorticity_viscosity;

static float HELPER_VARIABLEvorticity_nu_t{0.05f};
float* parameters::vorticity_nu_t::ptr = &HELPER_VARIABLEvorticity_nu_t;
float* parameters::vorticity_nu_t::unit_ptr = (float*) &HELPER_VARIABLEvorticity_nu_t;

static float HELPER_VARIABLExsph_viscosity{0.05f};
float* parameters::xsph_viscosity::ptr = &HELPER_VARIABLExsph_viscosity;
float* parameters::xsph_viscosity::unit_ptr = (float*) &HELPER_VARIABLExsph_viscosity;

static float HELPER_VARIABLEtension_akinci{0.015f};
float* parameters::tension_akinci::ptr = &HELPER_VARIABLEtension_akinci;
value_unit<float, SI::acceleration>* parameters::tension_akinci::unit_ptr = (value_unit<float, SI::acceleration>*) &HELPER_VARIABLEtension_akinci;

static float4 HELPER_VARIABLEair_velocity{1.f,0.f,0.f,0.f};
float4* parameters::air_velocity::ptr = &HELPER_VARIABLEair_velocity;
value_unit<float4, SI::velocity>* parameters::air_velocity::unit_ptr = (value_unit<float4, SI::velocity>*) &HELPER_VARIABLEair_velocity;

static float4 HELPER_VARIABLEexternal_force{0.f,0.f,-9.81f,0.f};
float4* parameters::external_force::ptr = &HELPER_VARIABLEexternal_force;
value_unit<float4, SI::acceleration>* parameters::external_force::unit_ptr = (value_unit<float4, SI::acceleration>*) &HELPER_VARIABLEexternal_force;

static float HELPER_VARIABLEtimestep_min{0.001f};
float* parameters::timestep_min::ptr = &HELPER_VARIABLEtimestep_min;
value_unit<float, SI::s>* parameters::timestep_min::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep_min;

static float HELPER_VARIABLEtimestep_max{0.01f};
float* parameters::timestep_max::ptr = &HELPER_VARIABLEtimestep_max;
value_unit<float, SI::s>* parameters::timestep_max::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep_max;

static bool HELPER_VARIABLEmovingBoundaries{false};
bool* parameters::movingBoundaries::ptr = &HELPER_VARIABLEmovingBoundaries;
bool* parameters::movingBoundaries::unit_ptr = (bool*) &HELPER_VARIABLEmovingBoundaries;

static float HELPER_VARIABLEboundaryDampening{0.97f};
float* parameters::boundaryDampening::ptr = &HELPER_VARIABLEboundaryDampening;
float* parameters::boundaryDampening::unit_ptr = (float*) &HELPER_VARIABLEboundaryDampening;

static std::string HELPER_VARIABLEboundaryObject{""};
std::string* parameters::boundaryObject::ptr = &HELPER_VARIABLEboundaryObject;
std::string* parameters::boundaryObject::unit_ptr = (std::string*) &HELPER_VARIABLEboundaryObject;

static std::string HELPER_VARIABLEdomainWalls{"x+-y+-z+-"};
std::string* parameters::domainWalls::ptr = &HELPER_VARIABLEdomainWalls;
std::string* parameters::domainWalls::unit_ptr = (std::string*) &HELPER_VARIABLEdomainWalls;

static int32_t HELPER_VARIABLEneighborlimit{150u};
int32_t* parameters::neighborlimit::ptr = &HELPER_VARIABLEneighborlimit;
int32_t* parameters::neighborlimit::unit_ptr = (int32_t*) &HELPER_VARIABLEneighborlimit;

static std::vector<moving_plane> HELPER_VARIABLEmoving_planes{};
std::vector<moving_plane>* parameters::moving_planes::ptr = &HELPER_VARIABLEmoving_planes;
std::vector<moving_plane>* parameters::moving_planes::unit_ptr = (std::vector<moving_plane>*) &HELPER_VARIABLEmoving_planes;

static int32_t HELPER_VARIABLEvolumeBoundaryCounter{0};
int32_t* parameters::volumeBoundaryCounter::ptr = &HELPER_VARIABLEvolumeBoundaryCounter;
int32_t* parameters::volumeBoundaryCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEvolumeBoundaryCounter;

static std::vector<boundary_volume> HELPER_VARIABLEboundary_volumes{};
std::vector<boundary_volume>* parameters::boundary_volumes::ptr = &HELPER_VARIABLEboundary_volumes;
std::vector<boundary_volume>* parameters::boundary_volumes::unit_ptr = (std::vector<boundary_volume>*) &HELPER_VARIABLEboundary_volumes;

static neighbor_list HELPER_VARIABLEneighborhood_kind{neighbor_list::constrained};
neighbor_list* parameters::neighborhood_kind::ptr = &HELPER_VARIABLEneighborhood_kind;
neighbor_list* parameters::neighborhood_kind::unit_ptr = (neighbor_list*) &HELPER_VARIABLEneighborhood_kind;

static int32_t HELPER_VARIABLEdumpNextframe{0};
int32_t* parameters::dumpNextframe::ptr = &HELPER_VARIABLEdumpNextframe;
int32_t* parameters::dumpNextframe::unit_ptr = (int32_t*) &HELPER_VARIABLEdumpNextframe;

static launch_config HELPER_VARIABLEtarget{launch_config::device};
launch_config* parameters::target::ptr = &HELPER_VARIABLEtarget;
launch_config* parameters::target::unit_ptr = (launch_config*) &HELPER_VARIABLEtarget;

static hash_length HELPER_VARIABLEhash_size{hash_length::bit_64};
hash_length* parameters::hash_size::ptr = &HELPER_VARIABLEhash_size;
hash_length* parameters::hash_size::unit_ptr = (hash_length*) &HELPER_VARIABLEhash_size;

static cell_ordering HELPER_VARIABLEcell_order{cell_ordering::z_order};
cell_ordering* parameters::cell_order::ptr = &HELPER_VARIABLEcell_order;
cell_ordering* parameters::cell_order::unit_ptr = (cell_ordering*) &HELPER_VARIABLEcell_order;

static cell_structuring HELPER_VARIABLEcell_structure{cell_structuring::hashed};
cell_structuring* parameters::cell_structure::ptr = &HELPER_VARIABLEcell_structure;
cell_structuring* parameters::cell_structure::unit_ptr = (cell_structuring*) &HELPER_VARIABLEcell_structure;

static int32_t HELPER_VARIABLEnum_ptcls{0u};
int32_t* parameters::num_ptcls::ptr = &HELPER_VARIABLEnum_ptcls;
int32_t* parameters::num_ptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEnum_ptcls;

static int32_t HELPER_VARIABLEboundaryCounter{0u};
int32_t* parameters::boundaryCounter::ptr = &HELPER_VARIABLEboundaryCounter;
int32_t* parameters::boundaryCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEboundaryCounter;

static int32_t HELPER_VARIABLEboundaryLUTSize{0u};
int32_t* parameters::boundaryLUTSize::ptr = &HELPER_VARIABLEboundaryLUTSize;
int32_t* parameters::boundaryLUTSize::unit_ptr = (int32_t*) &HELPER_VARIABLEboundaryLUTSize;

static int32_t HELPER_VARIABLEframe{0u};
int32_t* parameters::frame::ptr = &HELPER_VARIABLEframe;
int32_t* parameters::frame::unit_ptr = (int32_t*) &HELPER_VARIABLEframe;

static float HELPER_VARIABLEmax_velocity{1.f};
float* parameters::max_velocity::ptr = &HELPER_VARIABLEmax_velocity;
value_unit<float, SI::velocity>* parameters::max_velocity::unit_ptr = (value_unit<float, SI::velocity>*) &HELPER_VARIABLEmax_velocity;

static float3 HELPER_VARIABLEmin_domain{0.f,0.f,0.f};
float3* parameters::min_domain::ptr = &HELPER_VARIABLEmin_domain;
value_unit<float3, SI::m>* parameters::min_domain::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmin_domain;

static float3 HELPER_VARIABLEmax_domain{0.f,0.f,0.f};
float3* parameters::max_domain::ptr = &HELPER_VARIABLEmax_domain;
value_unit<float3, SI::m>* parameters::max_domain::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmax_domain;

static float3 HELPER_VARIABLEmin_coord{0.f,0.f,0.f};
float3* parameters::min_coord::ptr = &HELPER_VARIABLEmin_coord;
value_unit<float3, SI::m>* parameters::min_coord::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmin_coord;

static float3 HELPER_VARIABLEmax_coord{0.f,0.f,0.f};
float3* parameters::max_coord::ptr = &HELPER_VARIABLEmax_coord;
value_unit<float3, SI::m>* parameters::max_coord::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEmax_coord;

static float3 HELPER_VARIABLEcell_size{0.f,0.f,0.f};
float3* parameters::cell_size::ptr = &HELPER_VARIABLEcell_size;
value_unit<float3, SI::m>* parameters::cell_size::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEcell_size;

static int3 HELPER_VARIABLEgrid_size{0u,0u,0u};
int3* parameters::grid_size::ptr = &HELPER_VARIABLEgrid_size;
int3* parameters::grid_size::unit_ptr = (int3*) &HELPER_VARIABLEgrid_size;

static float HELPER_VARIABLEptcl_spacing{0.f};
float* parameters::ptcl_spacing::ptr = &HELPER_VARIABLEptcl_spacing;
value_unit<float, SI::m>* parameters::ptcl_spacing::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEptcl_spacing;

static float HELPER_VARIABLEptcl_support{0.f};
float* parameters::ptcl_support::ptr = &HELPER_VARIABLEptcl_support;
value_unit<float, SI::m>* parameters::ptcl_support::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEptcl_support;

static std::string HELPER_VARIABLEconfig_file{"DamBreakObstacle.json"};
std::string* parameters::config_file::ptr = &HELPER_VARIABLEconfig_file;
std::string* parameters::config_file::unit_ptr = (std::string*) &HELPER_VARIABLEconfig_file;

static std::string HELPER_VARIABLEconfig_folder{"D:/DamBreak"};
std::string* parameters::config_folder::ptr = &HELPER_VARIABLEconfig_folder;
std::string* parameters::config_folder::unit_ptr = (std::string*) &HELPER_VARIABLEconfig_folder;

static std::string HELPER_VARIABLEworking_directory{"D:/DamBreak"};
std::string* parameters::working_directory::ptr = &HELPER_VARIABLEworking_directory;
std::string* parameters::working_directory::unit_ptr = (std::string*) &HELPER_VARIABLEworking_directory;

static std::string HELPER_VARIABLEbuild_directory{"D:/DamBreak"};
std::string* parameters::build_directory::ptr = &HELPER_VARIABLEbuild_directory;
std::string* parameters::build_directory::unit_ptr = (std::string*) &HELPER_VARIABLEbuild_directory;

static std::string HELPER_VARIABLEsource_directory{"D:/DamBreak"};
std::string* parameters::source_directory::ptr = &HELPER_VARIABLEsource_directory;
std::string* parameters::source_directory::unit_ptr = (std::string*) &HELPER_VARIABLEsource_directory;

static std::string HELPER_VARIABLEbinary_directory{"D:/DamBreak"};
std::string* parameters::binary_directory::ptr = &HELPER_VARIABLEbinary_directory;
std::string* parameters::binary_directory::unit_ptr = (std::string*) &HELPER_VARIABLEbinary_directory;

static float HELPER_VARIABLEtimestep{0.f};
float* parameters::timestep::ptr = &HELPER_VARIABLEtimestep;
value_unit<float, SI::s>* parameters::timestep::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEtimestep;

static float HELPER_VARIABLEsimulationTime{0.f};
float* parameters::simulationTime::ptr = &HELPER_VARIABLEsimulationTime;
value_unit<float, SI::s>* parameters::simulationTime::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEsimulationTime;

static float HELPER_VARIABLEradius{0.5f};
float* parameters::radius::ptr = &HELPER_VARIABLEradius;
value_unit<float, SI::m>* parameters::radius::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEradius;

static std::string HELPER_VARIABLEdumpFile{"simulation.dump"};
std::string* parameters::dumpFile::ptr = &HELPER_VARIABLEdumpFile;
std::string* parameters::dumpFile::unit_ptr = (std::string*) &HELPER_VARIABLEdumpFile;

static int32_t HELPER_VARIABLEmax_numptcls{1000000u};
int32_t* parameters::max_numptcls::ptr = &HELPER_VARIABLEmax_numptcls;
int32_t* parameters::max_numptcls::unit_ptr = (int32_t*) &HELPER_VARIABLEmax_numptcls;

static uint32_t HELPER_VARIABLEhash_entries{UINT_MAX};
uint32_t* parameters::hash_entries::ptr = &HELPER_VARIABLEhash_entries;
uint32_t* parameters::hash_entries::unit_ptr = (uint32_t*) &HELPER_VARIABLEhash_entries;

static uint32_t HELPER_VARIABLEmlm_schemes{UINT_MAX};
uint32_t* parameters::mlm_schemes::ptr = &HELPER_VARIABLEmlm_schemes;
uint32_t* parameters::mlm_schemes::unit_ptr = (uint32_t*) &HELPER_VARIABLEmlm_schemes;

static std::string HELPER_VARIABLEdeviceRegex{""};
std::string* parameters::deviceRegex::ptr = &HELPER_VARIABLEdeviceRegex;
std::string* parameters::deviceRegex::unit_ptr = (std::string*) &HELPER_VARIABLEdeviceRegex;

static std::string HELPER_VARIABLEhostRegex{""};
std::string* parameters::hostRegex::ptr = &HELPER_VARIABLEhostRegex;
std::string* parameters::hostRegex::unit_ptr = (std::string*) &HELPER_VARIABLEhostRegex;

static std::string HELPER_VARIABLEdebugRegex{""};
std::string* parameters::debugRegex::ptr = &HELPER_VARIABLEdebugRegex;
std::string* parameters::debugRegex::unit_ptr = (std::string*) &HELPER_VARIABLEdebugRegex;

static std::string HELPER_VARIABLEdensity{"standard"};
std::string* parameters::density::ptr = &HELPER_VARIABLEdensity;
std::string* parameters::density::unit_ptr = (std::string*) &HELPER_VARIABLEdensity;

static float HELPER_VARIABLErest_density{998.f};
float* parameters::rest_density::ptr = &HELPER_VARIABLErest_density;
value_unit<float, SI::density>* parameters::rest_density::unit_ptr = (value_unit<float, SI::density>*) &HELPER_VARIABLErest_density;

static int32_t HELPER_VARIABLEdensitySteps{10u};
int32_t* parameters::densitySteps::ptr = &HELPER_VARIABLEdensitySteps;
int32_t* parameters::densitySteps::unit_ptr = (int32_t*) &HELPER_VARIABLEdensitySteps;

static float HELPER_VARIABLEdfsph_density_error{0.f};
float* parameters::dfsph_density_error::ptr = &HELPER_VARIABLEdfsph_density_error;
float* parameters::dfsph_density_error::unit_ptr = (float*) &HELPER_VARIABLEdfsph_density_error;

static int32_t HELPER_VARIABLEdfsph_iterations{0};
int32_t* parameters::dfsph_iterations::ptr = &HELPER_VARIABLEdfsph_iterations;
int32_t* parameters::dfsph_iterations::unit_ptr = (int32_t*) &HELPER_VARIABLEdfsph_iterations;

static float HELPER_VARIABLEdfsph_density_eta{0.1f};
float* parameters::dfsph_density_eta::ptr = &HELPER_VARIABLEdfsph_density_eta;
float* parameters::dfsph_density_eta::unit_ptr = (float*) &HELPER_VARIABLEdfsph_density_eta;

static float HELPER_VARIABLEdfsph_divergence_eta{0.1f};
float* parameters::dfsph_divergence_eta::ptr = &HELPER_VARIABLEdfsph_divergence_eta;
float* parameters::dfsph_divergence_eta::unit_ptr = (float*) &HELPER_VARIABLEdfsph_divergence_eta;

static float HELPER_VARIABLEdensity_error{0.f};
float* parameters::density_error::ptr = &HELPER_VARIABLEdensity_error;
float* parameters::density_error::unit_ptr = (float*) &HELPER_VARIABLEdensity_error;

static int32_t HELPER_VARIABLEiterations{0};
int32_t* parameters::iterations::ptr = &HELPER_VARIABLEiterations;
int32_t* parameters::iterations::unit_ptr = (int32_t*) &HELPER_VARIABLEiterations;

static float HELPER_VARIABLEeta{0.1f};
float* parameters::eta::ptr = &HELPER_VARIABLEeta;
float* parameters::eta::unit_ptr = (float*) &HELPER_VARIABLEeta;

static float HELPER_VARIABLEjacobi_omega{0.2f};
float* parameters::jacobi_omega::ptr = &HELPER_VARIABLEjacobi_omega;
float* parameters::jacobi_omega::unit_ptr = (float*) &HELPER_VARIABLEjacobi_omega;

static bool HELPER_VARIABLEparticleCleanUp{true};
bool* parameters::particleCleanUp::ptr = &HELPER_VARIABLEparticleCleanUp;
bool* parameters::particleCleanUp::unit_ptr = (bool*) &HELPER_VARIABLEparticleCleanUp;

static bool HELPER_VARIABLEvolumeInlets{false};
bool* parameters::volumeInlets::ptr = &HELPER_VARIABLEvolumeInlets;
bool* parameters::volumeInlets::unit_ptr = (bool*) &HELPER_VARIABLEvolumeInlets;

static bool HELPER_VARIABLEvolumeOutlets{false};
bool* parameters::volumeOutlets::ptr = &HELPER_VARIABLEvolumeOutlets;
bool* parameters::volumeOutlets::unit_ptr = (bool*) &HELPER_VARIABLEvolumeOutlets;

static std::vector<particle_volume> HELPER_VARIABLEparticle_volumes{};
std::vector<particle_volume>* parameters::particle_volumes::ptr = &HELPER_VARIABLEparticle_volumes;
std::vector<particle_volume>* parameters::particle_volumes::unit_ptr = (std::vector<particle_volume>*) &HELPER_VARIABLEparticle_volumes;

static std::vector<inlet_volume> HELPER_VARIABLEinlet_volumes{};
std::vector<inlet_volume>* parameters::inlet_volumes::ptr = &HELPER_VARIABLEinlet_volumes;
std::vector<inlet_volume>* parameters::inlet_volumes::unit_ptr = (std::vector<inlet_volume>*) &HELPER_VARIABLEinlet_volumes;

static int32_t HELPER_VARIABLEvolumeOutletCounter{0};
int32_t* parameters::volumeOutletCounter::ptr = &HELPER_VARIABLEvolumeOutletCounter;
int32_t* parameters::volumeOutletCounter::unit_ptr = (int32_t*) &HELPER_VARIABLEvolumeOutletCounter;

static float HELPER_VARIABLEvolumeOutletTime{-1.f};
float* parameters::volumeOutletTime::ptr = &HELPER_VARIABLEvolumeOutletTime;
value_unit<float, SI::s>* parameters::volumeOutletTime::unit_ptr = (value_unit<float, SI::s>*) &HELPER_VARIABLEvolumeOutletTime;

static std::vector<outlet_volume> HELPER_VARIABLEoutlet_volumes{};
std::vector<outlet_volume>* parameters::outlet_volumes::ptr = &HELPER_VARIABLEoutlet_volumes;
std::vector<outlet_volume>* parameters::outlet_volumes::unit_ptr = (std::vector<outlet_volume>*) &HELPER_VARIABLEoutlet_volumes;

static std::string HELPER_VARIABLEneighborhood{"constrained"};
std::string* parameters::neighborhood::ptr = &HELPER_VARIABLEneighborhood;
std::string* parameters::neighborhood::unit_ptr = (std::string*) &HELPER_VARIABLEneighborhood;

static float3 HELPER_VARIABLErender_clamp{0.f,0.f,0.f};
float3* parameters::render_clamp::ptr = &HELPER_VARIABLErender_clamp;
float3* parameters::render_clamp::unit_ptr = (float3*) &HELPER_VARIABLErender_clamp;

static float3 HELPER_VARIABLEcamera_position{125, 0, -50};
float3* parameters::camera_position::ptr = &HELPER_VARIABLEcamera_position;
value_unit<float3, SI::m>* parameters::camera_position::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEcamera_position;

static float3 HELPER_VARIABLEcamera_angle{-90, 0, 90};
float3* parameters::camera_angle::ptr = &HELPER_VARIABLEcamera_angle;
float3* parameters::camera_angle::unit_ptr = (float3*) &HELPER_VARIABLEcamera_angle;

static float2 HELPER_VARIABLEcamera_resolution{1920, 1080};
float2* parameters::camera_resolution::ptr = &HELPER_VARIABLEcamera_resolution;
float2* parameters::camera_resolution::unit_ptr = (float2*) &HELPER_VARIABLEcamera_resolution;

static float HELPER_VARIABLEcamera_fps{60.f};
float* parameters::camera_fps::ptr = &HELPER_VARIABLEcamera_fps;
float* parameters::camera_fps::unit_ptr = (float*) &HELPER_VARIABLEcamera_fps;

static std::string HELPER_VARIABLEgl_file{"gl.mp4"};
std::string* parameters::gl_file::ptr = &HELPER_VARIABLEgl_file;
std::string* parameters::gl_file::unit_ptr = (std::string*) &HELPER_VARIABLEgl_file;

static float HELPER_VARIABLErender_min{0.f};
float* parameters::render_min::ptr = &HELPER_VARIABLErender_min;
float* parameters::render_min::unit_ptr = (float*) &HELPER_VARIABLErender_min;

static float HELPER_VARIABLErender_max{1.f};
float* parameters::render_max::ptr = &HELPER_VARIABLErender_max;
float* parameters::render_max::unit_ptr = (float*) &HELPER_VARIABLErender_max;

static bool HELPER_VARIABLErender_auto{true};
bool* parameters::render_auto::ptr = &HELPER_VARIABLErender_auto;
bool* parameters::render_auto::unit_ptr = (bool*) &HELPER_VARIABLErender_auto;

static int HELPER_VARIABLEmap_flipped{0};
int* parameters::map_flipped::ptr = &HELPER_VARIABLEmap_flipped;
int* parameters::map_flipped::unit_ptr = (int*) &HELPER_VARIABLEmap_flipped;

static std::string HELPER_VARIABLErender_buffer{"density"};
std::string* parameters::render_buffer::ptr = &HELPER_VARIABLErender_buffer;
std::string* parameters::render_buffer::unit_ptr = (std::string*) &HELPER_VARIABLErender_buffer;

static std::string HELPER_VARIABLErender_colormap{"inferno"};
std::string* parameters::render_colormap::ptr = &HELPER_VARIABLErender_colormap;
std::string* parameters::render_colormap::unit_ptr = (std::string*) &HELPER_VARIABLErender_colormap;

static std::string HELPER_VARIABLEsorting{"hashed_cell"};
std::string* parameters::sorting::ptr = &HELPER_VARIABLEsorting;
std::string* parameters::sorting::unit_ptr = (std::string*) &HELPER_VARIABLEsorting;

static std::string HELPER_VARIABLEhash_width{"64bit"};
std::string* parameters::hash_width::ptr = &HELPER_VARIABLEhash_width;
std::string* parameters::hash_width::unit_ptr = (std::string*) &HELPER_VARIABLEhash_width;

static bool HELPER_VARIABLEalembic_export{false};
bool* parameters::alembic_export::ptr = &HELPER_VARIABLEalembic_export;
bool* parameters::alembic_export::unit_ptr = (bool*) &HELPER_VARIABLEalembic_export;

static bool HELPER_VARIABLEerror_checking{false};
bool* parameters::error_checking::ptr = &HELPER_VARIABLEerror_checking;
bool* parameters::error_checking::unit_ptr = (bool*) &HELPER_VARIABLEerror_checking;

static bool HELPER_VARIABLEgl_record{false};
bool* parameters::gl_record::ptr = &HELPER_VARIABLEgl_record;
bool* parameters::gl_record::unit_ptr = (bool*) &HELPER_VARIABLEgl_record;

static std::string HELPER_VARIABLElaunch_cfg{"device"};
std::string* parameters::launch_cfg::ptr = &HELPER_VARIABLElaunch_cfg;
std::string* parameters::launch_cfg::unit_ptr = (std::string*) &HELPER_VARIABLElaunch_cfg;

static bool HELPER_VARIABLEregex_cfg{false};
bool* parameters::regex_cfg::ptr = &HELPER_VARIABLEregex_cfg;
bool* parameters::regex_cfg::unit_ptr = (bool*) &HELPER_VARIABLEregex_cfg;

static int HELPER_VARIABLEresort_algorithm{0};
int* parameters::resort_algorithm::ptr = &HELPER_VARIABLEresort_algorithm;
int* parameters::resort_algorithm::unit_ptr = (int*) &HELPER_VARIABLEresort_algorithm;

static int HELPER_VARIABLEvalid_cells{0};
int* parameters::valid_cells::ptr = &HELPER_VARIABLEvalid_cells;
int* parameters::valid_cells::unit_ptr = (int*) &HELPER_VARIABLEvalid_cells;

static int HELPER_VARIABLEcollision_cells{0};
int* parameters::collision_cells::ptr = &HELPER_VARIABLEcollision_cells;
int* parameters::collision_cells::unit_ptr = (int*) &HELPER_VARIABLEcollision_cells;

static std::string HELPER_VARIABLEalembic_file_name{"export/alembic_$f.abc"};
std::string* parameters::alembic_file_name::ptr = &HELPER_VARIABLEalembic_file_name;
std::string* parameters::alembic_file_name::unit_ptr = (std::string*) &HELPER_VARIABLEalembic_file_name;

static int32_t HELPER_VARIABLEalembic_fps{24};
int32_t* parameters::alembic_fps::ptr = &HELPER_VARIABLEalembic_fps;
int32_t* parameters::alembic_fps::unit_ptr = (int32_t*) &HELPER_VARIABLEalembic_fps;

static std::string HELPER_VARIABLEsupport{"constrained"};
std::string* parameters::support::ptr = &HELPER_VARIABLEsupport;
std::string* parameters::support::unit_ptr = (std::string*) &HELPER_VARIABLEsupport;

static uint32_t HELPER_VARIABLEsupport_current_iteration{0};
uint32_t* parameters::support_current_iteration::ptr = &HELPER_VARIABLEsupport_current_iteration;
uint32_t* parameters::support_current_iteration::unit_ptr = (uint32_t*) &HELPER_VARIABLEsupport_current_iteration;

static int32_t HELPER_VARIABLEadjusted_particles{0};
int32_t* parameters::adjusted_particles::ptr = &HELPER_VARIABLEadjusted_particles;
int32_t* parameters::adjusted_particles::unit_ptr = (int32_t*) &HELPER_VARIABLEadjusted_particles;

static float HELPER_VARIABLEomega{0.97f};
float* parameters::omega::ptr = &HELPER_VARIABLEomega;
float* parameters::omega::unit_ptr = (float*) &HELPER_VARIABLEomega;

static int32_t HELPER_VARIABLEtarget_neighbors{0};
int32_t* parameters::target_neighbors::ptr = &HELPER_VARIABLEtarget_neighbors;
int32_t* parameters::target_neighbors::unit_ptr = (int32_t*) &HELPER_VARIABLEtarget_neighbors;

static int32_t HELPER_VARIABLEsupport_leeway{0};
int32_t* parameters::support_leeway::ptr = &HELPER_VARIABLEsupport_leeway;
int32_t* parameters::support_leeway::unit_ptr = (int32_t*) &HELPER_VARIABLEsupport_leeway;

static int32_t HELPER_VARIABLEoverhead_size{0};
int32_t* parameters::overhead_size::ptr = &HELPER_VARIABLEoverhead_size;
int32_t* parameters::overhead_size::unit_ptr = (int32_t*) &HELPER_VARIABLEoverhead_size;

static int32_t HELPER_VARIABLEerror_factor{3};
int32_t* parameters::error_factor::ptr = &HELPER_VARIABLEerror_factor;
int32_t* parameters::error_factor::unit_ptr = (int32_t*) &HELPER_VARIABLEerror_factor;

static bool HELPER_VARIABLEsurfaceDistance{true};
bool* parameters::surfaceDistance::ptr = &HELPER_VARIABLEsurfaceDistance;
bool* parameters::surfaceDistance::unit_ptr = (bool*) &HELPER_VARIABLEsurfaceDistance;

static float HELPER_VARIABLEsurface_levelLimit{-20.f};
float* parameters::surface_levelLimit::ptr = &HELPER_VARIABLEsurface_levelLimit;
value_unit<float, SI::m>* parameters::surface_levelLimit::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEsurface_levelLimit;

static int32_t HELPER_VARIABLEsurface_neighborLimit{40};
int32_t* parameters::surface_neighborLimit::ptr = &HELPER_VARIABLEsurface_neighborLimit;
int32_t* parameters::surface_neighborLimit::unit_ptr = (int32_t*) &HELPER_VARIABLEsurface_neighborLimit;

static float HELPER_VARIABLEsurface_phiMin{0.f};
float* parameters::surface_phiMin::ptr = &HELPER_VARIABLEsurface_phiMin;
value_unit<float, SI::m>* parameters::surface_phiMin::unit_ptr = (value_unit<float, SI::m>*) &HELPER_VARIABLEsurface_phiMin;

static float HELPER_VARIABLEsurface_phiChange{0.f};
float* parameters::surface_phiChange::ptr = &HELPER_VARIABLEsurface_phiChange;
float* parameters::surface_phiChange::unit_ptr = (float*) &HELPER_VARIABLEsurface_phiChange;

static float3 HELPER_VARIABLEsurface_distanceFieldDistances{4.f,4.f,4.f};
float3* parameters::surface_distanceFieldDistances::ptr = &HELPER_VARIABLEsurface_distanceFieldDistances;
value_unit<float3, SI::m>* parameters::surface_distanceFieldDistances::unit_ptr = (value_unit<float3, SI::m>*) &HELPER_VARIABLEsurface_distanceFieldDistances;

static int32_t HELPER_VARIABLEsurface_iterations{0};
int32_t* parameters::surface_iterations::ptr = &HELPER_VARIABLEsurface_iterations;
int32_t* parameters::surface_iterations::unit_ptr = (int32_t*) &HELPER_VARIABLEsurface_iterations;


std::tuple<parameters::adaptive, parameters::adaptive_resolution, parameters::adaptiveRatio, parameters::blendsteps, parameters::adaptive_delay, parameters::split_ptcls, parameters::blend_ptcls, parameters::merged_ptcls, parameters::shared_ptcls, parameters::pressure, parameters::volumeBoundary, parameters::xsph, parameters::drag, parameters::viscosity, parameters::tension, parameters::vorticity, parameters::monaghan_viscosity, parameters::intertia_inverse, parameters::vorticity_viscosity, parameters::vorticity_nu_t, parameters::xsph_viscosity, parameters::tension_akinci, parameters::air_velocity, parameters::external_force, parameters::timestep_min, parameters::timestep_max, parameters::movingBoundaries, parameters::boundaryDampening, parameters::boundaryObject, parameters::domainWalls, parameters::neighborlimit, parameters::moving_planes, parameters::volumeBoundaryCounter, parameters::boundary_volumes, parameters::neighborhood_kind, parameters::dumpNextframe, parameters::target, parameters::hash_size, parameters::cell_order, parameters::cell_structure, parameters::num_ptcls, parameters::boundaryCounter, parameters::boundaryLUTSize, parameters::frame, parameters::max_velocity, parameters::min_domain, parameters::max_domain, parameters::min_coord, parameters::max_coord, parameters::cell_size, parameters::grid_size, parameters::ptcl_spacing, parameters::ptcl_support, parameters::config_file, parameters::config_folder, parameters::working_directory, parameters::build_directory, parameters::source_directory, parameters::binary_directory, parameters::timestep, parameters::simulationTime, parameters::radius, parameters::dumpFile, parameters::max_numptcls, parameters::hash_entries, parameters::mlm_schemes, parameters::deviceRegex, parameters::hostRegex, parameters::debugRegex, parameters::density, parameters::rest_density, parameters::densitySteps, parameters::dfsph_density_error, parameters::dfsph_iterations, parameters::dfsph_density_eta, parameters::dfsph_divergence_eta, parameters::density_error, parameters::iterations, parameters::eta, parameters::jacobi_omega, parameters::particleCleanUp, parameters::volumeInlets, parameters::volumeOutlets, parameters::particle_volumes, parameters::inlet_volumes, parameters::volumeOutletCounter, parameters::volumeOutletTime, parameters::outlet_volumes, parameters::neighborhood, parameters::render_clamp, parameters::camera_position, parameters::camera_angle, parameters::camera_resolution, parameters::camera_fps, parameters::gl_file, parameters::render_min, parameters::render_max, parameters::render_auto, parameters::map_flipped, parameters::render_buffer, parameters::render_colormap, parameters::sorting, parameters::hash_width, parameters::alembic_export, parameters::error_checking, parameters::gl_record, parameters::launch_cfg, parameters::regex_cfg, parameters::resort_algorithm, parameters::valid_cells, parameters::collision_cells, parameters::alembic_file_name, parameters::alembic_fps, parameters::support, parameters::support_current_iteration, parameters::adjusted_particles, parameters::omega, parameters::target_neighbors, parameters::support_leeway, parameters::overhead_size, parameters::error_factor, parameters::surfaceDistance, parameters::surface_levelLimit, parameters::surface_neighborLimit, parameters::surface_phiMin, parameters::surface_phiChange, parameters::surface_distanceFieldDistances, parameters::surface_iterations> uniforms_list;

