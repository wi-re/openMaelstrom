#pragma once
#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility/template/nonesuch.h>
#include <utility/unit_math.h>
#include <utility>
#include <vector>

enum struct uniforms{modules_adaptive, adaptive_adaptivityScaling, adaptive_adaptivityThreshold, adaptive_adaptivityGamma, adaptive_resolution, adaptive_useVolume, adaptive_minVolume, adaptive_detailedAdaptiveStatistics, adaptive_ratio, adaptive_blendSteps, adaptive_delay, adaptive_splitPtcls, adaptive_blendedPtcls, adaptive_mergedPtcls, adaptive_sharedPtcls, modules_pressure, modules_volumeBoundary, modules_xsph, modules_drag, modules_viscosity, modules_tension, modules_vorticity, particle_settings_monaghan_viscosity, particle_settings_boundaryViscosity, particle_settings_xsph_viscosity, particle_settings_rigidAdhesion_akinci, particle_settings_boundaryAdhesion_akinci, particle_settings_tension_akinci, particle_settings_air_velocity, vorticitySettings_intertiaInverse, vorticitySettings_viscosityOmega, vorticitySettings_vorticityCoeff, simulation_settings_external_force, simulation_settings_timestep_min, simulation_settings_timestep_max, modules_movingBoundaries, simulation_settings_boundaryDampening, simulation_settings_LUTOffset, simulation_settings_boundaryObject, simulation_settings_domainWalls, simulation_settings_neighborlimit, moving_plane_movingPlanes, boundary_volumes_volumeBoundaryCounter, boundary_volumes_boundaryVolumes, modules_debug, internal_neighborhood_kind, internal_dumpNextframe, internal_dumpForSSSPH, internal_target, internal_hash_size, internal_cell_order, internal_cell_structure, internal_num_ptcls, internal_boundaryCounter, internal_boundaryLUTSize, internal_frame, internal_max_velocity, internal_min_domain, internal_max_domain, internal_min_coord, internal_max_coord, internal_cell_size, internal_grid_size, internal_ptcl_spacing, internal_ptcl_support, internal_config_file, internal_config_folder, internal_working_directory, internal_build_directory, internal_source_directory, internal_binary_directory, internal_timestep, internal_simulationTime, particle_settings_radius, simulation_settings_dumpFile, simulation_settings_max_numptcls, simulation_settings_hash_entries, simulation_settings_mlm_schemes, simulation_settings_deviceRegex, simulation_settings_hostRegex, simulation_settings_debugRegex, modules_density, particle_settings_rest_density, simulation_settings_densitySteps, dfsph_settings_densityError, dfsph_settings_divergenceError, dfsph_settings_densitySolverIterations, dfsph_settings_divergenceSolverIterations, dfsph_settings_densityEta, dfsph_settings_divergenceEta, iisph_settings_density_error, iisph_settings_iterations, iisph_settings_eta, iisph_settings_jacobi_omega, modules_particleCleanUp, modules_volumeInlets, modules_volumeOutlets, particle_volumes_particleVolumes, rigid_volumes_gamma, rigid_volumes_beta, rigid_volumes_rigidVolumes, particleSets_particleSets, inlet_volumes_inletVolumes, outlet_volumes_volumeOutletCounter, outlet_volumes_volumeOutletTime, outlet_volumes_outletVolumes, modules_neighborhood, modules_neighborSorting, rtxScene_rtxSpheres, rtxScene_rtxBoxes, modules_rayTracing, modules_anisotropicSurface, modules_renderMode, render_settings_apertureRadius, render_settings_anisotropicLambda, render_settings_anisotropicNepsilon, render_settings_anisotropicKs, render_settings_anisotropicKr, render_settings_anisotropicKn, render_settings_focalDistance, render_settings_vrtxNeighborLimit, render_settings_vrtxFluidBias, render_settings_vrtxSurfaceExtraction, render_settings_vrtxRenderMode, render_settings_vrtxRenderGrid, render_settings_vrtxRenderFluid, render_settings_vrtxRenderSurface, render_settings_vrtxDisplayStats, render_settings_vrtxRenderBVH, render_settings_vrtxBVHMaterial, render_settings_vrtxRenderNormals, render_settings_vrtxMaterial, render_settings_vrtxDomainEpsilon, render_settings_vrtxDomainMin, render_settings_vrtxDomainMax, render_settings_vrtxDebeerScale, render_settings_vrtxDebeer, render_settings_bvhColor, render_settings_vrtxFluidColor, render_settings_vrtxDepth, render_settings_vrtxDepthScale, render_settings_vrtxWMin, render_settings_vrtxR, render_settings_camera_fov, render_settings_vrtxWMax, render_settings_vrtxBounces, render_settings_auxScale, render_settings_vrtxIOR, render_settings_renderSteps, render_settings_internalLimit, render_settings_auxCellCount, render_settings_axesRender, render_settings_boundsRender, render_settings_floorRender, render_settings_axesScale, render_settings_render_clamp, render_settings_camera_position, render_settings_camera_angle, render_settings_camera_resolution, render_settings_camera_fps, render_settings_gl_file, color_map_transfer_mode, color_map_mapping_mode, color_map_vectorMode, color_map_visualizeDirection, color_map_vectorScale, color_map_vectorScaling, color_map_render_min, color_map_render_max, color_map_transfer_fn, color_map_pruneVoxel, color_map_mapping_fn, color_map_render_auto, color_map_map_flipped, color_map_render_buffer, color_map_render_colormap, modules_sorting, modules_hash_width, modules_alembic_export, modules_error_checking, modules_gl_record, modules_launch_cfg, modules_regex_cfg, resort_auxCells, resort_auxCollisions, resort_resort_algorithm, resort_valid_cells, resort_zOrderScale, resort_collision_cells, resort_occupiedCells, alembic_alembic_file_name, alembic_alembic_fps, modules_support, support_support_current_iteration, support_adjusted_particles, support_omega, support_target_neighbors, support_support_leeway, support_overhead_size, support_error_factor, modules_surfaceDistance, modules_surfaceDetection, surfaceDistance_surface_levelLimit, surfaceDistance_surface_neighborLimit, surfaceDistance_surface_phiMin, surfaceDistance_surface_phiChange, surfaceDistance_surface_distanceFieldDistances, surfaceDistance_surface_iterations};

#define PARAMETER_COUNT 208

#include <utility/identifier/resource_helper.h>

template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get(T&& = T{}) {
	return *T::ptr;
}
template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get_u(T&& = T{}) {
	return *T::unit_ptr;
}

struct movingPlane{
	using uniform_type = complex_uniform;
	
	complex_type<float3> plane_position{ "pos", {0.f,0.f,0.f}};
	complex_type<float3> plane_normal{ "norm", {0.f,0.f,0.f}};
	complex_type<float3> plane_direction{ "dir", {0.f,0.f,0.f}};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> magnitude{ "mag", 0.f};
	complex_type<float> frequency{ "freq", 0.f};
	complex_type<int32_t> index{ "idx", 0};
};


struct boundaryVolume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<float> density{ "density", 998.0f};
	complex_type<float3> position{ "position", {0.f,0.f,0.f}};
	complex_type<float3> velocity{ "velocity", {0.f,0.f,0.f}};
	complex_type<float4> angularVelocity{ "angularVelocity", {CUDART_PI_F * 0.5f,0.f,0.f,0.f}};
	complex_type<float3> angle{ "angle", {0.f,0.f,0.f}};
	complex_type<int32_t> kind{ "kind", 0};
	complex_type<std::string> animationPath{ "animationPath", ""};
};


struct particleVolume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<std::string> kind{ "kind", ""};
	complex_type<float> concentration{ "concentration", 0.f};
	complex_type<float> timeToEmit{ "timeToEmit", 0.f};
};


struct rigidVolume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<std::string> kind{ "kind", ""};
	complex_type<float> density{ "density", 1.f};
	complex_type<float3> shift{ "shift", {0.f,0.f,0.f}};
	complex_type<float> concentration{ "concentration", 0.f};
	complex_type<float> timeToEmit{ "timeToEmit", 0.f};
};


struct inletVolume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<int32_t> particles_emitted{ "ptcls", 0};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> inlet_radius{ "r", -1.f};
	complex_type<float4> emitter_velocity{ "vel", {0.f,0.f,0.f,0.f}};
};


struct outletVolume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> flowRate{ "rate", -1.f};
};


struct rtxSphere{
	using uniform_type = complex_uniform;
	
	complex_type<float> radius{ "radius", 1.f};
	complex_type<float3> position{ "position", 0.f,0.f,0.f};
	complex_type<float3> emission{ "emission", 0.f,0.f,0.f};
	complex_type<float3> color{ "color", 0.f,0.f,0.f};
	complex_type<int32_t> refl_t{ "material", 0};
};


struct rtxBox{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> maxPosition{ "maxPosition", "1.f 1.f 1.f"};
	complex_type<std::string> minPosition{ "minPosition", "0.f 0.f 0.f"};
	complex_type<float3> emission{ "emission", 0.f,0.f,0.f};
	complex_type<float3> color{ "color", 0.f,0.f,0.f};
	complex_type<int32_t> refl_t{ "material", 0};
};


namespace parameters{
namespace adaptive{

	struct sharedPtcls{
		using type = std::vector<int32_t>;
		using unit_type = std::vector<int32_t>;
		static constexpr const uniforms identifier = uniforms::adaptive_sharedPtcls;
		sharedPtcls(const type& val){*ptr = val;}
		sharedPtcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "sharedPtcls";
		static constexpr const auto description = R"()";
		static std::vector<int32_t>* ptr;
		static std::vector<int32_t>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.sharedPtcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.sharedPtcls;}		
	};
	struct adaptivityScaling{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_adaptivityScaling;
		adaptivityScaling(const type& val){*ptr = val;}
		adaptivityScaling() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "adaptivityScaling";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.adaptivityScaling";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptivityScaling;}		
	};
	struct adaptivityThreshold{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_adaptivityThreshold;
		adaptivityThreshold(const type& val){*ptr = val;}
		adaptivityThreshold() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "adaptivityThreshold";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.adaptivityThreshold";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptivityThreshold;}		
	};
	struct adaptivityGamma{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_adaptivityGamma;
		adaptivityGamma(const type& val){*ptr = val;}
		adaptivityGamma() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "adaptivityGamma";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.adaptivityGamma";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptivityGamma;}		
	};
	struct resolution{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_resolution;
		resolution(const type& val){*ptr = val;}
		resolution() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "resolution";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.resolution";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.resolution;}
		static constexpr const float min{1.f};
		static constexpr const float max{512.f};
		static constexpr const float step{1.f};		
	};
	struct useVolume{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::adaptive_useVolume;
		useVolume(const type& val){*ptr = val;}
		useVolume() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "useVolume";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.useVolume";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.useVolume;}		
	};
	struct minVolume{
		using type = float;
		using unit_type = value_unit<float, SI::volume>;
		static constexpr const uniforms identifier = uniforms::adaptive_minVolume;
		minVolume(const type& val){*ptr = val;}
		minVolume() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "minVolume";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::volume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.minVolume";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.minVolume;}		
	};
	struct detailedAdaptiveStatistics{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::adaptive_detailedAdaptiveStatistics;
		detailedAdaptiveStatistics(const type& val){*ptr = val;}
		detailedAdaptiveStatistics() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "detailedAdaptiveStatistics";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.detailedAdaptiveStatistics";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.detailedAdaptiveStatistics;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct ratio{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_ratio;
		ratio(const type& val){*ptr = val;}
		ratio() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "ratio";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.ratio";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.ratio;}		
	};
	struct blendSteps{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_blendSteps;
		blendSteps(const type& val){*ptr = val;}
		blendSteps() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "blendSteps";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.blendSteps";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.blendSteps;}		
	};
	struct delay{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::adaptive_delay;
		delay(const type& val){*ptr = val;}
		delay() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "delay";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.delay";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.delay;}		
	};
	struct splitPtcls{
		using type = std::vector<int32_t>;
		using unit_type = std::vector<int32_t>;
		static constexpr const uniforms identifier = uniforms::adaptive_splitPtcls;
		splitPtcls(const type& val){*ptr = val;}
		splitPtcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "splitPtcls";
		static constexpr const auto description = R"()";
		static std::vector<int32_t>* ptr;
		static std::vector<int32_t>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.splitPtcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.splitPtcls;}		
	};
	struct blendedPtcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::adaptive_blendedPtcls;
		blendedPtcls(const type& val){*ptr = val;}
		blendedPtcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "blendedPtcls";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.blendedPtcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.blendedPtcls;}		
	};
	struct mergedPtcls{
		using type = std::vector<int32_t>;
		using unit_type = std::vector<int32_t>;
		static constexpr const uniforms identifier = uniforms::adaptive_mergedPtcls;
		mergedPtcls(const type& val){*ptr = val;}
		mergedPtcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "mergedPtcls";
		static constexpr const auto description = R"()";
		static std::vector<int32_t>* ptr;
		static std::vector<int32_t>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "adaptive.mergedPtcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.mergedPtcls;}		
	};}
namespace alembic{

	struct alembic_fps{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::alembic_alembic_fps;
		alembic_fps(const type& val){*ptr = val;}
		alembic_fps() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "alembic_fps";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "alembic.fps";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_fps;}		
	};
	struct alembic_file_name{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::alembic_alembic_file_name;
		alembic_file_name(const type& val){*ptr = val;}
		alembic_file_name() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "alembic_file_name";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "alembic.file_name";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_file_name;}		
	};}
namespace boundary_volumes{

	struct volumeBoundaryCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::boundary_volumes_volumeBoundaryCounter;
		volumeBoundaryCounter(const type& val){*ptr = val;}
		volumeBoundaryCounter() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeBoundaryCounter";
		static constexpr const auto description = R"(Represents the number of valid boundary volumes.)";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "boundary_volumes.volumeBoundaryCounter";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryCounter;}		
	};
	struct boundaryVolumes{
		using type = std::vector<boundaryVolume>;
		using unit_type = std::vector<boundaryVolume>;
		static constexpr const uniforms identifier = uniforms::boundary_volumes_boundaryVolumes;
		boundaryVolumes(const type& val){*ptr = val;}
		boundaryVolumes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryVolumes";
		static constexpr const auto description = R"()";
		static std::vector<boundaryVolume>* ptr;
		static std::vector<boundaryVolume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "boundary_volumes.volume$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryVolumes;}		
	};}
namespace color_map{

	struct vectorMode{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::color_map_vectorMode;
		vectorMode(const type& val){*ptr = val;}
		vectorMode() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vectorMode";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.vectorMode";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vectorMode;}		
	};
	struct render_colormap{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::color_map_render_colormap;
		render_colormap(const type& val){*ptr = val;}
		render_colormap() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_colormap";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.map";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_colormap;}		
	};
	struct mapping_mode{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::color_map_mapping_mode;
		mapping_mode(const type& val){*ptr = val;}
		mapping_mode() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "mapping_mode";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.mapping_mode";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.mapping_mode;}		
	};
	struct vectorScaling{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::color_map_vectorScaling;
		vectorScaling(const type& val){*ptr = val;}
		vectorScaling() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vectorScaling";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.vectorScaling";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vectorScaling;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct pruneVoxel{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::color_map_pruneVoxel;
		pruneVoxel(const type& val){*ptr = val;}
		pruneVoxel() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "pruneVoxel";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.pruneVoxel";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.pruneVoxel;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct render_buffer{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::color_map_render_buffer;
		render_buffer(const type& val){*ptr = val;}
		render_buffer() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_buffer";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.buffer";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_buffer;}		
	};
	struct map_flipped{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::color_map_map_flipped;
		map_flipped(const type& val){*ptr = val;}
		map_flipped() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "map_flipped";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.map_flipped";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.map_flipped;}
		static constexpr const int min{0};
		static constexpr const int max{1};
		static constexpr const int step{1};		
	};
	struct render_auto{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::color_map_render_auto;
		render_auto(const type& val){*ptr = val;}
		render_auto() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_auto";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.auto";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_auto;}
		static constexpr const int min{0};
		static constexpr const int max{1};
		static constexpr const int step{1};		
	};
	struct mapping_fn{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::color_map_mapping_fn;
		mapping_fn(const type& val){*ptr = val;}
		mapping_fn() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "mapping_fn";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.mapping_fn";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.mapping_fn;}		
	};
	struct transfer_fn{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::color_map_transfer_fn;
		transfer_fn(const type& val){*ptr = val;}
		transfer_fn() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "transfer_fn";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.transfer_fn";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.transfer_fn;}		
	};
	struct transfer_mode{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::color_map_transfer_mode;
		transfer_mode(const type& val){*ptr = val;}
		transfer_mode() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "transfer_mode";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.transfer_mode";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.transfer_mode;}		
	};
	struct render_max{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::color_map_render_max;
		render_max(const type& val){*ptr = val;}
		render_max() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_max";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.max";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_max;}
		static constexpr const float min{-10.f};
		static constexpr const float max{10.f};
		static constexpr const float step{0.5f};		
	};
	struct render_min{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::color_map_render_min;
		render_min(const type& val){*ptr = val;}
		render_min() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_min";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.min";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_min;}
		static constexpr const float min{-10.f};
		static constexpr const float max{10.f};
		static constexpr const float step{0.5f};		
	};
	struct vectorScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::color_map_vectorScale;
		vectorScale(const type& val){*ptr = val;}
		vectorScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vectorScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.vectorScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vectorScale;}
		static constexpr const float min{0.f};
		static constexpr const float max{10.f};
		static constexpr const float step{0.1f};		
	};
	struct visualizeDirection{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::color_map_visualizeDirection;
		visualizeDirection(const type& val){*ptr = val;}
		visualizeDirection() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "visualizeDirection";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "color_map.visualizeDirection";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.visualizeDirection;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};}
namespace dfsph_settings{

	struct densityError{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_densityError;
		densityError(const type& val){*ptr = val;}
		densityError() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "densityError";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.densityError";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.densityError;}		
	};
	struct divergenceEta{
		using type = float;
		using unit_type = value_unit<float, SI::Hz>;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_divergenceEta;
		divergenceEta(const type& val){*ptr = val;}
		divergenceEta() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "divergenceEta";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::Hz>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.divergenceEta";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.divergenceEta;}		
	};
	struct densityEta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_densityEta;
		densityEta(const type& val){*ptr = val;}
		densityEta() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "densityEta";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.densityEta";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.densityEta;}		
	};
	struct divergenceError{
		using type = float;
		using unit_type = value_unit<float, SI::Hz>;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_divergenceError;
		divergenceError(const type& val){*ptr = val;}
		divergenceError() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "divergenceError";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::Hz>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.divergenceError";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.divergenceError;}		
	};
	struct divergenceSolverIterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_divergenceSolverIterations;
		divergenceSolverIterations(const type& val){*ptr = val;}
		divergenceSolverIterations() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "divergenceSolverIterations";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.divergenceSolverIterations";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.divergenceSolverIterations;}		
	};
	struct densitySolverIterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::dfsph_settings_densitySolverIterations;
		densitySolverIterations(const type& val){*ptr = val;}
		densitySolverIterations() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "densitySolverIterations";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "dfsph_settings.densitySolverIterations";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.densitySolverIterations;}		
	};}
namespace iisph_settings{

	struct jacobi_omega{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::iisph_settings_jacobi_omega;
		jacobi_omega(const type& val){*ptr = val;}
		jacobi_omega() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "jacobi_omega";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "iisph_settings.jacobi_omega";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.jacobi_omega;}		
	};
	struct iterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::iisph_settings_iterations;
		iterations(const type& val){*ptr = val;}
		iterations() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "iterations";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "iisph_settings.iterations";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.iterations;}		
	};
	struct eta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::iisph_settings_eta;
		eta(const type& val){*ptr = val;}
		eta() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "eta";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "iisph_settings.eta";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.eta;}		
	};
	struct density_error{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::iisph_settings_density_error;
		density_error(const type& val){*ptr = val;}
		density_error() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "density_error";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "iisph_settings.density_error";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.density_error;}		
	};}
namespace inlet_volumes{

	struct inletVolumes{
		using type = std::vector<inletVolume>;
		using unit_type = std::vector<inletVolume>;
		static constexpr const uniforms identifier = uniforms::inlet_volumes_inletVolumes;
		inletVolumes(const type& val){*ptr = val;}
		inletVolumes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "inletVolumes";
		static constexpr const auto description = R"()";
		static std::vector<inletVolume>* ptr;
		static std::vector<inletVolume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "inlet_volumes.volume$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.inletVolumes;}		
	};}
namespace internal{

	struct num_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_num_ptcls;
		num_ptcls(const type& val){*ptr = val;}
		num_ptcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "num_ptcls";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.num_ptcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.num_ptcls;}		
	};
	struct max_domain{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_max_domain;
		max_domain(const type& val){*ptr = val;}
		max_domain() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "max_domain";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.maxAABB";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_domain;}		
	};
	struct neighborhood_kind{
		using type = neighbor_list;
		using unit_type = neighbor_list;
		static constexpr const uniforms identifier = uniforms::internal_neighborhood_kind;
		neighborhood_kind(const type& val){*ptr = val;}
		neighborhood_kind() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "neighborhood_kind";
		static constexpr const auto description = R"()";
		static neighbor_list* ptr;
		static neighbor_list* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.neighborhood_kind";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.neighborhood_kind;}		
	};
	struct dumpNextframe{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_dumpNextframe;
		dumpNextframe(const type& val){*ptr = val;}
		dumpNextframe() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "dumpNextframe";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.dumpNextframe";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dumpNextframe;}		
	};
	struct dumpForSSSPH{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_dumpForSSSPH;
		dumpForSSSPH(const type& val){*ptr = val;}
		dumpForSSSPH() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "dumpForSSSPH";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.dumpForSSSPH";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dumpForSSSPH;}		
	};
	struct target{
		using type = launch_config;
		using unit_type = launch_config;
		static constexpr const uniforms identifier = uniforms::internal_target;
		target(const type& val){*ptr = val;}
		target() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "target";
		static constexpr const auto description = R"()";
		static launch_config* ptr;
		static launch_config* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.target";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.target;}		
	};
	struct hash_size{
		using type = hash_length;
		using unit_type = hash_length;
		static constexpr const uniforms identifier = uniforms::internal_hash_size;
		hash_size(const type& val){*ptr = val;}
		hash_size() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "hash_size";
		static constexpr const auto description = R"()";
		static hash_length* ptr;
		static hash_length* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.hash_size";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.hash_size;}		
	};
	struct cell_order{
		using type = cell_ordering;
		using unit_type = cell_ordering;
		static constexpr const uniforms identifier = uniforms::internal_cell_order;
		cell_order(const type& val){*ptr = val;}
		cell_order() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "cell_order";
		static constexpr const auto description = R"()";
		static cell_ordering* ptr;
		static cell_ordering* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.cell_ordering";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.cell_order;}		
	};
	struct cell_structure{
		using type = cell_structuring;
		using unit_type = cell_structuring;
		static constexpr const uniforms identifier = uniforms::internal_cell_structure;
		cell_structure(const type& val){*ptr = val;}
		cell_structure() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "cell_structure";
		static constexpr const auto description = R"()";
		static cell_structuring* ptr;
		static cell_structuring* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.cell_structure";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.cell_structure;}		
	};
	struct boundaryCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_boundaryCounter;
		boundaryCounter(const type& val){*ptr = val;}
		boundaryCounter() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryCounter";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.boundaryCounter";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryCounter;}		
	};
	struct boundaryLUTSize{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_boundaryLUTSize;
		boundaryLUTSize(const type& val){*ptr = val;}
		boundaryLUTSize() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryLUTSize";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.boundaryLUTSize";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryLUTSize;}		
	};
	struct frame{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::internal_frame;
		frame(const type& val){*ptr = val;}
		frame() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "frame";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.frame";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.frame;}		
	};
	struct max_velocity{
		using type = float;
		using unit_type = value_unit<float, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::internal_max_velocity;
		max_velocity(const type& val){*ptr = val;}
		max_velocity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "max_velocity";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::velocity>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.max_velocity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_velocity;}		
	};
	struct min_domain{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_min_domain;
		min_domain(const type& val){*ptr = val;}
		min_domain() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "min_domain";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.minAABB";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.min_domain;}		
	};
	struct simulationTime{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::internal_simulationTime;
		simulationTime(const type& val){*ptr = val;}
		simulationTime() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "simulationTime";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.simulationTime";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.simulationTime;}		
	};
	struct min_coord{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_min_coord;
		min_coord(const type& val){*ptr = val;}
		min_coord() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "min_coord";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.minCoord";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.min_coord;}		
	};
	struct max_coord{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_max_coord;
		max_coord(const type& val){*ptr = val;}
		max_coord() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "max_coord";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.maxCoord";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_coord;}		
	};
	struct cell_size{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_cell_size;
		cell_size(const type& val){*ptr = val;}
		cell_size() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "cell_size";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.cellSize";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.cell_size;}		
	};
	struct grid_size{
		using type = int3;
		using unit_type = int3;
		static constexpr const uniforms identifier = uniforms::internal_grid_size;
		grid_size(const type& val){*ptr = val;}
		grid_size() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "grid_size";
		static constexpr const auto description = R"()";
		static int3* ptr;
		static int3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.gridSize";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.grid_size;}		
	};
	struct ptcl_spacing{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_ptcl_spacing;
		ptcl_spacing(const type& val){*ptr = val;}
		ptcl_spacing() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "ptcl_spacing";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.ptcl_spacing";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.ptcl_spacing;}		
	};
	struct ptcl_support{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::internal_ptcl_support;
		ptcl_support(const type& val){*ptr = val;}
		ptcl_support() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "ptcl_support";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.ptcl_support";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.ptcl_support;}		
	};
	struct config_file{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_config_file;
		config_file(const type& val){*ptr = val;}
		config_file() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "config_file";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.config_file";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.config_file;}		
	};
	struct config_folder{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_config_folder;
		config_folder(const type& val){*ptr = val;}
		config_folder() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "config_folder";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.config_folder";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.config_folder;}		
	};
	struct working_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_working_directory;
		working_directory(const type& val){*ptr = val;}
		working_directory() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "working_directory";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.working_directory";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.working_directory;}		
	};
	struct build_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_build_directory;
		build_directory(const type& val){*ptr = val;}
		build_directory() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "build_directory";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.build_directory";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.build_directory;}		
	};
	struct source_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_source_directory;
		source_directory(const type& val){*ptr = val;}
		source_directory() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "source_directory";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.source_directory";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.source_directory;}		
	};
	struct binary_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::internal_binary_directory;
		binary_directory(const type& val){*ptr = val;}
		binary_directory() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "binary_directory";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.binary_directory";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.binary_directory;}		
	};
	struct timestep{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::internal_timestep;
		timestep(const type& val){*ptr = val;}
		timestep() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "timestep";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "internal.timestep";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep;}		
	};}
namespace modules{

	struct debug{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_debug;
		debug(const type& val){*ptr = val;}
		debug() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "debug";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.debug";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.debug;}		
	};
	struct sorting{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_sorting;
		sorting(const type& val){*ptr = val;}
		sorting() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "sorting";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.resorting";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.sorting;}		
	};
	struct hash_width{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_hash_width;
		hash_width(const type& val){*ptr = val;}
		hash_width() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "hash_width";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.hash_width";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hash_width;}		
	};
	struct alembic_export{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_alembic_export;
		alembic_export(const type& val){*ptr = val;}
		alembic_export() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "alembic_export";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.alembic";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_export;}		
	};
	struct movingBoundaries{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_movingBoundaries;
		movingBoundaries(const type& val){*ptr = val;}
		movingBoundaries() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "movingBoundaries";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.movingBoundaries";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.movingBoundaries;}		
	};
	struct launch_cfg{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_launch_cfg;
		launch_cfg(const type& val){*ptr = val;}
		launch_cfg() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "launch_cfg";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.launch_cfg";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.launch_cfg;}		
	};
	struct error_checking{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_error_checking;
		error_checking(const type& val){*ptr = val;}
		error_checking() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "error_checking";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.error_checking";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.error_checking;}		
	};
	struct renderMode{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::modules_renderMode;
		renderMode(const type& val){*ptr = val;}
		renderMode() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "renderMode";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.renderMode";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.renderMode;}		
	};
	struct anisotropicSurface{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_anisotropicSurface;
		anisotropicSurface(const type& val){*ptr = val;}
		anisotropicSurface() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicSurface";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.anisotropicSurface";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicSurface;}		
	};
	struct density{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_density;
		density(const type& val){*ptr = val;}
		density() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "density";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.density";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.density;}		
	};
	struct gl_record{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_gl_record;
		gl_record(const type& val){*ptr = val;}
		gl_record() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "gl_record";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.gl_record";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.gl_record;}		
	};
	struct rayTracing{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_rayTracing;
		rayTracing(const type& val){*ptr = val;}
		rayTracing() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rayTracing";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.rayTracing";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rayTracing;}		
	};
	struct regex_cfg{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_regex_cfg;
		regex_cfg(const type& val){*ptr = val;}
		regex_cfg() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "regex_cfg";
		static constexpr const auto description = R"(Used to select where the code should be run. Valid options are: gpu (runs almost everything on the gpu), cpu (runs almost everything on the cpu) and debug (same as cpu but single threaded).)";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.regex_cfg";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.regex_cfg;}		
	};
	struct volumeBoundary{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_volumeBoundary;
		volumeBoundary(const type& val){*ptr = val;}
		volumeBoundary() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeBoundary";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.volumeBoundary";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeBoundary;}		
	};
	struct adaptive{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_adaptive;
		adaptive(const type& val){*ptr = val;}
		adaptive() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "adaptive";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.adaptive";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptive;}		
	};
	struct pressure{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_pressure;
		pressure(const type& val){*ptr = val;}
		pressure() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "pressure";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.pressure";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.pressure;}		
	};
	struct support{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_support;
		support(const type& val){*ptr = val;}
		support() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "support";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.support";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support;}		
	};
	struct particleCleanUp{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_particleCleanUp;
		particleCleanUp(const type& val){*ptr = val;}
		particleCleanUp() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "particleCleanUp";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.particleCleanUp";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.particleCleanUp;}		
	};
	struct surfaceDistance{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_surfaceDistance;
		surfaceDistance(const type& val){*ptr = val;}
		surfaceDistance() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surfaceDistance";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.surfaceDistance";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surfaceDistance;}		
	};
	struct vorticity{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_vorticity;
		vorticity(const type& val){*ptr = val;}
		vorticity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vorticity";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.vorticity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vorticity;}		
	};
	struct tension{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_tension;
		tension(const type& val){*ptr = val;}
		tension() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "tension";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.tension";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.tension;}		
	};
	struct viscosity{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_viscosity;
		viscosity(const type& val){*ptr = val;}
		viscosity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "viscosity";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.viscosity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.viscosity;}		
	};
	struct neighborhood{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_neighborhood;
		neighborhood(const type& val){*ptr = val;}
		neighborhood() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "neighborhood";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.neighborhood";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.neighborhood;}		
	};
	struct volumeInlets{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_volumeInlets;
		volumeInlets(const type& val){*ptr = val;}
		volumeInlets() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeInlets";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.volumeInlets";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeInlets;}		
	};
	struct volumeOutlets{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_volumeOutlets;
		volumeOutlets(const type& val){*ptr = val;}
		volumeOutlets() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeOutlets";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.volumeOutlets";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutlets;}		
	};
	struct neighborSorting{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::modules_neighborSorting;
		neighborSorting(const type& val){*ptr = val;}
		neighborSorting() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "neighborSorting";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.neighborSorting";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.neighborSorting;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct surfaceDetection{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_surfaceDetection;
		surfaceDetection(const type& val){*ptr = val;}
		surfaceDetection() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surfaceDetection";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.surfaceDetection";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surfaceDetection;}		
	};
	struct drag{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::modules_drag;
		drag(const type& val){*ptr = val;}
		drag() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "drag";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.drag";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.drag;}		
	};
	struct xsph{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::modules_xsph;
		xsph(const type& val){*ptr = val;}
		xsph() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "xsph";
		static constexpr const auto description = R"()";
		static bool* ptr;
		static bool* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "modules.xsph";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.xsph;}		
	};}
namespace moving_plane{

	struct movingPlanes{
		using type = std::vector<movingPlane>;
		using unit_type = std::vector<movingPlane>;
		static constexpr const uniforms identifier = uniforms::moving_plane_movingPlanes;
		movingPlanes(const type& val){*ptr = val;}
		movingPlanes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "movingPlanes";
		static constexpr const auto description = R"()";
		static std::vector<movingPlane>* ptr;
		static std::vector<movingPlane>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "moving_plane.plane$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.movingPlanes;}		
	};}
namespace outlet_volumes{

	struct volumeOutletTime{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::outlet_volumes_volumeOutletTime;
		volumeOutletTime(const type& val){*ptr = val;}
		volumeOutletTime() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeOutletTime";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "outlet_volumes.volumeOutletTime";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutletTime;}		
	};
	struct volumeOutletCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::outlet_volumes_volumeOutletCounter;
		volumeOutletCounter(const type& val){*ptr = val;}
		volumeOutletCounter() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "volumeOutletCounter";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "outlet_volumes.volumeOutletCounter";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutletCounter;}		
	};
	struct outletVolumes{
		using type = std::vector<outletVolume>;
		using unit_type = std::vector<outletVolume>;
		static constexpr const uniforms identifier = uniforms::outlet_volumes_outletVolumes;
		outletVolumes(const type& val){*ptr = val;}
		outletVolumes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "outletVolumes";
		static constexpr const auto description = R"()";
		static std::vector<outletVolume>* ptr;
		static std::vector<outletVolume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "outlet_volumes.volume$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.outletVolumes;}		
	};}
namespace particleSets{

	struct particleSets{
		using type = std::vector<std::string>;
		using unit_type = std::vector<std::string>;
		static constexpr const uniforms identifier = uniforms::particleSets_particleSets;
		particleSets(const type& val){*ptr = val;}
		particleSets() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "particleSets";
		static constexpr const auto description = R"()";
		static std::vector<std::string>* ptr;
		static std::vector<std::string>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particleSets.set$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.particleSets;}		
	};}
namespace particle_settings{

	struct radius{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::particle_settings_radius;
		radius(const type& val){*ptr = val;}
		radius() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "radius";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.radius";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.radius;}		
	};
	struct boundaryAdhesion_akinci{
		using type = float;
		using unit_type = value_unit<float, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::particle_settings_boundaryAdhesion_akinci;
		boundaryAdhesion_akinci(const type& val){*ptr = val;}
		boundaryAdhesion_akinci() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryAdhesion_akinci";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::acceleration>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.boundaryAdhesion_akinci";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryAdhesion_akinci;}
		static constexpr const float min{0.000f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct rest_density{
		using type = float;
		using unit_type = value_unit<float, SI::density>;
		static constexpr const uniforms identifier = uniforms::particle_settings_rest_density;
		rest_density(const type& val){*ptr = val;}
		rest_density() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rest_density";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::density>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.rest_density";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rest_density;}		
	};
	struct air_velocity{
		using type = float4;
		using unit_type = value_unit<float4, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::particle_settings_air_velocity;
		air_velocity(const type& val){*ptr = val;}
		air_velocity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "air_velocity";
		static constexpr const auto description = R"()";
		static float4* ptr;
		static value_unit<float4, SI::velocity>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.air_velocity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.air_velocity;}
		static constexpr const float4 min{-10.f,-10.f,-10.f,0.f};
		static constexpr const float4 max{10.f,10.f,10.f,0.f};
		static constexpr const float4 step{0.01f,0.01f,0.01f,0.f};		
	};
	struct tension_akinci{
		using type = float;
		using unit_type = value_unit<float, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::particle_settings_tension_akinci;
		tension_akinci(const type& val){*ptr = val;}
		tension_akinci() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "tension_akinci";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::acceleration>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.tension_akinci";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.tension_akinci;}
		static constexpr const float min{0.000f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct rigidAdhesion_akinci{
		using type = float;
		using unit_type = value_unit<float, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::particle_settings_rigidAdhesion_akinci;
		rigidAdhesion_akinci(const type& val){*ptr = val;}
		rigidAdhesion_akinci() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rigidAdhesion_akinci";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::acceleration>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.rigidAdhesion_akinci";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rigidAdhesion_akinci;}
		static constexpr const float min{0.000f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct xsph_viscosity{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::particle_settings_xsph_viscosity;
		xsph_viscosity(const type& val){*ptr = val;}
		xsph_viscosity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "xsph_viscosity";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.xsph_viscosity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.xsph_viscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct monaghan_viscosity{
		using type = float;
		using unit_type = value_unit<float, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::particle_settings_monaghan_viscosity;
		monaghan_viscosity(const type& val){*ptr = val;}
		monaghan_viscosity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "monaghan_viscosity";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::velocity>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.viscosity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.monaghan_viscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{50.f};
		static constexpr const float step{0.1f};		
	};
	struct boundaryViscosity{
		using type = float;
		using unit_type = value_unit<float, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::particle_settings_boundaryViscosity;
		boundaryViscosity(const type& val){*ptr = val;}
		boundaryViscosity() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryViscosity";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::velocity>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_settings.boundaryViscosity";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryViscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{50.f};
		static constexpr const float step{0.1f};		
	};}
namespace particle_volumes{

	struct particleVolumes{
		using type = std::vector<particleVolume>;
		using unit_type = std::vector<particleVolume>;
		static constexpr const uniforms identifier = uniforms::particle_volumes_particleVolumes;
		particleVolumes(const type& val){*ptr = val;}
		particleVolumes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "particleVolumes";
		static constexpr const auto description = R"()";
		static std::vector<particleVolume>* ptr;
		static std::vector<particleVolume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "particle_volumes.volume$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.particleVolumes;}		
	};}
namespace render_settings{

	struct vrtxRenderGrid{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderGrid;
		vrtxRenderGrid(const type& val){*ptr = val;}
		vrtxRenderGrid() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderGrid";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderGrid";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderGrid;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxRenderFluid{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderFluid;
		vrtxRenderFluid(const type& val){*ptr = val;}
		vrtxRenderFluid() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderFluid";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderFluid";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderFluid;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxRenderSurface{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderSurface;
		vrtxRenderSurface(const type& val){*ptr = val;}
		vrtxRenderSurface() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderSurface";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderSurface";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderSurface;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxDisplayStats{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDisplayStats;
		vrtxDisplayStats(const type& val){*ptr = val;}
		vrtxDisplayStats() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDisplayStats";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDisplayStats";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDisplayStats;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxRenderBVH{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderBVH;
		vrtxRenderBVH(const type& val){*ptr = val;}
		vrtxRenderBVH() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderBVH";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderBVH";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderBVH;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxBVHMaterial{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxBVHMaterial;
		vrtxBVHMaterial(const type& val){*ptr = val;}
		vrtxBVHMaterial() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxBVHMaterial";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxBVHMaterial";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxBVHMaterial;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{4};
		static constexpr const int32_t step{1};		
	};
	struct vrtxMaterial{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxMaterial;
		vrtxMaterial(const type& val){*ptr = val;}
		vrtxMaterial() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxMaterial";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxMaterial";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxMaterial;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{4};
		static constexpr const int32_t step{1};		
	};
	struct vrtxDomainEpsilon{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDomainEpsilon;
		vrtxDomainEpsilon(const type& val){*ptr = val;}
		vrtxDomainEpsilon() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDomainEpsilon";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDomainEpsilon";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDomainEpsilon;}
		static constexpr const float min{-3.f};
		static constexpr const float max{3.f};
		static constexpr const float step{0.01f};		
	};
	struct vrtxDomainMin{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDomainMin;
		vrtxDomainMin(const type& val){*ptr = val;}
		vrtxDomainMin() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDomainMin";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDomainMin";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDomainMin;}		
	};
	struct vrtxDomainMax{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDomainMax;
		vrtxDomainMax(const type& val){*ptr = val;}
		vrtxDomainMax() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDomainMax";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDomainMax";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDomainMax;}		
	};
	struct vrtxDebeerScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDebeerScale;
		vrtxDebeerScale(const type& val){*ptr = val;}
		vrtxDebeerScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDebeerScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDebeerScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDebeerScale;}
		static constexpr const float min{0.001f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.001f};		
	};
	struct vrtxDebeer{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDebeer;
		vrtxDebeer(const type& val){*ptr = val;}
		vrtxDebeer() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDebeer";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDebeer";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDebeer;}
		static constexpr const float3 min{0.f,0.f,0.f};
		static constexpr const float3 max{1.f,1.f,1.f};
		static constexpr const float3 step{0.001f,0.001f,0.001f};		
	};
	struct bvhColor{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_bvhColor;
		bvhColor(const type& val){*ptr = val;}
		bvhColor() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "bvhColor";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.bvhColor";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.bvhColor;}
		static constexpr const float3 min{0.f,0.f,0.f};
		static constexpr const float3 max{1.f,1.f,1.f};
		static constexpr const float3 step{0.001f,0.001f,0.001f};		
	};
	struct vrtxFluidColor{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxFluidColor;
		vrtxFluidColor(const type& val){*ptr = val;}
		vrtxFluidColor() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxFluidColor";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxFluidColor";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxFluidColor;}
		static constexpr const float3 min{0.f,0.f,0.f};
		static constexpr const float3 max{1.f,1.f,1.f};
		static constexpr const float3 step{0.001f,0.001f,0.001f};		
	};
	struct vrtxDepth{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDepth;
		vrtxDepth(const type& val){*ptr = val;}
		vrtxDepth() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDepth";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDepth";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDepth;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct vrtxDepthScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxDepthScale;
		vrtxDepthScale(const type& val){*ptr = val;}
		vrtxDepthScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxDepthScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxDepthScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxDepthScale;}
		static constexpr const float min{0.f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.001f};		
	};
	struct vrtxWMin{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxWMin;
		vrtxWMin(const type& val){*ptr = val;}
		vrtxWMin() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxWMin";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxWMin";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxWMin;}
		static constexpr const float min{0.f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.001f};		
	};
	struct vrtxR{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxR;
		vrtxR(const type& val){*ptr = val;}
		vrtxR() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxR";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxR";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxR;}
		static constexpr const float min{0.f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.001f};		
	};
	struct camera_fov{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_camera_fov;
		camera_fov(const type& val){*ptr = val;}
		camera_fov() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "camera_fov";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.camera_fov";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_fov;}
		static constexpr const float min{0.f};
		static constexpr const float max{256.f};
		static constexpr const float step{0.1f};		
	};
	struct vrtxWMax{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxWMax;
		vrtxWMax(const type& val){*ptr = val;}
		vrtxWMax() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxWMax";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxWMax";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxWMax;}
		static constexpr const float min{0.f};
		static constexpr const float max{4.f};
		static constexpr const float step{0.001f};		
	};
	struct vrtxBounces{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxBounces;
		vrtxBounces(const type& val){*ptr = val;}
		vrtxBounces() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxBounces";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxBounces";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxBounces;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{64};
		static constexpr const int32_t step{1};		
	};
	struct auxScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_auxScale;
		auxScale(const type& val){*ptr = val;}
		auxScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "auxScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.auxScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.auxScale;}
		static constexpr const float min{0.25f};
		static constexpr const float max{16.f};
		static constexpr const float step{0.25f};		
	};
	struct vrtxIOR{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxIOR;
		vrtxIOR(const type& val){*ptr = val;}
		vrtxIOR() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxIOR";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxIOR";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxIOR;}
		static constexpr const float min{1.f};
		static constexpr const float max{5.f};
		static constexpr const float step{0.1f};		
	};
	struct renderSteps{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_renderSteps;
		renderSteps(const type& val){*ptr = val;}
		renderSteps() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "renderSteps";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.renderSteps";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.renderSteps;}
		static constexpr const int32_t min{1};
		static constexpr const int32_t max{50};
		static constexpr const int32_t step{1};		
	};
	struct internalLimit{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_internalLimit;
		internalLimit(const type& val){*ptr = val;}
		internalLimit() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "internalLimit";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.internalLimit";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.internalLimit;}
		static constexpr const float min{0.f};
		static constexpr const float max{64.f};
		static constexpr const float step{1.f};		
	};
	struct auxCellCount{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_auxCellCount;
		auxCellCount(const type& val){*ptr = val;}
		auxCellCount() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "auxCellCount";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.auxCellCount";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.auxCellCount;}		
	};
	struct axesRender{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_axesRender;
		axesRender(const type& val){*ptr = val;}
		axesRender() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "axesRender";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.axesRender";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.axesRender;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct boundsRender{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_boundsRender;
		boundsRender(const type& val){*ptr = val;}
		boundsRender() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundsRender";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.boundsRender";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundsRender;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct floorRender{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_floorRender;
		floorRender(const type& val){*ptr = val;}
		floorRender() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "floorRender";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.floorRender";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.floorRender;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct axesScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_axesScale;
		axesScale(const type& val){*ptr = val;}
		axesScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "axesScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.axesScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.axesScale;}
		static constexpr const float min{0.f};
		static constexpr const float max{64.f};
		static constexpr const float step{1.f};		
	};
	struct camera_position{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::render_settings_camera_position;
		camera_position(const type& val){*ptr = val;}
		camera_position() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "camera_position";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.camera_position";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_position;}
		static constexpr const float3 min{-1000.f,-1000.f,-1000.f};
		static constexpr const float3 max{1000.f,1000.f,1000.f};
		static constexpr const float3 step{1.f,1.f,1.f};		
	};
	struct camera_angle{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_camera_angle;
		camera_angle(const type& val){*ptr = val;}
		camera_angle() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "camera_angle";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.camera_angle";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_angle;}
		static constexpr const float3 min{-360.f,-360.f,-360.f};
		static constexpr const float3 max{360.f,360.f,360.f};
		static constexpr const float3 step{0.5f,0.5f,0.5f};		
	};
	struct camera_resolution{
		using type = float2;
		using unit_type = float2;
		static constexpr const uniforms identifier = uniforms::render_settings_camera_resolution;
		camera_resolution(const type& val){*ptr = val;}
		camera_resolution() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "camera_resolution";
		static constexpr const auto description = R"()";
		static float2* ptr;
		static float2* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.camera_resolution";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_resolution;}		
	};
	struct camera_fps{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_camera_fps;
		camera_fps(const type& val){*ptr = val;}
		camera_fps() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "camera_fps";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.camera_fps";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_fps;}		
	};
	struct gl_file{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::render_settings_gl_file;
		gl_file(const type& val){*ptr = val;}
		gl_file() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "gl_file";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.gl_file";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.gl_file;}		
	};
	struct vrtxRenderNormals{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderNormals;
		vrtxRenderNormals(const type& val){*ptr = val;}
		vrtxRenderNormals() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderNormals";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderNormals";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderNormals;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{1};
		static constexpr const int32_t step{1};		
	};
	struct apertureRadius{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_apertureRadius;
		apertureRadius(const type& val){*ptr = val;}
		apertureRadius() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "apertureRadius";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.apertureRadius";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.apertureRadius;}
		static constexpr const float min{0.f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct anisotropicLambda{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_anisotropicLambda;
		anisotropicLambda(const type& val){*ptr = val;}
		anisotropicLambda() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicLambda";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.anisotropicLambda";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicLambda;}
		static constexpr const float min{0.f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct anisotropicNepsilon{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_anisotropicNepsilon;
		anisotropicNepsilon(const type& val){*ptr = val;}
		anisotropicNepsilon() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicNepsilon";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.anisotropicNepsilon";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicNepsilon;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{60};
		static constexpr const int32_t step{1};		
	};
	struct anisotropicKs{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_anisotropicKs;
		anisotropicKs(const type& val){*ptr = val;}
		anisotropicKs() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicKs";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.anisotropicKs";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicKs;}
		static constexpr const float min{0.f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.001f};		
	};
	struct anisotropicKr{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_anisotropicKr;
		anisotropicKr(const type& val){*ptr = val;}
		anisotropicKr() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicKr";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.anisotropicKr";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicKr;}
		static constexpr const float min{0.f};
		static constexpr const float max{10.f};
		static constexpr const float step{0.1f};		
	};
	struct render_clamp{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_settings_render_clamp;
		render_clamp(const type& val){*ptr = val;}
		render_clamp() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "render_clamp";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static float3* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.render_clamp";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_clamp;}
		static constexpr const float3 min{-1.f,-1.f,-1.f};
		static constexpr const float3 max{1.f,1.f,1.f};
		static constexpr const float3 step{0.01f,0.01f,0.01f};		
	};
	struct anisotropicKn{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_anisotropicKn;
		anisotropicKn(const type& val){*ptr = val;}
		anisotropicKn() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "anisotropicKn";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.anisotropicKn";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.anisotropicKn;}
		static constexpr const float min{0.f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct focalDistance{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_focalDistance;
		focalDistance(const type& val){*ptr = val;}
		focalDistance() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "focalDistance";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.focalDistance";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.focalDistance;}
		static constexpr const float min{0.f};
		static constexpr const float max{100.f};
		static constexpr const float step{0.1f};		
	};
	struct vrtxNeighborLimit{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxNeighborLimit;
		vrtxNeighborLimit(const type& val){*ptr = val;}
		vrtxNeighborLimit() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxNeighborLimit";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxNeighborLimit";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxNeighborLimit;}
		static constexpr const int32_t min{0};
		static constexpr const int32_t max{100};
		static constexpr const int32_t step{1};		
	};
	struct vrtxFluidBias{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxFluidBias;
		vrtxFluidBias(const type& val){*ptr = val;}
		vrtxFluidBias() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxFluidBias";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxFluidBias";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxFluidBias;}
		static constexpr const float min{0.f};
		static constexpr const float max{2.5f};
		static constexpr const float step{0.01f};		
	};
	struct vrtxSurfaceExtraction{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxSurfaceExtraction;
		vrtxSurfaceExtraction(const type& val){*ptr = val;}
		vrtxSurfaceExtraction() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxSurfaceExtraction";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxSurfaceExtraction";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxSurfaceExtraction;}		
	};
	struct vrtxRenderMode{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::render_settings_vrtxRenderMode;
		vrtxRenderMode(const type& val){*ptr = val;}
		vrtxRenderMode() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vrtxRenderMode";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "render_settings.vrtxRenderMode";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vrtxRenderMode;}		
	};}
namespace resort{

	struct zOrderScale{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::resort_zOrderScale;
		zOrderScale(const type& val){*ptr = val;}
		zOrderScale() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "zOrderScale";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.zOrderScale";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.zOrderScale;}		
	};
	struct valid_cells{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_valid_cells;
		valid_cells(const type& val){*ptr = val;}
		valid_cells() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "valid_cells";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.valid_cells";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.valid_cells;}		
	};
	struct resort_algorithm{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_resort_algorithm;
		resort_algorithm(const type& val){*ptr = val;}
		resort_algorithm() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "resort_algorithm";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.algorithm";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.resort_algorithm;}		
	};
	struct auxCells{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_auxCells;
		auxCells(const type& val){*ptr = val;}
		auxCells() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "auxCells";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.auxCells";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.auxCells;}		
	};
	struct auxCollisions{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_auxCollisions;
		auxCollisions(const type& val){*ptr = val;}
		auxCollisions() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "auxCollisions";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.auxCollisions";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.auxCollisions;}		
	};
	struct occupiedCells{
		using type = std::vector<int32_t>;
		using unit_type = std::vector<int32_t>;
		static constexpr const uniforms identifier = uniforms::resort_occupiedCells;
		occupiedCells(const type& val){*ptr = val;}
		occupiedCells() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "occupiedCells";
		static constexpr const auto description = R"()";
		static std::vector<int32_t>* ptr;
		static std::vector<int32_t>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.occupiedCells";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.occupiedCells;}		
	};
	struct collision_cells{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_collision_cells;
		collision_cells(const type& val){*ptr = val;}
		collision_cells() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "collision_cells";
		static constexpr const auto description = R"()";
		static int* ptr;
		static int* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "resort.collision_cells";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.collision_cells;}		
	};}
namespace rigid_volumes{

	struct rigidVolumes{
		using type = std::vector<rigidVolume>;
		using unit_type = std::vector<rigidVolume>;
		static constexpr const uniforms identifier = uniforms::rigid_volumes_rigidVolumes;
		rigidVolumes(const type& val){*ptr = val;}
		rigidVolumes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rigidVolumes";
		static constexpr const auto description = R"()";
		static std::vector<rigidVolume>* ptr;
		static std::vector<rigidVolume>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "rigid_volumes.volume$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rigidVolumes;}		
	};
	struct beta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::rigid_volumes_beta;
		beta(const type& val){*ptr = val;}
		beta() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "beta";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "rigid_volumes.beta";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.beta;}
		static constexpr const float min{0.000f};
		static constexpr const float max{5.f};
		static constexpr const float step{0.02f};		
	};
	struct gamma{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::rigid_volumes_gamma;
		gamma(const type& val){*ptr = val;}
		gamma() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "gamma";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "rigid_volumes.gamma";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.gamma;}
		static constexpr const float min{0.000f};
		static constexpr const float max{5.f};
		static constexpr const float step{0.02f};		
	};}
namespace rtxScene{

	struct rtxSpheres{
		using type = std::vector<rtxSphere>;
		using unit_type = std::vector<rtxSphere>;
		static constexpr const uniforms identifier = uniforms::rtxScene_rtxSpheres;
		rtxSpheres(const type& val){*ptr = val;}
		rtxSpheres() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rtxSpheres";
		static constexpr const auto description = R"()";
		static std::vector<rtxSphere>* ptr;
		static std::vector<rtxSphere>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "rtxScene.sphere$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rtxSpheres;}		
	};
	struct rtxBoxes{
		using type = std::vector<rtxBox>;
		using unit_type = std::vector<rtxBox>;
		static constexpr const uniforms identifier = uniforms::rtxScene_rtxBoxes;
		rtxBoxes(const type& val){*ptr = val;}
		rtxBoxes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "rtxBoxes";
		static constexpr const auto description = R"()";
		static std::vector<rtxBox>* ptr;
		static std::vector<rtxBox>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "rtxScene.box$";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rtxBoxes;}		
	};}
namespace simulation_settings{

	struct boundaryObject{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_boundaryObject;
		boundaryObject(const type& val){*ptr = val;}
		boundaryObject() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryObject";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.boundaryObject";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryObject;}		
	};
	struct LUTOffset{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::simulation_settings_LUTOffset;
		LUTOffset(const type& val){*ptr = val;}
		LUTOffset() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "LUTOffset";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.LUTOffset";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.LUTOffset;}
		static constexpr const float min{-2.285391f};
		static constexpr const float max{2.285391f};
		static constexpr const float step{0.01f};		
	};
	struct boundaryDampening{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::simulation_settings_boundaryDampening;
		boundaryDampening(const type& val){*ptr = val;}
		boundaryDampening() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "boundaryDampening";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.boundaryDampening";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryDampening;}
		static constexpr const float min{0.0f};
		static constexpr const float max{1.0f};
		static constexpr const float step{0.001f};		
	};
	struct timestep_min{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::simulation_settings_timestep_min;
		timestep_min(const type& val){*ptr = val;}
		timestep_min() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "timestep_min";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.timestep_min";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep_min;}
		static constexpr const float min{0.001f};
		static constexpr const float max{0.01f};
		static constexpr const float step{0.001f};		
	};
	struct external_force{
		using type = float4;
		using unit_type = value_unit<float4, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::simulation_settings_external_force;
		external_force(const type& val){*ptr = val;}
		external_force() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "external_force";
		static constexpr const auto description = R"()";
		static float4* ptr;
		static value_unit<float4, SI::acceleration>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.external_force";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.external_force;}
		static constexpr const float4 min{-10.f,-10.f,-10.f,0.f};
		static constexpr const float4 max{10.f,10.f,10.f,0.f};
		static constexpr const float4 step{0.01f,0.01f,0.01f,0.f};		
	};
	struct densitySteps{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::simulation_settings_densitySteps;
		densitySteps(const type& val){*ptr = val;}
		densitySteps() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "densitySteps";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.densitySteps";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.densitySteps;}		
	};
	struct debugRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_debugRegex;
		debugRegex(const type& val){*ptr = val;}
		debugRegex() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "debugRegex";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.debugRegex";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.debugRegex;}		
	};
	struct mlm_schemes{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::simulation_settings_mlm_schemes;
		mlm_schemes(const type& val){*ptr = val;}
		mlm_schemes() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "mlm_schemes";
		static constexpr const auto description = R"()";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.mlm_schemes";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.mlm_schemes;}		
	};
	struct hostRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_hostRegex;
		hostRegex(const type& val){*ptr = val;}
		hostRegex() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "hostRegex";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.hostRegex";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hostRegex;}		
	};
	struct deviceRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_deviceRegex;
		deviceRegex(const type& val){*ptr = val;}
		deviceRegex() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "deviceRegex";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.deviceRegex";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.deviceRegex;}		
	};
	struct timestep_max{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::simulation_settings_timestep_max;
		timestep_max(const type& val){*ptr = val;}
		timestep_max() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "timestep_max";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.timestep_max";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep_max;}
		static constexpr const float min{0.001f};
		static constexpr const float max{0.01f};
		static constexpr const float step{0.001f};		
	};
	struct neighborlimit{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::simulation_settings_neighborlimit;
		neighborlimit(const type& val){*ptr = val;}
		neighborlimit() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "neighborlimit";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.neighborlimit";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.neighborlimit;}		
	};
	struct domainWalls{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_domainWalls;
		domainWalls(const type& val){*ptr = val;}
		domainWalls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "domainWalls";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.domainWalls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.domainWalls;}		
	};
	struct hash_entries{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::simulation_settings_hash_entries;
		hash_entries(const type& val){*ptr = val;}
		hash_entries() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "hash_entries";
		static constexpr const auto description = R"()";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.hash_entries";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hash_entries;}		
	};
	struct max_numptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::simulation_settings_max_numptcls;
		max_numptcls(const type& val){*ptr = val;}
		max_numptcls() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "max_numptcls";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.numptcls";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_numptcls;}		
	};
	struct dumpFile{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::simulation_settings_dumpFile;
		dumpFile(const type& val){*ptr = val;}
		dumpFile() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "dumpFile";
		static constexpr const auto description = R"()";
		static std::string* ptr;
		static std::string* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "simulation_settings.dumpFile";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dumpFile;}		
	};}
namespace support{

	struct support_current_iteration{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::support_support_current_iteration;
		support_current_iteration(const type& val){*ptr = val;}
		support_current_iteration() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "support_current_iteration";
		static constexpr const auto description = R"()";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.support_current_iteration";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support_current_iteration;}		
	};
	struct adjusted_particles{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_adjusted_particles;
		adjusted_particles(const type& val){*ptr = val;}
		adjusted_particles() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "adjusted_particles";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.adjusted_particles";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adjusted_particles;}		
	};
	struct omega{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::support_omega;
		omega(const type& val){*ptr = val;}
		omega() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "omega";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.omega";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.omega;}		
	};
	struct target_neighbors{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_target_neighbors;
		target_neighbors(const type& val){*ptr = val;}
		target_neighbors() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "target_neighbors";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.target_neighbors";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.target_neighbors;}		
	};
	struct support_leeway{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_support_leeway;
		support_leeway(const type& val){*ptr = val;}
		support_leeway() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "support_leeway";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.support_leeway";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support_leeway;}		
	};
	struct overhead_size{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_overhead_size;
		overhead_size(const type& val){*ptr = val;}
		overhead_size() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "overhead_size";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.overhead_size";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.overhead_size;}		
	};
	struct error_factor{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_error_factor;
		error_factor(const type& val){*ptr = val;}
		error_factor() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "error_factor";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "support.error_factor";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.error_factor;}		
	};}
namespace surfaceDistance{

	struct surface_levelLimit{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_levelLimit;
		surface_levelLimit(const type& val){*ptr = val;}
		surface_levelLimit() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_levelLimit";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.level_limit";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_levelLimit;}		
	};
	struct surface_neighborLimit{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_neighborLimit;
		surface_neighborLimit(const type& val){*ptr = val;}
		surface_neighborLimit() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_neighborLimit";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.neighborLimit";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_neighborLimit;}		
	};
	struct surface_phiMin{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_phiMin;
		surface_phiMin(const type& val){*ptr = val;}
		surface_phiMin() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_phiMin";
		static constexpr const auto description = R"()";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.phi_min";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_phiMin;}		
	};
	struct surface_phiChange{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_phiChange;
		surface_phiChange(const type& val){*ptr = val;}
		surface_phiChange() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_phiChange";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.phi_change";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_phiChange;}		
	};
	struct surface_distanceFieldDistances{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_distanceFieldDistances;
		surface_distanceFieldDistances(const type& val){*ptr = val;}
		surface_distanceFieldDistances() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_distanceFieldDistances";
		static constexpr const auto description = R"()";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.distanceFieldDistances";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_distanceFieldDistances;}		
	};
	struct surface_iterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::surfaceDistance_surface_iterations;
		surface_iterations(const type& val){*ptr = val;}
		surface_iterations() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "surface_iterations";
		static constexpr const auto description = R"()";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "surfaceDistance.surfaceIterations";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_iterations;}		
	};}
namespace vorticitySettings{

	struct vorticityCoeff{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::vorticitySettings_vorticityCoeff;
		vorticityCoeff(const type& val){*ptr = val;}
		vorticityCoeff() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "vorticityCoeff";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "vorticitySettings.vorticityCoeff";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vorticityCoeff;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct intertiaInverse{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::vorticitySettings_intertiaInverse;
		intertiaInverse(const type& val){*ptr = val;}
		intertiaInverse() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "intertiaInverse";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "vorticitySettings.intertiaInverse";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.intertiaInverse;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct viscosityOmega{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::vorticitySettings_viscosityOmega;
		viscosityOmega(const type& val){*ptr = val;}
		viscosityOmega() = default;
		operator type() const{return * ptr;}
		//static constexpr const auto variableName = "viscosityOmega";
		static constexpr const auto description = R"()";
		static float* ptr;
		static float* unit_ptr;

		static std::vector<std::string> getPresets();

		static constexpr const auto jsonName = "vorticitySettings.viscosityOmega";
		static constexpr const auto ambiguous = false;
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.viscosityOmega;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};}

}
namespace parameters{
using namespace adaptive;
using namespace alembic;
using namespace boundary_volumes;
using namespace color_map;
using namespace dfsph_settings;
using namespace iisph_settings;
using namespace inlet_volumes;
using namespace internal;
using namespace modules;
using namespace moving_plane;
using namespace outlet_volumes;
using namespace particleSets;
using namespace particle_settings;
using namespace particle_volumes;
using namespace render_settings;
using namespace resort;
using namespace rigid_volumes;
using namespace rtxScene;
using namespace simulation_settings;
using namespace support;
using namespace surfaceDistance;
using namespace vorticitySettings;
}


extern std::tuple<parameters::modules::adaptive, parameters::adaptive::adaptivityScaling, parameters::adaptive::adaptivityThreshold, parameters::adaptive::adaptivityGamma, parameters::adaptive::resolution, parameters::adaptive::useVolume, parameters::adaptive::minVolume, parameters::adaptive::detailedAdaptiveStatistics, parameters::adaptive::ratio, parameters::adaptive::blendSteps, parameters::adaptive::delay, parameters::adaptive::splitPtcls, parameters::adaptive::blendedPtcls, parameters::adaptive::mergedPtcls, parameters::adaptive::sharedPtcls, parameters::modules::pressure, parameters::modules::volumeBoundary, parameters::modules::xsph, parameters::modules::drag, parameters::modules::viscosity, parameters::modules::tension, parameters::modules::vorticity, parameters::particle_settings::monaghan_viscosity, parameters::particle_settings::boundaryViscosity, parameters::particle_settings::xsph_viscosity, parameters::particle_settings::rigidAdhesion_akinci, parameters::particle_settings::boundaryAdhesion_akinci, parameters::particle_settings::tension_akinci, parameters::particle_settings::air_velocity, parameters::vorticitySettings::intertiaInverse, parameters::vorticitySettings::viscosityOmega, parameters::vorticitySettings::vorticityCoeff, parameters::simulation_settings::external_force, parameters::simulation_settings::timestep_min, parameters::simulation_settings::timestep_max, parameters::modules::movingBoundaries, parameters::simulation_settings::boundaryDampening, parameters::simulation_settings::LUTOffset, parameters::simulation_settings::boundaryObject, parameters::simulation_settings::domainWalls, parameters::simulation_settings::neighborlimit, parameters::moving_plane::movingPlanes, parameters::boundary_volumes::volumeBoundaryCounter, parameters::boundary_volumes::boundaryVolumes, parameters::modules::debug, parameters::internal::neighborhood_kind, parameters::internal::dumpNextframe, parameters::internal::dumpForSSSPH, parameters::internal::target, parameters::internal::hash_size, parameters::internal::cell_order, parameters::internal::cell_structure, parameters::internal::num_ptcls, parameters::internal::boundaryCounter, parameters::internal::boundaryLUTSize, parameters::internal::frame, parameters::internal::max_velocity, parameters::internal::min_domain, parameters::internal::max_domain, parameters::internal::min_coord, parameters::internal::max_coord, parameters::internal::cell_size, parameters::internal::grid_size, parameters::internal::ptcl_spacing, parameters::internal::ptcl_support, parameters::internal::config_file, parameters::internal::config_folder, parameters::internal::working_directory, parameters::internal::build_directory, parameters::internal::source_directory, parameters::internal::binary_directory, parameters::internal::timestep, parameters::internal::simulationTime, parameters::particle_settings::radius, parameters::simulation_settings::dumpFile, parameters::simulation_settings::max_numptcls, parameters::simulation_settings::hash_entries, parameters::simulation_settings::mlm_schemes, parameters::simulation_settings::deviceRegex, parameters::simulation_settings::hostRegex, parameters::simulation_settings::debugRegex, parameters::modules::density, parameters::particle_settings::rest_density, parameters::simulation_settings::densitySteps, parameters::dfsph_settings::densityError, parameters::dfsph_settings::divergenceError, parameters::dfsph_settings::densitySolverIterations, parameters::dfsph_settings::divergenceSolverIterations, parameters::dfsph_settings::densityEta, parameters::dfsph_settings::divergenceEta, parameters::iisph_settings::density_error, parameters::iisph_settings::iterations, parameters::iisph_settings::eta, parameters::iisph_settings::jacobi_omega, parameters::modules::particleCleanUp, parameters::modules::volumeInlets, parameters::modules::volumeOutlets, parameters::particle_volumes::particleVolumes, parameters::rigid_volumes::gamma, parameters::rigid_volumes::beta, parameters::rigid_volumes::rigidVolumes, parameters::particleSets::particleSets, parameters::inlet_volumes::inletVolumes, parameters::outlet_volumes::volumeOutletCounter, parameters::outlet_volumes::volumeOutletTime, parameters::outlet_volumes::outletVolumes, parameters::modules::neighborhood, parameters::modules::neighborSorting, parameters::rtxScene::rtxSpheres, parameters::rtxScene::rtxBoxes, parameters::modules::rayTracing, parameters::modules::anisotropicSurface, parameters::modules::renderMode, parameters::render_settings::apertureRadius, parameters::render_settings::anisotropicLambda, parameters::render_settings::anisotropicNepsilon, parameters::render_settings::anisotropicKs, parameters::render_settings::anisotropicKr, parameters::render_settings::anisotropicKn, parameters::render_settings::focalDistance, parameters::render_settings::vrtxNeighborLimit, parameters::render_settings::vrtxFluidBias, parameters::render_settings::vrtxSurfaceExtraction, parameters::render_settings::vrtxRenderMode, parameters::render_settings::vrtxRenderGrid, parameters::render_settings::vrtxRenderFluid, parameters::render_settings::vrtxRenderSurface, parameters::render_settings::vrtxDisplayStats, parameters::render_settings::vrtxRenderBVH, parameters::render_settings::vrtxBVHMaterial, parameters::render_settings::vrtxRenderNormals, parameters::render_settings::vrtxMaterial, parameters::render_settings::vrtxDomainEpsilon, parameters::render_settings::vrtxDomainMin, parameters::render_settings::vrtxDomainMax, parameters::render_settings::vrtxDebeerScale, parameters::render_settings::vrtxDebeer, parameters::render_settings::bvhColor, parameters::render_settings::vrtxFluidColor, parameters::render_settings::vrtxDepth, parameters::render_settings::vrtxDepthScale, parameters::render_settings::vrtxWMin, parameters::render_settings::vrtxR, parameters::render_settings::camera_fov, parameters::render_settings::vrtxWMax, parameters::render_settings::vrtxBounces, parameters::render_settings::auxScale, parameters::render_settings::vrtxIOR, parameters::render_settings::renderSteps, parameters::render_settings::internalLimit, parameters::render_settings::auxCellCount, parameters::render_settings::axesRender, parameters::render_settings::boundsRender, parameters::render_settings::floorRender, parameters::render_settings::axesScale, parameters::render_settings::render_clamp, parameters::render_settings::camera_position, parameters::render_settings::camera_angle, parameters::render_settings::camera_resolution, parameters::render_settings::camera_fps, parameters::render_settings::gl_file, parameters::color_map::transfer_mode, parameters::color_map::mapping_mode, parameters::color_map::vectorMode, parameters::color_map::visualizeDirection, parameters::color_map::vectorScale, parameters::color_map::vectorScaling, parameters::color_map::render_min, parameters::color_map::render_max, parameters::color_map::transfer_fn, parameters::color_map::pruneVoxel, parameters::color_map::mapping_fn, parameters::color_map::render_auto, parameters::color_map::map_flipped, parameters::color_map::render_buffer, parameters::color_map::render_colormap, parameters::modules::sorting, parameters::modules::hash_width, parameters::modules::alembic_export, parameters::modules::error_checking, parameters::modules::gl_record, parameters::modules::launch_cfg, parameters::modules::regex_cfg, parameters::resort::auxCells, parameters::resort::auxCollisions, parameters::resort::resort_algorithm, parameters::resort::valid_cells, parameters::resort::zOrderScale, parameters::resort::collision_cells, parameters::resort::occupiedCells, parameters::alembic::alembic_file_name, parameters::alembic::alembic_fps, parameters::modules::support, parameters::support::support_current_iteration, parameters::support::adjusted_particles, parameters::support::omega, parameters::support::target_neighbors, parameters::support::support_leeway, parameters::support::overhead_size, parameters::support::error_factor, parameters::modules::surfaceDistance, parameters::modules::surfaceDetection, parameters::surfaceDistance::surface_levelLimit, parameters::surfaceDistance::surface_neighborLimit, parameters::surfaceDistance::surface_phiMin, parameters::surfaceDistance::surface_phiChange, parameters::surfaceDistance::surface_distanceFieldDistances, parameters::surfaceDistance::surface_iterations> uniforms_list;

template<typename T>
using parameter = typename T::type;
template<typename T>
using parameter_u = typename T::unit_type;

template<typename Func, typename... Ts>
auto iterateParameters(Func&& fn, Ts&&... args){
	fn(parameters::modules::adaptive(), args...);
	fn(parameters::adaptive::adaptivityScaling(), args...);
	fn(parameters::adaptive::adaptivityThreshold(), args...);
	fn(parameters::adaptive::adaptivityGamma(), args...);
	fn(parameters::adaptive::resolution(), args...);
	fn(parameters::adaptive::useVolume(), args...);
	fn(parameters::adaptive::minVolume(), args...);
	fn(parameters::adaptive::detailedAdaptiveStatistics(), args...);
	fn(parameters::adaptive::ratio(), args...);
	fn(parameters::adaptive::blendSteps(), args...);
	fn(parameters::adaptive::delay(), args...);
	fn(parameters::adaptive::splitPtcls(), args...);
	fn(parameters::adaptive::blendedPtcls(), args...);
	fn(parameters::adaptive::mergedPtcls(), args...);
	fn(parameters::adaptive::sharedPtcls(), args...);
	fn(parameters::modules::pressure(), args...);
	fn(parameters::modules::volumeBoundary(), args...);
	fn(parameters::modules::xsph(), args...);
	fn(parameters::modules::drag(), args...);
	fn(parameters::modules::viscosity(), args...);
	fn(parameters::modules::tension(), args...);
	fn(parameters::modules::vorticity(), args...);
	fn(parameters::particle_settings::monaghan_viscosity(), args...);
	fn(parameters::particle_settings::boundaryViscosity(), args...);
	fn(parameters::particle_settings::xsph_viscosity(), args...);
	fn(parameters::particle_settings::rigidAdhesion_akinci(), args...);
	fn(parameters::particle_settings::boundaryAdhesion_akinci(), args...);
	fn(parameters::particle_settings::tension_akinci(), args...);
	fn(parameters::particle_settings::air_velocity(), args...);
	fn(parameters::vorticitySettings::intertiaInverse(), args...);
	fn(parameters::vorticitySettings::viscosityOmega(), args...);
	fn(parameters::vorticitySettings::vorticityCoeff(), args...);
	fn(parameters::simulation_settings::external_force(), args...);
	fn(parameters::simulation_settings::timestep_min(), args...);
	fn(parameters::simulation_settings::timestep_max(), args...);
	fn(parameters::modules::movingBoundaries(), args...);
	fn(parameters::simulation_settings::boundaryDampening(), args...);
	fn(parameters::simulation_settings::LUTOffset(), args...);
	fn(parameters::simulation_settings::boundaryObject(), args...);
	fn(parameters::simulation_settings::domainWalls(), args...);
	fn(parameters::simulation_settings::neighborlimit(), args...);
	fn(parameters::moving_plane::movingPlanes(), args...);
	fn(parameters::boundary_volumes::volumeBoundaryCounter(), args...);
	fn(parameters::boundary_volumes::boundaryVolumes(), args...);
	fn(parameters::modules::debug(), args...);
	fn(parameters::internal::neighborhood_kind(), args...);
	fn(parameters::internal::dumpNextframe(), args...);
	fn(parameters::internal::dumpForSSSPH(), args...);
	fn(parameters::internal::target(), args...);
	fn(parameters::internal::hash_size(), args...);
	fn(parameters::internal::cell_order(), args...);
	fn(parameters::internal::cell_structure(), args...);
	fn(parameters::internal::num_ptcls(), args...);
	fn(parameters::internal::boundaryCounter(), args...);
	fn(parameters::internal::boundaryLUTSize(), args...);
	fn(parameters::internal::frame(), args...);
	fn(parameters::internal::max_velocity(), args...);
	fn(parameters::internal::min_domain(), args...);
	fn(parameters::internal::max_domain(), args...);
	fn(parameters::internal::min_coord(), args...);
	fn(parameters::internal::max_coord(), args...);
	fn(parameters::internal::cell_size(), args...);
	fn(parameters::internal::grid_size(), args...);
	fn(parameters::internal::ptcl_spacing(), args...);
	fn(parameters::internal::ptcl_support(), args...);
	fn(parameters::internal::config_file(), args...);
	fn(parameters::internal::config_folder(), args...);
	fn(parameters::internal::working_directory(), args...);
	fn(parameters::internal::build_directory(), args...);
	fn(parameters::internal::source_directory(), args...);
	fn(parameters::internal::binary_directory(), args...);
	fn(parameters::internal::timestep(), args...);
	fn(parameters::internal::simulationTime(), args...);
	fn(parameters::particle_settings::radius(), args...);
	fn(parameters::simulation_settings::dumpFile(), args...);
	fn(parameters::simulation_settings::max_numptcls(), args...);
	fn(parameters::simulation_settings::hash_entries(), args...);
	fn(parameters::simulation_settings::mlm_schemes(), args...);
	fn(parameters::simulation_settings::deviceRegex(), args...);
	fn(parameters::simulation_settings::hostRegex(), args...);
	fn(parameters::simulation_settings::debugRegex(), args...);
	fn(parameters::modules::density(), args...);
	fn(parameters::particle_settings::rest_density(), args...);
	fn(parameters::simulation_settings::densitySteps(), args...);
	fn(parameters::dfsph_settings::densityError(), args...);
	fn(parameters::dfsph_settings::divergenceError(), args...);
	fn(parameters::dfsph_settings::densitySolverIterations(), args...);
	fn(parameters::dfsph_settings::divergenceSolverIterations(), args...);
	fn(parameters::dfsph_settings::densityEta(), args...);
	fn(parameters::dfsph_settings::divergenceEta(), args...);
	fn(parameters::iisph_settings::density_error(), args...);
	fn(parameters::iisph_settings::iterations(), args...);
	fn(parameters::iisph_settings::eta(), args...);
	fn(parameters::iisph_settings::jacobi_omega(), args...);
	fn(parameters::modules::particleCleanUp(), args...);
	fn(parameters::modules::volumeInlets(), args...);
	fn(parameters::modules::volumeOutlets(), args...);
	fn(parameters::particle_volumes::particleVolumes(), args...);
	fn(parameters::rigid_volumes::gamma(), args...);
	fn(parameters::rigid_volumes::beta(), args...);
	fn(parameters::rigid_volumes::rigidVolumes(), args...);
	fn(parameters::particleSets::particleSets(), args...);
	fn(parameters::inlet_volumes::inletVolumes(), args...);
	fn(parameters::outlet_volumes::volumeOutletCounter(), args...);
	fn(parameters::outlet_volumes::volumeOutletTime(), args...);
	fn(parameters::outlet_volumes::outletVolumes(), args...);
	fn(parameters::modules::neighborhood(), args...);
	fn(parameters::modules::neighborSorting(), args...);
	fn(parameters::rtxScene::rtxSpheres(), args...);
	fn(parameters::rtxScene::rtxBoxes(), args...);
	fn(parameters::modules::rayTracing(), args...);
	fn(parameters::modules::anisotropicSurface(), args...);
	fn(parameters::modules::renderMode(), args...);
	fn(parameters::render_settings::apertureRadius(), args...);
	fn(parameters::render_settings::anisotropicLambda(), args...);
	fn(parameters::render_settings::anisotropicNepsilon(), args...);
	fn(parameters::render_settings::anisotropicKs(), args...);
	fn(parameters::render_settings::anisotropicKr(), args...);
	fn(parameters::render_settings::anisotropicKn(), args...);
	fn(parameters::render_settings::focalDistance(), args...);
	fn(parameters::render_settings::vrtxNeighborLimit(), args...);
	fn(parameters::render_settings::vrtxFluidBias(), args...);
	fn(parameters::render_settings::vrtxSurfaceExtraction(), args...);
	fn(parameters::render_settings::vrtxRenderMode(), args...);
	fn(parameters::render_settings::vrtxRenderGrid(), args...);
	fn(parameters::render_settings::vrtxRenderFluid(), args...);
	fn(parameters::render_settings::vrtxRenderSurface(), args...);
	fn(parameters::render_settings::vrtxDisplayStats(), args...);
	fn(parameters::render_settings::vrtxRenderBVH(), args...);
	fn(parameters::render_settings::vrtxBVHMaterial(), args...);
	fn(parameters::render_settings::vrtxRenderNormals(), args...);
	fn(parameters::render_settings::vrtxMaterial(), args...);
	fn(parameters::render_settings::vrtxDomainEpsilon(), args...);
	fn(parameters::render_settings::vrtxDomainMin(), args...);
	fn(parameters::render_settings::vrtxDomainMax(), args...);
	fn(parameters::render_settings::vrtxDebeerScale(), args...);
	fn(parameters::render_settings::vrtxDebeer(), args...);
	fn(parameters::render_settings::bvhColor(), args...);
	fn(parameters::render_settings::vrtxFluidColor(), args...);
	fn(parameters::render_settings::vrtxDepth(), args...);
	fn(parameters::render_settings::vrtxDepthScale(), args...);
	fn(parameters::render_settings::vrtxWMin(), args...);
	fn(parameters::render_settings::vrtxR(), args...);
	fn(parameters::render_settings::camera_fov(), args...);
	fn(parameters::render_settings::vrtxWMax(), args...);
	fn(parameters::render_settings::vrtxBounces(), args...);
	fn(parameters::render_settings::auxScale(), args...);
	fn(parameters::render_settings::vrtxIOR(), args...);
	fn(parameters::render_settings::renderSteps(), args...);
	fn(parameters::render_settings::internalLimit(), args...);
	fn(parameters::render_settings::auxCellCount(), args...);
	fn(parameters::render_settings::axesRender(), args...);
	fn(parameters::render_settings::boundsRender(), args...);
	fn(parameters::render_settings::floorRender(), args...);
	fn(parameters::render_settings::axesScale(), args...);
	fn(parameters::render_settings::render_clamp(), args...);
	fn(parameters::render_settings::camera_position(), args...);
	fn(parameters::render_settings::camera_angle(), args...);
	fn(parameters::render_settings::camera_resolution(), args...);
	fn(parameters::render_settings::camera_fps(), args...);
	fn(parameters::render_settings::gl_file(), args...);
	fn(parameters::color_map::transfer_mode(), args...);
	fn(parameters::color_map::mapping_mode(), args...);
	fn(parameters::color_map::vectorMode(), args...);
	fn(parameters::color_map::visualizeDirection(), args...);
	fn(parameters::color_map::vectorScale(), args...);
	fn(parameters::color_map::vectorScaling(), args...);
	fn(parameters::color_map::render_min(), args...);
	fn(parameters::color_map::render_max(), args...);
	fn(parameters::color_map::transfer_fn(), args...);
	fn(parameters::color_map::pruneVoxel(), args...);
	fn(parameters::color_map::mapping_fn(), args...);
	fn(parameters::color_map::render_auto(), args...);
	fn(parameters::color_map::map_flipped(), args...);
	fn(parameters::color_map::render_buffer(), args...);
	fn(parameters::color_map::render_colormap(), args...);
	fn(parameters::modules::sorting(), args...);
	fn(parameters::modules::hash_width(), args...);
	fn(parameters::modules::alembic_export(), args...);
	fn(parameters::modules::error_checking(), args...);
	fn(parameters::modules::gl_record(), args...);
	fn(parameters::modules::launch_cfg(), args...);
	fn(parameters::modules::regex_cfg(), args...);
	fn(parameters::resort::auxCells(), args...);
	fn(parameters::resort::auxCollisions(), args...);
	fn(parameters::resort::resort_algorithm(), args...);
	fn(parameters::resort::valid_cells(), args...);
	fn(parameters::resort::zOrderScale(), args...);
	fn(parameters::resort::collision_cells(), args...);
	fn(parameters::resort::occupiedCells(), args...);
	fn(parameters::alembic::alembic_file_name(), args...);
	fn(parameters::alembic::alembic_fps(), args...);
	fn(parameters::modules::support(), args...);
	fn(parameters::support::support_current_iteration(), args...);
	fn(parameters::support::adjusted_particles(), args...);
	fn(parameters::support::omega(), args...);
	fn(parameters::support::target_neighbors(), args...);
	fn(parameters::support::support_leeway(), args...);
	fn(parameters::support::overhead_size(), args...);
	fn(parameters::support::error_factor(), args...);
	fn(parameters::modules::surfaceDistance(), args...);
	fn(parameters::modules::surfaceDetection(), args...);
	fn(parameters::surfaceDistance::surface_levelLimit(), args...);
	fn(parameters::surfaceDistance::surface_neighborLimit(), args...);
	fn(parameters::surfaceDistance::surface_phiMin(), args...);
	fn(parameters::surfaceDistance::surface_phiChange(), args...);
	fn(parameters::surfaceDistance::surface_distanceFieldDistances(), args...);
	fn(parameters::surfaceDistance::surface_iterations(), args...);
}

