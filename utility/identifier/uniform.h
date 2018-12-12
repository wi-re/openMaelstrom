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

enum struct uniforms{adaptive, adaptive_resolution, adaptiveRatio, blendsteps, adaptive_delay, split_ptcls, blend_ptcls, merged_ptcls, shared_ptcls, pressure, volumeBoundary, xsph, drag, viscosity, tension, vorticity, monaghan_viscosity, intertia_inverse, vorticity_viscosity, vorticity_nu_t, xsph_viscosity, tension_akinci, air_velocity, external_force, timestep_min, timestep_max, movingBoundaries, boundaryDampening, boundaryObject, domainWalls, neighborlimit, moving_planes, volumeBoundaryCounter, boundary_volumes, neighborhood_kind, dumpNextframe, target, hash_size, cell_order, cell_structure, num_ptcls, boundaryCounter, boundaryLUTSize, frame, max_velocity, min_domain, max_domain, min_coord, max_coord, cell_size, grid_size, ptcl_spacing, ptcl_support, config_file, config_folder, working_directory, build_directory, source_directory, binary_directory, timestep, simulationTime, radius, dumpFile, max_numptcls, hash_entries, mlm_schemes, deviceRegex, hostRegex, debugRegex, density, rest_density, densitySteps, dfsph_density_error, dfsph_iterations, dfsph_density_eta, dfsph_divergence_eta, density_error, iterations, eta, jacobi_omega, particleCleanUp, volumeInlets, volumeOutlets, particle_volumes, inlet_volumes, volumeOutletCounter, volumeOutletTime, outlet_volumes, neighborhood, render_clamp, camera_position, camera_angle, camera_resolution, camera_fps, gl_file, render_min, render_max, render_auto, map_flipped, render_buffer, render_colormap, sorting, hash_width, alembic_export, error_checking, gl_record, launch_cfg, regex_cfg, resort_algorithm, valid_cells, collision_cells, alembic_file_name, alembic_fps, support, support_current_iteration, adjusted_particles, omega, target_neighbors, support_leeway, overhead_size, error_factor, surfaceDistance, surface_levelLimit, surface_neighborLimit, surface_phiMin, surface_phiChange, surface_distanceFieldDistances, surface_iterations};

#include <utility/identifier/resource_helper.h>

template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get(T&& = T{}) {
	return *T::ptr;
}
template<typename T, typename = std::enable_if_t<std::is_same<uniforms, std::decay_t<decltype(T::identifier)>>::value>, typename = void> auto& get_u(T&& = T{}) {
	return *T::unit_ptr;
}

struct moving_plane{
	using uniform_type = complex_uniform;
	
	complex_type<float3> plane_position{ "pos", {0.f,0.f,0.f}};
	complex_type<float3> plane_normal{ "norm", {0.f,0.f,0.f}};
	complex_type<float3> plane_direction{ "dir", {0.f,0.f,0.f}};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> magnitude{ "mag", 0.f};
	complex_type<float> frequency{ "freq", 0.f};
	complex_type<int32_t> index{ "idx", 0};
};


struct boundary_volume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
};


struct particle_volume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<std::string> kind{ "kind", ""};
	complex_type<float> concentration{ "concentration", 0.f};
	complex_type<float> timeToEmit{ "timeToEmit", 0.f};
};


struct inlet_volume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<int32_t> particles_emitted{ "ptcls", 0};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> inlet_radius{ "r", -1.f};
	complex_type<float4> emitter_velocity{ "vel", {0.f,0.f,0.f,0.f}};
};


struct outlet_volume{
	using uniform_type = complex_uniform;
	
	complex_type<std::string> fileName{ "file", ""};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> flowRate{ "rate", -1.f};
};


namespace parameters{



	struct adaptive{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::adaptive;
		adaptive(const type& val){*ptr = val;}
		adaptive() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "adaptive";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.adaptive";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptive;}		
	};
	struct adaptive_resolution{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptive_resolution;
		adaptive_resolution(const type& val){*ptr = val;}
		adaptive_resolution() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "adaptive_resolution";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "adaptive.resolution";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptive_resolution;}
		static constexpr const float min{1.f};
		static constexpr const float max{512.f};
		static constexpr const float step{1.f};		
	};
	struct adaptiveRatio{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::adaptiveRatio;
		adaptiveRatio(const type& val){*ptr = val;}
		adaptiveRatio() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "adaptiveRatio";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "adaptive.adaptiveRatio";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptiveRatio;}		
	};
	struct blendsteps{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::blendsteps;
		blendsteps(const type& val){*ptr = val;}
		blendsteps() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "blendsteps";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "adaptive.blendsteps";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.blendsteps;}		
	};
	struct adaptive_delay{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::adaptive_delay;
		adaptive_delay(const type& val){*ptr = val;}
		adaptive_delay() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "adaptive_delay";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "adaptive.delay";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adaptive_delay;}		
	};
	struct split_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::split_ptcls;
		split_ptcls(const type& val){*ptr = val;}
		split_ptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "split_ptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "adaptive.split";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.split_ptcls;}		
	};
	struct blend_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::blend_ptcls;
		blend_ptcls(const type& val){*ptr = val;}
		blend_ptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "blend_ptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "adaptive.blend";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.blend_ptcls;}		
	};
	struct merged_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::merged_ptcls;
		merged_ptcls(const type& val){*ptr = val;}
		merged_ptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "merged_ptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "adaptive.merged";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.merged_ptcls;}		
	};
	struct shared_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::shared_ptcls;
		shared_ptcls(const type& val){*ptr = val;}
		shared_ptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "shared_ptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "adaptive.shared";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.shared_ptcls;}		
	};
	struct pressure{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::pressure;
		pressure(const type& val){*ptr = val;}
		pressure() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "pressure";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.pressure";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.pressure;}		
	};
	struct volumeBoundary{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::volumeBoundary;
		volumeBoundary(const type& val){*ptr = val;}
		volumeBoundary() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeBoundary";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.volumeBoundary";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeBoundary;}		
	};
	struct xsph{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::xsph;
		xsph(const type& val){*ptr = val;}
		xsph() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "xsph";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.xsph";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.xsph;}		
	};
	struct drag{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::drag;
		drag(const type& val){*ptr = val;}
		drag() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "drag";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.drag";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.drag;}		
	};
	struct viscosity{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::viscosity;
		viscosity(const type& val){*ptr = val;}
		viscosity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "viscosity";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.viscosity";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.viscosity;}		
	};
	struct tension{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::tension;
		tension(const type& val){*ptr = val;}
		tension() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "tension";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.tension";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.tension;}		
	};
	struct vorticity{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::vorticity;
		vorticity(const type& val){*ptr = val;}
		vorticity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "vorticity";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.vorticity";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vorticity;}		
	};
	struct monaghan_viscosity{
		using type = float;
		using unit_type = value_unit<float, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::monaghan_viscosity;
		monaghan_viscosity(const type& val){*ptr = val;}
		monaghan_viscosity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "monaghan_viscosity";
		static float* ptr;
		static value_unit<float, SI::velocity>* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.viscosity";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.monaghan_viscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{50.f};
		static constexpr const float step{0.1f};		
	};
	struct intertia_inverse{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::intertia_inverse;
		intertia_inverse(const type& val){*ptr = val;}
		intertia_inverse() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "intertia_inverse";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.intertia_inverse";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.intertia_inverse;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct vorticity_viscosity{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::vorticity_viscosity;
		vorticity_viscosity(const type& val){*ptr = val;}
		vorticity_viscosity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "vorticity_viscosity";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.vorticity_viscosity";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vorticity_viscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct vorticity_nu_t{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::vorticity_nu_t;
		vorticity_nu_t(const type& val){*ptr = val;}
		vorticity_nu_t() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "vorticity_nu_t";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.vorticity_nu_t";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.vorticity_nu_t;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct xsph_viscosity{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::xsph_viscosity;
		xsph_viscosity(const type& val){*ptr = val;}
		xsph_viscosity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "xsph_viscosity";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.xsph_viscosity";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.xsph_viscosity;}
		static constexpr const float min{0.000f};
		static constexpr const float max{1.f};
		static constexpr const float step{0.01f};		
	};
	struct tension_akinci{
		using type = float;
		using unit_type = value_unit<float, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::tension_akinci;
		tension_akinci(const type& val){*ptr = val;}
		tension_akinci() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "tension_akinci";
		static float* ptr;
		static value_unit<float, SI::acceleration>* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.tension_akinci";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.tension_akinci;}
		static constexpr const float min{0.000f};
		static constexpr const float max{2.f};
		static constexpr const float step{0.01f};		
	};
	struct air_velocity{
		using type = float4;
		using unit_type = value_unit<float4, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::air_velocity;
		air_velocity(const type& val){*ptr = val;}
		air_velocity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "air_velocity";
		static float4* ptr;
		static value_unit<float4, SI::velocity>* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.air_velocity";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.air_velocity;}
		static constexpr const float4 min{-10.f,-10.f,-10.f,0.f};
		static constexpr const float4 max{10.f,10.f,10.f,0.f};
		static constexpr const float4 step{0.01f,0.01f,0.01f,0.f};		
	};
	struct external_force{
		using type = float4;
		using unit_type = value_unit<float4, SI::acceleration>;
		static constexpr const uniforms identifier = uniforms::external_force;
		external_force(const type& val){*ptr = val;}
		external_force() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "external_force";
		static float4* ptr;
		static value_unit<float4, SI::acceleration>* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.external_force";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.external_force;}
		static constexpr const float4 min{-10.f,-10.f,-10.f,0.f};
		static constexpr const float4 max{10.f,10.f,10.f,0.f};
		static constexpr const float4 step{0.01f,0.01f,0.01f,0.f};		
	};
	struct timestep_min{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::timestep_min;
		timestep_min(const type& val){*ptr = val;}
		timestep_min() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "timestep_min";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.timestep_min";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep_min;}
		static constexpr const float min{0.001f};
		static constexpr const float max{0.01f};
		static constexpr const float step{0.001f};		
	};
	struct timestep_max{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::timestep_max;
		timestep_max(const type& val){*ptr = val;}
		timestep_max() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "timestep_max";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.timestep_max";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep_max;}
		static constexpr const float min{0.001f};
		static constexpr const float max{0.01f};
		static constexpr const float step{0.001f};		
	};
	struct movingBoundaries{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::movingBoundaries;
		movingBoundaries(const type& val){*ptr = val;}
		movingBoundaries() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "movingBoundaries";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.movingBoundaries";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.movingBoundaries;}		
	};
	struct boundaryDampening{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::boundaryDampening;
		boundaryDampening(const type& val){*ptr = val;}
		boundaryDampening() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "boundaryDampening";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.boundaryDampening";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryDampening;}
		static constexpr const float min{0.0f};
		static constexpr const float max{1.0f};
		static constexpr const float step{0.001f};		
	};
	struct boundaryObject{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::boundaryObject;
		boundaryObject(const type& val){*ptr = val;}
		boundaryObject() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "boundaryObject";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.boundaryObject";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryObject;}		
	};
	struct domainWalls{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::domainWalls;
		domainWalls(const type& val){*ptr = val;}
		domainWalls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "domainWalls";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.domainWalls";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.domainWalls;}		
	};
	struct neighborlimit{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::neighborlimit;
		neighborlimit(const type& val){*ptr = val;}
		neighborlimit() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "neighborlimit";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.neighborlimit";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.neighborlimit;}		
	};
	struct moving_planes{
		using type = std::vector<moving_plane>;
		using unit_type = std::vector<moving_plane>;
		static constexpr const uniforms identifier = uniforms::moving_planes;
		moving_planes(const type& val){*ptr = val;}
		moving_planes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "moving_planes";
		static std::vector<moving_plane>* ptr;
		static std::vector<moving_plane>* unit_ptr;

		static constexpr const auto jsonName = "moving_plane.plane$";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.moving_planes;}		
	};
	struct volumeBoundaryCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::volumeBoundaryCounter;
		volumeBoundaryCounter(const type& val){*ptr = val;}
		volumeBoundaryCounter() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeBoundaryCounter";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "boundary_volumes.volumeBoundaryCounter";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeBoundaryCounter;}		
	};
	struct boundary_volumes{
		using type = std::vector<boundary_volume>;
		using unit_type = std::vector<boundary_volume>;
		static constexpr const uniforms identifier = uniforms::boundary_volumes;
		boundary_volumes(const type& val){*ptr = val;}
		boundary_volumes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "boundary_volumes";
		static std::vector<boundary_volume>* ptr;
		static std::vector<boundary_volume>* unit_ptr;

		static constexpr const auto jsonName = "boundary_volumes.volume$";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundary_volumes;}		
	};
	struct neighborhood_kind{
		using type = neighbor_list;
		using unit_type = neighbor_list;
		static constexpr const uniforms identifier = uniforms::neighborhood_kind;
		neighborhood_kind(const type& val){*ptr = val;}
		neighborhood_kind() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "neighborhood_kind";
		static neighbor_list* ptr;
		static neighbor_list* unit_ptr;

		static constexpr const auto jsonName = "internal.neighborhood_kind";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.neighborhood_kind;}		
	};
	struct dumpNextframe{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::dumpNextframe;
		dumpNextframe(const type& val){*ptr = val;}
		dumpNextframe() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dumpNextframe";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "internal.dumpNextframe";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dumpNextframe;}		
	};
	struct target{
		using type = launch_config;
		using unit_type = launch_config;
		static constexpr const uniforms identifier = uniforms::target;
		target(const type& val){*ptr = val;}
		target() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "target";
		static launch_config* ptr;
		static launch_config* unit_ptr;

		static constexpr const auto jsonName = "internal.target";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.target;}		
	};
	struct hash_size{
		using type = hash_length;
		using unit_type = hash_length;
		static constexpr const uniforms identifier = uniforms::hash_size;
		hash_size(const type& val){*ptr = val;}
		hash_size() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "hash_size";
		static hash_length* ptr;
		static hash_length* unit_ptr;

		static constexpr const auto jsonName = "internal.hash_size";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.hash_size;}		
	};
	struct cell_order{
		using type = cell_ordering;
		using unit_type = cell_ordering;
		static constexpr const uniforms identifier = uniforms::cell_order;
		cell_order(const type& val){*ptr = val;}
		cell_order() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "cell_order";
		static cell_ordering* ptr;
		static cell_ordering* unit_ptr;

		static constexpr const auto jsonName = "internal.cell_ordering";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.cell_order;}		
	};
	struct cell_structure{
		using type = cell_structuring;
		using unit_type = cell_structuring;
		static constexpr const uniforms identifier = uniforms::cell_structure;
		cell_structure(const type& val){*ptr = val;}
		cell_structure() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "cell_structure";
		static cell_structuring* ptr;
		static cell_structuring* unit_ptr;

		static constexpr const auto jsonName = "internal.cell_structure";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = false;

		template<class T> static inline auto& get_member(T& var) { return var.cell_structure;}		
	};
	struct num_ptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::num_ptcls;
		num_ptcls(const type& val){*ptr = val;}
		num_ptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "num_ptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "internal.num_ptcls";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.num_ptcls;}		
	};
	struct boundaryCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::boundaryCounter;
		boundaryCounter(const type& val){*ptr = val;}
		boundaryCounter() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "boundaryCounter";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "internal.boundaryCounter";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryCounter;}		
	};
	struct boundaryLUTSize{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::boundaryLUTSize;
		boundaryLUTSize(const type& val){*ptr = val;}
		boundaryLUTSize() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "boundaryLUTSize";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "internal.boundaryLUTSize";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.boundaryLUTSize;}		
	};
	struct frame{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::frame;
		frame(const type& val){*ptr = val;}
		frame() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "frame";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "internal.frame";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.frame;}		
	};
	struct max_velocity{
		using type = float;
		using unit_type = value_unit<float, SI::velocity>;
		static constexpr const uniforms identifier = uniforms::max_velocity;
		max_velocity(const type& val){*ptr = val;}
		max_velocity() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "max_velocity";
		static float* ptr;
		static value_unit<float, SI::velocity>* unit_ptr;

		static constexpr const auto jsonName = "internal.max_velocity";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_velocity;}		
	};
	struct min_domain{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::min_domain;
		min_domain(const type& val){*ptr = val;}
		min_domain() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "min_domain";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.minAABB";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.min_domain;}		
	};
	struct max_domain{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::max_domain;
		max_domain(const type& val){*ptr = val;}
		max_domain() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "max_domain";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.maxAABB";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_domain;}		
	};
	struct min_coord{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::min_coord;
		min_coord(const type& val){*ptr = val;}
		min_coord() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "min_coord";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.minCoord";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.min_coord;}		
	};
	struct max_coord{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::max_coord;
		max_coord(const type& val){*ptr = val;}
		max_coord() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "max_coord";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.maxCoord";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_coord;}		
	};
	struct cell_size{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::cell_size;
		cell_size(const type& val){*ptr = val;}
		cell_size() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "cell_size";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.cellSize";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.cell_size;}		
	};
	struct grid_size{
		using type = int3;
		using unit_type = int3;
		static constexpr const uniforms identifier = uniforms::grid_size;
		grid_size(const type& val){*ptr = val;}
		grid_size() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "grid_size";
		static int3* ptr;
		static int3* unit_ptr;

		static constexpr const auto jsonName = "internal.gridSize";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.grid_size;}		
	};
	struct ptcl_spacing{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::ptcl_spacing;
		ptcl_spacing(const type& val){*ptr = val;}
		ptcl_spacing() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "ptcl_spacing";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.ptcl_spacing";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.ptcl_spacing;}		
	};
	struct ptcl_support{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::ptcl_support;
		ptcl_support(const type& val){*ptr = val;}
		ptcl_support() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "ptcl_support";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "internal.ptcl_support";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.ptcl_support;}		
	};
	struct config_file{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::config_file;
		config_file(const type& val){*ptr = val;}
		config_file() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "config_file";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.config_file";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.config_file;}		
	};
	struct config_folder{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::config_folder;
		config_folder(const type& val){*ptr = val;}
		config_folder() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "config_folder";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.config_folder";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.config_folder;}		
	};
	struct working_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::working_directory;
		working_directory(const type& val){*ptr = val;}
		working_directory() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "working_directory";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.working_directory";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.working_directory;}		
	};
	struct build_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::build_directory;
		build_directory(const type& val){*ptr = val;}
		build_directory() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "build_directory";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.build_directory";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.build_directory;}		
	};
	struct source_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::source_directory;
		source_directory(const type& val){*ptr = val;}
		source_directory() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "source_directory";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.source_directory";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.source_directory;}		
	};
	struct binary_directory{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::binary_directory;
		binary_directory(const type& val){*ptr = val;}
		binary_directory() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "binary_directory";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "internal.binary_directory";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.binary_directory;}		
	};
	struct timestep{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::timestep;
		timestep(const type& val){*ptr = val;}
		timestep() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "timestep";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "internal.timestep";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.timestep;}		
	};
	struct simulationTime{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::simulationTime;
		simulationTime(const type& val){*ptr = val;}
		simulationTime() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "simulationTime";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "internal.simulationTime";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.simulationTime;}		
	};
	struct radius{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::radius;
		radius(const type& val){*ptr = val;}
		radius() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "radius";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.radius";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.radius;}		
	};
	struct dumpFile{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::dumpFile;
		dumpFile(const type& val){*ptr = val;}
		dumpFile() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dumpFile";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.dumpFile";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dumpFile;}		
	};
	struct max_numptcls{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::max_numptcls;
		max_numptcls(const type& val){*ptr = val;}
		max_numptcls() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "max_numptcls";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.numptcls";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.max_numptcls;}		
	};
	struct hash_entries{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::hash_entries;
		hash_entries(const type& val){*ptr = val;}
		hash_entries() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "hash_entries";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.hash_entries";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hash_entries;}		
	};
	struct mlm_schemes{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::mlm_schemes;
		mlm_schemes(const type& val){*ptr = val;}
		mlm_schemes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "mlm_schemes";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.mlm_schemes";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.mlm_schemes;}		
	};
	struct deviceRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::deviceRegex;
		deviceRegex(const type& val){*ptr = val;}
		deviceRegex() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "deviceRegex";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.deviceRegex";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.deviceRegex;}		
	};
	struct hostRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::hostRegex;
		hostRegex(const type& val){*ptr = val;}
		hostRegex() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "hostRegex";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.hostRegex";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hostRegex;}		
	};
	struct debugRegex{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::debugRegex;
		debugRegex(const type& val){*ptr = val;}
		debugRegex() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "debugRegex";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.debugRegex";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.debugRegex;}		
	};
	struct density{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::density;
		density(const type& val){*ptr = val;}
		density() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "density";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.density";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.density;}		
	};
	struct rest_density{
		using type = float;
		using unit_type = value_unit<float, SI::density>;
		static constexpr const uniforms identifier = uniforms::rest_density;
		rest_density(const type& val){*ptr = val;}
		rest_density() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "rest_density";
		static float* ptr;
		static value_unit<float, SI::density>* unit_ptr;

		static constexpr const auto jsonName = "particle_settings.rest_density";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.rest_density;}		
	};
	struct densitySteps{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::densitySteps;
		densitySteps(const type& val){*ptr = val;}
		densitySteps() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "densitySteps";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "simulation_settings.densitySteps";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.densitySteps;}		
	};
	struct dfsph_density_error{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::dfsph_density_error;
		dfsph_density_error(const type& val){*ptr = val;}
		dfsph_density_error() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dfsph_density_error";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "dfsph_settings.density_error";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dfsph_density_error;}		
	};
	struct dfsph_iterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::dfsph_iterations;
		dfsph_iterations(const type& val){*ptr = val;}
		dfsph_iterations() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dfsph_iterations";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "dfsph_settings.iterations";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dfsph_iterations;}		
	};
	struct dfsph_density_eta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::dfsph_density_eta;
		dfsph_density_eta(const type& val){*ptr = val;}
		dfsph_density_eta() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dfsph_density_eta";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "dfsph_settings.eta";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dfsph_density_eta;}		
	};
	struct dfsph_divergence_eta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::dfsph_divergence_eta;
		dfsph_divergence_eta(const type& val){*ptr = val;}
		dfsph_divergence_eta() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "dfsph_divergence_eta";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "dfsph_settings.eta_divergence";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.dfsph_divergence_eta;}		
	};
	struct density_error{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::density_error;
		density_error(const type& val){*ptr = val;}
		density_error() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "density_error";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "iisph_settings.density_error";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.density_error;}		
	};
	struct iterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::iterations;
		iterations(const type& val){*ptr = val;}
		iterations() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "iterations";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "iisph_settings.iterations";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.iterations;}		
	};
	struct eta{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::eta;
		eta(const type& val){*ptr = val;}
		eta() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "eta";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "iisph_settings.eta";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.eta;}		
	};
	struct jacobi_omega{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::jacobi_omega;
		jacobi_omega(const type& val){*ptr = val;}
		jacobi_omega() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "jacobi_omega";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "iisph_settings.jacobi_omega";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.jacobi_omega;}		
	};
	struct particleCleanUp{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::particleCleanUp;
		particleCleanUp(const type& val){*ptr = val;}
		particleCleanUp() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "particleCleanUp";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.particleCleanUp";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.particleCleanUp;}		
	};
	struct volumeInlets{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::volumeInlets;
		volumeInlets(const type& val){*ptr = val;}
		volumeInlets() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeInlets";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.volumeInlets";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeInlets;}		
	};
	struct volumeOutlets{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::volumeOutlets;
		volumeOutlets(const type& val){*ptr = val;}
		volumeOutlets() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeOutlets";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.volumeOutlets";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutlets;}		
	};
	struct particle_volumes{
		using type = std::vector<particle_volume>;
		using unit_type = std::vector<particle_volume>;
		static constexpr const uniforms identifier = uniforms::particle_volumes;
		particle_volumes(const type& val){*ptr = val;}
		particle_volumes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "particle_volumes";
		static std::vector<particle_volume>* ptr;
		static std::vector<particle_volume>* unit_ptr;

		static constexpr const auto jsonName = "particle_volumes.volume$";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.particle_volumes;}		
	};
	struct inlet_volumes{
		using type = std::vector<inlet_volume>;
		using unit_type = std::vector<inlet_volume>;
		static constexpr const uniforms identifier = uniforms::inlet_volumes;
		inlet_volumes(const type& val){*ptr = val;}
		inlet_volumes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "inlet_volumes";
		static std::vector<inlet_volume>* ptr;
		static std::vector<inlet_volume>* unit_ptr;

		static constexpr const auto jsonName = "inlet_volumes.volume$";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.inlet_volumes;}		
	};
	struct volumeOutletCounter{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::volumeOutletCounter;
		volumeOutletCounter(const type& val){*ptr = val;}
		volumeOutletCounter() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeOutletCounter";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "outlet_volumes.volumeOutletCounter";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutletCounter;}		
	};
	struct volumeOutletTime{
		using type = float;
		using unit_type = value_unit<float, SI::s>;
		static constexpr const uniforms identifier = uniforms::volumeOutletTime;
		volumeOutletTime(const type& val){*ptr = val;}
		volumeOutletTime() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "volumeOutletTime";
		static float* ptr;
		static value_unit<float, SI::s>* unit_ptr;

		static constexpr const auto jsonName = "outlet_volumes.volumeOutletTime";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.volumeOutletTime;}		
	};
	struct outlet_volumes{
		using type = std::vector<outlet_volume>;
		using unit_type = std::vector<outlet_volume>;
		static constexpr const uniforms identifier = uniforms::outlet_volumes;
		outlet_volumes(const type& val){*ptr = val;}
		outlet_volumes() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "outlet_volumes";
		static std::vector<outlet_volume>* ptr;
		static std::vector<outlet_volume>* unit_ptr;

		static constexpr const auto jsonName = "outlet_volumes.volume$";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.outlet_volumes;}		
	};
	struct neighborhood{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::neighborhood;
		neighborhood(const type& val){*ptr = val;}
		neighborhood() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "neighborhood";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.neighborhood";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.neighborhood;}		
	};
	struct render_clamp{
		using type = float3;
		using unit_type = float3;
		static constexpr const uniforms identifier = uniforms::render_clamp;
		render_clamp(const type& val){*ptr = val;}
		render_clamp() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_clamp";
		static float3* ptr;
		static float3* unit_ptr;

		static constexpr const auto jsonName = "render_settings.render_clamp";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_clamp;}
		static constexpr const float3 min{-1.f,-1.f,-1.f};
		static constexpr const float3 max{1.f,1.f,1.f};
		static constexpr const float3 step{0.01f,0.01f,0.01f};		
	};
	struct camera_position{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::camera_position;
		camera_position(const type& val){*ptr = val;}
		camera_position() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "camera_position";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "render_settings.camera_position";
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
		static constexpr const uniforms identifier = uniforms::camera_angle;
		camera_angle(const type& val){*ptr = val;}
		camera_angle() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "camera_angle";
		static float3* ptr;
		static float3* unit_ptr;

		static constexpr const auto jsonName = "render_settings.camera_angle";
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
		static constexpr const uniforms identifier = uniforms::camera_resolution;
		camera_resolution(const type& val){*ptr = val;}
		camera_resolution() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "camera_resolution";
		static float2* ptr;
		static float2* unit_ptr;

		static constexpr const auto jsonName = "render_settings.camera_resolution";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_resolution;}		
	};
	struct camera_fps{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::camera_fps;
		camera_fps(const type& val){*ptr = val;}
		camera_fps() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "camera_fps";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "render_settings.camera_fps";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.camera_fps;}		
	};
	struct gl_file{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::gl_file;
		gl_file(const type& val){*ptr = val;}
		gl_file() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "gl_file";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "render_settings.gl_file";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.gl_file;}		
	};
	struct render_min{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_min;
		render_min(const type& val){*ptr = val;}
		render_min() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_min";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "color_map.min";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_min;}
		static constexpr const float min{-100.f};
		static constexpr const float max{100.f};
		static constexpr const float step{0.5f};		
	};
	struct render_max{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::render_max;
		render_max(const type& val){*ptr = val;}
		render_max() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_max";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "color_map.max";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_max;}
		static constexpr const float min{-100.f};
		static constexpr const float max{100.f};
		static constexpr const float step{0.5f};		
	};
	struct render_auto{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::render_auto;
		render_auto(const type& val){*ptr = val;}
		render_auto() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_auto";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "color_map.auto";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_auto;}		
	};
	struct map_flipped{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::map_flipped;
		map_flipped(const type& val){*ptr = val;}
		map_flipped() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "map_flipped";
		static int* ptr;
		static int* unit_ptr;

		static constexpr const auto jsonName = "color_map.map_flipped";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.map_flipped;}		
	};
	struct render_buffer{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::render_buffer;
		render_buffer(const type& val){*ptr = val;}
		render_buffer() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_buffer";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "color_map.buffer";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_buffer;}		
	};
	struct render_colormap{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::render_colormap;
		render_colormap(const type& val){*ptr = val;}
		render_colormap() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "render_colormap";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "color_map.map";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.render_colormap;}		
	};
	struct sorting{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::sorting;
		sorting(const type& val){*ptr = val;}
		sorting() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "sorting";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.resorting";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.sorting;}		
	};
	struct hash_width{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::hash_width;
		hash_width(const type& val){*ptr = val;}
		hash_width() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "hash_width";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.hash_width";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.hash_width;}		
	};
	struct alembic_export{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::alembic_export;
		alembic_export(const type& val){*ptr = val;}
		alembic_export() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "alembic_export";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.alembic";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_export;}		
	};
	struct error_checking{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::error_checking;
		error_checking(const type& val){*ptr = val;}
		error_checking() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "error_checking";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.error_checking";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.error_checking;}		
	};
	struct gl_record{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::gl_record;
		gl_record(const type& val){*ptr = val;}
		gl_record() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "gl_record";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.gl_record";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.gl_record;}		
	};
	struct launch_cfg{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::launch_cfg;
		launch_cfg(const type& val){*ptr = val;}
		launch_cfg() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "launch_cfg";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.launch_cfg";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.launch_cfg;}		
	};
	struct regex_cfg{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::regex_cfg;
		regex_cfg(const type& val){*ptr = val;}
		regex_cfg() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "regex_cfg";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.regex_cfg";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.regex_cfg;}		
	};
	struct resort_algorithm{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::resort_algorithm;
		resort_algorithm(const type& val){*ptr = val;}
		resort_algorithm() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "resort_algorithm";
		static int* ptr;
		static int* unit_ptr;

		static constexpr const auto jsonName = "resort.algorithm";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.resort_algorithm;}		
	};
	struct valid_cells{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::valid_cells;
		valid_cells(const type& val){*ptr = val;}
		valid_cells() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "valid_cells";
		static int* ptr;
		static int* unit_ptr;

		static constexpr const auto jsonName = "resort.valid_cells";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.valid_cells;}		
	};
	struct collision_cells{
		using type = int;
		using unit_type = int;
		static constexpr const uniforms identifier = uniforms::collision_cells;
		collision_cells(const type& val){*ptr = val;}
		collision_cells() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "collision_cells";
		static int* ptr;
		static int* unit_ptr;

		static constexpr const auto jsonName = "resort.collision_cells";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.collision_cells;}		
	};
	struct alembic_file_name{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::alembic_file_name;
		alembic_file_name(const type& val){*ptr = val;}
		alembic_file_name() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "alembic_file_name";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "alembic.file_name";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_file_name;}		
	};
	struct alembic_fps{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::alembic_fps;
		alembic_fps(const type& val){*ptr = val;}
		alembic_fps() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "alembic_fps";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "alembic.fps";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.alembic_fps;}		
	};
	struct support{
		using type = std::string;
		using unit_type = std::string;
		static constexpr const uniforms identifier = uniforms::support;
		support(const type& val){*ptr = val;}
		support() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "support";
		static std::string* ptr;
		static std::string* unit_ptr;

		static constexpr const auto jsonName = "modules.support";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support;}		
	};
	struct support_current_iteration{
		using type = uint32_t;
		using unit_type = uint32_t;
		static constexpr const uniforms identifier = uniforms::support_current_iteration;
		support_current_iteration(const type& val){*ptr = val;}
		support_current_iteration() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "support_current_iteration";
		static uint32_t* ptr;
		static uint32_t* unit_ptr;

		static constexpr const auto jsonName = "support.support_current_iteration";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support_current_iteration;}		
	};
	struct adjusted_particles{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::adjusted_particles;
		adjusted_particles(const type& val){*ptr = val;}
		adjusted_particles() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "adjusted_particles";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "support.adjusted_particles";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.adjusted_particles;}		
	};
	struct omega{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::omega;
		omega(const type& val){*ptr = val;}
		omega() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "omega";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "support.omega";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.omega;}		
	};
	struct target_neighbors{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::target_neighbors;
		target_neighbors(const type& val){*ptr = val;}
		target_neighbors() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "target_neighbors";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "support.target_neighbors";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.target_neighbors;}		
	};
	struct support_leeway{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::support_leeway;
		support_leeway(const type& val){*ptr = val;}
		support_leeway() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "support_leeway";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "support.support_leeway";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.support_leeway;}		
	};
	struct overhead_size{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::overhead_size;
		overhead_size(const type& val){*ptr = val;}
		overhead_size() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "overhead_size";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "support.overhead_size";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.overhead_size;}		
	};
	struct error_factor{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::error_factor;
		error_factor(const type& val){*ptr = val;}
		error_factor() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "error_factor";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "support.error_factor";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.error_factor;}		
	};
	struct surfaceDistance{
		using type = bool;
		using unit_type = bool;
		static constexpr const uniforms identifier = uniforms::surfaceDistance;
		surfaceDistance(const type& val){*ptr = val;}
		surfaceDistance() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surfaceDistance";
		static bool* ptr;
		static bool* unit_ptr;

		static constexpr const auto jsonName = "modules.surfaceDistance";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surfaceDistance;}		
	};
	struct surface_levelLimit{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::surface_levelLimit;
		surface_levelLimit(const type& val){*ptr = val;}
		surface_levelLimit() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_levelLimit";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.level_limit";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_levelLimit;}		
	};
	struct surface_neighborLimit{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::surface_neighborLimit;
		surface_neighborLimit(const type& val){*ptr = val;}
		surface_neighborLimit() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_neighborLimit";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.neighborLimit";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_neighborLimit;}		
	};
	struct surface_phiMin{
		using type = float;
		using unit_type = value_unit<float, SI::m>;
		static constexpr const uniforms identifier = uniforms::surface_phiMin;
		surface_phiMin(const type& val){*ptr = val;}
		surface_phiMin() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_phiMin";
		static float* ptr;
		static value_unit<float, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.phi_min";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_phiMin;}		
	};
	struct surface_phiChange{
		using type = float;
		using unit_type = float;
		static constexpr const uniforms identifier = uniforms::surface_phiChange;
		surface_phiChange(const type& val){*ptr = val;}
		surface_phiChange() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_phiChange";
		static float* ptr;
		static float* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.phi_change";
		static constexpr const bool modifiable = false;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_phiChange;}		
	};
	struct surface_distanceFieldDistances{
		using type = float3;
		using unit_type = value_unit<float3, SI::m>;
		static constexpr const uniforms identifier = uniforms::surface_distanceFieldDistances;
		surface_distanceFieldDistances(const type& val){*ptr = val;}
		surface_distanceFieldDistances() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_distanceFieldDistances";
		static float3* ptr;
		static value_unit<float3, SI::m>* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.distanceFieldDistances";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_distanceFieldDistances;}		
	};
	struct surface_iterations{
		using type = int32_t;
		using unit_type = int32_t;
		static constexpr const uniforms identifier = uniforms::surface_iterations;
		surface_iterations(const type& val){*ptr = val;}
		surface_iterations() = default;
		operator type() const{return * ptr;}
		static constexpr const auto variableName = "surface_iterations";
		static int32_t* ptr;
		static int32_t* unit_ptr;

		static constexpr const auto jsonName = "surfaceDistance.surfaceIterations";
		static constexpr const bool modifiable = true;
		static constexpr const bool visible = true;

		template<class T> static inline auto& get_member(T& var) { return var.surface_iterations;}		
	};



}

extern std::tuple<parameters::adaptive, parameters::adaptive_resolution, parameters::adaptiveRatio, parameters::blendsteps, parameters::adaptive_delay, parameters::split_ptcls, parameters::blend_ptcls, parameters::merged_ptcls, parameters::shared_ptcls, parameters::pressure, parameters::volumeBoundary, parameters::xsph, parameters::drag, parameters::viscosity, parameters::tension, parameters::vorticity, parameters::monaghan_viscosity, parameters::intertia_inverse, parameters::vorticity_viscosity, parameters::vorticity_nu_t, parameters::xsph_viscosity, parameters::tension_akinci, parameters::air_velocity, parameters::external_force, parameters::timestep_min, parameters::timestep_max, parameters::movingBoundaries, parameters::boundaryDampening, parameters::boundaryObject, parameters::domainWalls, parameters::neighborlimit, parameters::moving_planes, parameters::volumeBoundaryCounter, parameters::boundary_volumes, parameters::neighborhood_kind, parameters::dumpNextframe, parameters::target, parameters::hash_size, parameters::cell_order, parameters::cell_structure, parameters::num_ptcls, parameters::boundaryCounter, parameters::boundaryLUTSize, parameters::frame, parameters::max_velocity, parameters::min_domain, parameters::max_domain, parameters::min_coord, parameters::max_coord, parameters::cell_size, parameters::grid_size, parameters::ptcl_spacing, parameters::ptcl_support, parameters::config_file, parameters::config_folder, parameters::working_directory, parameters::build_directory, parameters::source_directory, parameters::binary_directory, parameters::timestep, parameters::simulationTime, parameters::radius, parameters::dumpFile, parameters::max_numptcls, parameters::hash_entries, parameters::mlm_schemes, parameters::deviceRegex, parameters::hostRegex, parameters::debugRegex, parameters::density, parameters::rest_density, parameters::densitySteps, parameters::dfsph_density_error, parameters::dfsph_iterations, parameters::dfsph_density_eta, parameters::dfsph_divergence_eta, parameters::density_error, parameters::iterations, parameters::eta, parameters::jacobi_omega, parameters::particleCleanUp, parameters::volumeInlets, parameters::volumeOutlets, parameters::particle_volumes, parameters::inlet_volumes, parameters::volumeOutletCounter, parameters::volumeOutletTime, parameters::outlet_volumes, parameters::neighborhood, parameters::render_clamp, parameters::camera_position, parameters::camera_angle, parameters::camera_resolution, parameters::camera_fps, parameters::gl_file, parameters::render_min, parameters::render_max, parameters::render_auto, parameters::map_flipped, parameters::render_buffer, parameters::render_colormap, parameters::sorting, parameters::hash_width, parameters::alembic_export, parameters::error_checking, parameters::gl_record, parameters::launch_cfg, parameters::regex_cfg, parameters::resort_algorithm, parameters::valid_cells, parameters::collision_cells, parameters::alembic_file_name, parameters::alembic_fps, parameters::support, parameters::support_current_iteration, parameters::adjusted_particles, parameters::omega, parameters::target_neighbors, parameters::support_leeway, parameters::overhead_size, parameters::error_factor, parameters::surfaceDistance, parameters::surface_levelLimit, parameters::surface_neighborLimit, parameters::surface_phiMin, parameters::surface_phiChange, parameters::surface_distanceFieldDistances, parameters::surface_iterations> uniforms_list;

template<typename T>
using parameter = typename T::type;
template<typename T>
using parameter_u = typename T::unit_type;


