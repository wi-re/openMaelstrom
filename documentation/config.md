Configuration
---

Under construction

In general our simulation loads configurations from JSON files, which contain information about the simulation and rendering. An example might look like this
```JSON
{
  "modules": {
    "adaptive": true,
    "resorting": "compactMLM",
    "neighborhood": "constrained"
  },
  "adaptive": {
    "resolution": 64,
    "delay": 1.0
  },
  "inlet_volumes": {
    "volume1": {
      "file": "Volumes/Fluid.vdb",
      "dur": 10,
      "delay": 0.0,
      "vel": "0 20 0 0"
    }
  },
  "boundary_volumes": {
    "volume1": {
      "file": "Volumes/pillars.vdb",
      "kind": 2
    }
  },
  "particle_settings": {
    "radius": 0.5
  },
  "render_settings": {
    "camera_position": "70.7539 39.5768 60.3006",
    "camera_angle": "160.5 0 -124"
  },
  "simulation_settings": {
    "boundaryObject": "Objects/domain.obj",
    "domainWalls": "x+-y+-z+-",
    "numptcls": 3709000,
    "timestep_min": 0.0001,
    "timestep_max": 0.006,
    "neighborlimit": 120
  }
}
```
In this configuration multiple _namespaces_ of parameters are used, i.e. modules and particle_settings, to make the configuration more readable. Parameters are set directly based on their identifiers, where multi dimensional quantities are entered by separating dimensions by spaces and lists of parameters are entered by manually numbering the entries, i.e. volume1. As such in this configuration `modules.adaptive:true` means that the simulation uses adaptivity with an adaptive ratio of `adaptive.resolution:64`. These configurations can be overridden with command line arguments, i.e. `-j simulation_settings.numptcls=7000000`. 

All configurations are accompanied by a houdini (`.hipnic`) file that represents the simulation domain, and which is used to create and manage fluid volumes and boundaries. 

Configuration Parameters
---
In general there are a few basic ways to input parameters, based on the type of the parameter.

_float/int_ parameters are entered as normal json attribute values, i.e. `simulation_settings.numptcls=64000`

_multi dimensional_ parameters are entered as strings with components separated by spaces, i.e. `simulation_settings.external_force="10 0 0 0"`

_string_ parameters are entered as normal string attributes, i.e. `simulation_settings.domain_object="domain.obj"`

_vector_ parameters are entered as separate json entries with manual numbering, i.e. for a vector of integers requiring an input of `intVal$` the entry form would be 
```json
{intVal1:1, intVal2:2}
```

_complex_ parameters are entered based on their type description, i.e. a type 
```
file -  type:string, default:""
rate - type:float default:-1.0
```
would be entered as
```json
{
	file: '~/file.end',
	rate: 100.0
}
```

These inputs can be combined, i.e. for vectors of complex parameters.

Below is a list of all current (14.10.2019) simulation parameters listed by their namespaces. The listings are separated into option parameters, that allow a choice of inputs, scalar inputs, that are user defined, and computed parameters, that serve to observe the simulation.

Overview of parameters:


__[modules](#modules)__ contains parameters that are used to toggle or choose from options for the simulation, i.e. what pressure solver is being used and whether or not adaptivity is supported.

__[adaptive](#adaptive)__ contains parameters that control the behavior of our spacially adaptive method.

__[vorticitySettings](#vorticitySettings)__ contains parameters that control the behavior of the Micropolar Turblence model for SPH.

__[simulation_settings](#simulation_settings)__ contains generic simulation settings, i.e. timestep limits and maximum number of particles.

__[boundary_volumes](#boundary_volumes)__ contains information about rigid obstacles using our boundary integral based method.

__[moving_plane](#moving_plane)__ contains information about moving flat (infinite) plane obstacles, i.e. to create wave pools.

__[internal](#internal)__ contains parameters that should not be changed manually.

__[dfsph_settings](#dfsph_settings)__ contains parameters to control the solver of Divergence Free SPH

__[iisph_settings](#iisph_settings)__ contains parameters to control the solver of Implicit Incompressible SPH, and related methods like Interlinked Pressure Solvers.

__[particle_volumes](#particle_volumes)__ contains information about initial fluid volumes, i.e. dam break volumes, that are created on simulation start.

__[rigid_volumes](#rigid_volumes)__ contains information about rigid obstacles using particle based representations, i.e. Interlinked pressure solvers for SPH.

__[particleSets](#particleSets)__ contains a list of files, that contain particle data, that should be loaded into the simulation on startup.

__[inlet_volumes](#inlet_volumes)__ contains information about fluid inlets, i.e. streams, that create fluid for a specific duration.

__[outlet_volumes](#outlet_volumes)__ contains information about fluid outlets, so regions in which particles are removed, and their removal rates.

__[rtxScene](#rtxScene)__ contains information about boxes and spheres that should be rendered when using ray tracing.

__[render_settings](#render_settings)__ controls all rendering aspects, i.e. anisotropic surfaces, ray bounces and camera settings.

__[color_map](#color_map)__ contains parameters that control the behavior of the color mapping, i.e. what buffers in what coloring in what range.

__[resort](#resort)__ contains parameters that control various resorting methods contained within the simulation.

__[alembic](#alembic)__ contains parameters that control the simulation output as Alembic `*.abc` files.

__[support](#support)__ contains parameters that control the simulation with respect to constraining support radii, either using our 2016 SCA paper or the recent 2019 VMV paper.

__[surfaceDistance](#surfaceDistance)__ contains parameters that control aspects of surface distance calculations, i.e. the maximum depth that will be calculated.

<a name="Modules"></a>
Modules
---
These parameters, in general, are used to enable or disable whole parts of the simulation, i.e. by choosing pressure solvers or enabling spatial adaptivity.

Option parameters:

|name|default|options|description|
|---|---|---|---|
|adaptive|`false`|`false`,`true`| Enables/Disables spatial adaptivity in the simulation|
|pressure|`DFSPH`|`DFSPH`,<br/>`IISPH17`,<br/>`IISPH17_RBAND`|Selects a pressure solver to use to ensure incompressibility and possibly divergence freedom|
|volumeBoundary|`true`|`false`,`true`| Enables/Disables complex level set based boundary objects| 
|xsph|`true`|`false`,`true`| Enables/Disables an XSPH based artificial viscosity formulation|
|drag|`Gissler17`|`None`,`Gissler17`|Enables the air-fluid interaction model of Gissler et al 2017 |
|viscosity|`true`|`false`,`true`| Enables/Disables an artificial viscosity model based on Monaghan 1995|
|tension|`Akinci`|`None`,`Akinci` |Selects the surface tension model to use for the simulation|
|vorticity|`Bender17`|`None`,`Bender17`|Enables the micropolar vorticity model of Bender et al.|
|movingBoundaries|`true`|`false`,`true`|Enables/Disables support for moving wall boundaries, i.e. to create a wave pool|
|debug|`false`|`false`,`true`|Enables/Disables support for the particle picking function, slows down simulation slightly due to memory copying|
|density|||Deprecated parameter to pick a density calculation method |
|particleCleanUp|`true`|`false`,`true`|If set to false particles that are generated inside objects are not removed|
|volumeInlets|`true`|`false`,`true`|Enables/Disables support for fluid inlet streams|
|volumeOutlets|`true`|`false`,`true`|Enables/Disables support for fluid outlets|
|resorting|`compactMLM`|`linear_cell`,<br/>`hashed_cell`,<br/>`MLM`,`compactMLM`|Used to pick a data structure method, should always be `compactMLM` unless specific things are tested|
|hash_width|`64bit`|`32bit`,`64bit`|Length of the Morton code used for resorting particles. Should always be `64bit`|
|alembic_export|`false`|`false`,`true`|Enables/Disables support for export of simulation data as Alembic files|
|error_checking|`true`|`false`,`true`|Enables/Disables checking for CUDA errors after every function call|
|gl_record|`false`|`false`,`true`|Enables/Disables openGL recording, should be set via the `--record` command line argument|
|launch_cfg|||Internal parameter, should not be set|
|regex_cfg|||Internal parameter, should not be set|
|support|`constrained`|`none`,`constrained`|Enables/Disables support radius constrainment via histograms|
|surfaceDistance|`false`|`false`,`true`|Enables/Disables an SPH based surface detection (unstable for adaptive fluids)|
|surfaceDetection|`true`|`false`,`true`|Enables/Disables a geometric surface detection (fairly stable for adaptive fluids)|
|neighborhood|`constrained`|`cell_based`,`basic`,<br/>`compactCell`,`masked`,<br/>`compactMLM`|Used to select the neighbor list method. By default only support for constrained is compiled.|
|neighborhoodSorting|`false`|`false`,`true`|Enables/Disables sorting neighborlists by ascending distance of particle pairs|
|rayTracing|`true`|`false`,`true`|Enables/Disables ray tracing functionality|
|anisotropicSurface|`true`|`false`,`true`|Enables/Disables support for an anisotropic surface function based rendering|
|renderMode|||Deprecated parameter|

<a name="adaptive"></a>
adaptive
---

Option parameters:

|name|default|options|description|
|---|---|---|---|
|useVolume|`1`|`0`,`1`|This parameter changes how volumes and radii change in adaptive fluids. If set to `1` a particle at half of the maximum surface distance will have half the volume, with `0` it will have half the radius. This influences performance significantly but not visually until $1000:1$ adaptive ratios.|
|detailedAdaptiveStatistics|`1`|`0`,`1`|Displays Statistics about how many particles were split into how many particles on screen.|

Scalar parameters:

|name|default|description|
|---|---|---|
|adaptivityScaling|`1.0`| Parameter used internally for an optimization process, should be left unchanged.|
|adaptivityThreshold|`1.0`|Parameter used internally for an optimization process, should be left unchanged.|
|adaptivityGamma|`0.1`|Parameter that controls an internal gradient descent process, should be left unchanged.|
|resolution|`32.0`|Controls the adaptive ratio (with respect to volume), also known as $\alpha$ in equations.|
|blendSteps|`10.0`|The number of timesteps over which temporal blending should ideally take place, also known as $\Theta$|
|delay|`1.0`|Most simulations do not benefit from an adaptive ratio at $t=0$, as the interesting interactions usually take place a few seconds into the simulation, so controlling the delay after which adaptivity happens is very useful. Given in seconds of simulated time.|

Computed parameters:

|name|description|
|---|---|
|minVolume|Contains the volume represented by the smallest particle of the simulation.|
|ratio|Contains the ratio of minVolume to the volume of a particle of the base resolution.|
|splitPtcls|An array with 16 entries that contains the number of particles that were split during the last few timesteps into a given number of particles.|
|blendedPtcls|Contains the number of particles that are in the process of blending right now.|
|mergedPtcls|Contains the number of particles that were merged during the last few timesteps.|
|sharedPtcls|Contains the number of particles that participated in sharing during the last few timesteps.|

<a name="particle_settings"></a>
particle_settings
---
Scalar parameters:

|name|default|description|
|---|---|---|
|viscosity|`5.0`|Controls the artificial viscosity model of `Monaghan 1995`.|
|boundaryViscosity|`0.0375`|Controls the friction coefficient of boundary integral based obstacles (including the domain boundary), based on the Coloumb force from `Density maps for improved Boundary Handling`.|
|xsph_viscosity|`0.05`|Controls the `XSPH` based artificial viscosity.|
|rigidAdhesion_akinci|`0.0`|Controls the adhesion of fluids to rigid objects (excluding the domain boundary) based on `Versatile Surface Tension and Adhesion for SPH Fluids`|
|boundaryAdhesion_akinci|`0.0`|Controls the adhesion of fluids to the domain boundary (excluding rigid objects) based on `Versatile Surface Tension and Adhesion for SPH Fluids`|
|tension_akinci|`0.15`|Controls the surface tension parameter based on `Versatile Surface Tension and Adhesion for SPH Fluids`, also known as $\kappa$.|
|air_velocity|`0 0 0 0`|Controls the external air velocity that influences the fluid based on `Approximate Air-Fluid Interactions for SPH`|
|radius|`0.5`|Controls the basic, uniform, fluid particle radius, also known as $r$ or $r_\text{base}$|
|rest_density|`998.0`|Defines the uniform rest density of all fluid particles, also known as $\rho_0$. The simulation does not support multiple phases.|

<a name="vorticitySettings"></a>
vorticitySettings
---

These parameters all influence the `Turbulent Micropolar SPH Fluids with Foam` method

Scalar parameters:

|name|default|description|
|---|---|---|
|inertiaInverse|`0.5`|Defines $\frac{1}{\Theta}$, should always be set to 0.5, as per the reference paper.|
|viscosityOmega|`0.1`|Defines angular viscosity parameter, similar to XSPH, also known as $\eta$.|
|vorticityCoeff|`0.05`|Defines the strength of the vorticity model, also known as $\nu_t$.|

<a name="simulation_settings"></a>
simulation_settings
---

Scalar parameters:

|name|default|description|
|---|---|---|
|external_force|`0 0 -9.81 0`|Usually this parameter describes the external gravity that is uniformly applied to all particles everywhere.|
|timestep_min|`0.001`|The smallest timestep that can be used in the simulation. Theoretically a timestep based on a CFL Condition can be arbitrarily small, but requiring a minimum timestep can be useful due to accumulated numerical issues from very small timesteps outweighing the stability of a proper CFL timestep.|
|timestep_max|`0.01`|The maximum timestep that can be used in the simulation. Theoretically the largest timestep should be the CFL based one, but this leads to issues when trying to export a sequence of particle positions at a set rate, i.e. once every $\frac{1}{60}s$, and allowing arbitrarily large timesteps can lead to very high pressure solver iteration counts|
|boundaryDampening|`0.97`|Currently not used. Defines the dampening of impulse based boundary handling methods.|
|LUTOffset|`0.0`|Should not be changed|
|boundaryObject|`""`|Contains the path to an object file that describes the AABB of the simulation domain.|
|domainWalls|`"x+-y+-z+-"`|This parameter can be used to remove some of the domain walls based on the boundaryObjects, i.e. `"xyz-"` would only create a floor. One wall has to always be created due to memory allocation.|
|neighborlimit|`150`|Contains the maximum number of neighbors for a particle, should be adjusted based on on the kernel function and whether or not adaptivity is used.|
|dumpFile|`"simulation.dump"`|Contains a path to a file that simulation memory dumps are written to, i.e. via hitting `o` in the GUI.|
|numptcls|`1000000`|The maximum number of particles allowed.|
|deviceRegex|`""`|Contains a regex that is matched against called SPH functions where a match forces the function to be run on the GPU.|
|hostRegex|`""`|Contains a regex that is matched against called SPH functions where a match forces the function to be run on the CPU.|
|debugRegex|`""`|Contains a regex that is matched against called SPH functions where a match forces the function to be run on the CPU in a single threaded variant.|
|densitySteps|`10`|Deprecated parameter.|

Computed parameters:

|name|description|
|---|---|
|hash_entries|Contains the size of the hash table used for MLM data structures. Determined as the smallest prime larger than the maximum number of particles.|
|mlm_schemes|Contains the number of MLM levels that are created, based on the adaptive ratio at simulation startup.|

<a name="boundary_volumes"></a>
boundary_volumes
---

Computed parameters:

|name|description|
|---|---|
|volumeBoundaryCounter|Contains the number of valid rigid objects.|

Vector parameter: `volume$` of complex type boundaryVolume

Complex type _boundaryVolume_:
```
file: type:string, default:""
density: type:float, default:998
position: type:float3, default:"0 0 0"
velocity: type:float3, default:"0 0 0"
angularVelocity: type:float4, default: "0.5*pi 0 0 0"
angle: type:float3, default:"0 0 0"
kind: type:int, default:0
animationPath: type:string, default:""
```
<a name="moving_plane"></a>
moving_plane
---

Vector parameter: `plane$` of complex type movingPlane

Complex type _movingPlane_:
```
	complex_type<float3> plane_position{ "pos", {0.f,0.f,0.f}};
	complex_type<float3> plane_normal{ "norm", {0.f,0.f,0.f}};
	complex_type<float3> plane_direction{ "dir", {0.f,0.f,0.f}};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> magnitude{ "mag", 0.f};
	complex_type<float> frequency{ "freq", 0.f};
	complex_type<int32_t> index{ "idx", 0};
```

<a name="internal"></a>
internal
---
These parameters should not be set by the user.

Computed parameters:

|name|default|
|---|---|
|neighborhood_kind|Internal representation of the neighborlist algorithm used.|
|dumpNextframe|Internal flag that is set if a memory dump is written on the next timestep.|
|dumpforSSSPH|Internal deprecated flag.|
|target|Internal flag describing if the simulation runs on GPU or CPU.|
|hash_size|Internal parameter describing the Morton code length.|
|cell_ordering|Internal parameter describing the mapping function, i.e. Morton code or linear, used.|
|cell_structure|Internal parameter describing the used data structure method|
|num_ptcls|Contains the number of currently valid particles.|
|boundaryCounter|Contains the number of valid flat boundary obstacles.|
|boundaryLUTSize|Internal parameter for boundary integral based methods.|
|frame|Contains the current simulation frame, based on timesteps.|
|max_velocity|Contains the highest, scalar, velocity of any particle.|
|minAABB|Contains the minimum AABB of the initial simulation domain.|
|maxAABB|Contains the maximum AABB of the initial simulation domain.|
|minCoord|Contains the minimum of the current simulations AABB based on particle positions.|
|maxCoord|Contains the maximum of the current simulations AABB based on particle positions.|
|cellSize|Contains the size of the uniform cells used for data handling.|
|gridSize|Contains the size of the uniform grid (if it was dense) at the lowest level.|
|ptcl_spacing|Internal representation of the ideal spacing of particles in a hexahedral grid.|
|ptcl_support|Contains the support radius corresponding to $r_\text{base}$.|
|config_file|Contains the path to the current configuration file.|
|config_folder|Contains the path to the folder that contains the current configuration.|
|working_directory|Contains the path of the working directory.|
|build_directory|Contains the path under which the source code was built.|
|source_directory|Contains the path containing the source, at the time of compilation.|
|binary_directory|Contains the path to the executable.|
|timestep|Current timestep of the simulation.|
|simulationTime|Represents the total time that has been simulated in seconds.|

<a name="dfsph_settings"></a>
dfsph_settings
---

Scalar parameters:

|name|default|description|
|---|---|---|
|densityEta|`0.0001`|User defined parameter to set the stopping criterion for the incompressible part.|
|divergenceEta|`0.001`|User defined parameter to set the stopping criterion for the divergence free part.|

Computed parameters:

|name|description|
|---|---|
|densityError|The estimated density error when the incompressible solver stopped.|
|divergenceError|The estimated divergence error when the divergence free solver stopped.|
|densitySolverIterations|The number of incompressible solver iterations.|
|divergenceSolverIterations|The number of divergence free solver iterations.|

<a name="iisph_settings"></a>
iisph_settings
---

Scalar parameters:

|name|default|description|
|---|---|---|
|eta|`0.1`|User defined parameter to set the stopping criterion.|
|jacobi_omega|`0.2`|Relaxation parameter of the relaxed Jacobi method.|

Computed parameters:

|name|description|
|---|---|
|density_error|The estimated density error when the solver stopped.|
|iterations|The number of solver iterations.|

<a name="particle_volumes"></a>
particle_volumes
---

Vector parameter: `volume$` of complex type particleVolume

Complex type _particleVolume_:
```
	complex_type<std::string> fileName{ "file", ""};
	complex_type<float> density{ "density", 998.0f};
	complex_type<float3> position{ "position", {0.f,0.f,0.f}};
	complex_type<float3> velocity{ "velocity", {0.f,0.f,0.f}};
	complex_type<float4> angularVelocity{ "angularVelocity", {CUDART_PI_F * 0.5f,0.f,0.f,0.f}};
	complex_type<float3> angle{ "angle", {0.f,0.f,0.f}};
	complex_type<int32_t> kind{ "kind", 0};
	complex_type<std::string> animationPath{ "animationPath", ""};
```

<a name="rigid_volumes"></a>
rigid_volumes
---

Scalar parameters:

|name|default|description|
|---|---|---|
|gamma|`0.7`|Parameter determining particle sampling density|
|beta|`0.1`|Parameter determining particle sampling density|

Vector parameter: `volume$` of complex type rigidVolume

Complex type _rigidVolume_:
```
	complex_type<std::string> fileName{ "file", ""};
	complex_type<std::string> kind{ "kind", ""};
	complex_type<float> density{ "density", 1.f};
	complex_type<float3> shift{ "shift", {0.f,0.f,0.f}};
	complex_type<float> concentration{ "concentration", 0.f};
	complex_type<float> timeToEmit{ "timeToEmit", 0.f};

```

<a name="particleSets"></a>
particleSets
---

Vector parameter: `set$` of type `string`

<a name="inlet_volumes"></a>
inlet_volumes
---

Vector parameter: `volume$` of complex type inletVolume

Complex type _inletVolume_:
```
	complex_type<std::string> fileName{ "file", ""};
	complex_type<int32_t> particles_emitted{ "ptcls", 0};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> inlet_radius{ "r", -1.f};
	complex_type<float4> emitter_velocity{ "vel", {0.f,0.f,0.f,0.f}};
```

<a name="outlet_volumes"></a>
outlet_volumes
---

Computed parameters:

|name|description|
|---|---|
|volumeOutletCounter|Number of valid fluid outlets.|
|volumeOutletTime|Internal parameter.|

Vector parameter: `volume$` of complex type outletVolume

Complex type _outletVolume_:
```
	complex_type<std::string> fileName{ "file", ""};
	complex_type<float> duration{ "dur", -1.f};
	complex_type<float> delay{ "del", -1.f};
	complex_type<float> flowRate{ "rate", -1.f};
```

<a name="rtxScene"></a>
rtxScene
---

Vector parameter: `box$` of complex type rtxBox
Vector parameter: `sphere$` of complex type rtxSphere

Complex type _rtxSphere_:
```
	complex_type<float> radius{ "radius", 1.f};
	complex_type<float3> position{ "position", 0.f,0.f,0.f};
	complex_type<float3> emission{ "emission", 0.f,0.f,0.f};
	complex_type<float3> color{ "color", 0.f,0.f,0.f};
	complex_type<int32_t> refl_t{ "material", 0};
```
Complex type _rtxBox_:
```
	complex_type<std::string> maxPosition{ "maxPosition", "1.f 1.f 1.f"};
	complex_type<std::string> minPosition{ "minPosition", "0.f 0.f 0.f"};
	complex_type<float3> emission{ "emission", 0.f,0.f,0.f};
	complex_type<float3> color{ "color", 0.f,0.f,0.f};
	complex_type<int32_t> refl_t{ "material", 0};
```

<a name="render_settings"></a>
render_settings
---

Option parameters

|name|default|options|description|
|---|---|---|---|
|vrtxRenderGrid|`0`|`0`,`1`|Toggles rendering of the sparse data structure|
|vrtxRenderFluid|`1`|`0`,`1`|Toggles rendering of the fluid in general|
|vrtxRenderSurface|`1`|`0`,`1`|Toggles surface extraction|
|vrtxDisplayStats|`1`|`0`,`1`|Toggles display of additional rendering statistics in the GUI|
|vrtxRenderBVH|`1`|`0`,`1`|Toggles rendering of boundary rigid using BVHs|
|vrtxBVHMaterial|`1`|`[0-4]`|Selects the shading used for rigid objects.|
|vrtxDepth|`0`|`0`,`1`|Toggles rendering of depth instead of color.|
|axesRender|`1`|`0`,`1`|Toggles display of xyz axes, only in openGL|
|boundsRender|`1`|`0`,`1`|Toggles display of the simulation Domain as a wireframe box, only in openGL.|
|floorRender|`0`|`0`,`1`|Toggles display of the lower simulation bound as a plane, only in openGL.|
|vrtxRenderNormals|`0`|`0`,`1`|Toggles rendering of normals instead of color.|
|vrtxSurfaceExtraction|`0`|`0`,`1`|Deprecated|
|vrtxRenderMode|`0`|`[0-4]`|Selects the shading used for the fluid volume.|

Scalar parameters

|name|default|description|
|---|---|---|
|apertureRadius|`0.15`|Controls the aperture of the ray traced camera, a $0$ value disables depth of field effects|
|anisotropicLambda|`0.980198`|Used for `Reconstructing  Surfaces of Particle-Based Fluids Using Anisotropic Kernels`, also known as $\lambda$|
|anisotropicNepsilon|`40`|Used for `Reconstructing  Surfaces of Particle-Based Fluids Using Anisotropic Kernels`, also known as $N_\epsilon$|
|anisotropicKs|`1.0`|Used for `Reconstructing  Surfaces of Particle-Based Fluids Using Anisotropic Kernels`, also known as $k_s$|
|anisotropicKr|`3.0`|Used for `Reconstructing  Surfaces of Particle-Based Fluids Using Anisotropic Kernels`, also known as $k_r$|
|anisotropicKn|`0.188806`|Used for `Reconstructing  Surfaces of Particle-Based Fluids Using Anisotropic Kernels`, also known as $k_n$|
|focalDistance|`100.0`|Sets the focus distance, relevant when using depth of field effects.|
|vrtxNeighborLimit|`0`|Controls if lone particles are displayed or not.|
|vrtxFluidBias|`0.05`|Controls a bias introduced when bouncing rays to avoid self intersections.|
|vrtxDomainEpsilon|`-1.762063`|Controls the minimum distance to the rendering domain for things to be visible.|
|vrtxDebeersScale|`0.056`|Controls the exponent of a simple DeBeer Absorption model.|
|vrtxDebeer|`0.94902 0.76863 0.50582`|Controls the DeBeer Absortion color.|
|bvhColor|`0.566 0.621 0.641`|Controls the color with which rigid objects are rendered.|
|vrtxFluidColor|`0.897 0.917 1.0`|Controls the color used for the fluid surface.|
|vrtxDepthScale|`0.1`|Used to scale the depth rendering to make the values visible.|
|vrtxWMin|`0.4`|Deprecated|
|vrtxR|`0.586`|Deprecated|
|camera_fov|`96.0`|Controls the field of view of the camera. Behaves slightly differently between openGL and the ray tracer.|
|vrtxWMax|`2.0`|Deprecated|
|vrtxBounces|`5`|The maximum number of bounces for a single ray before it is terminated.|
|auxScale|`1.0`|Deprecated|
|vrtxIOR|`1.3`|Index Of Refraction for transparent rendering.|
|renderSteps|`25`|Number of Monte Carlo path tracing steps taken when outputting the rendered images to file.|
|internalLimit|`40`|Threshold to help detecting interior particles.|
|axesScale|`1.0`|Scales the axes controlled by axesRender.|
|render_clamp|`0 0 0`|Allows cuts through the simulation in openGL. Sign indicates what side of the cut plane is removed and 0 means no cut.|
|camera_position|`125, 0, -50`|Controls the position of the camera, in world space coordinates.|
|camera_angle|`-90, 0, 90`|Controls the angle of the camera, in degrees in world space.|
|camera_resolution|`1920, 1080`|Should not be changed.|
|camera_fps|`60.0`|Controls the rate at which images are written to file.|
|gl_File|`"gl.mp4"`|Controls the output file when capturing the openGL output. Should not be set manually, use the `--record` command line option instead..|

Computed parameters:

|name|description|
|---|---|
|vrtxDomainMin|Contains the minimum coordinate that is rendered when using ray tracing, can be used to cut parts of the simulation.|
|vrtxDomainMax|Contains the maximum coordinate that is rendered when using ray tracing, can be used to cut parts of the simulation.|
|auxCellCount|Contains the number of occupied cells in the auxiliary rendering grid.|

<a name="color_map"></a>
color_map
---

Option parameters:

|name|default|options|description|
|---|---|---|---|
|transfer_mode|`"linear"`|`"linear"`,`"cubicRoot"`,<br/>`"cubic"`,`"squareRoot"`,<br/>`"square"`,`"log"`|Determines the kind of function used to map the input range to $[0,1]$|
|mapping_mode|`"linear"`|`"linear"`,`"cubicRoot"`,<br/>`"cubic"`,`"squareRoot"`,<br/>`"square"`,`"log"`|Applied to the $[0,1]$ scaled range to modify the visual appearance.|
|vectorMode|`"length"`|`"length"`,`"x"`,`"y"`,`"z"`,`"w"`|Controls the kind of mapping used for vector quantities. Length refers to the standard vector length, but only considering the first three vector components.|
|visualizeDirection|`0`|`0`,`1`|If `1` all particles will display the direction of a vector quantity as small lines using geometry shaders.|
|vectorScale|`1.0`||Controls the scaling of the visualized direction to control visibility.|
|vectorScaling|`0`|`0`,`1`|Controls if the length of visualized vectors should first be scaled to $[0,1]$, or visualized as is.|
|pruneVoxel|`0`|`0`,`1`|Internal parameter.|
|auto|`1`|`0`,`1`|If `1` the minimum and maximum visualized value depends on the input data and is not fixed.|
|map_flipped|`0`|`0`,`1`|If `1` the $[0,1]$ range is flipped to change the visual appearance.|

Scalar parameters:

|name|default|options|description|
|---|---|---|---|
|min|`0.0`||The lowest valid input value, mapped to $0$.|
|max|`1.0`||The highest valid input value, mapped to $1$|
|buffer|`density`|all qualified array names|Determines what array is being visualized.|
|map|`infero`|all maps in the cfg folder|Determines what color map is used to color the $[0,1$ range.|

Computed parameters:

|name|description|
|---|---|
|transfer_fn|Internal representation of the transfer mode.|
|mapping_fn|Internal representation of the mapping mode.|

<a name="resort"></a>
resort
---

Computed parameters:

|name|description|
|---|---|
|auxCells|The number of cells occupied in the auxiliary data structure used for rendering.|
|auxCollisions|The number of hash collisions for the auxiliary data structure used for rendering.|
|algorithm|Internal representation of the resorting method used|
|valid_cells|Deprecated parameter.|
|zOrderScale|Contains the ratio of $C_\text{fine}$ and $C_\text{max}$.|
|collision_cells|Deprecated parameter.|
|occupiedCells|Contains an array representing the number of occupied cells in each data structure level.|

<a name="alembic"></a>
alembic
---

Scalar parameters:

|name|default|description|
|---|---|---|
|file_name|`"export/alembic_$f.abc"`|Controls the output path of the alembic output. the `$f` in the string represents the frame counter.|
|fps|`24.0`|The framerate at which the particle data should be exported.|

<a name="support"></a>
support
---

Scalar parameters:

|name|default|description|
|---|---|---|
|omega|`0.97`|Scaling parameter of the 2016 SCA constrainment method.|
|error_factor|3|Scaling parameter of the 2016 SCA constrainment method.|

Computed parameters:

|name|description|
|---|---|
|support_current_iteration|Number of iterations required for the 2016 method.|
|adjusted_particles|Number of particles adjusted during the last step of the 2016 method.|
|target_neighbors|Contains $N_H$|
|support_leeway|Describes the difference of $N_H$ to the maximum number of neighbors allowed per particle $N_C$.|
|overhead_size|Internal parameter.|

<a name="surfaceDistance"></a>
surfaceDistance
---

Scalar parameters:

|name|default|description|
|---|---|---|
|level_limit|`-20.0`|The farthest distance to the surface that is calculated. All deeper particles get assigned this value.|
|neighborLimit|`40`|Threshold value to help detect bulk particles.|
|distanceFieldDistances|`0 0 1.5`|Distance thresholds to the boundary (in x,y,z absolute distance) that cause the particles to be classified as interior. Useful for the floor of the simulation.|

Computed parameters:

|name|description|
|---|---|
|phiMin|Contains the surface distance of the particle closest to the surface.|
|phiChange|Internal parameter.|
|surfaceIterations|Contains the number of iterations required to create the surface distance values.|