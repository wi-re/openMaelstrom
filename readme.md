<img style="width:100%;" id="image" src="./banner.png">

__openMaelstrom__ (opensource Multiplatform Adaptable Expressive and Lean Sph Toolkit for Realism Oriented Methods) is an open-source library for the simulation and rendering of fluids. This simulation is based on the Smoothed Particle Hydrodynamics (SPH) method, which is a grid-free physically based Lagrangian simulation scheme. SPH, in general, works by estimating flow quantities based on local interactions of particles weighed by kernel functions, which can easily be adapted to many physical processes. This approach can be used to simulate a variety of effects, ranging from astrophysics to engineering, but we limit our framework to the simulation of incompressible fluids for Computer Graphics. openMaelstrom implements a variety of methods, including spatially adaptive simulations, that allow for a wide range of flows to be simulated. 

This framework is intended to be run on an Nvidia GPU using CUDA, support for CPUs is experimental and AMD GPUs are not yet supported. This framework is supported on Windows (using Visual Studio 2017) and Linux (using GCC 7.4.0).

The following external dependencies are required: [Alembic](https://github.com/alembic/alembic), [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb), [CUDA 10.0](https://developer.nvidia.com/cuda-downloads), [Bullet](https://github.com/bulletphysics/bullet3), [Qt 5](https://www.qt.io/), [Boost 1.70](https://www.boost.org/) 

The simulation output is in Alembic files, which can be utilized in Houdini using built-in functions (see the example files in the Configurations). The simulation configuration is done using JSON files and supports inputs for fluid volumes and rigid objects as object files, VDB level-sets or ply files.

Author: [Rene Winchenbach](https://www.cg.informatik.uni-siegen.de/), License: MIT

Build Instructions
---

The simulation build itself is based on CMake and intended for 64 bit applications only. Tested configurations are:

- Windows 10 64 Bit, MSVC 19.16.27025.1, CMake 3.15, CUDA 10.0
- Ubuntu 18.04 64 Bit, GCC 7.4.0, CMake 3.15, CUDA 10.0

For some information on how to build the simulation under Windows see [Build Guide](/documentation/build.md)

---

Documentation
---
The usage of the simulation is described [here](/documentation/usage.md)

The configuration system is described [here](/documentation/config.md)

For more information on the design and the unit system see my [Master Thesis](http://www.cg.informatik.uni-siegen.de/data/Publications/2019/thesis-winchenbach.pdf).

---

Features
---
- an open source 3D SPH fluid simulation
- a compile time enforced SI unit system
- a meta modelling system for memory handling
- full GPU support with no reliance on CPU based methods
- incompressible fluid simulation methods (IISPH, DFSPH)
- spatially highly adaptive incompressible flows
- surface tension effects and fluid-air-drag forces
- rigid-fluid coupling for static and dynamic bodies
- two-way coupled rigid-fluid interactions (including for adaptive methods)
- rigid-rigid interactions
- fluid inlet and outlets
- JSON based configuration files
- Alembic output
- built-in ray tracing of particles and opaque/transparent fluid surfaces
- support for anisotropic surfaces
- a variety of neighbor list and data structure methods

---

Pressure solvers
---
- Implicit Incompressible SPH (IISPH)
- Divergence-free smoothed particle hydrodynamics (DFSPH)
- Pressure Boundaries for Implicit Incompressible SPH
- Interlinked SPH Pressure Solvers for Strong Fluid-Rigid Coupling

Boundary Handling
---
- Versatile Rigid-Fluid Coupling for Incompressible SPH (not for adaptive fluids)
- Pressure Boundaries for Implicit Incompressible SPH (not for adaptive fluids)
- Interlinked SPH Pressure Solvers for Strong Fluid-Rigid Coupling(not for adaptive fluids)
- MLS Pressure Extrapolation for the Boundary Handling in
Divergence-Free SPH (not for adaptive fluids)
- A boundary integral based method (supports adaptive fluids)

Fluid effects
---
- Versatile Surface Tension and Adhesion for SPH Fluids
- Robust Simulation of Small-Scale Thin Features
in SPH-based Free Surface Flows
- Approximate Air-Fluid Interactions for SPH
- XSPH based viscosity and traditional artificial viscosity

---

Screenshots
---
![](/documentation/Screenshot_01.png "Screenshot 1")

![](/documentation/Screenshot_02.png "Screenshot 2")

---

Example Simulations
---
Click on the images to open the video on YouTube. All simulations shown here are included and ready to use in the framework under the Configurations folder, including all required models and settings.

[![Watch the video](https://img.youtube.com/vi/KXMKdU_wy2I/maxresdefault.jpg)](https://youtu.be/KXMKdU_wy2I)

[![Watch the video](https://img.youtube.com/vi/QsGAY2UdlyE/maxresdefault.jpg)](https://youtu.be/QsGAY2UdlyE)

[![Watch the video](https://img.youtube.com/vi/OnQGROhHsyA/maxresdefault.jpg)](https://youtu.be/OnQGROhHsyA)

[![Watch the video](https://img.youtube.com/vi/is8Il8afhy0/maxresdefault.jpg)](https://youtu.be/is8Il8afhy0)

[![Watch the video](https://img.youtube.com/vi/SiKOIUJTb14/maxresdefault.jpg)](https://youtu.be/SiKOIUJTb14)

[![Watch the video](https://img.youtube.com/vi/QqVAtQ2lY6A/maxresdefault.jpg)](https://youtu.be/QqVAtQ2lY6A)

[![Watch the video](https://img.youtube.com/vi/qEJBWlRxJ94/maxresdefault.jpg)](https://youtu.be/qEJBWlRxJ94)

[![Watch the video](https://img.youtube.com/vi/ap_02lxHHdw/maxresdefault.jpg)](https://youtu.be/ap_02lxHHdw)

[![Watch the video](https://img.youtube.com/vi/mTmuGcu9Aj8/maxresdefault.jpg)](https://youtu.be/mTmuGcu9Aj8)

[![Watch the video](https://img.youtube.com/vi/mmAaqoA82rA/maxresdefault.jpg)](https://youtu.be/mmAaqoA82rA)

[![Watch the video](https://img.youtube.com/vi/D00wmD8Twgc/maxresdefault.jpg)](https://youtu.be/D00wmD8Twgc)

[![Watch the video](https://img.youtube.com/vi/9VI7gqha8ps/maxresdefault.jpg)](https://youtu.be/9VI7gqha8ps)

[![Watch the video](https://img.youtube.com/vi/86usjsuwVHo/maxresdefault.jpg)](https://youtu.be/86usjsuwVHo)

[![Watch the video](https://img.youtube.com/vi/eAkXCMcxuAc/maxresdefault.jpg)](https://youtu.be/eAkXCMcxuAc)

[![Watch the video](https://img.youtube.com/vi/iAE4p4BvvhQ/maxresdefault.jpg)](https://youtu.be/iAE4p4BvvhQ)

[![Watch the video](https://img.youtube.com/vi/IPUmonPWE-A/maxresdefault.jpg)](https://youtu.be/IPUmonPWE-A)
