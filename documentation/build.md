

For now this readme only covers how to build the program.

# How to build
In order to build the simulation a set of dependencies is required. These steps assume them to be installed in the standard paths and tools like git and svn should be available on the command line. In general this guide assumes all code to be based around some basic root directory for all components called CODE_BASE as an environment variable or via
```bash
SET CODE_BASE=C:/dev
```

The following dependencies are installed:

- Cuda 9.0 or newer, 10.0 preferred for future compatability, available at https://developer.nvidia.com/cuda-toolkit/whatsnew
- Visual Studio 2017 with at least version 15.6, current version 15.9 also works. https://visualstudio.microsoft.com/downloads/ vscode works as well but is not covered here and requires a simular build toolkit of version 15.6+
- Git for checking out various dependencies https://git-scm.com/download/win
- SVN for the same purpose https://tortoisesvn.net/downloads.html
- CMake 3.12 or newer https://cmake.org/download/
- Python is required for some dependencies, it is recommended to use for example Anaconda https://www.anaconda.com/ which also supplies a working qt version.

The following dependencies are not installed:

- Intel Thread Building Blocks https://github.com/01org/tbb/releases should be placed under CODE_BASE/Base/tbb
- Boost https://sourceforge.net/projects/boost/files/boost-binaries/ should be placed under CODE_BASE/Base/boost
- GLEW https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0-win32.zip/downloads should be placed under CODE_BASE/Base/glew
- GLFW http://www.glfw.org/download.html should be placed under CODE_BASE/Base/glfw
- GLUT https://www.transmissionzero.co.uk/software/freeglut-devel/ should be placed under CODE_BASE/Base/glut

Additionally, having clang installed might be useful for linting via clang-tidy but is not required. Once all these requirements are in place (noting that boost might take a while to download due to windows checking it for potential malware) the next step is to build all dependencies that are only available as source distributions.

## checking out things
OpenEXR and Ilmbase are in general available via http://www.openexr.com/downloads however these are notoriously difficult to build, so instead using a branch by Rama Hoetzlein from https://github.com/rchoetzlein/win_openexr.git is advisable. Place the batches at CODE_BASE/patches

```bash
mkdir source
cd source
git clone https://github.com/rchoetzlein/win_openexr.git
git clone https://github.com/rchoetzlein/zlib.git
cd win_openexr
git apply ../../patches/openexr.patch
cd ..
git clone https://github.com/dreamworksanimation/openvdb
cd openvdb\openvdb
git apply ..\..\..\patches\openvdb.patch
cd ..\..
git clone https://github.com/alembic/alembic
cd alembic\lib
git apply ..\..\..\patches\alembic.patch
cd ..\..
git clone https://github.com/Blosc/c-blosc
cd ..
mkdir build
cd build
```

## building 

Builds will all be placed within CODE_BASE/build. First build zlib:
```bash
mkdir zlib
cd zlib 
cmake %CODE_BASE%\source\zlib -G "Visual Studio 15 2017 Win64" -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++latest /we4290"
devenv zlib.sln /build "RelWithDebInfo"
cd ..
```

Next we build IlmBase from the Hoetzlein branch in order to be compatible with windows, for linux builds the standard source code can be used instead.
```bash
mkdir IlmBase
cd IlmBase
cmake %CODE_BASE%\source\win_openexr\IlmBase  -G "Visual Studio 15 2017 Win64" -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++latest /we4290"
devenv ilmbase.sln /build "RelWithDebInfo"
cd ..
```

Next build openEXR using the previously built Ilmbase. Paths are provided via -D statements for cmake to make sure everything is found properly, this means that the lines are significantly longer than can fit in the code block so mark the full line including the start of the next line to get all parts of the cmake command.
```bash
mkdir openEXR
cd openEXR
cmake %CODE_BASE%\source\win_openexr\openEXR  -G "Visual Studio 15 2017 Win64"  -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++latest /we4290" -DILMBASE_HALF_LIBRARY_DEBUG=%CODE_BASE%\build\IlmBase\lib\Half.lib -DILMBASE_HALF_LIBRARY_RELEASE=%CODE_BASE%\build\IlmBase\lib\Half.lib -DILMBASE_IEX_LIBRARY_DEBUG=%CODE_BASE%\build\IlmBase\lib\Iex.lib -DILMBASE_IEX_LIBRARY_RELEASE=%CODE_BASE%\build\IlmBase\lib\Iex.lib -DILMBASE_ILMTHREAD_LIBRARY_DEBUG=%CODE_BASE%\build\IlmBase\lib\IlmThread.lib -DILMBASE_ILMTHREAD_LIBRARY_RELEASE=%CODE_BASE%\build\IlmBase\lib\IlmThread.lib -DILMBASE_IMATH_LIBRARY_DEBUG=%CODE_BASE%\build\IlmBase\lib\Imath.lib -DILMBASE_IMATH_LIBRARY_RELEASE=%CODE_BASE%\build\IlmBase\lib\Imath.lib -DZLIB_INCLUDE_PATH=%CODE_BASE%\build\zlib\include -DZLIB_LIBRARY=%CODE_BASE%\build\zlib\lib\zlib.lib -DZLIB_INCLUDE_DIR=%CODE_BASE%\build\zlib\include 
devenv openexr.sln /build "RelWithDebInfo"
cd ..
```

Blosc is required for certain OpenVDB features and built from standard source.
```bash
mkdir blosc
cd blosc 
cmake %CODE_BASE%\source\c-blosc  -G "Visual Studio 15 2017 Win64" -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++latest /we4290" 
devenv blosc.sln /build "RelWithDebInfo"
cd ..
```

Next up we need to fix some include paths due to inconsistencies in includes for IlmBase and OpenEXR and OpenVDB code which makes them incompatible with each other in certain versions without this hack.
```bash
pushd openEXR
cd include
mkdir openEXR
cd openEXR
xcopy .. .
popd
pushd IlmBase
cd include
mkdir openEXR
cd openEXR
xcopy .. .
popd
cd blosc
xcopy ..\..\source\c-blosc\blosc include /s /i
cd ..
```

OpenVDB is built from source with ABI compatibility set to 5.
```bash
mkdir openVDB
cd openVDB
cmake %CODE_BASE%\source\openvdb -G "Visual Studio 15 2017 Win64" -DCMAKE_CXX_FLAGS="/DOPENVDB_USE_GLFW_3 /DOPENEXR_DLL /bigobj /DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++latest /we4290" -DGLEW_LOCATION=%CODE_BASE%\base\glew -DGLEW_LIBRARY_PATH=%CODE_BASE%\base\glew\lib\Release\x64\glew32.lib -DGLEW_GLEW_LIBRARY=%CODE_BASE%\base\glew\lib\Release\x64\glew32.lib -DBLOSC_LOCATION=%CODE_BASE%\build\blosc -DBLOSC_blosc_LIBRARY=%CODE_BASE%\build\blosc\blosc\RelWithDebInfo\blosc.lib -DZLIB_INCLUDE_PATH=%CODE_BASE%\build\zlib\include -DZLIB_INCLUDE_DIR=%CODE_BASE%\build\zlib\include -DZLIB_LIBRARY=%CODE_BASE%\build\zlib\lib\zlib.lib -DILMBASE_LOCATION=%CODE_BASE%\build\IlmBase -DOPENEXR_LOCATION=%CODE_BASE%\build\openEXR -DOPENVDB_BUILD_UNITTESTS=FALSE -DOPENVDB_BUILD_PYTHON_MODULE=FALSE -DIlmbase_IEX_LIBRARY=%CODE_BASE%\build\IlmBase\lib\Iex.lib -DIlmbase_ILMTHREAD_LIBRARY=%CODE_BASE%\build\IlmBase\lib\IlmThread.lib -DOpenexr_ILMIMF_LIBRARY=%CODE_BASE%\build\openEXR\lib\IlmImf.lib -DTBB_LOCATION=%CODE_BASE%\base\tbb -DTBB_LIBRARY_PATH=%CODE_BASE%\base\tbb\lib\intel64\vc14 -DTbb_TBB_LIBRARY=%CODE_BASE%\base\tbb/lib/intel64/vc14/tbb.lib -DTbb_TBBMALLOC_LIBRARY=%CODE_BASE%\base\tbb/lib/intel64/vc14/tbbmalloc.lib -DTBBMALLOC_LIBRARY_PATH=%CODE_BASE%\base\tbb/lib/intel64/vc14/tbbmalloc.lib -DTBB_PREVIEW_LIBRARY_PATH=%CODE_BASE%\base\tbb/lib/intel64/vc14/tbb_preview.lib -DTbb_TBB_PREVIEW_LIBRARY=%CODE_BASE%\base\tbb/lib/intel64/vc14/tbb_preview.lib -DBOOST_ROOT=%CODE_BASE%\base\boost -DBoost_INCLUDE_DIR=%CODE_BASE%\base\boost -DBOOST_LIBRARYDIR=%CODE_BASE%\base\boost\lib64-msvc-14.1 -DBoost_SYSTEM_LIBRARY_RELEASE=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_system-vc141-mt-x64-1_68.lib -DBoost_SYSTEM_LIBRARY_DEBUG=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_system-vc141-mt-gd-x64-1_68.lib -DBoost_THREAD_LIBRARY_RELEASE=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_thread-vc141-mt-x64-1_68.lib -DBoost_THREAD_LIBRARY_DEBUG=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_thread-vc141-mt-gd-x64-1_68.lib -DBoost_IOSTREAMS_LIBRARY_RELEASE=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_iostreams-vc141-mt-x64-1_68.lib -DBoost_IOSTREAMS_LIBRARY_DEBUG=%CODE_BASE%\base\boost/lib64-msvc-14.1/boost_iostreams-vc141-mt-gd-x64-1_68.lib -DOPENVDB_ABI_VERSION_NUMBER=5 -DGLFW_LOCATION=%CODE_BASE%\base\glfw -DGLFW_glfw_LIBRARY=%CODE_BASE%\base\glfw\lib-vc2015\glfw3.lib -DGLFW_LIBRARY_PATH=%CODE_BASE%\base\glfw\lib-vc2015\glfw3.lib -DGLFW_INCLUDE_DIR=%CODE_BASE%\base\glfw\include -DGLFW_INCLUDE_DIRECTORY=%CODE_BASE%\base\glfw\include
devenv openVDB.sln /build "RelWithDebInfo"
cd ..
```

Alembic is built from source as well with some hacks applied to the include files to make them found properly by certain components (the build script for alembic doesn't copy the headers into the include folder).
```bash
mkdir alembic
cd alembic
cmake %CODE_BASE%\source\alembic  -G "Visual Studio 15 2017 Win64" -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /DOPENEXR_DLL /DOPENVDB_DLL /W3 /GR /EHsc /std:c++latest /we4290" -DILMBASE_INCLUDE_DIR=%CODE_BASE%\build\IlmBase\include -DILMBASE_ROOT=%CODE_BASE%\build\IlmBase -DUSE_TESTS=OFF
devenv alembic.sln /build "RelWithDebInfo"
mkdir include
cd include
xcopy ..\..\..\source\alembic\lib\*.h . /sc
xcopy ..\lib\*.h . /sc
cd ..
cd ..
```

The next 2 steps are optional but recommended for an easier to use environment where only one path needs to be added to %PATH% instead of multiple. However, if this is done the step needs to be repeated when any of the copied dependencies are updated.
```bash
cd %CODE_BASE%
mkdir bin
cd bin
for /R ..\build %f in (*.dll) do xcopy %f . /Dy
for /R ..\base %f in (*.dll) do xcopy %f . /Dy

cd ..
mkdir lib
cd lib

for /R ..\build %f in (*.lib) do xcopy %f . /Dy
for /R ..\base %f in (*.lib) do xcopy %f . /Dy
cd ..
```

Finally we build the simulation from source:
```bash
cd build
mkdir maelstrom
cd maelstrom
cmake %CODE_BASE%\source\maelstrom -G "Visual Studio 15 2017 Win64" -DBLOSC_LOCATION=%CODE_BASE%/build/blosc -DBLOSC_blosc_LIBRARY=%CODE_BASE%/build/blosc/blosc/RelWithDebInfo/blosc.lib -DZLIB_INCLUDE_DIR=%CODE_BASE%/build/zlib/include -DZLIB_LIBRARY=%CODE_BASE%/build/zlib/lib/zlib.lib -DBOOST_ROOT=%CODE_BASE%/base/boost -DBoost_INCLUDE_DIR=%CODE_BASE%/base/boost -DALEMBIC_ABC_LIBRARY=%CODE_BASE%/build/alembic/lib/Alembic/RelWithDebInfo/Alembic.lib -DALEMBIC_INCLUDE_DIR=%CODE_BASE%/build/alembic/include -DTBB_LIBRARY=%CODE_BASE%/base/tbb/lib/intel64/vc14/tbb.lib -DTBB_LIBRARY_DEBUG=%CODE_BASE%/base/tbb/lib/intel64/vc14/tbb_debug.lib -DTBB_MALLOC_LIBRARY=%CODE_BASE%/base/tbb/lib/intel64/vc14/tbbmalloc.lib -DTBB_MALLOC_LIBRARY_DEBUG=%CODE_BASE%/base/tbb/lib/intel64/vc14/tbbmalloc_debug.lib -DTBB_ARCH_PLATFORM=intel64/vc14 -DTBB_INSTALL_DIR=%CODE_BASE%/base/tbb -DILMBASE_HALF_LIBRARY_DEBUG=%CODE_BASE%/build/IlmBase/lib/Half.lib -DILMBASE_HALF_LIBRARY_RELEASE=%CODE_BASE%/build/IlmBase/lib/Half.lib -DILMBASE_IEX_LIBRARY_DEBUG=%CODE_BASE%/build/IlmBase/lib/Iex.lib -DILMBASE_IEX_LIBRARY_RELEASE=%CODE_BASE%/build/IlmBase/lib/Iex.lib -DILMBASE_ILMTHREAD_LIBRARY_DEBUG=%CODE_BASE%/build/IlmBase/lib/IlmThread.lib -DILMBASE_ILMTHREAD_LIBRARY_RELEASE=%CODE_BASE%/build/IlmBase/lib/IlmThread.lib -DILMBASE_IMATH_LIBRARY_DEBUG=%CODE_BASE%/build/IlmBase/lib/Imath.lib -DILMBASE_IMATH_LIBRARY_RELEASE=%CODE_BASE%/build/IlmBase/lib/Imath.lib -DILMBASE_INCLUDE_DIR=%CODE_BASE%/build/IlmBase/include -DOPENEXR_INCLUDE_DIR=%CODE_BASE%/build/openEXR/include -DOPENEXR_ILMIMF_LIBRARY_DEBUG=%CODE_BASE%/build/openEXR/lib/IlmImf.lib -DOPENEXR_ILMIMF_LIBRARY_RELEASE=%CODE_BASE%/build/openEXR/lib/IlmImf.lib -DOPENVDB_INCLUDE_DIR=%CODE_BASE%/source/openvdb -DOPENVDB_LIBRARY=%CODE_BASE%/build/openVDB/openvdb/RelWithDebInfo/openvdb.lib -DBoost_SYSTEM_LIBRARY_RELEASE=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_system-vc141-mt-x64-1_68.lib -DBoost_SYSTEM_LIBRARY_DEBUG=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_system-vc141-mt-gd-x64-1_68.lib -DBoost_THREAD_LIBRARY_RELEASE=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_thread-vc141-mt-x64-1_68.lib -DBoost_THREAD_LIBRARY_DEBUG=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_thread-vc141-mt-gd-x64-1_68.lib -DBoost_IOSTREAMS_LIBRARY_RELEASE=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_iostreams-vc141-mt-x64-1_68.lib -DBoost_IOSTREAMS_LIBRARY_DEBUG=%CODE_BASE%/base/boost/lib64-msvc-14.1/boost_iostreams-vc141-mt-gd-x64-1_68.lib
```

If this does build but the exe crashes on startup run `windeployqt.exe \path\to\gui.exe` to copy over the platform settings. If you are using anaconda to provide Qt then only release builds (either RelWithDebInfo or Release) will be able to run as anaconda doesn't provide debug versions of Qt.
