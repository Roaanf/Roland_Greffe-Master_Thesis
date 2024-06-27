# Master Thesis repository: Visual Tools for Computed Tomography Volume Representation: Large Data Visualisation and Surface Extraction

## Repository structure:

This repository began as a fork of the following GitLab https://gitlab.inria.fr/neyret/gigavoxelsdist.

It was tested for CUDA 11.2 and Visual Studio 2019.

- Surface : This folder contains the code related to the Surface Extraction part of the thesis
- Data : Contains Images and files used by the GigaVoxel software
- Development : Contains the source code of the GigaVoxel software
- External : Contains binaries of external libraries used by the library
- Install : Contains the installation scripts used to setup the compiling
Note: The "SurfaceComp", "SurfaceCreation" and "SurfaceDetection" folders are older version of the Surface Extraction part of the thesis that are only kept for history.

## Original README.txt

GigaVoxels - GigaSpace

[GigaVoxels] is a ray-guided out-of-core, on demand production, and smart caching library
used for efficient 3D real-time visualization of highly large and detailed
sparse volumetric scenes (SVO : Sparse Voxel Octree).

[GigaSpace] is a full-customizable out-of-core, on demand production, and smart caching library
based on user-defined hierachical arbitrary space partitionning, space & brick visitors,
and brick producer using arbitrary data types.

GigaVoxels and GigaSpace are indeed the same tool.

Website: http://gigavoxels.inrialpes.fr/

The GigaVoxels/GigaSpace core is under the BSD3 license, but the demos
and tutorials use various third-party libraries (QT, QGLViewer, ...),
some of which have non-BSD license. The most constraining is
QGLViewer, with dual license commercial and GPL with lesser exceptions
to open source. For this reason we do not put any license on our demos
and tutorials files : if you reuse these for your projects, verify the
license terms of the libs you pull with them.


Check in the header of files which license apply to them.

Demos and tools are located in Development\Tools, Development\Tutorials\Demos, Development\Tutorials\ViewerPlugins
Third party libs (binaries, sources and headers) are located in [ External\Windows or External\Linux ].
GigaVoxels/GigaSpace core is located in [ Development\Library ]