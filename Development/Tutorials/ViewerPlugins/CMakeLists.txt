#----------------------------------------------------------------
# GigaVoxels LIBRARY CMake file
# Main user file
#----------------------------------------------------------------

# Check CMAKE version
cmake_minimum_required (VERSION 2.8) # needed to have the findCUDA module

# Use colored output
set (CMAKE_COLOR_MAKEFILE ON)

# Force CMake to not put in the rules that re-run CMake 
# CMakeLists.txt files are not added and not used when (re)building projects
# set (CMAKE_SUPPRESS_REGENERATION ON)

#----------------------------------------------------------------
# Project
#----------------------------------------------------------------

# Invite message
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : GigaVoxels Tutorials" )
MESSAGE ( STATUS "-----------------------------------------------------" )

# Project name
project (ViewerPluginTutorials)

# Tell CMake to first search in directories listed in CMAKE_MODULE_PATH when 
# using FIND_PACKAGE() or INCLUDE()
SET (GV_ROOT "${CMAKE_SOURCE_DIR}/../../..")
SET (CMAKE_MODULE_PATH "${GV_ROOT}/Development/CMake")

# Include common settings
INCLUDE (GvSettings_CMakeImport)

#----------------------------------------------------------------
# Initialize some environment variables
#----------------------------------------------------------------
SET (RELEASE_BIN_DIR ${GV_RELEASE}/Bin)

#########################################################################
# Add all the sub projects
# SDK Viewer : plugin integration
#########################################################################

####
# Demos / Tutorials
####

# Streaming
# - procedural generation on host
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleSphereCPU")
# - loading files from host
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDynamicLoad")
# - import RAW data files from host
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRAWDataLoader")
# - import DICOM data files from host
IF (WIN32)
	add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDicomViewer")
ELSE ()
	find_package( DCMTK )
	IF ( DCMTK_FOUND )
		add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDicomViewer")
	ELSE ()
		MESSAGE ( STATUS "" )
		MESSAGE ( STATUS "WARNING : Unable to generate the GvDicomViewer project because DCMTK has not been found on the system" )
	ENDIF ()
ENDIF ()

# Noise
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAmplifiedSurface")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAmplifiedVolume")

# Voxel data synthesis
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvMengerSponge")

# Procedural generation
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleSphere")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvMandelbrotSet")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSlisesix")

# Graphics interoperability
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRendererGLSL")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleShapeGLSL")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRayMapGenerator")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDepthPeeling")

# Effects
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDepthOfField")

# Hypertexture
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvHypertexture")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDynamicHypertexture")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvNoiseInheritance")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvLazyHypertexture")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvShellNoise")

# Demos
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvInfiniteSky")

# Proxy geometry
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvProxyGeometryManager")

# GigaSpace
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSphereRayTracing")
IF (WIN32)
	# cudpp is modified on LINUX, there is no cudppScan => TO DO : add access to cudppScan()
	add_subdirectory ("${CMAKE_SOURCE_DIR}/GvVBOGenerator")
ELSE ()
ENDIF ()

# Voxelization
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSignedDistanceFieldVoxelization")

# Instancing
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvInstancing")

# BVH
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleTriangles")

# Environment Mapping
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvEnvironmentMapping")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCastShadows")

# Collisions
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCollisionBBox")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCollision")

# Animation
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedSnake")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedFan")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedLUT")

# Environment mapping
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvNoiseInAShellGLSL")

# Production policies
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvProductionPolicies")

# Quality Renderer
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvQualityRenderer")
