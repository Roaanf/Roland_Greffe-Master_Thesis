#----------------------------------------------------------------
# GigaSpace LIBRARY CMake file
# Main user file
#----------------------------------------------------------------

# Check CMAKE version
cmake_minimum_required (VERSION 2.8) # needed to have the findCUDA module

#----------------------------------------------------------------
# Project
#----------------------------------------------------------------

# Invite message
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : GigaSpace Demo Tutorials" )
MESSAGE ( STATUS "-----------------------------------------------------" )

# Project name
project (DemoTutorials)

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
#SET (RELEASE_LIB_DIR ${GV_RELEASE}/Lib)
#SET (RELEASE_INC_DIR ${GV_RELEASE}/Inc)

#########################################################################
# Add all the sub projects
# Others
#########################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/DepthOfField")

#########################################################################
# Add all the sub projects
# IOStreaming
#########################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/IOStreaming/DynamicLoad")

#########################################################################
# Add all the sub projects
# Procedural Technics
#########################################################################

# Noise
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Noise/AmplifiedSurface")
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Noise/AmplifiedVolume")
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Noise/SmartPerlinNoise")
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Noise/ProceduralTerrain")
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Noise/NoiseInheritance")

# Geometry
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/SimpleSphere")
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/SimpleSphereCPU")

# Fractal
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/MandelbrotSet")

# Fractal + Instancing
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/MengerSponge")

# Distance field
add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/Slisesix")

# Clipping planes
add_subdirectory ("${CMAKE_SOURCE_DIR}/ClippingPlane")

# BVH - kd-tree
# - temporary removed => need to pass to new API
add_subdirectory ("${CMAKE_SOURCE_DIR}/SimpleTriangles")

#########################################################################
# Add all the sub projects
# GraphicsInteroperability
#########################################################################

# OpenGL
add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/RendererGLSL")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/RendererGLSLSphere")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/OpenGLScene")

# Environment Mapping
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/CastShadows")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/GlossySurface")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/RefractingLayer")

# Proxy geometry
add_subdirectory ("${CMAKE_SOURCE_DIR}/GraphicsInteroperability/ProxyGeometry")

#########################################################################
# Add all the sub projects
# GUI integration
#########################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/GUI/GLUTWindow")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GUI/QtWindow")

#########################################################################
# Add all the sub projects
# Voxelization
#########################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/Voxelization/Voxelization")
add_subdirectory ("${CMAKE_SOURCE_DIR}/Voxelization/VoxelizationSignedDistanceField")

#########################################################################
# Add all the sub projects
# Multi-Objects
#########################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/ProceduralTechnics/MultiObjects")
