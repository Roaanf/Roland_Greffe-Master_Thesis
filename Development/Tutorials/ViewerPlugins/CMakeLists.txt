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
# - import RAW data files from host
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRAWDataLoader")

