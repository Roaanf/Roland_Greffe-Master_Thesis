#----------------------------------------------------------------
# GigaVoxels TOOLS CMake file
# Main user file
#----------------------------------------------------------------

# Check CMAKE version
cmake_minimum_required (VERSION 2.8) # needed to have the findCUDA module

#----------------------------------------------------------------
# Project
#----------------------------------------------------------------

# Invite message
MESSAGE (STATUS "")
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : GigaVoxels Tools" )
MESSAGE ( STATUS "-----------------------------------------------------" )

# Project name
project (Tools)

# Tell CMake to first search in directories listed in CMAKE_MODULE_PATH when 
# using FIND_PACKAGE() or INCLUDE()
SET (GV_ROOT "${CMAKE_SOURCE_DIR}/../..")
SET (CMAKE_MODULE_PATH "${GV_ROOT}/Development/CMake")

# Include common settings
INCLUDE (GvSettings_CMakeImport)

#########################################################################
# Add all the sub projects
#########################################################################

# Viewer
# Invite message
MESSAGE ( STATUS "" )
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : Viewer Tools" )
MESSAGE ( STATUS "-----------------------------------------------------" )
add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsViewer")

# Voxelizer
MESSAGE ( STATUS "" )
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : Voxelizer Tools" )
MESSAGE ( STATUS "-----------------------------------------------------" )
add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsVoxelizer")

# Dicom Voxelizer
MESSAGE ( STATUS "" )
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : DicomVoxelizer Tools" )
MESSAGE ( STATUS "-----------------------------------------------------" )
IF (WIN32)
	add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsDicomVoxelizer")
ELSE ()
#	find_package( DCMTK )
#	IF ( DCMTK_FOUND )
#		add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsDicomVoxelizer")
#	ELSE ()
#		MESSAGE ( STATUS "" )
#		MESSAGE ( STATUS "WARNING : Unable to generate the GigaVoxelsDicomVoxelizer project because DCMTK has not been found on the system" )
#	ENDIF ()
ENDIF ()

# LEGACY : Data Converter
MESSAGE ( STATUS "" )
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : DataConvertor Tools" )
MESSAGE ( STATUS "-----------------------------------------------------" )
add_subdirectory ("${CMAKE_SOURCE_DIR}/Legacy/GigaVoxelsDataConvertor")
