#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GvDicomVoxelizer)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")
MESSAGE (STATUS "")

# Tell CMake to first search in directories listed in CMAKE_MODULE_PATH when 
# using FIND_PACKAGE() or INCLUDE()
SET (GV_ROOT "${CMAKE_SOURCE_DIR}/../..")
SET (CMAKE_MODULE_PATH "${GV_ROOT}/Development/CMake")

# Include common settings
INCLUDE (GvSettings_CMakeImport)

#----------------------------------------------------------------
# Target type
#----------------------------------------------------------------

# Can be GV_EXE or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_CUDA_EXE")

SET(RELEASE_BIN_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Bin)
SET(RELEASE_LIB_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Lib)
SET(RELEASE_INC_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Inc)

SET(GIGASPACE_RELEASE_BIN_DIR ${GV_RELEASE}/Bin)

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

# Set third party dependencies LINK library name if needed
#     . Example :
#     . When using the Qt_CMakeImport.cmake file to add Qt dependency with the command
#     .     INCLUDE (Qt_CMakeImport)
#     . the file use the ${qtLib} value to test if the user has set name different names to default ones.
#     . The DEFAULT library name is
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml")
#     . but user can only use
#     .     set (qtLib "QtCore" "QtGui")
#     . to reduce dependencies if needed
#     . or add dependencies
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml" "QtSvg")

IF (WIN32)
	INCLUDE(DCMTK_CMakeImport)
ELSE ()
	find_package(DCMTK REQUIRED)
	INCLUDE(DCMTK_CMakeImport)
	include_directories(${DCMTK_INCLUDE_DIRS})
	#target_link_libraries(dcmtk_test ${DCMTK_LIBRARIES} z)
ENDIF ()

# GigaSpace library
SET (gigaspaceLib "GigaSpace")
INCLUDE (GigaVoxels_CMakeImport)

INCLUDE (GigaSpace_ThirdPartyDependencies_CMakeImport)

# Add IHM third party dependencies
INCLUDE (Qt_CMakeImport)

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

IF (WIN32)
	# Add Windows dependencies
	LINK_LIBRARIES (user32)
	LINK_LIBRARIES (gdi32)
	LINK_LIBRARIES (shell32)
	
	# Required by DCMTK
	LINK_LIBRARIES (netapi32)
	LINK_LIBRARIES (wsock32)
	LINK_LIBRARIES (Advapi32)
	
ENDIF ()

# Add the common CMAKE settings to generate a GigaVoxels tool
INCLUDE (GV_CMakeCommonTools)

# - debug on CPU
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -pedantic -Wextra")
