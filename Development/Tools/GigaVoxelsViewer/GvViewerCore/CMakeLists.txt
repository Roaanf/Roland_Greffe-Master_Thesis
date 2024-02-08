#----------------------------------------------------------------
# DEMO CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GvViewerCore)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")
MESSAGE (STATUS "")

#----------------------------------------------------------------
# Target yype
#----------------------------------------------------------------

# Can be GV_EXE or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_SHARED_LIB")

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

INCLUDE (OpenGL_CMakeImport)
INCLUDE (GLU_CMakeImport)
INCLUDE (glew_CMakeImport)
INCLUDE (freeglut_CMakeImport)

# Add math 3D maths, matrix/vector library
INCLUDE (GLM_CMakeImport)

# - XML library
INCLUDE (TinyXML_CMakeImport)

# Linux special features
if (WIN32)
else ()
	INCLUDE (dl_CMakeImport)
endif()

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

INCLUDE (GV_CMakeCommonTools)
