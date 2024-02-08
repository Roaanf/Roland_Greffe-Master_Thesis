#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cudpp" for configuration "Release"
set_property(TARGET cudpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cudpp PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/cudpp64.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cudart.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/cudpp64.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp "${_IMPORT_PREFIX}/lib/cudpp64.lib" "${_IMPORT_PREFIX}/lib/cudpp64.dll" )

# Import target "cudpp_hash" for configuration "Release"
set_property(TARGET cudpp_hash APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cudpp_hash PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/cudpp_hash64.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cudart.lib;cudpp"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/cudpp_hash64.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp_hash )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp_hash "${_IMPORT_PREFIX}/lib/cudpp_hash64.lib" "${_IMPORT_PREFIX}/lib/cudpp_hash64.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
