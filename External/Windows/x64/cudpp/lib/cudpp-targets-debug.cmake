#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cudpp" for configuration "Debug"
set_property(TARGET cudpp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cudpp PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/cudpp64d.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cudart.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/cudpp64d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp "${_IMPORT_PREFIX}/lib/cudpp64d.lib" "${_IMPORT_PREFIX}/lib/cudpp64d.dll" )

# Import target "cudpp_hash" for configuration "Debug"
set_property(TARGET cudpp_hash APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cudpp_hash PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/cudpp_hash64d.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cudart.lib;cudpp"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/cudpp_hash64d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp_hash )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp_hash "${_IMPORT_PREFIX}/lib/cudpp_hash64d.lib" "${_IMPORT_PREFIX}/lib/cudpp_hash64d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
