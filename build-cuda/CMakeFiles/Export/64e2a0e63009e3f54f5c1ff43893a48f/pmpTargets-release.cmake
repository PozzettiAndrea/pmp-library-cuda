#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pmp" for configuration "Release"
set_property(TARGET pmp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pmp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpmp.so.3.0.0"
  IMPORTED_SONAME_RELEASE "libpmp.so.3.0.0"
  )

list(APPEND _cmake_import_check_targets pmp )
list(APPEND _cmake_import_check_files_for_pmp "${_IMPORT_PREFIX}/lib/libpmp.so.3.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
