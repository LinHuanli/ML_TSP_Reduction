find_path(LIBLINEAR_INCLUDE_DIR
  NAMES linear.h
  HINTS
    /usr/pkg/include
    /usr/include
    /usr/local/include)

find_library(LIBLINEAR_LIBRARY
  NAMES linear liblinear
  HINTS
    /usr/pkg/lib
    /usr/lib
    /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(liblinear
  REQUIRED_VARS
    LIBLINEAR_INCLUDE_DIR
    LIBLINEAR_LIBRARY)

if(liblinear_FOUND AND NOT TARGET liblinear::liblinear)
  add_library(liblinear::liblinear UNKNOWN IMPORTED)
  set_target_properties(liblinear::liblinear PROPERTIES
    IMPORTED_LOCATION "${LIBLINEAR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBLINEAR_INCLUDE_DIR}")
endif()

mark_as_advanced(LIBLINEAR_INCLUDE_DIR LIBLINEAR_LIBRARY)

