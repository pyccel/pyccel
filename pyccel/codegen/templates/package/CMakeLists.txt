# Create local variables
set( __SUFFIX___LIBS "" )  # Internal libraries (part of the __PROJECT__ project)
set( __SUFFIX___EXT  "" )  # External libraries (built with __PROJECT__)
set( __SUFFIX___DEPS "" )  # Dependencies (pre-installed libraries)

# Extract names of all libraries built in __PROJECT__
get_property( library_targets GLOBAL PROPERTY LIBRARY_TARGETS )

# Distinguish between "internal" and "external" libraries
foreach( lib ${library_targets} )
  if( ${lib} MATCHES "pyccelext_[a-zA-Z0-9]" )
    list( APPEND __SUFFIX___LIBS ${lib} )
  else()
    get_target_property( libpath ${lib} SOURCE_DIR )
    if( libpath MATCHES "${CMAKE_SOURCE_DIR}/external" )
      list( APPEND __SUFFIX___EXT ${lib} )
    endif()
  endif()
endforeach()

# Extract system dependencies
foreach( lib ${library_targets} )
  get_target_property( deps ${lib} INTERFACE_LINK_LIBRARIES )
  if( deps )
    foreach( dep ${deps} )
      list( FIND library_targets ${dep} idx )
      if( ${idx} EQUAL -1 )
        #get_filename_component(libdep "${dep}" NAME_WE)
        #list( APPEND __SUFFIX___DEPS ${libdep} )
	      list( APPEND __SUFFIX___DEPS ${dep} )
      endif()
    endforeach()
  endif()
endforeach()

# Remove duplicates without breaking dependency order
list( REVERSE __SUFFIX___DEPS )
list( REMOVE_DUPLICATES __SUFFIX___DEPS )
list( REVERSE __SUFFIX___DEPS )

#===============================================================================

# Print names of all sub-libraries in __PROJECT__
message( STATUS "__PROJECT__ LIBRARIES:" )
foreach( lib ${__SUFFIX___LIBS} )
  message( STATUS "  ${lib}" )
endforeach()

# Print names of built-in external 
message( STATUS "BUILT-IN EXTERNAL LIBRARIES:" )
foreach( lib ${__SUFFIX___EXT} )
  message( STATUS "  ${lib}" )
endforeach()

# Print all external dependencies
message( STATUS "EXTERNAL DEPENDENCIES:" )
foreach( lib ${__SUFFIX___DEPS} )
  message( STATUS "  ${lib}" )
endforeach()

#===============================================================================

# Extract file paths of all libraries
set( libpaths )
foreach( lib ${__SUFFIX___LIBS} )
  list( APPEND libpaths $<TARGET_FILE:${lib}> )
endforeach()

# CMake custom command: repackage __PROJECT__ into single archive "lib__PROJECT__.a"
# NOTE: Command does not run if file is newer than its dependencies
set( __SUFFIX___LIB "${CMAKE_CURRENT_BINARY_DIR}/lib__LIBNAME__.a" )
add_custom_command( OUTPUT ${__SUFFIX___LIB}
  DEPENDS ${__SUFFIX___LIBS}
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_package.py ${__SUFFIX___LIB} ${libpaths}
  COMMENT "Repackage all static libraries into a single archive: 'lib__LIBNAME__.a'"
  VERBATIM )

# CMake target: update "lib__PROJECT__.a" if obsolete, otherwise do nothing
# USAGE: make __PROJECT__-archive
add_custom_target( __PROJECT__-archive ALL DEPENDS ${__SUFFIX___LIB} )

# Wrap "lib__PROJECT__.a" with CMake imported library "__PROJECT__", which depends on
# above "__PROJECT__-archive" target and therefore creates archive only if needed.
# USAGE: target_link_libraries( <MY_LIB> __PROJECT__ ${__SUFFIX___EXT} ${__SUFFIX___DEPS} )
add_library( __PROJECT__ STATIC IMPORTED GLOBAL )
set_target_properties( __PROJECT__ PROPERTIES IMPORTED_LOCATION ${__SUFFIX___LIB} )
add_dependencies( __PROJECT__ __PROJECT__-archive )

#===============================================================================

# Store cache variables (available to simulations)
set( __SUFFIX___LIB  ${__SUFFIX___LIB}  CACHE INTERNAL "Path to full __PROJECT__ library" )
set( __SUFFIX___EXT  ${__SUFFIX___EXT}  CACHE INTERNAL "List of built-in external libraries" )
set( __SUFFIX___DEPS ${__SUFFIX___DEPS} CACHE INTERNAL "List of external dependencies" )

# Installation (library, headers, mod files)
install( FILES ${__SUFFIX___LIB}    DESTINATION lib )
install( DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}/"
  DESTINATION "include/__LIBNAME__" FILES_MATCHING PATTERN "*.mod" )

#===============================================================================
# Print PALFConfig.cmake for external use with cmake

set( filepath "${CMAKE_CURRENT_BINARY_DIR}/__PROJECT__Config.cmake" )
file( WRITE  ${filepath} "SET( CLAPP_DIR ${CMAKE_INSTALL_PREFIX})\n" )

file( APPEND ${filepath} "\nSET( __PROJECT___LIBRARIES \${CLAPP_DIR}/lib/lib__LIBNAME__.a" )

foreach( lib ${__SUFFIX___EXT} )
  file( APPEND ${filepath} "\n" )
  file( APPEND ${filepath} "	\${CLAPP_DIR}/lib/lib${lib}.a" )
endforeach()
foreach( lib ${__SUFFIX___DEPS} )
  file( APPEND ${filepath} "\n" )
  #file( APPEND ${filepath} "	\${CLAPP_DIR}/lib/${lib}.a" )
  file( APPEND ${filepath} "	${lib}" )
endforeach()
file( APPEND ${filepath} ")\n" )

# Set include path
file( APPEND ${filepath} "\nSET( __PROJECT___INCLUDE_PATH  \${CLAPP_DIR}/include/__LIBNAME__)\n" )

# Add include path
file( APPEND ${filepath} "\nINCLUDE_DIRECTORIES( \${__PROJECT___INCLUDE_PATH} )\n" )

# Installation
install( CODE "file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/cmake)" )
install( FILES ${filepath} DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake )

