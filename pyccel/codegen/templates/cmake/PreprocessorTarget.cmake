# CMake module to create a target to preprocess all the fortran libraries
#
# Usage:
# Add the following line to the beginning of the main CMakeLists.txt file
# include(PreprocessorTarget)
# 
# At the end of the same CMakeLists.txt call
# add_preprocessor_target()
# 
# After generating the makefiles one can call
# make all_preproc
#
# This module overrides the add_library cmake command to create a list of source
# files that can be given to Forcheck for analysis. Therefore, we should place the
# include(PreprocessorTarget) command before any add_library or add_subdirectory commands.
#
# The source files are preprocessed individually using the C preprocessor.
# The add_preprocessor_target() function generates the commands for the
# preprocessor. It should be called after all the libraries and 
# subdirectories are included.
# 
# Author of ForcheckTargets.cmake: Tamas Feher <tamas.bela.feher@ipp.mpg.de>
#
# Modifications
# -------------
#   - 22 Oct 2015: only preprocess files (Yaman Güçlü [YG], IPP Garching).
#   - 02 Nov 2015: also preprocess executable sources (YG).
#   - 26 Nov 2015: add OpenMP flag (YG).
#   - 02 Dec 2015: fix dependency bug (YG).
#   - 15 Jan 2016: store names of all libraries (YG).
#   - 19 Jan 2016: 'collect_source_info' handles libraries with no sources (YG)

if(__add_all_preproc)
   return()
endif()

set(__add_all_preproc YES)

# List of targets created by "add_library" instructions
set_property(GLOBAL PROPERTY LIBRARY_TARGETS "")

# List of source files to be analyzed
set_property(GLOBAL PROPERTY CPP_SOURCES "")
set_property(GLOBAL PROPERTY CPP_PREPROC_SOURCES "")

# List of include directories
set_property(GLOBAL PROPERTY CPP_INCLUDES "")

## Preprocessor flags
#if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
#  set(preprocessor_only_flags -EP)
#elseif(Fortran_COMPILER_NAME MATCHES gfortran)
#  set(preprocessor_only_flags -cpp -E -P)
#else()
#  message(SEND_ERROR "Unknown preprocessor flags for current compiler")
#endif()

#==============================================================================
# FUNCTION: collect_source_info
#==============================================================================
# Create a list of source files, to be later used to run the preprocessor
function( collect_source_info _name )

  # The include directories will be added to the global list
  get_property(_cpp_includes GLOBAL PROPERTY CPP_INCLUDES)
  get_target_property(_dirs ${_name} INCLUDE_DIRECTORIES)
  if(_dirs)
    list(APPEND _cpp_includes "${_dirs}")
  endif()
  list(REMOVE_DUPLICATES _cpp_includes)
  set_property(GLOBAL PROPERTY CPP_INCLUDES ${_cpp_includes})

  # We also extend the list of source files
  get_property(_cpp_sources  GLOBAL PROPERTY CPP_SOURCES)
  get_property(_cpp_preproc_sources  GLOBAL PROPERTY CPP_PREPROC_SOURCES)

  get_target_property(_forcheck_sources ${_name} SOURCES)
  if(_forcheck_sources)
    foreach(_source ${_forcheck_sources})
      get_source_file_property(_forcheck_lang "${_source}" LANGUAGE)
      get_source_file_property(_forcheck_loc "${_source}" LOCATION)

      if("${_forcheck_lang}" MATCHES "Fortran")
        # first we check if the source file is already in the source list
        list(FIND _cpp_sources ${_forcheck_loc} _list_idx)
        if( ${_list_idx} EQUAL -1)
          # Not yet in the source list
          list(APPEND _cpp_sources "${_forcheck_loc}")      

          # Here we generate the name of the preprocessed source file
          get_filename_component(e "${_source}" EXT)
          get_filename_component(n "${_source}" NAME_WE)
          string(REGEX REPLACE "F" "f" e "${e}")
          set(of "${CMAKE_CURRENT_BINARY_DIR}/${n}${e}")
          list(APPEND _cpp_preproc_sources ${of})
        endif()
      endif()
    endforeach()
  endif()

  # save the updated source list
  set_property(GLOBAL PROPERTY CPP_SOURCES  ${_cpp_sources})
  set_property(GLOBAL PROPERTY CPP_PREPROC_SOURCES  ${_cpp_preproc_sources})

endfunction()

#==============================================================================
# FUNCTION: collect_library_name
#==============================================================================
# Collect names of all library targets.
function( collect_library_name _name )
  get_property( _library_targets GLOBAL PROPERTY LIBRARY_TARGETS )
  list( APPEND _library_targets "${_name}" )
  set_property( GLOBAL PROPERTY LIBRARY_TARGETS ${_library_targets} )
endfunction()

#==============================================================================
# FUNCTION: store_current_dir
#==============================================================================
# Add property to target: directory with the currently processed CMakeLists.txt
function( store_current_dir _name )
  set_target_properties( ${_name} PROPERTIES
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
endfunction()

#==============================================================================
# FUNCTION: add_library
#==============================================================================
# We override the add_library built in function.
function( add_library _name )
  _add_library( ${_name} ${ARGN} ) # Call the original function
  collect_library_name( ${_name} ) # Store library name in proper list
  collect_source_info ( ${_name} ) # Create a list of source files
  store_current_dir   ( ${_name} ) # Store current directory in target property
endfunction()

#==============================================================================
# FUNCTION: add_executable
#==============================================================================
# We override the add_executable built in function.
function( add_executable _name )
  _add_executable( ${_name} ${ARGN} ) # Call the original function
  collect_source_info( ${_name} )     # Create a list of source files
endfunction()

##==============================================================================
## FUNCTION: add_preprocessor_target
##==============================================================================
## adds a custom target for running the C preprocessor on all source files
## call this function at the end of the CMakeLists.txt
#function(add_preprocessor_target)
#
#  # If needed, add OpenMP flag to preprocessor flags
#  if(OPENMP_ENABLED)
#    set(preprocessor_only_flags ${preprocessor_only_flags} ${OpenMP_Fortran_FLAGS})
#  endif()
#
#  # Retrieve the lists that were created by add_library:
#  get_property(_cpp_sources GLOBAL PROPERTY CPP_SOURCES)
#  get_property(_cpp_preproc_sources GLOBAL PROPERTY CPP_PREPROC_SOURCES)
#  get_property(_cpp_includes GLOBAL PROPERTY CPP_INCLUDES)
#
#  # Set up include flags for preprocessing
#  set(incflags)
#  foreach(i ${_cpp_includes})
#    set(incflags ${incflags} -I${i})
#  endforeach()
#
#  # List of compiler definitions to be used for preprocessing
#  foreach(_source ${_cpp_sources})
#    get_source_file_property(_defs "${_source}" COMPILE_DEFINITIONS)
#    list(APPEND _cpp_defines "${_defs}")
#    get_filename_component(_dir ${_source} PATH)
#    get_property( _defs  DIRECTORY ${_dir} PROPERTY COMPILE_DEFINITIONS)
#    if(_defs)
#      list(APPEND _cpp_defines "${_defs}")
#    endif()
#  endforeach()
#
#  if(_cpp_defines)
#    list(REMOVE_DUPLICATES _cpp_defines)
#  endif()
#
#  set(defflags)
#  foreach(d ${_cpp_defines})
#    set(defflags ${defflags} -D${d})
#  endforeach()
#
#  # Create custom commands for preprocessing the Fortran files
#  while(_cpp_sources)
#    list(GET _cpp_sources 0 _src)
#    list(GET _cpp_preproc_sources 0 _preproc_src)
#    list(REMOVE_AT _cpp_sources 0)
#    list(REMOVE_AT _cpp_preproc_sources 0)
#    add_custom_command(OUTPUT "${_preproc_src}"
#        COMMAND ${CMAKE_Fortran_COMPILER} ${incflags} ${defflags} ${preprocessor_only_flags} ${_src} > ${_preproc_src}
###        IMPLICIT_DEPENDS Fortran "${_source}"
#        DEPENDS "${_src}"
#        COMMENT "Preprocessing ${_src}"
#        VERBATIM
#      )
#  endwhile()
#
#  # Group all preprocessing commands into one target
#  get_property(_cpp_preproc_sources GLOBAL PROPERTY CPP_PREPROC_SOURCES)
#  add_custom_target(all_preproc DEPENDS ${_cpp_preproc_sources})
#  set_target_properties(all_preproc PROPERTIES EXCLUDE_FROM_ALL TRUE)
#
#  # Clean up *.s files just generated (if any)
#  add_custom_command( TARGET all_preproc POST_BUILD
#    COMMAND find . -iname "*.s" -type f -delete
#    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} ) 
#
#endfunction()
