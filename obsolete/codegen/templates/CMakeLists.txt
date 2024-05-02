#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)

IF(OPTIONS_FILE)
  INCLUDE(${OPTIONS_FILE})
  MESSAGE("OPTIONS_FILE ARE READ")
ENDIF()

#Name your project here
project(__PROJECT__)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
SET(CMAKE_SHARED_LIBRARY_LINK_FORTRAN_FLAGS "")

enable_language (Fortran)

IF(DEFINED CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} 
  CACHE STRING "Choose the type of build, options are: Debug Release.")
ELSE()
  SET(CMAKE_BUILD_TYPE Debug 
  CACHE STRING "Choose the type of build, options are: Debug Release.")
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  ADD_DEFINITIONS(-DDEBUG)
ENDIF()

set(__SUFFIX___BUILD_DIR $ENV{PWD} )

IF(APPLE)
  cmake_policy(SET CMP0042 NEW)
ENDIF(APPLE)

################################
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

# the RPATH to be used when installing
SET(CMAKE_INSTALL_RPATH "")

# don't add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
################################

# Add new directory to find the FIND<PACKAGE>.cmake files. CMAKE_SOURCE_DIR is
# the directory where the project sources are located. CMAKE_MODULE_PATH is 
# the list of directories in which to search for CMake modules.
SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} )

SET(BUILD_TESTING         ON  CACHE BOOL "Build __PROJECT__ library tests")
SET(BUILD_AGMG            OFF CACHE BOOL "Build AGMG libraries")
SET(MUMPS_ENABLED         OFF CACHE BOOL "Enable MUMPS solver ")
SET(SUPERLU_ENABLED       OFF CACHE BOOL "Enable SuplerLU solver")
SET(HDF5_ENABLED          OFF CACHE BOOL "Use HDF5 library for data output")
SET(MPI_ENABLED           OFF CACHE BOOL "Enable use of MPI library")
SET(METIS_ENABLED         OFF CACHE BOOL "Enable METIS")
SET(LAPACK_ENABLED        OFF CACHE BOOL "Enable LAPACK")
SET(PETSC_ENABLED         OFF CACHE BOOL "Enable PETSc solvers")
SET(FFTW_ENABLED          OFF CACHE BOOL "Enable FFTW")

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  GET_FILENAME_COMPONENT(INSTALL_DIR "${CMAKE_SOURCE_DIR}/../usr" ABSOLUTE)
  SET(CMAKE_INSTALL_PREFIX ${INSTALL_DIR} CACHE PATH
    "Default installation directory" FORCE)
ENDIF()

INCLUDE(FortranConfig)
INCLUDE(CTest)
INCLUDE(PreprocessorTarget)

set( CLAPP_DIR $ENV{CLAPP_DIR} )
if( NOT CLAPP_DIR )
  set( CLAPP_DIR ${CMAKE_INSTALL_PREFIX} )
endif()


IF(BUILD_AGMG)
  ADD_DEFINITIONS(-DAGMG_ENABLED)
ENDIF(BUILD_AGMG)

# if BLASLAPACK_DIR was specified
IF(LAPACK_ENABLED)
  IF(DEFINED BLASLAPACK_DIR)
    SET(BLASLAPACK_DIR ${BLASLAPACK_DIR} CACHE STRING "blas-lapack directory.")
    FIND_PACKAGE(BlasLapack)
  ELSE()
    INCLUDE(LapackConfig)
  ENDIF()
ENDIF()

IF(MUMPS_ENABLED)
  FIND_PACKAGE(MUMPS)
  ADD_DEFINITIONS(-DMUMPS_ENABLED)
  MESSAGE(STATUS "MUMPS_FOUND:${MUMPS_FOUND}")
  MESSAGE(STATUS "MUMPS_INCLUDE_DIRS:${MUMPS_INCLUDE_DIRS}")
ENDIF(MUMPS_ENABLED)

IF(SUPERLU_ENABLED)
  FIND_PACKAGE(SuperLU)
  ADD_DEFINITIONS(-DSUPERLU_ENABLED)
  MESSAGE(STATUS "SUPERLU_FOUND:${SUPERLU_FOUND}")
  MESSAGE(STATUS "SUPERLU_INCLUDE_DIRS:${SUPERLU_INCLUDE_DIRS}")
ENDIF(SUPERLU_ENABLED)

IF(PETSC_ENABLED)
  FIND_PACKAGE(PETSc)
  ADD_DEFINITIONS(-DPETSC_ENABLED)
  INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
  MESSAGE(STATUS "PETSC_INCLUDES:${PETSC_INCLUDES}")
  MESSAGE(STATUS "PETSC_LIBRARIES:${PETSC_LIBRARIES}")
  MESSAGE(STATUS "PETSC_VERSION:${PETSC_VERSION}")
ENDIF(PETSC_ENABLED)

IF(HDF5_ENABLED)
  INCLUDE(HDF5Config)
  ADD_DEFINITIONS(-DHDF5_ENABLED)
  INCLUDE_DIRECTORIES(${HDF5_INCLUDES})
ENDIF()

IF(MPI_ENABLED)
  INCLUDE(MPIConfig)
  INCLUDE(ProcessorCount)
  INCLUDE(POEConfig)
  ADD_DEFINITIONS(-DMPI_ENABLED)
  SET(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
ENDIF()

IF(METIS_ENABLED)
  FIND_PACKAGE(METIS)
  IF(METIS_FOUND)
    ADD_DEFINITIONS(-DMETIS_ENABLED)
  ENDIF(METIS_FOUND)
ENDIF(METIS_ENABLED)

if(FFTW_FOUND)
  FIND_PACKAGE(FFTW_ENABLED)
endif()

IF(FFTW_ENABLED)
  FIND_PACKAGE(FFTW)
  ADD_DEFINITIONS(-DFFTW_ENABLED)
  INCLUDE_DIRECTORIES(${FFTW_INCLUDES})
  MESSAGE(STATUS "FFTW Has been found")
  MESSAGE(STATUS "FFTW_DIR:${FFTW_DIR}")
  MESSAGE(STATUS "FFTW_INCLUDES:${FFTW_INCLUDES}")
  MESSAGE(STATUS "FFTW_LIB:${FFTW_LIB}")
ENDIF(FFTW_ENABLED)


FIND_PACKAGE(PythonInterp 2.7.3)
SET(PYTHON_FOUND FALSE)
MESSAGE(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")
MESSAGE(STATUS "PYTHON_VERSION_STRING:${PYTHON_VERSION_STRING}")
IF(PYTHON_EXECUTABLE AND PYTHON_VERSION_STRING MATCHES "2.*")
  STRING(COMPARE GREATER ${PYTHON_VERSION_STRING} "2.6.0" PYTHON_FOUND)
ENDIF()
MESSAGE(STATUS "PYTHON_FOUND:${PYTHON_FOUND}")

file(MAKE_DIRECTORY ${__SUFFIX___BUILD_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${__SUFFIX___BUILD_DIR}/bin)

file(MAKE_DIRECTORY ${__SUFFIX___BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${__SUFFIX___BUILD_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${__SUFFIX___BUILD_DIR}/lib)

# Define folder to be deleted by make clean         
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/modules/)
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/)
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/include/)

ADD_CUSTOM_TARGET( clear
  COMMAND find . -name parameter* -delete
  COMMAND find . -name *.mod -delete
  COMMAND find . -name *.a -delete
  COMMAND find . -name *.o -delete
  COMMAND find . -name *.tp -delete
  COMMAND find . -name *.cmake -delete
  COMMAND find . -name *.xmf -delete
  COMMAND find . -name *.h5 -delete
  COMMAND find . -name *.gnu -delete
  COMMAND find . -name *.dat -delete
  COMMAND find . -name *.mat -delete
  COMMAND find . -name *.mtx -delete
  COMMAND find . -name *.txt -delete
  COMMAND find . -name *.log -delete
  COMMAND rm -rf bin
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Delete all .a, .mod, .o, .tp, .cmake, .xmf, .h5, .gnu, .dat, .mat, .mtx, .txt, .log and folders /modules, /bin" VERBATIM)

##########################################################################
# 
##########################################################################
ENABLE_TESTING()
##########################################################################

##########################################################################
# library and models
ADD_SUBDIRECTORY("src")
##########################################################################

##########################################################################
IF(CMAKE_Fortran_COMPILER MATCHES "gfortran*")
  SET(COMPILER_ID 1 CACHE INTEGER "Identifier for the GCC compiler")

ELSEIF(CMAKE_Fortran_COMPILER MATCHES "ifort")
  SET(COMPILER_ID 2 CACHE INTEGER "Identifier for the IFORT compiler")

ELSEIF(CMAKE_Fortran_COMPILER MATCHES "pgi")
  SET(COMPILER_ID 3 CACHE INTEGER "Identifier for the PGI compiler")
ENDIF()
##########################################################################

##########################################################################
MESSAGE(STATUS "############################################################")
MESSAGE(STATUS "####")
MESSAGE(STATUS "#### CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "#### Fortran_COMPILER_NAME:${Fortran_COMPILER_NAME}")
MESSAGE(STATUS "#### CMAKE_Fortran_COMPILER:${CMAKE_Fortran_COMPILER}")
MESSAGE(STATUS "#### Fortran FLAGS ")
IF(CMAKE_BUILD_TYPE MATCHES "Debug") 
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS_DEBUG: ${CMAKE_Fortran_FLAGS_DEBUG}")
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release") 
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}")
ELSE()
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}")
ENDIF()
MESSAGE(STATUS "####")
MESSAGE(STATUS "############################################################")
##########################################################################

# Create single library 
ADD_SUBDIRECTORY( package )
