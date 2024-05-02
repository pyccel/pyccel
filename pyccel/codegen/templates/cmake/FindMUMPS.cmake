# This configuration is for LRZ Culusters
# todo add other configurations

IF(MPI_ENABLED)

  IF(Fortran_COMPILER_NAME MATCHES "mpi")
    set(OTHERS_LIB "-lpthread -qopenmp")
  ENDIF()

  # ===========================================================================================
  
  SET(MUMPS_BASE    "$ENV{MUMPS_BASE}")
  SET(MUMPS_INCLUDE "${MUMPS_BASE}/include")
  SET(MUMPS_LIB_SEQ "${MUMPS_BASE}/libseq")
  SET(MUMPS_LIB_DIR "${MUMPS_BASE}/lib")

  FIND_LIBRARY(MUMPS_COMMON_LIB NAMES libmumps_common.a
               HINTS ${MUMPS_LIB_DIR})

  FIND_LIBRARY(DMUMPS_LIB NAMES libdmumps.a
	             HINTS ${MUMPS_LIB_DIR})

  SET(MUMPS_LIB ${DMUMPS_LIB} ${MUMPS_COMMON_LIB})
  
  # ===========================================================================================
  
  SET(PARMETIS_LIB_DIR    "$ENV{PARMETIS_LIBDIR}")
  
  FIND_LIBRARY(PARMETIS_LIB NAMES libparmetis.a
               HINTS ${PARMETIS_LIB_DIR})

  FIND_LIBRARY(METIS_LIB NAMES libmetis.a
               HINTS ${PARMETIS_LIB_DIR})
  
  FIND_LIBRARY(MUMPS_PORD_LIB NAMES libpord.a
               HINTS ${MUMPS_LIB_DIR})
  
  # For performing only sequential analysis 
  SET(ORDERINGS_LIB ${METIS_LIB} ${MUMPS_PORD_LIB})

  # For performing only both parallel and sequential analysis 
  #SET(ORDERINGS_LIB ${PARMETIS_LIB} ${METIS_LIB} ${MUMPS_PORD_LIB})
  
  # ===========================================================================================
  # Make this variable point to the path where the Intel MKL library is installed
  # It is set to the default install directory for Intel MKL 
  SET(MKL_ROOT "/lrz/sys/intel/compiler/composer_xe_2015.5.223/mkl/lib/intel64")

  # SCALAPACK, BLACS and BLAS 
  SET(SCALAPACK_LIB  "-L${MKL_ROOT} -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64")
  SET(BLAS_LIB "-L${MKL_ROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core")
  
  # ===========================================================================================
  
  SET(MUMPS_LIBRARIES ${MUMPS_LIB} ${ORDERINGS_LIB} ${SCALAPACK_LIB} ${BLAS_LIB} ${OTHERS_LIB})
  SET(MUMPS_INCLUDES ${MUMPS_INCLUDE} ${MUMPS_LIB_SEQ} )
  
  SET (MUMPS_INCLUDES  ${MUMPS_INCLUDES}  CACHE STRING "MUMPS include path" FORCE)
  SET (MUMPS_LIBRARIES ${MUMPS_LIBRARIES} CACHE STRING "MUMPS libraries"    FORCE)

  MARK_AS_ADVANCED(MUMPS_INCLUDES MUMPS_LIBRARIES)

  INCLUDE(FindPackageHandleStandardArgs)
  find_package_handle_standard_args (MUMPS
	                  "MUMPS could not be found.  Be sure to set MUMPS_BASE."
	                  MUMPS_INCLUDES MUMPS_LIBRARIES)

  IF (MUMPS_FOUND)
    INCLUDE_DIRECTORIES(${MUMPS_INCLUDES})
    MESSAGE(STATUS "MUMPS Has been found")
    MESSAGE(STATUS "MUMPS_DIR: ${MUMPS_BASE}")
    MESSAGE(STATUS "MUMPS_INCLUDES: ${MUMPS_INCLUDES}")
    MESSAGE(STATUS "MUMPS_LIBRARIES: ${MUMPS_LIBRARIES}")
  ENDIF(MUMPS_FOUND)

ELSE(NOT MPI_ENABLED)

  MESSAGE(STATUS "MUMPS librairies: MPI is required")

ENDIF(MPI_ENABLED)
