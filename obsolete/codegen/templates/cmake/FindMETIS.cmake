# - Try to find Jorek 
# Once done this will define
#
#  METIS_DIR          - Install Directory for Jorek 
#  METIS_FOUND        - system has Jorek
#  METIS_INCLUDES     - Jorek include directories
#  METIS_LIBRARIES    - Link these to use Jorek
#  The following variables are TODO
#  METIS_COMPILER     - Compiler used by Jorek, helpful to find a compatible MPI
#  METIS_DEFINITIONS  - Compiler switches for using Jorek
#  METIS_MPIEXEC      - Executable for running MPI programs
#  METIS_VERSION      - Version string (MAJOR.MINOR.SUBMINOR)
#
#  Usage:
#  find_package(METIS)                  
#
# Setting these changes the behavior of the search
#  METIS_DIR - directory in which PETSc resides
#  The following variable is TODO
#  METIS_ARCH - build architecture
#

SET(METIS_DIR "$ENV{METIS_DIR}")
SET(METIS_INCLUDES "${METIS_DIR}/include")
SET(METIS_LIB_DIRS "${METIS_DIR}/lib")

FIND_LIBRARY(METIS_LIB NAMES metis 
	                  HINTS ${METIS_LIB_DIRS})

SET(METIS_LIBRARIES 
	${METIS_LIB}
	)

SET (METIS_INCLUDES  ${METIS_INCLUDES} CACHE STRING "METIS include path" FORCE)
SET (METIS_LIBRARIES ${METIS_LIBRARIES} CACHE STRING "METIS libraries" FORCE)
MARK_AS_ADVANCED(METIS_INCLUDES METIS_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args (METIS
	"METIS could not be found.  Be sure to set METIS_DIR."
	METIS_INCLUDES METIS_LIBRARIES)

##########################################################################
IF (METIS_FOUND)
	INCLUDE_DIRECTORIES(${METIS_INCLUDES})
	MESSAGE(STATUS "METIS Has been found")
	MESSAGE(STATUS "METIS_DIR:${METIS_DIR}")
	MESSAGE(STATUS "METIS_INCLUDES:${METIS_INCLUDES}")
	MESSAGE(STATUS "METIS_LIBRARIES:${METIS_LIBRARIES}")
ENDIF(METIS_FOUND)
##########################################################################
