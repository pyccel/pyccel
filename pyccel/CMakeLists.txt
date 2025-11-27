cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS ${SKBUILD_STATE})

include(ExternalProject)

# Build and install STC
ExternalProject_Add(
    stc-install
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extensions/STC
    CONFIGURE_COMMAND meson setup <BINARY_DIR> <SOURCE_DIR> --buildtype=release --prefix=${SKBUILD_PLATLIB_DIR}/pyccel/extensions/stc-install -Dpkgconfig.relocatable=true
    BUILD_COMMAND meson compile -C <BINARY_DIR>
    INSTALL_COMMAND meson install -C <BINARY_DIR>
)

# Build and pack gFTL ready for installation
ExternalProject_Add(
    gftl-install
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extensions/gFTL
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR} -DGFTL_TOP_DIR=${SKBUILD_PLATLIB_DIR}/GFTL-1.13
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
)

# Install gFTL
install(CODE "
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
        \"${CMAKE_BINARY_DIR}/GFTL-1.13\"
        \"${SKBUILD_PLATLIB_DIR}/pyccel/extensions/gftl-install/GFTL-1.13\"
    )
")
