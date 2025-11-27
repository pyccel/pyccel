cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS ${SKBUILD_STATE})

include(ExternalProject)

function(install_stc c_compiler compiler_family)
    find_program(MY_CC_PATH ${c_compiler} NO_CACHE)

    if (MY_CC_PATH)
        message(STATUS "Found compiler ${c_compiler} ${MY_CC_PATH}")
        # Build and install STC
        ExternalProject_Add(
            stc_install_${compiler_family}
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/extensions/STC
            CONFIGURE_COMMAND CC=${MY_CC_PATH} meson setup <BINARY_DIR> <SOURCE_DIR> --buildtype=release --prefix=${SKBUILD_PLATLIB_DIR}/pyccel/extensions/stc_install_${compiler_family} -Dpkgconfig.relocatable=true
            BUILD_COMMAND meson compile -C <BINARY_DIR>
            INSTALL_COMMAND meson install -C <BINARY_DIR>
        )

        # Create an empty __init__.py so Python treats STC as a package
        file(WRITE "${SKBUILD_PLATLIB_DIR}/pyccel/extensions/stc_install_${compiler_family}/__init__.py" "")
        message(STATUS "Building STC with ${compiler_family}")
    else()
        message(WARNING "Skipping ${compiler_family} (compiler not found)")
    endif()
    unset(MY_CC_PATH)
endfunction()

install_stc(gcc GNU)
install_stc(icx intel)
install_stc(nvc nvidia)
install_stc(clang LLVM)

# Build and pack gFTL ready for installation
ExternalProject_Add(
    gftl_install
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extensions/gFTL
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR} -DGFTL_TOP_DIR=${SKBUILD_PLATLIB_DIR}/GFTL-1.13
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
)

install(DIRECTORY
    ${CMAKE_BINARY_DIR}/GFTL-1.13
    DESTINATION ${SKBUILD_PROJECT_NAME}/extensions/gftl_install
)
# Create an empty __init__.py so Python treats gFTL as a package
file(WRITE "${SKBUILD_PLATLIB_DIR}/pyccel/extensions/gftl_install/__init__.py" "")
