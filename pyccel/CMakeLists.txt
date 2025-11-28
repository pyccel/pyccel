cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(COMMAND git submodule update --init)

include(ExternalProject)

function(install_stc c_compiler compiler_family)
    # Install STC on the chosen compiler.
    #
    # c_compiler : The name of the C compiler you would like to use.
    # compiler_family : The name of the compiler family (this is
    #               used in the name of the installed folder).

    # Try to locate the compiler
    find_program(MY_CC_PATH ${c_compiler} NO_CACHE)

    if (MY_CC_PATH)
        # If compiler exists install
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

    # Reset variable ready to rerun function
    unset(MY_CC_PATH)
endfunction()

if (NOT "${SKBUILD_STATE}" STREQUAL "sdist")
    # STC is required for the wheel only, so the setup is not required for sdist
    install_stc(gcc GNU)
    install_stc(icx intel)
    install_stc(nvc nvidia)
    install_stc(clang LLVM)
endif()

if ("${SKBUILD_STATE}" STREQUAL "sdist")
    # When preparing a sdist file install gFTL to source during CMake configuration
    # gFTL contains no compiled files so pre-installing removes the requirement
    # for users to install m4.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/gFTL)

    # Prepare to build and install gFTL
    execute_process(COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/extensions/gFTL -B ${CMAKE_BINARY_DIR}/gFTL -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/extensions/gftl_install -DGFTL_TOP_DIR=${SKBUILD_PLATLIB_DIR}/GFTL-1.13
        COMMAND_ERROR_IS_FATAL ANY
        COMMAND_ECHO STDOUT
    )
    # Build gFTL
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/gFTL
        COMMAND_ERROR_IS_FATAL ANY
        COMMAND_ECHO STDOUT
    )
    # Install gFTL
    execute_process(COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}/gFTL
        COMMAND_ERROR_IS_FATAL ANY
        COMMAND_ECHO STDOUT
    )

    # Create an empty __init__.py so Python treats gFTL as a package for importlib
    file(WRITE "${CMAKE_SOURCE_DIR}/extensions/gftl_install/__init__.py" "")

elseif(NOT EXISTS "${CMAKE_SOURCE_DIR}/extensions/gftl_install/__init__.py")
    # When preparing a wheel or installing normally, gFTL is installed in the usual
    # CMake build stage. This prevents the source folder being modified unnecessarily.
    ExternalProject_Add(
        gftl_install
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/extensions/gFTL
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR} -DGFTL_TOP_DIR=${SKBUILD_PLATLIB_DIR}/GFTL-1.13
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    )

    # Use install as otherwise files don't seem to be copied as expected
    install(DIRECTORY
        ${CMAKE_BINARY_DIR}/GFTL-1.13
        DESTINATION ${SKBUILD_PROJECT_NAME}/extensions/gftl_install
    )

    # Create an empty __init__.py so Python treats gFTL as a package for importlib
    file(WRITE "${SKBUILD_PLATLIB_DIR}/pyccel/extensions/gftl_install/__init__.py" "")
else()
    message(STATUS "gFTL files already generated")
endif()
