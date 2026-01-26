
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

if("Fortran" IN_LIST languages)
    add_library(pyc_math_f90 OBJECT pyc_math_f90.f90)

    target_include_directories(pyc_math_f90
        PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
    )
else()

add_library(pyc_math_c OBJECT pyc_math_c.c)

target_include_directories(pyc_math_c
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pyc_math_c
    PRIVATE
    m
)

endif()
