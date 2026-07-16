add_library(cwrapper OBJECT cwrapper.c)

target_include_directories(cwrapper
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cwrapper
    PUBLIC
    Python::NumPy
)
