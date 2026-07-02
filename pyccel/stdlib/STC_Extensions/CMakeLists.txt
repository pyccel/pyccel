
add_library(STC_Extensions INTERFACE)
target_include_directories(STC_Extensions
    INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)


add_library(CSpan_extensions INTERFACE)
target_include_directories(CSpan_extensions
    INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

