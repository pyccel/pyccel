IF(NOT DEFINED PROCESSOR_COUNT)
  # Unknown:
  SET(PROCESSOR_COUNT 0)

  # Linux:
  set(cpuinfo_file "/proc/cpuinfo")
  if(EXISTS "${cpuinfo_file}")
    file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
    list(LENGTH procs PROCESSOR_COUNT)
  endif()

  # Mac:
  if(APPLE)
    find_program(cmd_sys_pro "system_profiler -detailLevel mini")
    if(cmd_sys_pro)
      execute_process(COMMAND ${cmd_sys_pro} OUTPUT_VARIABLE info)
      string(REGEX REPLACE "^.*Total Number of Cores: ([0-9]+).*$" "\\1"
        PROCESSOR_COUNT "${info}")
    endif()
  endif()

  # Windows:
  if(WIN32)
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
  endif()

  SET(PROCESSOR_COUNT ${PROCESSOR_COUNT} CACHE INTEGER "Processor number")

endif()

MESSAGE(STATUS "NUMBER OF PROCESSORS = ${PROCESSOR_COUNT}")
