
if (pyccel_main_language == 'fortran')
  pyc_math_f90 = static_library('pyc_math_f90', 'pyc_math_f90.F90', install: true)
  pyc_math_f90_dep = declare_dependency(link_with: pyc_math_f90)
endif

if (pyccel_main_language == 'c)
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : true)

pyc_math_c = static_library('pyc_math_c',
  'pyc_math_c.c',
  include_directories: ['.'],
  dependencies: [m_dep, stc_dep],
  install: true)

pyc_math_c_dep = declare_dependency(
  include_directories: ['.'],
  dependencies: [stc_dep],
  link_with: pyc_math_c)
endif()
