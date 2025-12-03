
if add_languages('fortran', required : false, native: false)
  pyc_math_f90 = library('pyc_math_f90', 'pyc_math_f90.f90', install: true)
  pyc_math_f90_dep = declare_dependency(link_with: pyc_math_f90)
endif

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : true)

pyc_math_c = library('pyc_math_c',
  'pyc_math_c.c',
  include_directories: ['.'],
  dependencies: m_dep,
  install: true)

pyc_math_c_dep = declare_dependency(
  include_directories: ['.'],
  link_with: pyc_math_c)
