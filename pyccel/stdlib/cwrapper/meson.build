py_dep = py.dependency()
numpy_dep = dependency('numpy')

cwrapper_incdir = include_directories('.')

cwrapper_dep = declare_dependency(sources: 'cwrapper.c',
                                  include_directories : cwrapper_incdir,
                                  dependencies: [py_dep, numpy_dep])
