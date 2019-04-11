# coding: utf-8

# ... internal applications
# functors
_internal_map_functors = ['map', 'xmap', 'tmap']
_internal_functors = _internal_map_functors + ['reduce']

_internal_zip_functions = ['zip']
_internal_product_functions = ['product']
_internal_functions = _internal_zip_functions + _internal_product_functions

_internal_applications = _internal_functions + _internal_functors
# ...

# ...
# TODO atan2, pow
_elemental_math_functions = ['acos',
                             'asin',
                             'atan',
                             'cos',
                             'cosh',
                             'exp',
                             'log',
                             'log10',
                             'sin',
                             'sinh',
                             'sqrt',
                             'tan',
                             'tanh',
                            ]

# TODO add cross, etc
_math_vector_functions = ['dot']

# TODO
_math_matrix_functions = ['matmul']

_math_functions = _elemental_math_functions + _math_vector_functions + _math_matrix_functions
# ...
