# pylint: disable=missing-function-docstring, missing-module-docstring
def declare_target(data : 'float[:]', n : 'int'):
    import numpy as np
    #$ omp declare target
    lookup_table = [1, 2, 3]
    #$ omp end declare target
    result = np.zeros(n)
    #$ omp target map(to: data) map(from: result)
    for i in range(n):
        x = i % 3
        result[i] = data[i] * lookup_table[i]
    #$ omp end target