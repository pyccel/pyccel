# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8

#$ header class Parallel(public, with, openmp)
#$ header method __init__(Parallel, str, str, str [:], str [:], str [:], str [:], str, str [:], str)
#$ header method __del__(Parallel)
#$ header method __enter__(Parallel)
#$ header method __exit__(Parallel, str, str, str)

class Parallel(object):

    def __init__(self, num_threads, if_test,
                 private, firstprivate, shared,
                 reduction, default,
                 copyin, proc_bind):

        self._num_threads  = num_threads
        self._if_test      = if_test
        self._private      = private
        self._firstprivate = firstprivate
        self._shared       = shared
        self._reduction    = reduction
        self._default      = default
        self._copyin       = copyin
        self._proc_bind    = proc_bind

    def __del__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, dtype, value, tb):
        pass


x = 0.0
para    = Parallel()

with para:
    for i in range(10):
        x += 2 * i

print('x = ', x)


