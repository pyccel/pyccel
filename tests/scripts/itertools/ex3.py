# coding: utf-8

from pyccel.stdlib.parallel.openmp import omp_get_thread_num

#$ header class StopIteration(public, hide)
#$ header method __init__(StopIteration)
#$ header method __del__(StopIteration)
class StopIteration(object):

    def __init__(self):
        pass

    def __del__(self):
        pass

#$ header class Range(public, iterable, openmp)
#$ header method __init__(Range, int, int, int, bool, int, str [:], str [:], str [:], str [:], str [:], int, str [:])
#$ header method __del__(Range)
#$ header method __iter__(Range)
#$ header method __next__(Range)
class Range(object):

    def __init__(self, start, stop, step, nowait=None, collapse=None,
                 private=None, firstprivate=None, lastprivate=None,
                 reduction=None, schedule=None, ordered=None, linear=None):

        self.start = start
        self.stop  = stop
        self.step  = step

        self._ordered      = ordered
        self._private      = private
        self._firstprivate = firstprivate
        self._lastprivate  = lastprivate
        self._linear       = linear
        self._reduction    = reduction
        self._schedule     = schedule
        self._collapse     = collapse
        self._nowait       = nowait

        self.i = start

    def __del__(self):
        print('> free')

    def __iter__(self):
        self.i = 0

    def __next__(self):
        if (self.i < self.stop):
            i = self.i
            self.i = self.i + 1
        else:
            raise StopIteration()

x = 0.0

#$ omp parallel
for i in Range(-2, 5, 1, nowait=True, private=['i', 'idx'], reduction=['+', 'x'], schedule='static', ordered=True):
    idx = omp_get_thread_num()

    x += 2 * i
#    print("> thread id : ", idx, " working on ", i)
#$ omp end parallel

print('x = ', x)
