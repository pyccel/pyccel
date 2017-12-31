# coding: utf-8

# TODO raise StopIteration() instead of 'StopIteration'

#$ header class StopIteration(public, hide)
#$ header method __init__(StopIteration)
#$ header method __del__(StopIteration)
class StopIteration(object):

    def __init__(self):
        pass

    def __del__(self):
        pass

#$ header class Parallel(public, with, openacc)
#$ header method __init__(Parallel, str [:], str [:], int, int, int, str [:], bool, str [:], str [:], str [:], str [:], str [:], str [:], str [:], str [:], str [:], str)
#$ header method __del__(Parallel)
#$ header method __enter__(Parallel)
#$ header method __exit__(Parallel, str, str, str)
class Parallel(object):

    def __init__(self, Async=None, wait=None, numGangs=None,
                 numWorkers=None, vectorLength=None, deviceType=None,
                 If=None, reduction=None,
                 copy=None, copyin=None, copyout=None,
                 create=None, present=None, devicePtr=None,
                 private=None, firstPrivate=None, default=None):

        self._async        = Async
        self._wait         = wait
        self._numGangs     = numGangs
        self._numWorkers   = numWorkers
        self._vectorLength = vectorLength
        self._deviceType   = deviceType
        self._if           = If
        self._reduction    = reduction
        self._copy         = copy
        self._copyin       = copyin
        self._copyout      = copyout
        self._create       = create
        self._present      = present
        self._devicePtr    = devicePtr
        self._private      = private
        self._firstPrivate = firstPrivate
        self._default      = default

    def __del__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass
