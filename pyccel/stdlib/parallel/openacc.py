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

    def __init__(self, Async=None, wait=None, num_gangs=None,
                 num_workers=None, vector_length=None, device_type=None,
                 If=None, reduction=None,
                 copy=None, copyin=None, copyout=None,
                 create=None, present=None, deviceptr=None,
                 private=None, firstprivate=None, default=None):

        self._async         = Async
        self._wait          = wait
        self._num_gangs     = num_gangs
        self._num_workers   = num_workers
        self._vector_length = vector_length
        self._device_type   = device_type
        self._if            = If
        self._reduction     = reduction
        self._copy          = copy
        self._copyin        = copyin
        self._copyout       = copyout
        self._create        = create
        self._present       = present
        self._deviceptr     = deviceptr
        self._private       = private
        self._firstprivate  = firstprivate
        self._default       = default

    def __del__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass
