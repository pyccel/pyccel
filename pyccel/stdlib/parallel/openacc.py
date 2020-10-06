# coding: utf-8

#$ header class PyccelStopIteration(public, hide)
#$ header method __init__(PyccelStopIteration)
#$ header method __del__(PyccelStopIteration)
class PyccelStopIteration(object):

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

    def __exit__(self, dtype, value, tb):
        pass

#$ header class Range(public, iterable, openacc)
#$ header method __init__(Range, int, int, int, int, str [:], str [:], str [:], str, str, str [:], str [:], str, str [:], str[:])
#$ header method __del__(Range)
#$ header method __iter__(Range)
#$ header method __next__(Range)
class Range(object):

    def __init__(self, start, stop, step, collapse=None,
                 gang=None, worker=None, vector=None,
                 seq=None, auto=None, tile=None,
                 device_type=None, independent=None,
                 private=None, reduction=None):

        self.start = start
        self.stop  = stop
        self.step  = step

        self._collapse    = collapse
        self._gang        = gang
        self._worker      = worker
        self._vector      = vector
        self._seq         = seq
        self._auto        = auto
        self._tile        = tile
        self._device_type = device_type
        self._independent = independent
        self._private     = private
        self._reduction   = reduction

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
            raise StopIteration

