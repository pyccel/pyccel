# coding: utf-8

#$ header class OpenmpRange(public, iterable)
#$ header method __init__(OpenmpRange, int, int, int, int, int, int)
#$ header method __del__(OpenmpRange)
#$ header method __next__(OpenmpRange)
#$ header method __iter__(OpenmpRange)

class OpenmpRange(object):

    def __init__(self, start, stop, step, schedule, chunksize, num_threads):
        self.start = start
        self.stop  = stop
        self.step  = step

    def __del__(self):
        print('> free')

    def __next__(self):
        print('> next')

    def __iter__(self):
        print('> iter')

p = OpenmpRange(0,3,1,-1,-1,-1)
