# coding: utf-8

#$ header class Range(public, iterable)
#$ header method __init__(Range, int, int, int)
#$ header method __del__(Range)
#$ header method __iter__(Range)
#$ header method __next__(Range)

class Range(object):

    def __init__(self, start, stop, step):
        self.start = start
        self.stop  = stop
        self.step  = step

        self.i = 0
        self.j = 0

    def __del__(self):
        print('> free')

    def __iter__(self):
        self.i = 0
        self.j = 0

    def __next__(self):
        i = self.i
        self.i = self.i + 1

p = Range(0,3,1)

for i,j in p:
    print(k)
