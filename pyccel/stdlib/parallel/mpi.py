# coding: utf-8

# TODO : steps, periods, reorder to be passed as arguments

#$ header class Cart(public)
#$ header method __init__(Cart)
#$ header method __del__(Cart)

class Cart(object):
    def __init__(self):

        steps   = [1, 1]
        periods = [False, True]
        reorder = False

        # ... TODO : to be computed using 'len'
        self.ndims       = 2
        self.n_neighbour = 4
        # ...

        # ... Constants
        north = 0
        east  = 1
        south = 2
        west  = 3
        # ...

        # ... TODO : use steps, periods, reorder arguments
        self.neighbour = zeros(self.n_neighbour, int)
        self.coords    = zeros(self.ndims, int)
        self.dims      = zeros(self.ndims, int)

        self.steps   = [1, 1]
        self.periods = [False, True]
        self.reorder = False
        # ...

    def __del__(self):
        pass

#p = Cart()
#
#del p
