# coding: utf-8

#$ header class MyMatrix(abstract, public)
#$ header method __init__(MyMatrix, int, int)

class MyMatrix(object):
    def __init__(self, n_rows, n_cols):
        x = n_rows + n_cols
