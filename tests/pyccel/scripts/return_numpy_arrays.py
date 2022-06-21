import numpy as np

def single_return():
    a = np.array([1,2,3,4])
    return a

def multi_returns():
    x = single_return()
    z = np.array([1,2,3,4])
    return x, z

a = single_return()
b,c = multi_returns()

print(a, b, c)
