import numpy as np

def single_return(x: 'int'):
    return np.ones(x)

def multi_returns(x: 'int'):
    return single_return(x), np.array([1,2,3,4])

a = single_return(5)
b = multi_returns(5)

print(a)
print(b)
