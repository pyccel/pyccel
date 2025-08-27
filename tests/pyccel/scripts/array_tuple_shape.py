import numpy as np

def g():
    return (1,2)

def f():
    shape = (1, 2)
    a = np.zeros(g())
    b = np.zeros((1, 2))
    c = a[0,0]
    print(a.shape)
    print(b.shape)

if __name__ == '__main__':
    f() 
