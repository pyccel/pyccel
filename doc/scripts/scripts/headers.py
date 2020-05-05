#$ header function f(int, int) &
#$                 results (int)

def f(a,b):
    c = a + b
    return c


#$ header function g(int, int) &
#$                 results &
#$ (int, int)

def g(a,b):
    c = a + b
    d = a - b
    return c, d

print('hello world')
