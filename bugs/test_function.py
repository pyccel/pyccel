

#$ header function decr(double)
def decr(x):
    y = x - 1
    return y

#$ header procedure f2_py(int, double [:])
def f2_py(m1, x):
    for i in range(0, m1):
        y = x[i]
        x[i] = decr(y)
