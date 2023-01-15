# pylint: disable=missing-function-docstring, missing-module-docstring/
#$ header function f(double[:],double[:,:,:],int)
@sympy
def g(v,w,i):
    from sympy import Lambda, Function ,symbols ,IndexedBase,Idx ,Max, Sum
    x = Function('x')
    i, n, j, dim, k =symbols('i, n, j, dim, k')
    v=IndexedBase('v')
    w=IndexedBase('w')
    net = Lambda((i, n, dim, k), Max(0.0, Sum(x(k)*w[n, k, i], (k, 0, dim-1))))
    dim = [10**4]*10
    index =[symbols('i%s'%m) for m in range(len(dim)+1)]
    y = [0]*len(dim)
    new = v[index[0]]
    for n in range(len(dim)):
        y[n] = net(index[n+1], n, dim[n], index[n])
        y[n] = y[n].subs(x(index[n]), new)
        new = y[n]
    return y[-1]

f=lambdify(g)

