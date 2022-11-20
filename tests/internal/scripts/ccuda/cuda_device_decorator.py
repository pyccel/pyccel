from pyccel.decorators import kernel, types, device

@device
@types('int[:]')
def funcb(a):
    a[0] += 1

@kernel
@types('int[:]')
def funca(a):
    a[0] += 1
    funcb(a)