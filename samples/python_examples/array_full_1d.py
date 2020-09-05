from pyccel.decorators import types

@types('real')
def array_init_from_list_val():
    from numpy import full
    a = full(4, 5)
    return a

for a in array_init_from_list_val():
    print("%f" % a)