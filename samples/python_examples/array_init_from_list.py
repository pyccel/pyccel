from pyccel.decorators import types

@types('real')
def array_init_from_list_val():
    from numpy import array
    a = array([2, 3, 4, 1])
    return a

for a in array_init_from_list_val():
    print("%f" % a)
