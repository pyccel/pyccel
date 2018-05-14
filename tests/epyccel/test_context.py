# coding: utf-8

import numpy as np

from pyccel.epyccel import ContextPyccel
from pyccel.epyccel import epyccel

def test_1():
    context = ContextPyccel(name='context_1')

    # ... insert functions
    def decr(x):
        y = x - 1
        return y

    def f1(m1, x):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    context.insert_function(decr, ['int'], kind='function')
    context.insert_function(f1, ['int', 'double [:]'], kind='procedure')
    # ...

    # ... insert constants
    context.insert_constant({'c_float': 1., 'c_int':2})
    # ...

    print(context)

    context.compile()


def test_2():
    context = ContextPyccel(name='context_2')

    # ... insert functions
    def decr(x):
        y = x - 1
        return y

    context.insert_function(decr, ['double'], kind='function', results=['double'])
    # ...

    context.compile()

    header = '#$ header procedure f2_py(int, double [:])'
    def f2_py(m1, x):
        for i in range(0, m1):
            y = x[i]
            z = decr(y)
            x[i] = z

    f = epyccel(f2_py, header, context=context)

    m1 = 3
    x = np.asarray(range(0, m1), dtype='float')
    f(m1, x)

    x_expected = np.array([-1., 0., 1.])
    assert(np.allclose(x, x_expected))



if __name__ == '__main__':
#    test_1()
    test_2()
