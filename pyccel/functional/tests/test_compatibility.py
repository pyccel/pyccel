import numpy as np

from pyccel.functional import xmap, tmap, xproduct

def test_map():
    f = lambda x,y:x+y

    xs = range(0, 4)
    ys = range(5, 8)

    assert(list(map(f, xs, ys)) == [5, 7, 9] )

def test_xmap():
    f = lambda x,y:x+y

    xs = range(0, 4)
    ys = range(5, 8)

    assert(np.allclose(list(xmap(f, xs, ys)), [5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]))

def test_tmap():
    f = lambda x,y:x+y

    xs = range(0, 4)
    ys = range(5, 8)

    assert(np.allclose(tmap(f, xs, ys), [[5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]]))

#########################################
if __name__ == '__main__':
    test_map()
    test_tmap()
    test_xmap()
