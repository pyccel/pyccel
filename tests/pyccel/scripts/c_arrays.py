# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

if __name__ == '__main__':
    # ---------------------------- Array creation ---------------------------------
    ac0 = np.array([0])

    ac1 = np.array([0, 0, 0, 0])

    ac2 = np.array([1, 2, 3, 4, 5])

    ac3 = np.array([-1, -2, -3, -4, -5])

    print(ac0[0])
    for i in range(4):
        print(ac1[i])
    print()

    for i in range(5):
        print(ac2[i])
    print()

    for i in range(5):
        print(ac3[i])
    print()

    # ------------------------------- Array full ----------------------------------

    af1 = np.full(5, 2.0, dtype=float)
    for i in range(5):
        print(af1[i])
    print()

    af2 = np.full((5, 5), 31254626, dtype=int)
    for i in range(5):
        for j in range(5):
            print(af2[i][j])
        print()

    af3 = np.full((20, 5), -1.58, dtype=np.double)
    for i in range(20):
        for j in range(5):
            print(af3[i][j])
        print()

    af4 = np.full((3, 10), 1+2j, dtype=complex)
    for i in range(3):
        for j in range(10):
            print(af4[i][j])
        print()

    af5 = np.full(5, complex(12.2, 13), dtype=complex)
    for i in range(5):
        print(af5[i])
    print()
    # ------------------------------ Array empty ----------------------------------
    ao3 = np.ones(10)
    for i in range(10):
        print(ao3[i])
    print()

    ao4 = np.ones((2,3))
    for i in range(2):
        for j in range(3):
            print(ao4[i][j])
        print()

    ae1 = np.empty((2,3))

    afl = np.full_like(ae1, 5.21, float)
    for i in range(2):
        for j in range(3):
            print(afl[i][j])
        print()
    aol = np.ones_like(af4)
    for i in range(3):
        for j in range(10):
            print(aol[i][j])
        print()

    azl = np.zeros_like(afl)
    for i in range(2):
        for j in range(3):
            print(azl[i][j])
        print()

    # ------------------------------ Array init ----------------------------------

    a = np.array([1,2,3])
    b = np.array(a)
    print(b[0], b[1], b[2])
    print()

