from numba import njit

@njit(fastmath=True)
def ackermann(m : int, n : int) -> int:
    """  Total computable function that is not primitive recursive.
    This function is useful for testing recursion
    """
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))
