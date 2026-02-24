if __name__ == "__main__":
    from numpy import array, full, ones, zeros

    x1 = (1+3j)
    x2 = (2+4j)
    x3 = array([(1+2j), (2+3j)])
    x4 = zeros((10, 2), complex); x4 += 1j
    x5 = ones ((50, 7), complex); x5 += 2j
    x6 = full((5, 5), (1+2j))

    r1 = x1
    r2 = x2
    r3 = x3.sum()
    r4 = x4.sum()
    r5 = x5.sum()
    r6 = x6.sum()

    print(r1, r2, r3, r4, r5, r6)
