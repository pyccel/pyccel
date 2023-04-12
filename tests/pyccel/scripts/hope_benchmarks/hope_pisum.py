# pylint: disable=missing-function-docstring, missing-module-docstring

def pisum():
    for _ in range(1, 501):
        pi_sum = 0.0
        for k in range(1, 10001):
            pi_sum += 1.0/(k*k)
    return pi_sum

if __name__ == '__main__':
    print(pisum())
