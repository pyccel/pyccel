from mod_addmul_pyccel import wrap_addmul as paddmul
import numpy as np
from time import perf_counter

def div_up(a, b):
    return (((a) + (b)-1) // (b))

if __name__ == "__main__":
    size = 1000
    num_iter = 10000
    tpb = 256
    x1 = np.random.uniform(0, 1, size)
    x2 = np.random.uniform(0, 1, size)
    out = np.empty(size)
    ng = div_up(size, tpb)

    hTimerS = perf_counter()
    for i in range(num_iter):
        a = paddmul(ng, tpb, x1, x2, out)
    hTimerE = perf_counter()
    gpuTime = ((hTimerE - hTimerS) * 1000)
    print("pyccel --- addmul total %f ms avg kernel time %f ms" % (gpuTime, gpuTime / num_iter))
