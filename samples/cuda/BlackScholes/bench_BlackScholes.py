import numpy as np
from mod_pyccel_BlackScholes import wrap_blackscholes
from BlackScholes import blackscholesCPU
from time import perf_counter
import math


if __name__ == "__main__":
    opt_n = 4000000
    num_iter = 1024
    opt_sz = opt_n * 4
    riskfree = 0.02
    volatility = 0.30

    print("Initializing data...\n")
    print("...allocating CPU memory for options.\n")
    h_CallResultCPU =  np.zeros(opt_n, dtype=np.float64)
    h_PutResultCPU = np.full(opt_n, -1.0, dtype=np.float64)
    h_StockPrice = np.empty(opt_n, dtype=np.float64)
    h_OptionStrike = np.empty(opt_n, dtype=np.float64)
    h_OptionYears = np.empty(opt_n, dtype=np.float64)

    # np.random.seed(1)
    for i in range(opt_n):
        h_StockPrice[i] = np.random.uniform(5.0, 30.0)
        h_OptionStrike[i] = np.random.uniform(1.0, 100.0)
        h_OptionYears[i] = np.random.uniform(0.25, 10.0)

    hTimerS = perf_counter()
    h_CallResultGPU = wrap_blackscholes(h_StockPrice, h_OptionStrike, h_OptionYears, opt_n, num_iter)
    hTimerE = perf_counter()
    gpuTime = ((hTimerE - hTimerS) * 1000) / num_iter


    print("Options count             : %i     \n" % (2 * opt_n))
    print("BlackScholesGPU() time    : %f msec\n" % gpuTime)
    print("Effective memory bandwidth: %f GB/s\n" % 
         (((5 * opt_sz) * 1E-9) / (gpuTime * 1E-3)))
    print("Gigaoptions per second    : %f     \n\n" %
         (((2 * opt_n) * 1E-9) / (gpuTime * 1E-3)))
    print(
    "BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, NumDevsUsed = %u, Workgroup = %u\n" %
      ((((2.0 * opt_n) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime * 1e-3,
      (2 * opt_n), 1, 128))
    print("Checking the results...\n")
    print("...running CPU calculations.\n\n")
    # Calculate options values on CPU
    blackscholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                  h_OptionYears, riskfree, volatility, opt_n)

    print("Comparing the results...\n")

    sum_delta = 0
    sum_ref = 0
    max_delta = 0

    print(h_CallResultCPU[0], h_CallResultGPU[0])
    for i in range(opt_n):
        ref = h_CallResultCPU[i]
        delta = math.fabs(h_CallResultCPU[i] - h_CallResultGPU[i])

        if delta > max_delta:
            max_delta = delta
        
        sum_delta = sum_delta + delta
        sum_ref = sum_ref + math.fabs(ref)
    
    l1norm = sum_delta / sum_ref
    print("L1 norm: %E\n"% l1norm)
    print("Max absolute error: %E\n\n"% max_delta)