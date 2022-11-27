from numba import cuda
import numpy as np
from time import perf_counter
import math

def div_up(a, b):
    return (((a) + (b)-1) // (b))

@cuda.jit(device=True, inline=True)
def cndGPU(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    cnd = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    if (d > 0):
        cnd = 1.0 - cnd
    return cnd

@cuda.jit(device=True, inline=True)
def BlackScholesBodyGPU(S, X, T, R, V):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / X) + (R + 0.5 * V * V) * T)
    d2 = d1 - V * sqrtT
    CNDD1 = cndGPU(d1)
    CNDD2 = cndGPU(d2)
    expRT = math.exp(-R * T)
    CallResult = S * CNDD1 - X * expRT * CNDD2
    PutResult = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1)
    return CallResult, PutResult



@cuda.jit()
def blackscholesGPU(d_CallResult, d_PutResult, d_StockPrice, d_OptionStrike, d_OptionYears, Riskfree, Volatility, optn):
    opt = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if opt < (optn / 2):
        for i in range(opt, optn, optn/2):
            d_CallResult[i], d_PutResult[i] = BlackScholesBodyGPU(d_StockPrice[i], d_OptionStrike[i], d_OptionYears[i], Riskfree, Volatility)

def cndCPU(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    cnd = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    if (d > 0):
        cnd = 1.0 - cnd
    return cnd

def BlackScholesBodyCPU(S, X, T, R, V):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / X) + (R + 0.5 * V * V) * T)
    d2 = d1 - V * sqrtT
    CNDD1 = cndCPU(d1)
    CNDD2 = cndCPU(d2)
    # Calculate Call and Put simultaneously
    expRT = math.exp(-R * T)
    callResult = (S * CNDD1 - X * expRT * CNDD2)
    putResult = (X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1))
    return callResult, putResult

def blackscholesCPU(h_CallResult, h_PutResult, h_StockPrice, h_OptionStrike, h_OptionYears, Riskfree, Volatility, opt_n):
    for opt in range(opt_n):
        h_CallResult[opt], h_PutResult[opt] = BlackScholesBodyCPU(h_StockPrice[opt],
                        h_OptionStrike[opt], h_OptionYears[opt], Riskfree,
                        Volatility)



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
    h_CallResultGPU = np.empty(opt_n, dtype=np.float64)
    h_PutResultGPU = np.empty(opt_n, dtype=np.float64)
    h_StockPrice = np.empty(opt_n, dtype=np.float64)
    h_OptionStrike = np.empty(opt_n, dtype=np.float64)
    h_OptionYears = np.empty(opt_n, dtype=np.float64)

    # d_StockPrice = cuda.device_array(opt_n, dtype=np.float64) 
    # d_OptionStrike = cuda.device_array(opt_n, dtype=np.float64)
    # d_OptionYears = cuda.device_array(opt_n, dtype=np.float64)

    print("...generating input data in CPU mem.\n")
    np.random.seed(1)
    for i in range(opt_n):
        h_StockPrice[i] = np.random.uniform(5.0, 30.0)
        h_OptionStrike[i] = np.random.uniform(1.0, 100.0)
        h_OptionYears[i] = np.random.uniform(0.25, 10.0)
    
    hTimerS = perf_counter()
    print("...allocating GPU memory for options.\n")
    d_CallResult = cuda.device_array(opt_n, dtype=np.float64)
    d_PutResult = cuda.device_array(opt_n, dtype=np.float64)
    print("...copying input data to GPU mem.\n")
    d_StockPrice = cuda.to_device(h_StockPrice)
    d_OptionStrike = cuda.to_device(h_OptionStrike)
    d_OptionYears = cuda.to_device(h_OptionYears)
    print("Data init done.\n\n")

    print("Executing Black-Scholes GPU kernel (%d iterations)...\n" % num_iter)
    cuda.synchronize()

    for i in range(num_iter):
        blackscholesGPU[int(div_up(opt_n / 2, 128)), 128](d_CallResult,
                d_PutResult, d_StockPrice, d_OptionStrike,
                d_OptionYears, riskfree, volatility, opt_n)
    
    print("\nReading back GPU results...\n")

    cuda.synchronize()
    h_CallResultGPU = d_CallResult.copy_to_host()
    h_PutResultGPU = d_PutResult.copy_to_host()
    cuda.synchronize()

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
