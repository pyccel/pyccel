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

if __name__ == "__main__":
    opt_n = 20000000
    num_iter = 512
    opt_sz = opt_n * 4
    riskfree = 0.02
    volatility = 0.30

    h_CallResultGPU = np.empty(opt_n, dtype=np.float64)
    h_PutResultGPU = np.empty(opt_n, dtype=np.float64)
    h_StockPrice = np.empty(opt_n, dtype=np.float64)
    h_OptionStrike = np.empty(opt_n, dtype=np.float64)
    h_OptionYears = np.empty(opt_n, dtype=np.float64)

    for i in range(opt_n):
        h_StockPrice[i] = np.random.uniform(5.0, 30.0)
        h_OptionStrike[i] = np.random.uniform(1.0, 100.0)
        h_OptionYears[i] = np.random.uniform(0.25, 10.0)
    
    d_CallResult = cuda.device_array(opt_n, dtype=np.float64)
    d_PutResult = cuda.device_array(opt_n, dtype=np.float64)
    d_StockPrice = cuda.to_device(h_StockPrice)
    d_OptionStrike = cuda.to_device(h_OptionStrike)
    d_OptionYears = cuda.to_device(h_OptionYears)
    cuda.synchronize()

    for i in range(num_iter):
        blackscholesGPU[int(div_up(opt_n / 2, 128)), 128](d_CallResult,
                d_PutResult, d_StockPrice, d_OptionStrike,
                d_OptionYears, riskfree, volatility, opt_n)

    cuda.synchronize()
    h_CallResultGPU = d_CallResult.copy_to_host()
    cuda.synchronize()
