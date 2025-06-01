def omp_taskloop(n : 'int'):
    func_result = 0
    #$omp parallel num_threads(n)
    #$omp taskloop
    for i in range(0, 10): # pylint: disable=unused-variable
        #$omp atomic
        func_result = func_result + 1
    #$omp end parallel
    return func_result
