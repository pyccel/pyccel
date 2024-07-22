from pyccel import cuda

def addition_cuda_host_2Darrays():
    a = cuda.host_empty((10,10))
    b = cuda.host_empty((10,10))

    for i in range(10):
        for j in range(10):
            a[i][j] = 1
            b[i][j] = 1
    b+=a
    b+=1

    for i in range(10):
        for j in range(10):
            print(b[i][j])
if __name__ == '__main__':
    addition_cuda_host_2Darrays()

