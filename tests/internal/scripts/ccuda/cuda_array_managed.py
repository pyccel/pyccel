from pyccel import cuda

if __name__ == '__main__':
    a = cuda.array([0,1,2,3,4], memory_location = 'managed')
