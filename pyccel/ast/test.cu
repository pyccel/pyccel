#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 512;
    int size = n * sizeof(int);
    int *a, *b, *c;

    // Allocate unified memory - accessible from CPU or GPU
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize arrays on the host (CPU)
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Launch kernel with n threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(a, b, c, n);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Verify the result
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Error at index " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl
                      << std::endl;
            return 1;
        }
    }

    std::cout << "Success!" << std::endl;
    return 0;
}
