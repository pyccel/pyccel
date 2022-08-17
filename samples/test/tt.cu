#include <stdlib.h>
#include <stdio.h>

__global__ void func(int *p)
{
    p[0] = 5;
    printf("%s\n", "Hello World!");
}

int main()
{
    int *p1;

    cudaMallocManaged(&p1, 3*sizeof(int));
    p1[0] = 0;
    func<<<1,2>>>(p1);
    cudaDeviceSynchronize();
    printf("%d\n", p1[0]);
    return 0;
}