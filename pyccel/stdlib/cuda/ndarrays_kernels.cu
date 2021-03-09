#include "ndarrays_kenels.h"

__global__
void cuda_array_arange(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int64[i] = (i + start);
}

__global__
void cuda_array_fill_int(int64_t c, t_ndarray arr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int64[i] = c;
}