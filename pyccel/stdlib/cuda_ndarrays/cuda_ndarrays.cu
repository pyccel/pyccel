#include "cuda_ndarrays.h"

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

t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    switch (type)
    {
        case nd_int8:
            arr.type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr.type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr.type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr.type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr.type_size = sizeof(float);
            break;
        case nd_double:
            arr.type_size = sizeof(double);
            break;
        case nd_bool:
            arr.type_size = sizeof(bool);
            break;
    }
    arr.is_view = false;
    arr.length = 1;
    cudaMallocManaged(&(arr.shape), arr.nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    cudaMallocManaged(&(arr.strides), nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int32_t j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    cudaMallocManaged(&(arr.raw_data), arr.buffer_size);
    return (arr);
}
