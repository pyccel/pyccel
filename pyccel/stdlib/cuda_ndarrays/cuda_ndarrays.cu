#include "cuda_ndarrays.h"

__global__
void cuda_array_arange_int8(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = index ; i < arr.length; i+=1)
		arr.nd_int8[i] = (i + start);
}
__global__
void cuda_array_arange_int32(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = index ; i < arr.length; i+=1)
		arr.nd_int32[i] = (i + start);
}
__global__
void cuda_array_arange_int64(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = index ; i < arr.length; i+=1)
		arr.nd_int64[i] = (i + start);
}
__global__
void cuda_array_arange_double(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = index ; i < arr.length; i+=1)
		arr.nd_double[i] = (i + start);
}

__global__
void cuda_array_fill_int8(int8_t c, t_ndarray arr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int8[i] = c;
}

__global__
void cuda_array_fill_int32(int32_t c, t_ndarray arr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int32[i] = c;
}

__global__
void cuda_array_fill_int64(int64_t c, t_ndarray arr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int64[i] = c;
}
__global__
void cuda_array_fill_double(double c, t_ndarray arr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_double[i] = c;
}

void    device_memory(void* devPtr, size_t size)
{
    cudaMalloc(&devPtr, size);
}

void    managed_memory(void* devPtr, size_t size)
{
    cudaMallocManaged(&devPtr, size);
}

void    host_memory(void* devPtr, size_t size)
{
    cudaMallocHost(&devPtr, size);
}

t_ndarray   cuda_array_create(int32_t nd, int64_t *shape,
        enum e_types type, bool is_view, enum e_memory_locations location)
{
    t_ndarray arr;
    void (*fun_ptr_arr[])(void*, size_t) = {managed_memory, host_memory, device_memory};

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
    arr.is_view = is_view;
    arr.length = 1;
    (*fun_ptr_arr[location])(&(arr.shape), arr.nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    (*fun_ptr_arr[location])(&(arr.strides), nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int32_t j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    if (!is_view)
        (*fun_ptr_arr[location])(&(arr.raw_data), arr.buffer_size);
    return (arr);
}

void cupy_ravel(t_ndarray *dest, t_ndarray src, enum e_memory_locations location)
{
    *dest = src;
    void (*fun_ptr_arr[])(void*, size_t) = {managed_memory, host_memory, device_memory};

    dest->nd = 1;
    (*fun_ptr_arr[location])(&(dest->shape), sizeof(int64_t));
    (*fun_ptr_arr[location])(&(dest->strides), sizeof(int64_t));
    *(dest->shape) = src.length;
    *(dest->strides) = 1;
    dest->is_view = true;
}

int32_t cuda_free_array(t_ndarray arr)
{
    if (arr.shape == NULL)
        return (0);
    free(arr.raw_data);
    arr.raw_data = NULL;
    free(arr.shape);
    arr.shape = NULL;
    free(arr.strides);
    arr.strides = NULL;
    return (1);
}


int32_t cuda_free_pointer(t_ndarray arr)
{
    if (arr.is_view == false || arr.shape == NULL)
        return (0);
    free(arr.shape);
    arr.shape = NULL;
    free(arr.strides);
    arr.strides = NULL;
    return (1);
}
