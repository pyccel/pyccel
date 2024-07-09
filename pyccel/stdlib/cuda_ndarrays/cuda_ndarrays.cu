#include "cuda_ndarrays.h"

void    device_memory(void** devPtr, size_t size)
{
    cudaMalloc(devPtr, size);
}

void    managed_memory(void** devPtr, size_t size)
{
    cudaMallocManaged(devPtr, size);
}

void    host_memory(void** devPtr, size_t size)
{
    cudaMallocHost(devPtr, size);
}
t_ndarray   cuda_array_create(enum e_memory_locations location, int32_t nd, int64_t *shape,
        enum e_types type, bool is_view)
{
    t_ndarray arr;
    void (*fun_ptr_arr[])(void**, size_t) = {managed_memory, host_memory, device_memory};

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
    arr.shape = (int64_t *)malloc(arr.nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;

    if (!is_view)
        (*fun_ptr_arr[location])(&(arr.raw_data), arr.buffer_size);
    return (arr);
}

int32_t cuda_free_host(t_ndarray arr)
{
    if (arr.shape == NULL)
        return (0);
    cudaFreeHost(arr.raw_data);
    arr.raw_data = NULL;
    cudaFree(arr.shape);
    arr.shape = NULL;
    cudaFree(arr.strides);
    arr.strides = NULL;
    return (1);
}

__host__ __device__
int32_t cuda_free(t_ndarray arr)
{
    if (arr.shape == NULL)
        return (0);
    cudaFree(arr.raw_data);
    arr.raw_data = NULL;
    cudaFree(arr.shape);
    arr.shape = NULL;
    return (0);
}

__host__ __device__
int32_t cuda_free_pointer(t_ndarray arr)
{
    if (arr.is_view == false || arr.shape == NULL)
        return (0);
    cudaFree(arr.shape);
    arr.shape = NULL;
    return (0);
}
