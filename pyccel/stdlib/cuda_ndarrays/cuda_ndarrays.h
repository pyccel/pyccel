#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

#include "../ndarrays/ndarrays.h"

__global__
void cuda_array_arange_int8(t_ndarray arr, int start);
__global__
void cuda_array_arange_int32(t_ndarray arr, int start);
__global__
void cuda_array_arange_int64(t_ndarray arr, int start);
__global__
void cuda_array_arange_double(t_ndarray arr, int start);

__global__
void _cuda_array_fill_int8(int8_t c, t_ndarray arr);
__global__
void _cuda_array_fill_int32(int32_t c, t_ndarray arr);
__global__
void _cuda_array_fill_int64(int64_t c, t_ndarray arr);
__global__
void _cuda_array_fill_double(double c, t_ndarray arr);

t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type, bool is_view, enum e_memory_locations location);
int32_t         cuda_free_array(t_ndarray dump);

int32_t cuda_free_host(t_ndarray arr);

__host__ __device__
int32_t cuda_free(t_ndarray arr);

__host__ __device__
int32_t cuda_free_pointer(t_ndarray arr);
#endif
