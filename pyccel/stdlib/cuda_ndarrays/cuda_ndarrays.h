#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

# include <cuda_runtime.h>
# include <iostream>
#include "../ndarrays/ndarrays.h"



enum e_memory_locations
{
        allocateMemoryOnHost,
        allocateMemoryOnDevice
};



t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type, bool is_view ,
enum e_memory_locations location);
int32_t cuda_free_host(t_ndarray  arr);
__host__ __device__
int32_t cuda_free(t_ndarray  arr);


using namespace std;

#endif