#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H
# include "../ndarrays/ndarrays.h"
__global__
void cuda_array_arange(t_ndarray arr, int start);
__global__
void cuda_array_fill(int64_t c, t_ndarray arr);
t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type);
int32_t         cuda_free_array(t_ndarray dump);
int32_t         cuda_free_pointer(t_ndarray dump);
#endif
