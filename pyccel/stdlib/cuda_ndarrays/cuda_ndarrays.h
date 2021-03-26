#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

# include "ndarrays.h"
# include "cuda_ndarrays.h"

void cuda_array_arange(t_ndarray arr, int start);
void cuda_array_fill(int64_t c, t_ndarray arr);
#endif
