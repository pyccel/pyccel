#ifndef NDARRAYS_KERNELS_H
# define NDARRAYS_KERNELS_H

# include "ndarrays.h"

void cuda_array_arange(t_ndarray arr, int start);
void cuda_array_fill(int64_t c, t_ndarray arr);
#endif