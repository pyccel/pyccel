#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

#include "../ndarrays/ndarrays.h"

// enum e_cuda_types
// {
//         nd_bool     = 0,
//         nd_int8     = 1,
//         nd_int16    = 3,
//         nd_int32    = 5,
//         nd_int64    = 7,
//         nd_float    = 11,
//         nd_double   = 12
// };

// typedef struct  s_cuda_ndarray
// {
//     /* raw data buffer*/
//     union {
//             void            *raw_data;
//             int8_t          *nd_int8;
//             int16_t         *nd_int16;
//             int32_t         *nd_int32;
//             int64_t         *nd_int64;
//             float           *nd_float;
//             double          *nd_double;
//             bool            *nd_bool;
//             };
//     /* number of dimensions */
//     int32_t                 nd;
//     /* shape 'size of each dimension' */
//     int64_t                 *shape;
//     /* strides 'number of bytes to skip to get the next element' */
//     int64_t                 *strides;
//     /* type of the array elements */
//     enum e_types            type;
//     /* type size of the array elements */
//     int32_t                 type_size;
//     /* number of element in the array */
//     int32_t                 length;
//     /* size of the array */
//     int32_t                 buffer_size;
//     /* True if the array does not own the data */
//     bool                    is_view;
// }               t_cuda_ndarray;

__global__
void cuda_array_arange(t_ndarray arr, int start);
__global__
void cuda_array_fill(int64_t c, t_ndarray arr);
t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type, bool is_view);
int32_t         cuda_free_array(t_ndarray dump);
int32_t         cuda_free_pointer(t_ndarray dump);
#endif
