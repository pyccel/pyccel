#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

# include <cuda_runtime.h>
# include <iostream>

typedef enum cu_types
{
        cu_bool     = 0,
        cu_int8     = 1,
        cu_int16    = 3,
        cu_int32    = 5,
        cu_int64    = 7,
        cu_float    = 11,
        cu_double   = 12,
        cu_cfloat   = 14,
        cu_cdouble  = 15
} t_cu_types;


enum e_memory_locations
{
        allocateMemoryOnHost,
        allocateMemoryOnDevice
};

typedef enum e_order
{
    order_f,
    order_c,
} t_order;

typedef struct  s_cuda_ndarray
{
    void            *raw_data;
    /* number of dimensions */
    int32_t                 nd;
    /* shape 'size of each dimension' */
    int64_t                 *shape;
    /* strides 'number of elements to skip to get the next element' */
    cu_types            type;
    /* type size of the array elements */
    int32_t                 type_size;
    /* number of element in the array */
    int32_t                 length;
    /* size of the array */
    int32_t                 buffer_size;
    /* True if the array does not own the data */
    bool                    is_view;
    /* stores the order of the array: order_f or order_c */
    t_order            order;
}               t_cuda_ndarray;


t_cuda_ndarray  cuda_array_create(int32_t nd, int64_t *shape, enum cu_types type, bool is_view ,
enum e_memory_locations location);
int32_t cuda_free_host(t_cuda_ndarray arr);
__host__ __device__
int32_t cuda_free(t_cuda_ndarray arr);


using namespace std;

#endif