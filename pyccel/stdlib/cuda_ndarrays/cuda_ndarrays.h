#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

# include <cuda_runtime.h>
# include <iostream>

typedef enum e_types
{
        nd_bool     = 0,
        nd_int8     = 1,
        nd_int16    = 3,
        nd_int32    = 5,
        nd_int64    = 7,
        nd_float    = 11,
        nd_double   = 12,
        nd_cfloat   = 14,
        nd_cdouble  = 15
} t_types;


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
    t_types            type;
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


t_cuda_ndarray  cuda_array_create(int32_t nd, int64_t *shape, enum e_types type, bool is_view ,
enum e_memory_locations location);
int32_t cuda_free_host(t_cuda_ndarray arr);



using namespace std;


#endif