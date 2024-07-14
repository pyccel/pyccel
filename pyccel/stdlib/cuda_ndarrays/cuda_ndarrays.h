#ifndef CUDA_NDARRAYS_H
# define CUDA_NDARRAYS_H

# include <cuda_runtime.h>
# include <iostream>

#define GET_INDEX_EXP1(t, arr, a) t(arr, 0, a)
#define GET_INDEX_EXP2(t, arr, a, b) GET_INDEX_EXP1(t, arr, a) + t(arr, 1, b)
#define GET_INDEX_EXP3(t, arr, a, b, c) GET_INDEX_EXP2(t, arr, a, b) + t(arr, 2, c)
#define GET_INDEX_EXP4(t, arr, a, b, c, d) GET_INDEX_EXP3(t, arr, a, b, c) + t(arr, 3, d)
#define GET_INDEX_EXP5(t, arr, a, b, c, d, e) GET_INDEX_EXP4(t, arr, a, b, c, d) + t(arr, 4, e)
#define GET_INDEX_EXP6(t, arr, a, b, c, d, e, f) GET_INDEX_EXP5(t, arr, a, b, c, d, e) + t(arr, 5, f)
#define GET_INDEX_EXP7(t, arr, a, b, c, d, e, f, g) GET_INDEX_EXP6(t, arr, a, b, c, d, e, f) + t(arr, 6, g)
#define GET_INDEX_EXP8(t, arr, a, b, c, d, e, f, g, h) GET_INDEX_EXP7(t, arr, a, b, c, d, e, f, g) + t(arr, 7, h)
#define GET_INDEX_EXP9(t, arr, a, b, c, d, e, f, g, h, i) GET_INDEX_EXP8(t, arr, a, b, c, d, e, f, g, h) + t(arr, 8, i)
#define GET_INDEX_EXP10(t, arr, a, b, c, d, e, f, g, h, i, j) GET_INDEX_EXP9(t, arr, a, b, c, d, e, f, g, h, i) + t(arr, 9, j)
#define GET_INDEX_EXP11(t, arr, a, b, c, d, e, f, g, h, i, j, k) GET_INDEX_EXP10(t, arr, a, b, c, d, e, f, g, h, i, j) + t(arr, 10, k)
#define GET_INDEX_EXP12(t, arr, a, b, c, d, e, f, g, h, i, j, k, l) GET_INDEX_EXP11(t, arr, a, b, c, d, e, f, g, h, i, j, k) + t(arr, 11, l)
#define GET_INDEX_EXP13(t, arr, a, b, c, d, e, f, g, h, i, j, k, l, m) GET_INDEX_EXP12(t, arr, a, b, c, d, e, f, g, h, i, j, k, l) + t(arr, 12, m)
#define GET_INDEX_EXP14(t, arr, a, b, c, d, e, f, g, h, i, j, k, l, m, n) GET_INDEX_EXP13(t, arr, a, b, c, d, e, f, g, h, i, j, k, l, m) + t(arr, 13, n)
#define GET_INDEX_EXP15(t, arr, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) GET_INDEX_EXP14(t, arr, a, b, c, d, e, f, g, h, i, j, k, l, m, n) + t(arr, 14, o)

#define NUM_ARGS_H1(dummy, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0, ...) x0
#define NUM_ARGS(...) NUM_ARGS_H1(dummy, __VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define GET_INDEX_FUNC_H2(t, arr, ndim, ...) GET_INDEX_EXP##ndim(t, arr, __VA_ARGS__)
#define GET_INDEX_FUNC(t, arr, ndim, ...) GET_INDEX_FUNC_H2(t, arr, ndim, __VA_ARGS__)

#define GET_INDEX(arr, ...) GET_INDEX_FUNC(INDEX, arr, NUM_ARGS(__VA_ARGS__), __VA_ARGS__)
#define INDEX(arr, dim, a) (arr.strides[dim] * (a))
#define GET_ELEMENT(arr, type, ...) arr.type[GET_INDEX(arr, __VA_ARGS__)]

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
    int64_t                 *strides;
    /* data type of the array elements */
    t_cu_types            type;
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