/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <complex.h>
# include <stdbool.h>
# include <stdint.h>

/* mapping the function array_fill to the correct type */
# define array_fill(c, arr) _Generic((c), int64_t : _array_fill_int64,\
                                        int32_t : _array_fill_int32,\
                                        int16_t : _array_fill_int16,\
                                        int8_t : _array_fill_int8,\
                                        float : _array_fill_float,\
                                        double : _array_fill_double,\
                                        bool : _array_fill_bool,\
                                        float complex : _array_fill_cfloat,\
                                        double complex : _array_fill_cdouble)(c, arr)

typedef enum e_slice_type { ELEMENT, RANGE } t_slice_type;

typedef struct  s_slice
{
    int32_t             start;
    int32_t             end;
    int32_t             step;
    t_slice_type   type;
}               t_slice;

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

/*
** Map e_types enum to numpy NPY_TYPES enum
** ref: numpy_repo: numpy/numpy/core/include/numpy/ndarraytypes.h
*/
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

typedef enum e_order
{
    order_f,
    order_c,
} t_order;

typedef struct  s_ndarray
{
    /* raw data buffer*/
    union {
            void            *raw_data;
            int8_t          *nd_int8;
            int16_t         *nd_int16;
            int32_t         *nd_int32;
            int64_t         *nd_int64;
            float           *nd_float;
            double          *nd_double;
            bool            *nd_bool;
            double complex  *nd_cdouble;
            float  complex  *nd_cfloat;
            };
    /* number of dimensions */
    int32_t                 nd;
    /* shape 'size of each dimension' */
    int64_t                 *shape;
    /* strides 'number of elements to skip to get the next element' */
    int64_t                 *strides;
    /* type of the array elements */
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
}               t_ndarray;

/* functions prototypes */

/* allocations */
void        stack_array_init(t_ndarray *arr);
t_ndarray   array_create(int32_t nd, int64_t *shape,
        t_types type, bool is_view, t_order order);
void        _array_fill_int8(int8_t c, t_ndarray arr);
void        _array_fill_int16(int16_t c, t_ndarray arr);
void        _array_fill_int32(int32_t c, t_ndarray arr);
void        _array_fill_int64(int64_t c, t_ndarray arr);
void        _array_fill_float(float c, t_ndarray arr);
void        _array_fill_double(double c, t_ndarray arr);
void        _array_fill_bool(bool c, t_ndarray arr);
void        _array_fill_cfloat(float complex c, t_ndarray arr);
void        _array_fill_cdouble(double complex c, t_ndarray arr);

/* slicing */
                /* creating a Slice object */
t_slice new_slice(int32_t start, int32_t end, int32_t step, t_slice_type type);
                /* creating an array view */
t_ndarray   array_slicing(t_ndarray arr, int n, ...);

/* assigns */
void        alias_assign(t_ndarray *dest, t_ndarray src);
void        transpose_alias_assign(t_ndarray *dest, t_ndarray src);

/* free */
int32_t         free_array(t_ndarray* dump);
int32_t         free_pointer(t_ndarray* dump);

/* indexing */
int64_t         get_index(t_ndarray arr, ...);

/* data converting between numpy and ndarray */
int64_t     *numpy_to_ndarray_strides(int64_t *np_strides, int type_size, int nd);
int64_t     *numpy_to_ndarray_shape(int64_t *np_shape, int nd);
void print_ndarray_memory(t_ndarray nd);
/* copy data from ndarray */
void array_copy_data(t_ndarray* dest, t_ndarray src, uint32_t offset);

/* numpy sum */

int64_t            numpy_sum_bool(t_ndarray arr);
int64_t            numpy_sum_int8(t_ndarray arr);
int64_t            numpy_sum_int16(t_ndarray arr);
int64_t            numpy_sum_int32(t_ndarray arr);
int64_t            numpy_sum_int64(t_ndarray arr);
float              numpy_sum_float32(t_ndarray arr);
double             numpy_sum_float64(t_ndarray arr);
float complex      numpy_sum_complex64(t_ndarray arr);
double complex     numpy_sum_complex128(t_ndarray arr);

#endif
