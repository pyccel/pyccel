/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

# include "ndarrays.h"
# include <string.h>
# include <stdarg.h>
# include <stdlib.h>

/*
** allocation
*/

t_ndarray   array_create(int32_t nd, int64_t *shape,
        enum e_types type, bool is_view)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    switch (type)
    {
        case nd_int8:
            arr.type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr.type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr.type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr.type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr.type_size = sizeof(float);
            break;
        case nd_double:
            arr.type_size = sizeof(double);
            break;
        case nd_bool:
            arr.type_size = sizeof(bool);
            break;
        case nd_cfloat:
            arr.type_size = sizeof(float complex);
            break;
        case nd_cdouble:
            arr.type_size = sizeof(double complex);
            break;
    }
    arr.is_view = is_view;
    arr.length = 1;
    arr.shape = malloc(arr.nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    arr.strides = malloc(nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int32_t j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    if (!is_view)
        arr.raw_data = malloc(arr.buffer_size);
    return (arr);
}

void    stack_array_init(t_ndarray *arr)
{
    switch (arr->type)
    {
        case nd_int8:
            arr->type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr->type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr->type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr->type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr->type_size = sizeof(float);
            break;
        case nd_double:
            arr->type_size = sizeof(double);
            break;
        case nd_bool:
            arr->type_size = sizeof(bool);
            break;
        case nd_cfloat:
            arr->type_size = sizeof(float complex);
            break;
        case nd_cdouble:
            arr->type_size = sizeof(double complex);
            break;
    }
    arr->length = 1;
    for (int32_t i = 0; i < arr->nd; i++)
        arr->length *= arr->shape[i];
    arr->buffer_size = arr->length * arr->type_size;
    for (int32_t i = 0; i < arr->nd; i++)
    {
        arr->strides[i] = 1;
        for (int32_t j = i + 1; j < arr->nd; j++)
            arr->strides[i] *= arr->shape[j];
    }
}

void   _array_fill_int8(int8_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int8[i] = c;
}

void   _array_fill_int16(int16_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int16[i] = c;
}

void   _array_fill_int32(int32_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int32[i] = c;
}

void   _array_fill_int64(int64_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int64[i] = c;
}

void   _array_fill_bool(bool c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_bool[i] = c;
}

void   _array_fill_float(float c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_float[i] = c;
}

void   _array_fill_double(double c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_double[i] = c;
}

void   _array_fill_cfloat(float complex c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_cfloat[i] = c;
}


void   _array_fill_cdouble(double complex c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_cdouble[i] = c;
}

/*
** deallocation
*/

int32_t free_array(t_ndarray arr)
{
    if (arr.shape == NULL)
        return (0);
    free(arr.raw_data);
    arr.raw_data = NULL;
    free(arr.shape);
    arr.shape = NULL;
    free(arr.strides);
    arr.strides = NULL;
    return (1);
}


int32_t free_pointer(t_ndarray arr)
{
    if (arr.is_view == false || arr.shape == NULL)
        return (0);
    free(arr.shape);
    arr.shape = NULL;
    free(arr.strides);
    arr.strides = NULL;
    return (1);
}

/*
** slices
*/

t_slice new_slice(int32_t start, int32_t end, int32_t step, enum e_slice_type type)
{
    t_slice slice;

    slice.start = start;
    slice.end = end;
    slice.step = step;
    slice.type = type;
    return (slice);
}

t_ndarray array_slicing(t_ndarray arr, int n, ...)
{
    t_ndarray view;
    va_list  va;
    t_slice slice;
    int32_t start = 0;
    int32_t j;

    view.nd = n;
    view.type = arr.type;
    view.type_size = arr.type_size;
    view.shape = malloc(sizeof(int64_t) * view.nd);
    view.strides = malloc(sizeof(int64_t) * view.nd);
    view.is_view = true;

    va_start(va, n);
    j = 0;
    for (int32_t i = 0; i < arr.nd; i++)
    {
        slice = va_arg(va, t_slice);
        if (slice.type == RANGE)
        {
            view.shape[j] = (slice.end - slice.start + (slice.step - 1)) / slice.step;
            view.strides[j] = arr.strides[i] * slice.step;
            j++;
        }
        start += slice.start * arr.strides[i];
    }
    va_end(va);

    view.raw_data = arr.raw_data + start * arr.type_size;
    view.length = 1;
    for (int32_t i = 0; i < view.nd; i++)
            view.length *= view.shape[i];
    return (view);
}

/*
** assigns
*/

void        alias_assign(t_ndarray *dest, t_ndarray src)
{
    /*
    ** copy src to dest
    ** allocate new memory for shape and strides
    ** setting is_view to true for the garbage collector to deallocate
    */

    *dest = src;
    dest->shape = malloc(sizeof(int64_t) * src.nd);
    memcpy(dest->shape, src.shape, sizeof(int64_t) * src.nd);
    dest->strides = malloc(sizeof(int64_t) * src.nd);
    memcpy(dest->strides, src.strides, sizeof(int64_t) * src.nd);
    dest->is_view = true;
}

void        transpose_alias_assign(t_ndarray *dest, t_ndarray src)
{
    /*
    ** copy src to dest
    ** allocate new memory for shape and strides
    ** setting is_view to true for the garbage collector to deallocate
    */

    *dest = src;
    dest->shape = malloc(sizeof(int64_t) * src.nd);
    dest->strides = malloc(sizeof(int64_t) * src.nd);
    for (int32_t i = 0; i < src.nd; i++)
    {
        dest->shape[i] = src.shape[src.nd-1-i];
        dest->strides[i] = src.strides[src.nd-1-i];
    }
    dest->is_view = true;
}

/*
** indexing
*/

int64_t     get_index(t_ndarray arr, ...)
{
    va_list va;
    int32_t index;

    va_start(va, arr);
    index = 0;
    for (int32_t i = 0; i < arr.nd; i++)
    {
        index += va_arg(va, int64_t) * arr.strides[i];
    }
    va_end(va);
    return (index);
}

/*
** convert numpy strides to nd_array strides, and return it in a new array, to
** avoid the problem of different implementations of strides in numpy and ndarray.
*/
int64_t     *numpy_to_ndarray_strides(int64_t *np_strides, int type_size, int nd)
{
    int64_t *ndarray_strides;

    ndarray_strides = (int64_t*)malloc(sizeof(int64_t) * nd);
    for (int i = 0; i < nd; i++)
        ndarray_strides[i] = np_strides[i] / type_size;
    return ndarray_strides;

}

/*
** copy numpy shape to nd_array shape, and return it in a new array, to
** avoid the problem of variation of system architecture because numpy shape
** is not saved in fixed length type.
*/
int64_t     *numpy_to_ndarray_shape(int64_t *np_shape, int nd)
{
    int64_t *nd_shape;

    nd_shape = (int64_t*)malloc(sizeof(int64_t) * nd);
    for (int i = 0; i < nd; i++)
        nd_shape[i] = np_shape[i];
    return nd_shape;

}

/*
** sum of ndarray
*/

static int64_t     get_index_from_array(t_ndarray arr, int64_t *nd_indices)
{
    /*
    ** returns the one dimentional index equivalent to
    ** the indices in each dimension stored in nd_indices
    */
    int64_t idx = 0;
    for (int64_t dim = 0; dim<arr.nd; ++dim)
    {
        idx += arr.strides[dim] * (nd_indices[dim]);
    }
    return idx;
}

/*
** Calculate the sum of a numpy array of bools by
** looping over the length of the array and computing
** the n-dimensional indices of each element
**
** Example:
**      For a two dimentional array of shape (2, 3),
**  nd_indices is initialized to be [0, 0],
**  the main loop will run for 6 iterations
**  each iteration increments nd_indices[0] by 1, then
**  a carry is performed when nd_indices[i] is equal to shape[i]
**  iteration 0:
**      nd_indices = [0, 0] -> no carry
**  iteration 1:
**      nd_indices = [1, 0] -> no carry
**  iteration 2:
**      nd_indices = [2, 0] -> nd_indices[0] == shape[0]
**                          -> carry -> [0, 1]
**  iteration 3:
**      nd_indices = [1, 1] -> no carry
**  iteration 4:
**      nd_indices = [2, 1] -> nd_indices[0] == shape[0]
**                          -> carry -> [0, 2]
**  iteration 5:
**      nd_indices = [1, 2] -> no carry
*/
#define NUMPY_SUM_(NAME, TYPE, CTYPE) \
    TYPE numpy_sum_##NAME(t_ndarray arr) \
    { \
        int64_t nd_indices[arr.nd]; \
        memset(nd_indices, 0, sizeof(int64_t) * arr.nd); \
        TYPE output = 0; \
        for (int32_t i = 0; i < arr.length; i++) \
        { \
            output += arr.nd_##CTYPE[get_index_from_array(arr, nd_indices)]; \
            nd_indices[0]++; \
            for (int32_t j = 0; j < arr.nd - 1; j++) \
                if (nd_indices[j] == arr.shape[j]) \
                { \
                    nd_indices[j] = 0; \
                    nd_indices[j + 1]++; \
                } \
        } \
        return output; \
    }

NUMPY_SUM_(bool, int64_t, bool)
NUMPY_SUM_(int8, int64_t, int8)
NUMPY_SUM_(int16, int64_t, int16)
NUMPY_SUM_(int32, int64_t, int32)
NUMPY_SUM_(int64, int64_t, int64)
NUMPY_SUM_(float32, float, float)
NUMPY_SUM_(float64, double, double)
NUMPY_SUM_(complex64, float complex, cfloat)
NUMPY_SUM_(complex128, double complex, cdouble)
