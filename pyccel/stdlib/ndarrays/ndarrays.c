/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

# include "ndarrays.h"
# include <string.h>
# include <stdarg.h>
# include <stdlib.h>
# include <stdio.h>

/*
 * Takes an array, and prints its elements the laid out in memory (similar to ravel)
*/

void print_ndarray_memory(t_ndarray nd)
{
    int i = 0;
    while (i < nd.length)
    {
        switch (nd.type)
        {
            case nd_int8:
                printf("[%hhd]", nd.nd_int8[i]);
                break;
            case nd_int16:
                printf("[%hd]", nd.nd_int16[i]);
                break;
            case nd_int32:
                printf("[%d]", nd.nd_int32[i]);
                break;
            case nd_int64:
                printf("[%ld]", nd.nd_int64[i]);
                break;
            case nd_float:
                printf("[%f]", nd.nd_float[i]);
                break;
            case nd_double:
                printf("[%lf]", nd.nd_double[i]);
                break;
            case nd_bool:
                printf("[%d]", nd.nd_bool[i]);
                break;
            case nd_cfloat:
            {
                double real = creal(nd.nd_cfloat[i]);
                double imag = cimag(nd.nd_cfloat[i]);
                printf("[%lf%s%lfj]", real, imag >= 0 ? "+" : "", imag);
                break;
            }
            case nd_cdouble:
            {
                double real = creal(nd.nd_cdouble[i]);
                double imag = cimag(nd.nd_cdouble[i]);
                printf("[%lf%s%lfj]", real, imag >= 0 ? "+" : "", imag);
                break;
            }
            default:
                return;
        }
        ++i;
    }
    printf("\n");
    return;
}

/*
** allocation
*/

t_ndarray   array_create(int32_t nd, int64_t *shape,
        enum e_types type, bool is_view, enum e_order order)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    arr.order = order;
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
    if (arr.order == order_c)
    {
        for (int32_t i = 0; i < arr.nd; i++)
        {
            arr.strides[i] = 1;
            for (int32_t j = i + 1; j < arr.nd; j++)
                arr.strides[i] *= arr.shape[j];
        }
    }
    else if (arr.order == order_f)
    {
        for (int32_t i = 0; i < arr.nd; i++)
        {
            arr.strides[i] = 1;
            for (int32_t j = 0; j < i; j++)
                arr.strides[i] *= arr.shape[j];
        }
    }
    if (!is_view)
        arr.raw_data = malloc(arr.buffer_size);
    arr.current_length = 0;
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

t_slice new_slice(int32_t start, int32_t end, int32_t step)
{
    t_slice slice;

    slice.start = start;
    slice.end = end;
    slice.step = step;
    return (slice);
}

t_ndarray array_slicing(t_ndarray arr, int n, ...)
{
    t_ndarray view;
    va_list  va;
    t_slice slice;
    int32_t start = 0;

    view.nd = n;
    view.type = arr.type;
    view.type_size = arr.type_size;
    view.shape = malloc(sizeof(int64_t) * arr.nd);
    view.strides = malloc(sizeof(int64_t) * arr.nd);
    memcpy(view.strides, arr.strides, sizeof(int64_t) * arr.nd);
    view.is_view = true;
    va_start(va, n);
    for (int32_t i = 0; i < arr.nd ; i++)
    {
        slice = va_arg(va, t_slice);
        view.shape[i] = (slice.end - slice.start + (slice.step - 1)) / slice.step; // we need to round up the shape
        start += slice.start * arr.strides[i];
        view.strides[i] *= slice.step;
    }
    va_end(va);
    int32_t j = arr.nd - view.nd;
    if (j)
    {
        int64_t *tmp_strides = malloc(sizeof(int32_t) * view.nd);
        int64_t *tmp_shape = malloc(sizeof(int32_t) * view.nd);
        for (int32_t i = 0; i < view.nd; i++)
        {
            tmp_strides[i] = view.strides[j];
            tmp_shape[i] = view.shape[j];
            j++;
        }
        free(view.shape);
        free(view.strides);
        view.strides = tmp_strides;
        view.shape = tmp_shape;
    }
    view.raw_data = arr.raw_data + start * arr.type_size;
    view.length = 1;
    for (int32_t i = 0; i < view.nd; i++)
            view.length *= view.shape[i];
    view.buffer_size =  view.length * view.type_size;
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
** takes an array containing the shape of an array 'shape', number of a certain dimension 'nd', and the number of the array's dimensions
** returns the stride (number of elements to jump in a dimension) of the 'nd`th dimension
*/

int get_dimension_stride(int64_t *shape, int nd, int max_nd)
{
    int product = 1;

    for (int i = nd; i < max_nd; ++i)
        product *= shape[i];
    return (product);
}

int element_index(t_ndarray arr, uint32_t element_num, int nd)
{
    if (arr.order == order_c)
        return element_num;
    if (nd == 0)
        return (0);
    if (nd == arr.nd)
        return (element_num % arr.shape[nd - 1]) * arr.strides[nd - 1] + element_index(arr, element_num, nd - 1);
    int true_index = (element_num / (get_dimension_stride(arr.shape, nd, arr.nd)));
    if (true_index >= arr.shape[nd - 1])
        true_index = true_index % arr.shape[nd - 1];
    return (true_index * arr.strides[nd - 1] + element_index(arr, element_num, nd - 1));
}

void array_copy_data(t_ndarray *dest, t_ndarray src)
{
    unsigned char *d = (unsigned char*)dest->raw_data;
    unsigned char *s = (unsigned char*)src.raw_data;

    if (dest->order == order_c && src.order == order_c)
        memcpy(d + dest->current_length * dest->type_size, s, src.buffer_size);
    else
    {
        for (uint32_t element_num = 0; element_num < src.length; ++element_num)
        {
            memcpy(d + ((element_index(*dest, element_num, dest->nd) + dest->current_length) * dest->type_size),
                s + (element_index(src, element_num, src.nd) * src.type_size), src.type_size);
        }
    }
    dest->current_length += src.length;
}
